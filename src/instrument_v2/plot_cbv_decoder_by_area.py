from __future__ import annotations
"""Validation-only diagnostic: does an area's TRAINING star count explain the
low-Q1 tail of the K=8 CBV-weight decoder?

Nothing is trained or modified. For every validation star it reuses the EXISTING
decode pipeline from decode_single_star_k8 (same frozen instrument JEPA, same K=8
weight decoder, same regional CBV bases, same validation split, same
reference_target / decode / masked_metrics) and records the reconstruction
Pearson/R2/RMSE against the regional CBV target. Results are grouped by area and
correlated against training stars per area.

    python -m src.instrument_v2.plot_cbv_decoder_by_area
"""

import json
import os

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy import stats

from src.instrument_v2.area_commonmode_dataset import Sector14GroupStatDataset, ensure_area_column
from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.regional_group_teacher import state_hash
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_sector14_jepa import git_commit

# Reuse the decoder, config and validation/reconstruction code verbatim -- do NOT
# reimplement the pipeline.
from src.instrument_v2.decode_single_star_k8 import (
    SEED, GROUP_SIZE, CBV_RANK, MIN_VALID_STARS, DEVICE,
    S14_DATA, SPLIT_DIR, BASE_ART_DIR, GROUP_ART_DIR, STAGE_B_CKPT,
    build_decoder, deterministic_area_rows, reference_target, decode, masked_metrics,
)

DECODER_CKPT = os.environ.get(
    "DECODER_CKPT", os.path.join(GROUP_ART_DIR, "single_star_weight_decode", "decoder.pth"))
BASES_NPZ = os.environ.get(
    "CBV_BASES_NPZ",
    os.path.join(GROUP_ART_DIR,
                 "area_group_cbv_r8_g32_mv16_q16437_tglc0_1607e67857039a07.npz"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(GROUP_ART_DIR, "decoder_by_area"))


def load_bases_npz(path):
    """Load the EXISTING area -> (1024, 8) CBV bases; never rebuild."""
    if not os.path.exists(path):
        raise RuntimeError(f"missing CBV bases npz: {path}")
    d = np.load(path, allow_pickle=True)
    bases = {int(a): d[f"B_{int(a)}"] for a in d["areas"]}
    for a, B in bases.items():
        if B.shape != (1024, CBV_RANK):
            raise RuntimeError(f"area {a} basis shape {B.shape} != (1024, {CBV_RANK})")
    return bases


def area_quartiles(per_example):
    """Per-area Q1/median/Q3 of each metric (+ the area's train/size metadata)."""
    rows = []
    for area, g in per_example.groupby("area"):
        row = {"area": int(area),
               "n_train_stars": int(g["n_train_stars"].iloc[0]),
               "n_cbv_groups": int(g["n_cbv_groups"].iloc[0]),
               "n_val_examples": int(len(g))}
        for m in ("pearson", "r2", "rmse"):
            q1, med, q3 = np.percentile(g[m].to_numpy(), [25, 50, 75])
            row[f"{m}_q1"] = float(q1)
            row[f"{m}_median"] = float(med)
            row[f"{m}_q3"] = float(q3)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_train_stars").reset_index(drop=True)


def _panel(ax, x, y, sizes, worst_ids, area_ids, xlabel, ylabel, title_metric):
    """One scatter panel: training stars vs an area-level Q1 metric, with a
    fitted trend line, zero line, Spearman(rho, p) in the title, and the 10
    worst-area labels."""
    ax.scatter(x, y, s=sizes, alpha=0.6, color="tab:blue", edgecolor="none")
    ax.axhline(0.0, color="0.5", lw=0.8, ls="--")
    if len(x) >= 2 and np.ptp(x) > 0:
        b, a = np.polyfit(x, y, 1)                       # y = b*x + a
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, b * xs + a, color="tab:red", lw=1.2, label=f"fit slope={b:.2e}")
        ax.legend(fontsize=7, loc="lower right")
        rho, p = stats.spearmanr(x, y)
    else:
        rho, p = np.nan, np.nan
    for aid in worst_ids:                                # label the 10 worst areas
        j = np.where(area_ids == aid)[0]
        if len(j):
            k = j[0]
            ax.annotate(str(int(aid)), (x[k], y[k]), fontsize=6, color="tab:red",
                        xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.set_title(f"{title_metric}   Spearman rho={rho:.3f}  p={p:.2e}", fontsize=9)
    return float(rho), float(p)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    print("================ resolved configuration ================", flush=True)
    print(f"  git commit    : {git_commit()}", flush=True)
    print(f"  STAGE_B_CKPT  : {STAGE_B_CKPT}", flush=True)
    print(f"  DECODER_CKPT  : {DECODER_CKPT}", flush=True)
    print(f"  BASES_NPZ     : {BASES_NPZ}", flush=True)
    print(f"  G/R/MV        : {GROUP_SIZE}/{CBV_RANK}/{MIN_VALID_STARS}", flush=True)
    print(f"  OUT_DIR       : {OUT_DIR}", flush=True)
    print("========================================================", flush=True)

    # --- exact same data/split setup as decode_single_star_k8.main ------------
    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    df = ensure_area_column(df)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, train_tics)
    train_ds = Sector14GroupStatDataset(df, train_tics, t_range, "area", GROUP_SIZE,
                                        min_valid=MIN_VALID_STARS)
    val_ds = Sector14GroupStatDataset(df, val_tics, t_range, "area", GROUP_SIZE,
                                      min_valid=MIN_VALID_STARS)
    assert not (set(train_ds.tics) | set(val_ds.tics)) & test_tics, "test TIC leaked"

    bases = load_bases_npz(BASES_NPZ)

    # --- training stars per area (drives n_cbv_groups = floor(n_train/32)) ----
    train_area_rows = deterministic_area_rows(train_ds)
    n_train_by_area = {int(a): len(rows) for a, rows in train_area_rows.items()}

    # --- frozen instrument JEPA + frozen K=8 weight decoder ------------------
    model = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=256,
                                       n_layers=4, readout="mean",
                                       predictor_type="mlp").to(DEVICE)
    model.load_state_dict(torch.load(STAGE_B_CKPT, map_location=DEVICE))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    decoder = build_decoder(CBV_RANK).to(DEVICE)          # 8-weight decoder
    decoder.load_state_dict(torch.load(DECODER_CKPT, map_location=DEVICE))
    decoder.eval()
    hashes = {"teacher": state_hash(model.teacher), "student": state_hash(model.student),
              "predictor": state_hash(model.predictor), "decoder": state_hash(decoder)}
    print(f"model hashes: teacher {hashes['teacher'][:12]} student {hashes['student'][:12]} "
          f"predictor {hashes['predictor'][:12]} decoder {hashes['decoder'][:12]}", flush=True)

    # --- per validation example: reuse reference_target + decode + metrics ----
    val_area_rows = deterministic_area_rows(val_ds)
    records = []
    for i in range(len(val_ds.X)):
        ref, valid = reference_target(val_ds, val_area_rows, i, bases)
        if ref is None:
            continue                                     # area has no basis / <32 neighbors
        a = int(val_ds.areas[i])
        pred = decode(model, decoder, val_ds.X[i], val_ds.M[i], bases[a])
        c, rm, r2 = masked_metrics(pred, ref, valid)
        if not (np.isfinite(c) and np.isfinite(rm) and np.isfinite(r2)):
            continue
        n_train = n_train_by_area.get(a, 0)
        records.append({"tic": str(val_ds.tics[i]), "area": a,
                        "n_train_stars": n_train,
                        "n_cbv_groups": n_train // GROUP_SIZE,
                        "pearson": c, "r2": r2, "rmse": rm})

    # frozen models must be bit-identical after inference
    assert state_hash(model.teacher) == hashes["teacher"], "teacher changed"
    assert state_hash(model.student) == hashes["student"], "student changed"
    assert state_hash(model.predictor) == hashes["predictor"], "predictor changed"
    assert state_hash(decoder) == hashes["decoder"], "decoder changed"

    per_example = pd.DataFrame.from_records(records)
    if per_example.empty:
        raise RuntimeError("no scorable validation examples -- check bases/split")
    per_example.to_csv(os.path.join(OUT_DIR, "per_example_metrics.csv"), index=False)

    per_area = area_quartiles(per_example)
    per_area.to_csv(os.path.join(OUT_DIR, "per_area_metrics.csv"), index=False)

    # --- worst 10 areas by Q1 Pearson ---------------------------------------
    worst = per_area.sort_values("pearson_q1").head(10).reset_index(drop=True)
    worst_ids = worst["area"].to_numpy()

    # --- figure: training stars vs Q1 Pearson (L) and Q1 R2 (R) -------------
    x = per_area["n_train_stars"].to_numpy(float)
    area_ids = per_area["area"].to_numpy()
    sizes = 8.0 + 4.0 * per_area["n_val_examples"].to_numpy(float)  # point size ~ val count
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6))
    rho_p, p_p = _panel(axL, x, per_area["pearson_q1"].to_numpy(float), sizes, worst_ids,
                        area_ids, "training stars per area", "Q1 Pearson", "Q1 Pearson vs area size")
    rho_r, p_r = _panel(axR, x, per_area["r2_q1"].to_numpy(float), sizes, worst_ids,
                        area_ids, "training stars per area", "Q1 R^2", "Q1 R^2 vs area size")
    fig.suptitle("CBV K=8 weight-decoder validation performance vs area training size "
                 "(point size ~ #val examples; 10 worst areas labeled)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, "performance_vs_area_size.png"), dpi=130)
    plt.close(fig)

    corr = {
        "spearman_q1_pearson_vs_ntrain": {"rho": rho_p, "p_value": p_p},
        "spearman_q1_r2_vs_ntrain": {"rho": rho_r, "p_value": p_r},
        "n_areas": int(len(per_area)), "n_val_examples_scored": int(len(per_example)),
        "group_size": GROUP_SIZE, "cbv_rank": CBV_RANK, "min_valid": MIN_VALID_STARS,
        "stage_b_ckpt": STAGE_B_CKPT, "decoder_ckpt": DECODER_CKPT, "bases_npz": BASES_NPZ,
        "model_hashes": hashes, "git_commit": git_commit(),
        "worst10_by_q1_pearson": [
            {"area": int(r.area), "n_train_stars": int(r.n_train_stars),
             "n_cbv_groups": int(r.n_cbv_groups), "pearson_q1": round(float(r.pearson_q1), 4),
             "r2_q1": round(float(r.r2_q1), 4), "n_val_examples": int(r.n_val_examples)}
            for r in worst.itertuples()],
    }
    with open(os.path.join(OUT_DIR, "area_size_correlation.json"), "w") as fh:
        json.dump(corr, fh, indent=2)

    print("\n10 WORST AREAS BY Q1 PEARSON:", flush=True)
    print(f"{'area':>6} {'n_train':>8} {'cbv_grp':>8} {'Q1_pearson':>11} {'Q1_r2':>9}", flush=True)
    for r in worst.itertuples():
        print(f"{int(r.area):>6} {int(r.n_train_stars):>8} {int(r.n_cbv_groups):>8} "
              f"{float(r.pearson_q1):>11.4f} {float(r.r2_q1):>9.4f}", flush=True)
    print(f"\nSpearman Q1-Pearson vs n_train: rho={rho_p:.3f} p={p_p:.2e}", flush=True)
    print(f"Spearman Q1-R^2     vs n_train: rho={rho_r:.3f} p={p_r:.2e}", flush=True)
    print(f"wrote per_example_metrics.csv, per_area_metrics.csv, "
          f"performance_vs_area_size.png, area_size_correlation.json to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
