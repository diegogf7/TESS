from __future__ import annotations
"""Validation-only ORACLE CEILING test for the K=8 CBV-weight decoder.

Nothing is trained or rebuilt. For each validation star it reuses the EXISTING
pipeline from decode_single_star_k8 (same split, masks, preprocessing, area IDs,
K=8 bases, frozen instrument JEPA + trained weight decoder) and isolates WHERE
performance is lost by comparing three things against the 32-star median target
taken BEFORE the CBV projection:

  target      = 32-star cadence-wise median (group_statistics), pre-CBV
  oracle      = B_area @ argmin_w ||mask * (target - B_area @ w)||^2   (best K=8)
  decoder     = B_area @ (weights predicted by the frozen decoder)

  1. oracle  vs target   -> can 8 fixed components even represent the target? (basis capacity)
  2. decoder vs oracle   -> given the ceiling, how good is the latent/decoder?
  3. decoder vs target   -> end-to-end performance

The oracle is NOT compared to the existing ridge K=8 reconstruction (that would
be near-perfect by construction); it is compared to the pre-projection median.

    python -m src.instrument_v2.eval_cbv_oracle_ceiling
"""

import json
import os

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.instrument_v2.area_commonmode_dataset import (
    Sector14GroupStatDataset, ensure_area_column, group_statistics,
)
from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.regional_group_teacher import state_hash
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_sector14_jepa import git_commit

# Reuse config + validation/reconstruction code verbatim.
from src.instrument_v2.decode_single_star_k8 import (
    SEED, GROUP_SIZE, CBV_RANK, MIN_VALID_STARS, DEVICE,
    S14_DATA, SPLIT_DIR, BASE_ART_DIR, GROUP_ART_DIR, STAGE_B_CKPT,
    build_decoder, deterministic_area_rows, decode, masked_metrics,
)

DECODER_CKPT = os.environ.get(
    "DECODER_CKPT", os.path.join(GROUP_ART_DIR, "single_star_weight_decode", "decoder.pth"))
BASES_NPZ = os.environ.get(
    "CBV_BASES_NPZ",
    os.path.join(GROUP_ART_DIR,
                 "area_group_cbv_r8_g32_mv16_q16437_tglc0_1607e67857039a07.npz"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(GROUP_ART_DIR, "oracle_ceiling"))

COMPARISONS = ("oracle_vs_target", "decoder_vs_oracle", "decoder_vs_target")
METRICS = ("pearson", "r2", "rmse")


def load_bases_npz(path):
    if not os.path.exists(path):
        raise RuntimeError(f"missing CBV bases npz: {path}")
    d = np.load(path, allow_pickle=True)
    bases = {int(a): d[f"B_{int(a)}"] for a in d["areas"]}
    for a, B in bases.items():
        if B.shape != (1024, CBV_RANK):
            raise RuntimeError(f"area {a} basis shape {B.shape} != (1024, {CBV_RANK})")
    return bases


def reference_median(ds, area_rows, i, bases):
    """The 32-star cadence-wise median target BEFORE CBV projection -- identical
    neighbor selection to decode_single_star_k8.reference_target, but returns the
    raw median (not the ridge reconstruction)."""
    a = int(ds.areas[i])
    if a not in bases:
        return None, None
    neigh = [r for r in area_rows.get(a, []) if r != i]
    if len(neigh) < GROUP_SIZE:
        return None, None
    group = neigh[:GROUP_SIZE]
    median, _log_mad, valid, _ = group_statistics(ds.X[group], ds.M[group], ds.min_valid)
    return median, valid


def oracle_weights(B, target, valid):
    """argmin_w ||mask * (target - B w)||^2 with a binary mask = masked least
    squares over the observed cadences (the K=8 ceiling; no ridge penalty)."""
    obs = valid > 0
    Bv = np.asarray(B[obs], dtype=np.float64)
    tv = np.asarray(target[obs], dtype=np.float64)
    w, *_ = np.linalg.lstsq(Bv, tv, rcond=None)
    return w


def _target_rms(target, valid):
    obs = valid > 0
    t = np.asarray(target[obs], dtype=np.float64)
    return float(np.sqrt(np.mean((t - t.mean()) ** 2))) if t.size else np.nan


def _quart(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"q1": np.nan, "median": np.nan, "q3": np.nan, "n": 0}
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    return {"q1": float(q1), "median": float(med), "q3": float(q3), "n": int(a.size)}


def per_area_quartiles(per_example):
    rows = []
    for area, g in per_example.groupby("area"):
        row = {"area": int(area),
               "n_train_stars": int(g["n_train_stars"].iloc[0]),
               "n_cbv_groups": int(g["n_cbv_groups"].iloc[0]),
               "n_val_examples": int(len(g))}
        for c in COMPARISONS:
            for m in METRICS:
                q = _quart(g[f"{c}_{m}"].to_numpy())
                row[f"{c}_{m}_q1"] = q["q1"]
                row[f"{c}_{m}_median"] = q["median"]
                row[f"{c}_{m}_q3"] = q["q3"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_train_stars").reset_index(drop=True)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    print("================ resolved configuration ================", flush=True)
    print(f"  git commit    : {git_commit()}", flush=True)
    print(f"  STAGE_B_CKPT  : {STAGE_B_CKPT}", flush=True)
    print(f"  DECODER_CKPT  : {DECODER_CKPT}", flush=True)
    print(f"  BASES_NPZ     : {BASES_NPZ}", flush=True)
    print(f"  OUT_DIR       : {OUT_DIR}", flush=True)
    print("========================================================", flush=True)

    # --- same data/split/dataset setup as decode_single_star_k8.main ---------
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
    n_train_by_area = {int(a): len(rows)
                       for a, rows in deterministic_area_rows(train_ds).items()}

    # --- frozen instrument JEPA + frozen K=8 weight decoder ------------------
    model = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=256,
                                       n_layers=4, readout="mean",
                                       predictor_type="mlp").to(DEVICE)
    model.load_state_dict(torch.load(STAGE_B_CKPT, map_location=DEVICE))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    decoder = build_decoder(CBV_RANK).to(DEVICE)
    decoder.load_state_dict(torch.load(DECODER_CKPT, map_location=DEVICE))
    decoder.eval()
    hashes = {"teacher": state_hash(model.teacher), "student": state_hash(model.student),
              "predictor": state_hash(model.predictor), "decoder": state_hash(decoder)}
    print(f"model hashes: teacher {hashes['teacher'][:12]} decoder {hashes['decoder'][:12]}", flush=True)

    # --- per validation example: target / oracle / decoder -------------------
    val_area_rows = deterministic_area_rows(val_ds)
    records = []
    for i in range(len(val_ds.X)):
        target, valid = reference_median(val_ds, val_area_rows, i, bases)
        if target is None:
            continue
        a = int(val_ds.areas[i])
        B = np.asarray(bases[a], dtype=np.float64)
        oracle = (B @ oracle_weights(B, target, valid)).astype(np.float32)   # best K=8
        decoder_recon = decode(model, decoder, val_ds.X[i], val_ds.M[i], bases[a])  # B @ w_pred

        c1 = masked_metrics(oracle, target, valid)                # oracle vs pre-CBV target
        c2 = masked_metrics(decoder_recon, oracle, valid)         # decoder vs oracle
        c3 = masked_metrics(decoder_recon, target, valid)         # decoder vs pre-CBV target
        if not all(np.all(np.isfinite(c)) for c in (c1, c2, c3)):
            continue
        n_train = n_train_by_area.get(a, 0)
        rec = {"tic": str(val_ds.tics[i]), "area": a,
               "n_train_stars": n_train, "n_cbv_groups": n_train // GROUP_SIZE,
               "target_rms": _target_rms(target, valid),
               "valid_frac": float((valid > 0).mean())}
        for name, (corr, rmse, r2) in zip(COMPARISONS, (c1, c2, c3)):
            rec[f"{name}_pearson"] = corr
            rec[f"{name}_r2"] = r2
            rec[f"{name}_rmse"] = rmse
        records.append(rec)

    assert state_hash(model.teacher) == hashes["teacher"], "teacher changed"
    assert state_hash(model.student) == hashes["student"], "student changed"
    assert state_hash(model.predictor) == hashes["predictor"], "predictor changed"
    assert state_hash(decoder) == hashes["decoder"], "decoder changed"

    pe = pd.DataFrame.from_records(records)
    if pe.empty:
        raise RuntimeError("no scorable validation examples")
    pe.to_csv(os.path.join(OUT_DIR, "per_example_metrics.csv"), index=False)
    per_area = per_area_quartiles(pe)
    per_area.to_csv(os.path.join(OUT_DIR, "per_area_metrics.csv"), index=False)

    # --- global quartiles ----------------------------------------------------
    global_q = {c: {m: _quart(pe[f"{c}_{m}"].to_numpy()) for m in METRICS} for c in COMPARISONS}

    # --- low- vs high-target-RMS end-to-end Pearson (weak-signal instability) -
    rms_med = float(pe["target_rms"].median())
    lo = pe[pe["target_rms"] <= rms_med]["decoder_vs_target_pearson"]
    hi = pe[pe["target_rms"] > rms_med]["decoder_vs_target_pearson"]
    e2e_lo, e2e_hi = float(np.nanmedian(lo)), float(np.nanmedian(hi))

    oracle_med = global_q["oracle_vs_target"]["pearson"]["median"]
    dec_oracle_med = global_q["decoder_vs_oracle"]["pearson"]["median"]
    e2e_med = global_q["decoder_vs_target"]["pearson"]["median"]

    if oracle_med < 0.5:
        conclusion = (f"K=8 BASIS INSUFFICIENT: even the best-possible 8 weights fit the "
                      f"pre-CBV median only at median Pearson {oracle_med:.3f}.")
    elif dec_oracle_med < 0.7:
        conclusion = (f"LATENT/DECODER FAILING: basis capacity is fine (oracle vs target "
                      f"median Pearson {oracle_med:.3f}) but decoder vs oracle is only "
                      f"{dec_oracle_med:.3f}.")
    elif (e2e_hi - e2e_lo) > 0.3 and e2e_hi > 0.6:
        conclusion = (f"PEARSON UNSTABLE ON WEAK SIGNALS: end-to-end median Pearson is "
                      f"{e2e_lo:.3f} on low-RMS targets vs {e2e_hi:.3f} on high-RMS "
                      f"(oracle {oracle_med:.3f}, decoder-vs-oracle {dec_oracle_med:.3f}).")
    else:
        conclusion = (f"No single failure mode dominates: oracle {oracle_med:.3f}, "
                      f"decoder-vs-oracle {dec_oracle_med:.3f}, end-to-end {e2e_med:.3f}, "
                      f"low/high-RMS end-to-end {e2e_lo:.3f}/{e2e_hi:.3f}.")

    # --- three-panel figure --------------------------------------------------
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(19, 6))
    # 1) per-example oracle Pearson vs decoder (end-to-end) Pearson
    a1.scatter(pe["oracle_vs_target_pearson"], pe["decoder_vs_target_pearson"],
               s=5, alpha=0.25, color="tab:blue", edgecolor="none")
    a1.plot([-1, 1], [-1, 1], color="0.5", lw=0.8, ls="--")
    a1.set_xlabel("oracle Pearson (vs pre-CBV target)")
    a1.set_ylabel("decoder Pearson (vs pre-CBV target)")
    a1.set_title("per-example: oracle vs decoder Pearson", fontsize=9)
    # 2) per-area oracle Q1 vs decoder Q1 (Pearson)
    a2.scatter(per_area["oracle_vs_target_pearson_q1"], per_area["decoder_vs_target_pearson_q1"],
               s=8 + 4 * per_area["n_val_examples"], alpha=0.6, color="tab:green", edgecolor="none")
    a2.plot([-1, 1], [-1, 1], color="0.5", lw=0.8, ls="--")
    a2.set_xlabel("per-area oracle Q1 Pearson")
    a2.set_ylabel("per-area decoder Q1 Pearson")
    a2.set_title("per-area Q1: oracle vs decoder (size ~ #val)", fontsize=9)
    # 3) end-to-end Pearson vs target RMS
    a3.scatter(pe["target_rms"], pe["decoder_vs_target_pearson"],
               s=5, alpha=0.25, color="tab:purple", edgecolor="none")
    a3.axhline(0.0, color="0.5", lw=0.8, ls="--")
    a3.axvline(rms_med, color="tab:red", lw=0.8, ls=":", label=f"median RMS={rms_med:.3f}")
    a3.set_xscale("log")
    a3.set_xlabel("target RMS (log)")
    a3.set_ylabel("decoder Pearson (vs pre-CBV target)")
    a3.set_title("Pearson vs target RMS (weak-signal test)", fontsize=9)
    a3.legend(fontsize=7, loc="lower right")
    fig.suptitle("CBV K=8 oracle-ceiling diagnostic (validation only)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(OUT_DIR, "oracle_ceiling.png"), dpi=130)
    plt.close(fig)

    report = {
        "conclusion": conclusion,
        "global_quartiles": global_q,
        "weak_signal_split": {"target_rms_median": rms_med,
                              "e2e_pearson_median_low_rms": e2e_lo,
                              "e2e_pearson_median_high_rms": e2e_hi},
        "n_val_examples_scored": int(len(pe)), "n_areas": int(per_area.shape[0]),
        "group_size": GROUP_SIZE, "cbv_rank": CBV_RANK, "min_valid": MIN_VALID_STARS,
        "stage_b_ckpt": STAGE_B_CKPT, "decoder_ckpt": DECODER_CKPT, "bases_npz": BASES_NPZ,
        "model_hashes": hashes, "git_commit": git_commit(),
        "note": "oracle compared to the pre-CBV 32-star median, NOT the ridge K=8 reconstruction",
    }
    with open(os.path.join(OUT_DIR, "oracle_ceiling.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    def line(tag, c):
        q = global_q[c]
        print(f"  {tag:<22} Pearson med {q['pearson']['median']:.3f} "
              f"[Q1 {q['pearson']['q1']:.3f}, Q3 {q['pearson']['q3']:.3f}]  "
              f"R2 med {q['r2']['median']:.3f}  RMSE med {q['rmse']['median']:.3f}", flush=True)
    print(f"\nglobal quartiles (n={len(pe)} val examples, {per_area.shape[0]} areas):", flush=True)
    line("oracle vs target", "oracle_vs_target")
    line("decoder vs oracle", "decoder_vs_oracle")
    line("decoder vs target", "decoder_vs_target")
    print(f"  weak-signal: end-to-end Pearson median low-RMS {e2e_lo:.3f} | high-RMS {e2e_hi:.3f}", flush=True)
    print(f"\nCONCLUSION: {conclusion}", flush=True)
    print(f"wrote per_example_metrics.csv, per_area_metrics.csv, oracle_ceiling.png, "
          f"oracle_ceiling.json to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
