from __future__ import annotations
"""Validation-only comparison of the area-conditioned K=8 weight decoder against
the existing GLOBAL weight decoder, on identical validation rows, using the exact
oracle-ceiling metrics (target = pre-CBV 32-star median; oracle = best masked K=8
fit). Nothing is trained or rebuilt here.

  global decoder : B_area @ decoder(latent)
  area decoder   : B_area @ decoder(concat(latent, area_one_hot))

Reports, for each decoder: decoder-vs-oracle and decoder-vs-target Pearson (Q1,
median) + R2/RMSE, overall negative-Pearson rate, and failure % by camera / CCD /
subregion / area -- then explicitly diffs against the frozen baseline numbers.

    python -m src.instrument_v2.eval_cbv_area_conditioned
"""

import json
import os

import numpy as np
import pandas as pd
import torch

from src.instrument_v2.area_commonmode_dataset import Sector14GroupStatDataset, ensure_area_column
from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.regional_group_teacher import state_hash
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_sector14_jepa import git_commit
from src.instrument_v2.decode_single_star_k8 import (
    SEED, GROUP_SIZE, CBV_RANK, MIN_VALID_STARS, DEVICE,
    S14_DATA, SPLIT_DIR, BASE_ART_DIR, GROUP_ART_DIR, STAGE_B_CKPT,
    build_decoder, deterministic_area_rows, decode, masked_metrics,
    area_index_map, area_one_hot,
)
from src.instrument_v2.eval_cbv_oracle_ceiling import (
    load_bases_npz, reference_median, oracle_weights, _quart,
)

GLOBAL_DECODER = os.environ.get(
    "GLOBAL_DECODER", os.path.join(GROUP_ART_DIR, "single_star_weight_decode", "decoder.pth"))
AREA_DECODER = os.environ.get(
    "AREA_DECODER", os.path.join(GROUP_ART_DIR, "single_star_weight_decode_area_conditioned", "decoder.pth"))
BASES_NPZ = os.environ.get(
    "CBV_BASES_NPZ",
    os.path.join(GROUP_ART_DIR, "area_group_cbv_r8_g32_mv16_q16437_tglc0_1607e67857039a07.npz"))
OUT_DIR = os.environ.get(
    "OUT_DIR", os.path.join(GROUP_ART_DIR, "single_star_weight_decode_area_conditioned"))

# frozen baseline (global decoder, from the oracle-ceiling + failure-correlation runs)
BASELINE = {"decoder_vs_oracle_pearson_median": 0.815, "decoder_vs_oracle_pearson_q1": -0.404,
            "decoder_vs_target_pearson_median": 0.757, "decoder_vs_target_pearson_q1": -0.366,
            "neg_pearson_rate_pct": 31.139, "camera1_failure_pct": 63.269}


def _detector(area):
    a = int(area)
    return a // 100, (a // 10) % 10, a % 10


def _summarize(df, tag):
    """Q1/median for the two Pearson comparisons + R2/RMSE + neg-rate + failure
    rates by detector level, for one decoder's per-example rows."""
    out = {
        "decoder_vs_oracle_pearson": {"q1": _quart(df[f"{tag}_oracle_pearson"])["q1"],
                                      "median": _quart(df[f"{tag}_oracle_pearson"])["median"]},
        "decoder_vs_target_pearson": {"q1": _quart(df[f"{tag}_target_pearson"])["q1"],
                                      "median": _quart(df[f"{tag}_target_pearson"])["median"]},
        "decoder_vs_target_r2": {"q1": _quart(df[f"{tag}_target_r2"])["q1"],
                                 "median": _quart(df[f"{tag}_target_r2"])["median"]},
        "decoder_vs_target_rmse": {"q1": _quart(df[f"{tag}_target_rmse"])["q1"],
                                   "median": _quart(df[f"{tag}_target_rmse"])["median"]},
        "neg_pearson_rate_pct": round(100.0 * (df[f"{tag}_target_pearson"] < 0).mean(), 3),
        "failure_pct_by": {},
    }
    fail = (df[f"{tag}_target_pearson"] < 0).astype(int)
    for col in ("camera", "ccd", "subregion", "area"):
        g = fail.groupby(df[col]).agg(["size", "mean"])
        out["failure_pct_by"][col] = {int(k): {"n": int(r["size"]), "failure_pct": round(100 * r["mean"], 3)}
                                      for k, r in g.iterrows()}
    return out


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"git commit: {git_commit()}", flush=True)
    print(f"global decoder: {GLOBAL_DECODER}", flush=True)
    print(f"area decoder  : {AREA_DECODER}", flush=True)

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

    # --- frozen instrument JEPA -----------------------------------------------
    model = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=256,
                                       n_layers=4, readout="mean",
                                       predictor_type="mlp").to(DEVICE)
    model.load_state_dict(torch.load(STAGE_B_CKPT, map_location=DEVICE))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    frozen = {"teacher": state_hash(model.teacher), "student": state_hash(model.student),
              "predictor": state_hash(model.predictor)}

    # --- global weight decoder (plain state_dict) -----------------------------
    global_dec = build_decoder(CBV_RANK).to(DEVICE)
    global_dec.load_state_dict(torch.load(GLOBAL_DECODER, map_location=DEVICE))
    global_dec.eval()

    # --- area-conditioned decoder (checkpoint carries the area map) -----------
    ck = torch.load(AREA_DECODER, map_location=DEVICE)
    if not (isinstance(ck, dict) and "area_to_index" in ck):
        raise RuntimeError(f"{AREA_DECODER} is not an area-conditioned checkpoint")
    a2i = {int(k): int(v) for k, v in ck["area_to_index"].items()}
    n_areas = int(ck["n_areas"])
    area_dec = build_decoder(int(ck["cbv_rank"]), in_dim=int(ck["in_dim"])).to(DEVICE)
    area_dec.load_state_dict(ck["state_dict"])
    area_dec.eval()
    # the checkpointed map must equal the deterministic sorted training-area map
    if a2i != area_index_map(sorted(bases)):
        raise RuntimeError("checkpointed area map != deterministic sorted training-area map")
    print(f"area map: {n_areas} areas, decoder in_dim={ck['in_dim']}", flush=True)

    # --- per-example: identical rows scored by both decoders ------------------
    val_area_rows = deterministic_area_rows(val_ds)
    rows = []
    for i in range(len(val_ds.X)):
        target, valid = reference_median(val_ds, val_area_rows, i, bases)
        if target is None:
            continue
        a = int(val_ds.areas[i])
        B = np.asarray(bases[a], dtype=np.float64)
        oracle = (B @ oracle_weights(B, target, valid)).astype(np.float32)
        g_rec = decode(model, global_dec, val_ds.X[i], val_ds.M[i], bases[a], None)
        a_rec = decode(model, area_dec, val_ds.X[i], val_ds.M[i], bases[a], area_one_hot(a2i, a))
        cam, ccd, sub = _detector(a)
        rec = {"tic": str(val_ds.tics[i]), "area": a, "camera": cam, "ccd": ccd, "subregion": sub}
        for tag, rc in (("global", g_rec), ("area", a_rec)):
            oc, orm, or2 = masked_metrics(rc, oracle, valid)     # decoder vs oracle
            tc, trm, tr2 = masked_metrics(rc, target, valid)     # decoder vs pre-CBV target
            rec[f"{tag}_oracle_pearson"] = oc
            rec[f"{tag}_target_pearson"] = tc
            rec[f"{tag}_target_r2"] = tr2
            rec[f"{tag}_target_rmse"] = trm
        if all(np.isfinite(rec[k]) for k in rec if k.endswith(("pearson", "r2", "rmse"))):
            rows.append(rec)

    assert state_hash(model.teacher) == frozen["teacher"], "teacher changed"
    assert state_hash(model.student) == frozen["student"], "student changed"
    assert state_hash(model.predictor) == frozen["predictor"], "predictor changed"

    pe = pd.DataFrame(rows)
    if pe.empty:
        raise RuntimeError("no scorable validation examples")
    pe.to_csv(os.path.join(OUT_DIR, "per_example_metrics_area_conditioned.csv"), index=False)

    global_sum = _summarize(pe, "global")
    area_sum = _summarize(pe, "area")

    def _delta(area_v, base_v):
        return None if base_v is None else round(area_v - base_v, 4)

    vs_baseline = {
        "decoder_vs_oracle_pearson_median": _delta(area_sum["decoder_vs_oracle_pearson"]["median"],
                                                   BASELINE["decoder_vs_oracle_pearson_median"]),
        "decoder_vs_oracle_pearson_q1": _delta(area_sum["decoder_vs_oracle_pearson"]["q1"],
                                               BASELINE["decoder_vs_oracle_pearson_q1"]),
        "decoder_vs_target_pearson_median": _delta(area_sum["decoder_vs_target_pearson"]["median"],
                                                   BASELINE["decoder_vs_target_pearson_median"]),
        "decoder_vs_target_pearson_q1": _delta(area_sum["decoder_vs_target_pearson"]["q1"],
                                               BASELINE["decoder_vs_target_pearson_q1"]),
        "neg_pearson_rate_pct": _delta(area_sum["neg_pearson_rate_pct"], BASELINE["neg_pearson_rate_pct"]),
        "camera1_failure_pct": _delta(
            area_sum["failure_pct_by"]["camera"].get(1, {}).get("failure_pct"),
            BASELINE["camera1_failure_pct"]),
    }

    report = {
        "n_val_examples": int(len(pe)),
        "global_decoder": global_sum,
        "area_conditioned_decoder": area_sum,
        "baseline_reference": BASELINE,
        "area_conditioned_minus_baseline": vs_baseline,
        "n_areas": n_areas,
        "frozen_hashes_unchanged": True,
        "identical_rows_both_decoders": True,
        "paths": {"stage_b_ckpt": STAGE_B_CKPT, "global_decoder": GLOBAL_DECODER,
                  "area_decoder": AREA_DECODER, "bases_npz": BASES_NPZ},
        "git_commit": git_commit(),
    }
    with open(os.path.join(OUT_DIR, "area_conditioned_comparison.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    def _line(tag, s):
        print(f"  {tag:<16} dec-vs-oracle med {s['decoder_vs_oracle_pearson']['median']:.3f} "
              f"Q1 {s['decoder_vs_oracle_pearson']['q1']:.3f} | dec-vs-target med "
              f"{s['decoder_vs_target_pearson']['median']:.3f} Q1 {s['decoder_vs_target_pearson']['q1']:.3f} "
              f"| neg {s['neg_pearson_rate_pct']:.2f}% | cam1 "
              f"{s['failure_pct_by']['camera'].get(1, {}).get('failure_pct')}%", flush=True)
    print(f"\nvalidation examples scored (both decoders, identical rows): {len(pe)}", flush=True)
    _line("global (this run)", global_sum)
    _line("area-conditioned", area_sum)
    print("\nbaseline (frozen):  dec-vs-oracle med 0.815 Q1 -0.404 | dec-vs-target med 0.757 "
          "Q1 -0.366 | neg 31.139% | cam1 63.269%", flush=True)
    print(f"area-conditioned MINUS baseline: {json.dumps(vs_baseline)}", flush=True)
    print(f"wrote per_example_metrics_area_conditioned.csv + area_conditioned_comparison.json to {OUT_DIR}",
          flush=True)


if __name__ == "__main__":
    main()
