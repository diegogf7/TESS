from __future__ import annotations
"""Validation-only report for the dynamically-retrained encoder + retrained
area-head decoder. Uses the oracle-ceiling metric (target = pre-CBV 32-star
median) and reports decoder-vs-target Pearson (median, Q1), Q1 R2, negative-
Pearson rate, overall and by camera. Nothing is trained here.

    python -m src.instrument_v2.eval_dynamic_area_heads
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
    GROUP_SIZE, CBV_RANK, MIN_VALID_STARS, DEVICE,
    load_area_decoder, area_one_hot, deterministic_area_rows, decode, masked_metrics,
)
from src.instrument_v2.eval_cbv_oracle_ceiling import load_bases_npz, reference_median, oracle_weights, _quart

S14_DATA = os.environ.get("S14_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2.parquet")
SPLIT_DIR = os.environ.get("SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "dense_v2_split"))
BASE_ART_DIR = os.environ.get("BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa_dense_v2"))
GROUP_ART_DIR = os.environ.get("GROUP_ART_DIR", os.path.join("artifacts", "instrument_v2", "custom_group32_cbv8_mlp_dynamic1000_v1"))
STAGE_B_CKPT = os.environ["STAGE_B_CKPT"]                     # new dynamic encoder (full model)
AREA_DECODER = os.environ.get("AREA_DECODER", os.path.join(GROUP_ART_DIR, "single_star_weight_decode_area_heads", "decoder.pth"))
BASES_NPZ = os.environ.get("CBV_BASES_NPZ",
                           os.path.join(GROUP_ART_DIR, f"area_group_cbv_r{CBV_RANK}_g{GROUP_SIZE}_mv{MIN_VALID_STARS}_q16437_tglc0_*.npz"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(GROUP_ART_DIR, "single_star_weight_decode_area_heads"))


def _resolve_bases():
    import glob
    if "*" in BASES_NPZ:
        hits = sorted(glob.glob(BASES_NPZ))
        if len(hits) != 1:
            raise RuntimeError(f"need one bases npz matching {BASES_NPZ}, found {hits}")
        return hits[0]
    return BASES_NPZ


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    df = ensure_area_column(df)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, train_tics)
    val_ds = Sector14GroupStatDataset(df, val_tics, t_range, "area", GROUP_SIZE, min_valid=MIN_VALID_STARS)
    assert not set(val_ds.tics) & test_tics, "test TIC leaked into validation"

    bases = load_bases_npz(_resolve_bases())
    model = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=256, n_layers=4,
                                       readout="mean", predictor_type="mlp").to(DEVICE)
    model.load_state_dict(torch.load(STAGE_B_CKPT, map_location=DEVICE))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    frozen = (state_hash(model.teacher), state_hash(model.student), state_hash(model.predictor))
    dec, a2i, kind = load_area_decoder(AREA_DECODER, DEVICE)
    print(f"encoder {STAGE_B_CKPT}\narea decoder {AREA_DECODER} (kind={kind}, {len(a2i)} areas)", flush=True)

    rows = []
    var = deterministic_area_rows(val_ds)
    for i in range(len(val_ds.X)):
        target, valid = reference_median(val_ds, var, i, bases)
        if target is None:
            continue
        a = int(val_ds.areas[i])
        if a not in a2i:                                     # decoder has no head for this area
            continue
        rec = decode(model, dec, val_ds.X[i], val_ds.M[i], bases[a], area_one_hot(a2i, a))
        c, rm, r2 = masked_metrics(rec, target, valid)
        if np.isfinite(c) and np.isfinite(r2) and np.isfinite(rm):
            rows.append({"area": a, "camera": a // 100, "pearson": c, "r2": r2, "rmse": rm})

    assert (state_hash(model.teacher), state_hash(model.student), state_hash(model.predictor)) == frozen, \
        "frozen instrument model changed during eval"

    pe = pd.DataFrame(rows)
    if pe.empty:
        raise RuntimeError("no scorable validation examples")
    pe.to_csv(os.path.join(OUT_DIR, "dynamic_area_heads_per_example.csv"), index=False)

    def block(g):
        return {"n": int(len(g)),
                "pearson_median": round(float(np.median(g["pearson"])), 4),
                "pearson_q1": round(float(np.percentile(g["pearson"], 25)), 4),
                "r2_q1": round(float(np.percentile(g["r2"], 25)), 4),
                "neg_pearson_rate_pct": round(100.0 * float((g["pearson"] < 0).mean()), 3)}

    report = {"encoder_ckpt": STAGE_B_CKPT, "area_decoder": AREA_DECODER, "decoder_kind": kind,
              "n_val_examples": int(len(pe)),
              "overall": block(pe),
              "by_camera": {int(cam): block(g) for cam, g in pe.groupby("camera")},
              "select_metric": "validation Q1 Pearson + negative-Pearson rate",
              "frozen_hashes_unchanged": True, "git_commit": git_commit()}
    with open(os.path.join(OUT_DIR, "dynamic_area_heads_summary.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    o = report["overall"]
    print(f"\nOVERALL  median Pearson {o['pearson_median']}  Q1 Pearson {o['pearson_q1']}  "
          f"Q1 R2 {o['r2_q1']}  neg-rate {o['neg_pearson_rate_pct']}%  (n={o['n']})", flush=True)
    print("by camera:", flush=True)
    for cam, b in sorted(report["by_camera"].items()):
        print(f"  cam {cam}: median {b['pearson_median']}  Q1 {b['pearson_q1']}  "
              f"Q1 R2 {b['r2_q1']}  neg {b['neg_pearson_rate_pct']}%  (n={b['n']})", flush=True)
    print(f"wrote dynamic_area_heads_summary.json + per_example.csv to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
