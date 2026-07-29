from __future__ import annotations
"""Validation-only 3-way decoder comparison on identical rows, using the exact
oracle-ceiling metrics (target = pre-CBV 32-star median; oracle = best masked K=8
fit). Nothing is trained or rebuilt here.

  global      : B_area @ decoder(latent)
  area one-hot: B_area @ decoder(concat(latent, area_one_hot))
  area heads  : B_area @ AreaHeadDecoder(concat(latent, area_one_hot))  (shared trunk + per-area head)

Reports decoder-vs-target and decoder-vs-oracle Pearson (Q1, median) + R2/RMSE,
negative-Pearson rate, and failure % by camera and area for all three, and diffs
the area-head decoder against the frozen area-one-hot numbers.

    python -m src.instrument_v2.eval_cbv_area_heads
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
    build_decoder, load_area_decoder, area_index_map, area_one_hot,
    deterministic_area_rows, decode, masked_metrics,
)
from src.instrument_v2.eval_cbv_oracle_ceiling import load_bases_npz, reference_median, oracle_weights, _quart

GLOBAL_DECODER = os.environ.get(
    "GLOBAL_DECODER", os.path.join(GROUP_ART_DIR, "single_star_weight_decode", "decoder.pth"))
ONEHOT_DECODER = os.environ.get(
    "ONEHOT_DECODER", os.path.join(GROUP_ART_DIR, "single_star_weight_decode_area_conditioned", "decoder.pth"))
HEAD_DECODER = os.environ.get(
    "HEAD_DECODER", os.path.join(GROUP_ART_DIR, "single_star_weight_decode_area_heads", "decoder.pth"))
BASES_NPZ = os.environ.get(
    "CBV_BASES_NPZ",
    os.path.join(GROUP_ART_DIR, "area_group_cbv_r8_g32_mv16_q16437_tglc0_1607e67857039a07.npz"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(GROUP_ART_DIR, "single_star_weight_decode_area_heads"))

# frozen area-one-hot reference (from the prior run)
ONEHOT_REF = {"decoder_vs_target_pearson_q1": 0.279, "decoder_vs_target_pearson_median": 0.811,
              "neg_pearson_rate_pct": 18.28, "camera1_failure_pct": 50.23}


def _summarize(df, tag):
    s = {"decoder_vs_oracle_pearson": {"q1": _quart(df[f"{tag}_oracle_pearson"])["q1"],
                                       "median": _quart(df[f"{tag}_oracle_pearson"])["median"]},
         "decoder_vs_target_pearson": {"q1": _quart(df[f"{tag}_target_pearson"])["q1"],
                                       "median": _quart(df[f"{tag}_target_pearson"])["median"]},
         "decoder_vs_target_r2": {"q1": _quart(df[f"{tag}_target_r2"])["q1"],
                                  "median": _quart(df[f"{tag}_target_r2"])["median"]},
         "decoder_vs_target_rmse": {"q1": _quart(df[f"{tag}_target_rmse"])["q1"],
                                    "median": _quart(df[f"{tag}_target_rmse"])["median"]},
         "neg_pearson_rate_pct": round(100.0 * (df[f"{tag}_target_pearson"] < 0).mean(), 3),
         "failure_pct_by": {}}
    fail = (df[f"{tag}_target_pearson"] < 0).astype(int)
    for col in ("camera", "area"):
        g = fail.groupby(df[col]).agg(["size", "mean"])
        s["failure_pct_by"][col] = {int(k): {"n": int(r["size"]), "failure_pct": round(100 * r["mean"], 3)}
                                    for k, r in g.iterrows()}
    return s


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"git commit: {git_commit()}", flush=True)

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    df = ensure_area_column(df)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, train_tics)
    train_ds = Sector14GroupStatDataset(df, train_tics, t_range, "area", GROUP_SIZE, min_valid=MIN_VALID_STARS)
    val_ds = Sector14GroupStatDataset(df, val_tics, t_range, "area", GROUP_SIZE, min_valid=MIN_VALID_STARS)
    assert not (set(train_ds.tics) | set(val_ds.tics)) & test_tics, "test TIC leaked"

    bases = load_bases_npz(BASES_NPZ)
    a2i_ref = area_index_map(sorted(bases))

    model = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=256, n_layers=4,
                                       readout="mean", predictor_type="mlp").to(DEVICE)
    model.load_state_dict(torch.load(STAGE_B_CKPT, map_location=DEVICE))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    frozen = {"teacher": state_hash(model.teacher), "student": state_hash(model.student),
              "predictor": state_hash(model.predictor)}

    global_dec = build_decoder(CBV_RANK).to(DEVICE)
    global_dec.load_state_dict(torch.load(GLOBAL_DECODER, map_location=DEVICE)); global_dec.eval()
    oh_dec, a2i_oh, oh_kind = load_area_decoder(ONEHOT_DECODER, DEVICE)
    hd_dec, a2i_hd, hd_kind = load_area_decoder(HEAD_DECODER, DEVICE)
    if hd_kind != "area_heads":
        raise RuntimeError(f"{HEAD_DECODER} is kind={hd_kind}, expected area_heads")
    if not (a2i_oh == a2i_hd == a2i_ref):
        raise RuntimeError("decoder area maps disagree with the deterministic training-area map")
    print(f"decoders: global | one-hot ({oh_kind}) | heads ({hd_kind}); {len(a2i_ref)} areas", flush=True)

    val_area_rows = deterministic_area_rows(val_ds)
    rows = []
    for i in range(len(val_ds.X)):
        target, valid = reference_median(val_ds, val_area_rows, i, bases)
        if target is None:
            continue
        a = int(val_ds.areas[i])
        B = np.asarray(bases[a], dtype=np.float64)
        oracle = (B @ oracle_weights(B, target, valid)).astype(np.float32)
        av = area_one_hot(a2i_ref, a)
        recons = {"global": decode(model, global_dec, val_ds.X[i], val_ds.M[i], bases[a], None),
                  "onehot": decode(model, oh_dec, val_ds.X[i], val_ds.M[i], bases[a], av),
                  "heads": decode(model, hd_dec, val_ds.X[i], val_ds.M[i], bases[a], av)}
        rec = {"tic": str(val_ds.tics[i]), "area": a, "camera": a // 100}
        for tag, rc in recons.items():
            oc, orm, or2 = masked_metrics(rc, oracle, valid)
            tc, trm, tr2 = masked_metrics(rc, target, valid)
            rec[f"{tag}_oracle_pearson"] = oc
            rec[f"{tag}_target_pearson"] = tc
            rec[f"{tag}_target_r2"] = tr2
            rec[f"{tag}_target_rmse"] = trm
        if all(np.isfinite(v) for k, v in rec.items() if k.endswith(("pearson", "r2", "rmse"))):
            rows.append(rec)

    assert state_hash(model.teacher) == frozen["teacher"], "teacher changed"
    assert state_hash(model.student) == frozen["student"], "student changed"
    assert state_hash(model.predictor) == frozen["predictor"], "predictor changed"

    pe = pd.DataFrame(rows)
    if pe.empty:
        raise RuntimeError("no scorable validation examples")
    pe.to_csv(os.path.join(OUT_DIR, "per_example_metrics_area_heads.csv"), index=False)

    summ = {k: _summarize(pe, k) for k in ("global", "onehot", "heads")}
    heads_t = summ["heads"]["decoder_vs_target_pearson"]
    heads_cam1 = summ["heads"]["failure_pct_by"]["camera"].get(1, {}).get("failure_pct")
    vs_onehot = {
        "target_pearson_q1": round(heads_t["q1"] - ONEHOT_REF["decoder_vs_target_pearson_q1"], 4),
        "target_pearson_median": round(heads_t["median"] - ONEHOT_REF["decoder_vs_target_pearson_median"], 4),
        "neg_pearson_rate_pct": round(summ["heads"]["neg_pearson_rate_pct"] - ONEHOT_REF["neg_pearson_rate_pct"], 3),
        "camera1_failure_pct": (None if heads_cam1 is None
                                else round(heads_cam1 - ONEHOT_REF["camera1_failure_pct"], 3)),
    }
    # decision rule (item 10): improves Q1 or camera-1 without reducing the median
    improved = ((heads_t["q1"] > ONEHOT_REF["decoder_vs_target_pearson_q1"]
                 or (heads_cam1 is not None and heads_cam1 < ONEHOT_REF["camera1_failure_pct"]))
                and heads_t["median"] >= ONEHOT_REF["decoder_vs_target_pearson_median"] - 1e-6)

    report = {"n_val_examples": int(len(pe)),
              "global_decoder": summ["global"], "area_onehot_decoder": summ["onehot"],
              "area_head_decoder": summ["heads"],
              "area_onehot_reference": ONEHOT_REF, "area_heads_minus_onehot": vs_onehot,
              "area_heads_improves_over_onehot": bool(improved),
              "n_areas": len(a2i_ref), "frozen_hashes_unchanged": True,
              "identical_rows_all_decoders": True,
              "paths": {"global": GLOBAL_DECODER, "onehot": ONEHOT_DECODER, "heads": HEAD_DECODER,
                        "bases_npz": BASES_NPZ, "stage_b_ckpt": STAGE_B_CKPT},
              "git_commit": git_commit()}
    with open(os.path.join(OUT_DIR, "area_heads_comparison.json"), "w") as fh:
        json.dump(report, fh, indent=2)

    def _ln(tag, s):
        t = s["decoder_vs_target_pearson"]
        print(f"  {tag:<10} target med {t['median']:.3f} Q1 {t['q1']:.3f} | neg "
              f"{s['neg_pearson_rate_pct']:.2f}% | cam1 "
              f"{s['failure_pct_by']['camera'].get(1, {}).get('failure_pct')}%", flush=True)
    print(f"\nvalidation examples (identical rows, 3 decoders): {len(pe)}", flush=True)
    _ln("global", summ["global"]); _ln("one-hot", summ["onehot"]); _ln("heads", summ["heads"])
    print(f"one-hot ref: target med 0.811 Q1 0.279 | neg 18.28% | cam1 50.23%", flush=True)
    print(f"area-heads MINUS one-hot: {json.dumps(vs_onehot)}", flush=True)
    print(f"AREA-HEADS IMPROVES OVER ONE-HOT: {improved}", flush=True)
    print(f"wrote per_example_metrics_area_heads.csv + area_heads_comparison.json to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
