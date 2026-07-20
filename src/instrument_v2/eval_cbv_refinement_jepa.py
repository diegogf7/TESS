# All this code is from Claude
"""Frozen-encoder evaluation for the CBV-refinement pilot.

Discards the transformer predictor, the frozen regional teacher, and all
PCA/CBV machinery: the only representation evaluated is

    raw star -> frozen trained online S4D -> identical linear probe.

Four encoders through one probe harness (validation-only, nothing frozen is
updated):
  random_s4d          random-init S4D
  mlp_jepa_encoder    fixed-teacher MLP-JEPA encoder
  tx_jepa_encoder     Transformer-JEPA encoder (0.4559 reference)
  cbv_refined         CBV-refined Transformer-JEPA encoder  <- candidate

PASS iff the CBV-refined encoder exceeds the Transformer-JEPA encoder
(recomputed in-harness, ~0.4559). Beat-random is also reported.

    python -m src.instrument_v2.eval_cbv_refinement_jepa
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch

from src.instrument_v2.area_commonmode_dataset import (
    Sector14GroupStatDataset,
    ensure_area_column,
)
from src.instrument_v2.fixed_teacher_instrument_jepa import (
    FixedTeacherInstrumentJEPA,
)
from src.instrument_v2.regional_group_teacher import state_hash
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_group_level_jepa import fast_probe, individual_latents
from src.instrument_v2.train_sector14_jepa import effective_rank, git_commit

SEED = int(os.environ.get("SEED", "0"))
K = int(os.environ.get("K", "8"))
TX_REFERENCE = 0.4559
ART_DIR = os.environ.get(
    "CBV_ART_DIR",
    os.path.join("artifacts", "instrument_v2", "cbv_refinement_screen"))
S14_DATA = os.environ.get(
    "S14_DATA",
    "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get(
    "SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
BASE_ART_DIR = os.environ.get(
    "BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))
MLP_CKPT = os.environ.get(
    "MLP_CKPT",
    "/orcd/scratch/orcd/006/diegogon/checkpoints/fixed_regional_teacher_v1/"
    "frtstudent_k8_s0_best.pth")
TX_CKPT = os.environ.get(
    "TX_CKPT",
    "/orcd/scratch/orcd/006/diegogon/checkpoints/transformer_encoder_screen/"
    "frtstudent_tx_k8_s0_best.pth")
CBV_CKPT = os.environ.get(
    "CBV_CKPT",
    "/orcd/scratch/orcd/006/diegogon/checkpoints/cbv_refinement_screen/"
    "cbv_refine_tx_k8_s0_best.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build(predictor_type):
    return FixedTeacherInstrumentJEPA(
        n_tokens=16, token_dim=16, d_model=256, n_layers=4,
        readout="mean", predictor_type=predictor_type).to(DEVICE).eval()


def probe_encoder(model, train_ds, val_ds):
    before = state_hash(model)
    tr = individual_latents(model, train_ds, "online")
    va = individual_latents(model, val_ds, "online")
    assert state_hash(model) == before, "probe modified frozen representation"
    return float(fast_probe(tr, train_ds.chips, va, val_ds.chips)), \
        effective_rank(va)


def main():
    os.makedirs(ART_DIR, exist_ok=True)
    print(f"git commit: {git_commit()}", flush=True)

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    df = ensure_area_column(df)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, train_tics)
    train_ds = Sector14GroupStatDataset(df, train_tics, t_range, "area", K)
    val_ds = Sector14GroupStatDataset(df, val_tics, t_range, "area", K)
    assert not (set(train_ds.tics) | set(val_ds.tics)) & test_tics

    results = {}

    torch.manual_seed(SEED)
    results["random_s4d"] = probe_encoder(build("transformer"), train_ds, val_ds)

    for name, ckpt, ptype in (("mlp_jepa_encoder", MLP_CKPT, "mlp"),
                              ("tx_jepa_encoder", TX_CKPT, "transformer"),
                              ("cbv_refined", CBV_CKPT, "transformer")):
        model = build(ptype)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        results[name] = probe_encoder(model, train_ds, val_ds)

    baccs = {name: bacc for name, (bacc, _) in results.items()}
    eranks = {name: er for name, (_, er) in results.items()}
    for name in ("random_s4d", "mlp_jepa_encoder", "tx_jepa_encoder",
                 "cbv_refined"):
        print(f"{name:18s} camccd={baccs[name]:.4f} erank={eranks[name]:.1f}",
              flush=True)

    candidate = baccs["cbv_refined"]
    diffs = {"cbv_minus_tx": candidate - baccs["tx_jepa_encoder"],
             "cbv_minus_tx_reference": candidate - TX_REFERENCE,
             "cbv_minus_mlp": candidate - baccs["mlp_jepa_encoder"],
             "cbv_minus_random": candidate - baccs["random_s4d"]}
    verdict = "PASS" if candidate > baccs["tx_jepa_encoder"] else "FAIL"

    summary = {"git_commit": git_commit(),
               "results": {n: {"val_camccd_bacc": b, "effective_rank": e}
                           for n, (b, e) in results.items()},
               "tx_reference": TX_REFERENCE, "diffs": diffs, "verdict": verdict,
               "beats_random": bool(candidate > baccs["random_s4d"]),
               "statements": [
                   "METADATA-GUIDED pretraining: per-chip CBV bases are "
                   "defined by camera/CCD membership, so chip metadata enters "
                   "the teacher target -- this is NOT fully label-free JEPA.",
                   "Transformer predictor, frozen regional teacher, and "
                   "PCA/CBV machinery were all discarded for evaluation; only "
                   "the frozen S4D encoder is probed.",
                   "No fine-tuning; test TICs never loaded."]}
    with open(os.path.join(ART_DIR, "results.json"), "w") as handle:
        json.dump(summary, handle, indent=2, default=float)

    lines = ["# CBV-refinement pilot -- frozen encoder benchmark "
             "(validation-only)", "", f"git commit: {summary['git_commit']}",
             "", "## Frozen S4D encoders (identical probe harness)"]
    for name in ("random_s4d", "mlp_jepa_encoder", "tx_jepa_encoder",
                 "cbv_refined"):
        mark = "  <- candidate" if name == "cbv_refined" else (
            "  (0.4559 reference)" if name == "tx_jepa_encoder" else "")
        lines.append(f"- {name}: {baccs[name]:.4f} "
                     f"(erank {eranks[name]:.1f}){mark}")
    lines += ["",
              f"cbv - tx (in-harness): {diffs['cbv_minus_tx']:+.4f}",
              f"cbv - tx (0.4559 ref): {diffs['cbv_minus_tx_reference']:+.4f}",
              f"cbv - mlp:             {diffs['cbv_minus_mlp']:+.4f}",
              f"cbv - random:          {diffs['cbv_minus_random']:+.4f}", "",
              f"## VERDICT: {verdict}",
              "(PASS requires the CBV-refined encoder to exceed the "
              "Transformer-JEPA encoder)",
              f"beats random S4D: {summary['beats_random']}", ""]
    lines += [f"- {s}" for s in summary["statements"]] + [""]
    with open(os.path.join(ART_DIR, "final_summary.md"), "w") as handle:
        handle.write("\n".join(lines))
    print(f"report -> {os.path.join(ART_DIR, 'final_summary.md')}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    main()
