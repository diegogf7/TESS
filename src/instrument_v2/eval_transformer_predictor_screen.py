# All this code is from Claude
"""Frozen five-way probe screen for the Transformer-predictor variant.

Validation-only. All representation parameters are frozen; only the sklearn
probe classifier is fit. Every baseline is RECOMPUTED through the identical
probe harness (no hard-coded reference numbers). Representations:

  1. mlp_jepa_encoder      existing fixed-teacher MLP-JEPA student encoder
  2. tx_jepa_encoder       Transformer-JEPA student encoder
  3. tx_jepa_transformer   Transformer-JEPA predictor output  <- candidate
  4. random_s4d            random frozen S4D encoder
  5. random_s4d_tx         same random S4D + identical random Transformer

Verdict PASS only if (3) beats BOTH random controls (4) and (5).

    python -m src.instrument_v2.eval_transformer_predictor_screen
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

K = int(os.environ.get("K", "8"))
SEED = int(os.environ.get("SEED", "0"))
ART_DIR = os.environ.get(
    "TXS_ART_DIR",
    os.path.join("artifacts", "instrument_v2", "transformer_predictor_screen"))
MLP_SELECTION = os.environ.get(
    "MLP_SELECTION",
    os.path.join("artifacts", "instrument_v2", "fixed_regional_teacher_v1",
                 f"selection_frtstudent_k{K}_s{SEED}.json"))
TX_SELECTION = os.environ.get(
    "TX_SELECTION",
    os.path.join(ART_DIR, f"selection_frtstudent_tx_k{K}_s{SEED}.json"))
S14_DATA = os.environ.get(
    "S14_DATA",
    "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get(
    "SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
BASE_ART_DIR = os.environ.get(
    "BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(selection_path, predictor_type):
    with open(selection_path) as handle:
        selection = json.load(handle)
    model = FixedTeacherInstrumentJEPA(predictor_type=predictor_type).to(DEVICE)
    model.load_state_dict(torch.load(selection["checkpoint"],
                                     map_location=DEVICE))
    model.eval()
    return model, selection


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
    assert not (set(train_ds.tics) | set(val_ds.tics)) & test_tics, \
        "test TIC leaked into screen"

    mlp_model, _ = load_model(MLP_SELECTION, "mlp")
    tx_model, tx_selection = load_model(TX_SELECTION, "transformer")
    torch.manual_seed(SEED)
    random_model = FixedTeacherInstrumentJEPA(
        predictor_type="transformer").to(DEVICE).eval()

    representations = {
        "mlp_jepa_encoder": (mlp_model, "online"),
        "tx_jepa_encoder": (tx_model, "online"),
        "tx_jepa_transformer": (tx_model, "predicted"),
        "random_s4d": (random_model, "online"),
        "random_s4d_tx": (random_model, "predicted"),
    }

    results = {}
    for name, (model, view) in representations.items():
        before = state_hash(model)
        train_z = individual_latents(model, train_ds, view)
        val_z = individual_latents(model, val_ds, view)
        bacc = fast_probe(train_z, train_ds.chips, val_z, val_ds.chips)
        assert state_hash(model) == before, \
            f"probe modified representation parameters for {name}"
        results[name] = {"val_camccd_bacc": float(bacc),
                         "val_effective_rank": effective_rank(val_z)}
        print(f"{name:22s} camccd={bacc:.4f} "
              f"erank={results[name]['val_effective_rank']:.1f}", flush=True)

    candidate = results["tx_jepa_transformer"]["val_camccd_bacc"]
    diffs = {"candidate_minus_random_s4d":
                 candidate - results["random_s4d"]["val_camccd_bacc"],
             "candidate_minus_random_s4d_tx":
                 candidate - results["random_s4d_tx"]["val_camccd_bacc"]}
    verdict = "PASS" if (diffs["candidate_minus_random_s4d"] > 0
                         and diffs["candidate_minus_random_s4d_tx"] > 0) else "FAIL"

    summary = {"git_commit": git_commit(),
               "results": results, "diffs": diffs, "verdict": verdict,
               "tx_best_epoch": tx_selection["best"]["epoch"],
               "tx_selection": tx_selection,
               "teacher_hash_verified":
                   tx_selection.get("teacher_hash_verified_every_epoch", False),
               "statements": [
                   "No encoder or predictor fine-tuning occurred in this "
                   "screen; all representation parameters were frozen and "
                   "only the probe classifier was fit.",
                   "Test TICs were never loaded or evaluated."]}
    with open(os.path.join(ART_DIR, "results.json"), "w") as handle:
        json.dump(summary, handle, indent=2, default=float)

    lines = ["# transformer_predictor_screen -- final summary "
             "(validation-only, frozen)", "",
             f"git commit: {summary['git_commit']}", "",
             "## Frozen validation camCCD (identical probe harness)"]
    for name, entry in results.items():
        marker = "  <- candidate" if name == "tx_jepa_transformer" else ""
        lines.append(f"- {name}: {entry['val_camccd_bacc']:.4f} "
                     f"(erank {entry['val_effective_rank']:.1f}){marker}")
    lines += ["",
              f"candidate - random_s4d:    "
              f"{diffs['candidate_minus_random_s4d']:+.4f}",
              f"candidate - random_s4d_tx: "
              f"{diffs['candidate_minus_random_s4d_tx']:+.4f}", "",
              f"Transformer best epoch: {summary['tx_best_epoch']}",
              f"teacher hash verified every training epoch: "
              f"{summary['teacher_hash_verified']}", "",
              f"## VERDICT: {verdict}",
              "(PASS requires the pretrained Transformer output to beat both "
              "random controls)", ""]
    lines += [f"- {s}" for s in summary["statements"]] + [""]
    md_path = os.path.join(ART_DIR, "final_summary.md")
    with open(md_path, "w") as handle:
        handle.write("\n".join(lines))
    print(f"report -> {md_path}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    main()
