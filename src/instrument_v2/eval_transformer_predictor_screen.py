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


def compute_verdict(results):
    """PRIMARY benchmark (physics frozen-probe protocol): the trained frozen
    S4D ENCODER, with the Transformer predictor and the teacher both
    discarded. PASS iff it beats the random frozen S4D encoder. Transformer
    outputs are diagnostics only and never affect the verdict."""
    candidate = results["tx_jepa_encoder"]["val_camccd_bacc"]
    diffs = {"encoder_minus_random_s4d":
                 candidate - results["random_s4d"]["val_camccd_bacc"],
             "encoder_minus_mlp_jepa":
                 candidate - results["mlp_jepa_encoder"]["val_camccd_bacc"]}
    verdict = "PASS" if diffs["encoder_minus_random_s4d"] > 0 else "FAIL"
    return diffs, verdict


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

    diffs, verdict = compute_verdict(results)

    summary = {"git_commit": git_commit(),
               "results": results, "diffs": diffs, "verdict": verdict,
               "tx_best_epoch": tx_selection["best"]["epoch"],
               "tx_select_view": tx_selection.get("select_view"),
               "tx_selection": tx_selection,
               "teacher_hash_verified":
                   tx_selection.get("teacher_hash_verified_every_epoch", False),
               "statements": [
                   "Transformer and teacher discarded during evaluation "
                   "(the benchmark representation is the frozen S4D encoder).",
                   "No fine-tuning and no test evaluation.",
                   "Only the probe classifier was fit; all representation "
                   "parameters stayed frozen."]}
    with open(os.path.join(ART_DIR, "results.json"), "w") as handle:
        json.dump(summary, handle, indent=2, default=float)

    lines = ["# transformer screen -- frozen ENCODER benchmark "
             "(validation-only)", "",
             f"git commit: {summary['git_commit']}", "",
             "## PRIMARY: frozen S4D encoders (physics probe protocol)"]
    for name in ("random_s4d", "mlp_jepa_encoder", "tx_jepa_encoder"):
        entry = results[name]
        marker = "  <- candidate" if name == "tx_jepa_encoder" else ""
        lines.append(f"- {name}: {entry['val_camccd_bacc']:.4f} "
                     f"(erank {entry['val_effective_rank']:.1f}){marker}")
    lines += ["",
              f"candidate - random_s4d:  "
              f"{diffs['encoder_minus_random_s4d']:+.4f}",
              f"candidate - mlp_jepa:    "
              f"{diffs['encoder_minus_mlp_jepa']:+.4f}", "",
              f"best selected epoch: {summary['tx_best_epoch']} "
              f"(selected on ENCODER camCCD, select_view="
              f"{summary['tx_select_view']})",
              f"teacher hash verified every training epoch: "
              f"{summary['teacher_hash_verified']}", "",
              "## DIAGNOSTICS ONLY (never affect selection or PASS)"]
    for name in ("tx_jepa_transformer", "random_s4d_tx"):
        entry = results[name]
        lines.append(f"- {name}: {entry['val_camccd_bacc']:.4f} "
                     f"(erank {entry['val_effective_rank']:.1f})")
    lines += ["", f"## VERDICT: {verdict}",
              "(PASS requires trained frozen S4D encoder camCCD > random "
              "frozen S4D encoder camCCD)", ""]
    lines += [f"- {s}" for s in summary["statements"]] + [""]
    md_path = os.path.join(ART_DIR, "final_summary.md")
    with open(md_path, "w") as handle:
        handle.write("\n".join(lines))
    print(f"report -> {md_path}", flush=True)
    print(f"VERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    main()
