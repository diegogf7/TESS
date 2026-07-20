# All this code is from Claude
"""Seed-distribution screen: is the fixed-teacher MLP-JEPA encoder actually
above the RANDOM-INIT distribution, or inside its spread?

Probes, through the identical harness:
  - N_RANDOM random-init S4D encoders (different seeds, no training)
  - every trained fixed-teacher student listed in STUDENT_SELECTIONS

Reports each value, mean +/- std per group, and the gap. Validation-only.

    python -m src.instrument_v2.seed_distribution_screen
Env: N_RANDOM, STUDENT_SELECTIONS (comma-separated selection JSONs),
     SDS_ART_DIR, K, S14_DATA, SPLIT_DIR, BASE_ART_DIR
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
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_group_level_jepa import fast_probe, individual_latents
from src.instrument_v2.train_sector14_jepa import git_commit

K = int(os.environ.get("K", "8"))
N_RANDOM = int(os.environ.get("N_RANDOM", "8"))
ART_DIR = os.environ.get(
    "SDS_ART_DIR",
    os.path.join("artifacts", "instrument_v2", "seed_distribution_screen"))
DEFAULT_SELECTIONS = ",".join([
    "artifacts/instrument_v2/fixed_regional_teacher_v1/selection_frtstudent_k8_s0.json",
    os.path.join(ART_DIR, "selection_frtstudent_k8_s1.json"),
    os.path.join(ART_DIR, "selection_frtstudent_k8_s2.json"),
])
STUDENT_SELECTIONS = [p for p in os.environ.get(
    "STUDENT_SELECTIONS", DEFAULT_SELECTIONS).split(",") if p]
S14_DATA = os.environ.get(
    "S14_DATA",
    "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get(
    "SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
BASE_ART_DIR = os.environ.get(
    "BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def probe(model):
        model.eval()
        train_z = individual_latents(model, train_ds, "online")
        val_z = individual_latents(model, val_ds, "online")
        return float(fast_probe(train_z, train_ds.chips, val_z, val_ds.chips))

    random_scores = []
    for i in range(N_RANDOM):
        torch.manual_seed(1000 + i)
        model = FixedTeacherInstrumentJEPA().to(DEVICE)
        bacc = probe(model)
        random_scores.append(bacc)
        print(f"random_s4d[{i}] (seed {1000 + i}): camccd={bacc:.4f}", flush=True)

    student_scores = {}
    for path in STUDENT_SELECTIONS:
        if not os.path.exists(path):
            print(f"WARNING: missing {path} -- skipped", flush=True)
            continue
        with open(path) as handle:
            selection = json.load(handle)
        model = FixedTeacherInstrumentJEPA().to(DEVICE)
        model.load_state_dict(torch.load(selection["checkpoint"],
                                         map_location=DEVICE))
        bacc = probe(model)
        student_scores[selection["tag"]] = bacc
        print(f"{selection['tag']}: camccd={bacc:.4f}", flush=True)

    random_arr = np.asarray(random_scores)
    student_arr = np.asarray(list(student_scores.values()))
    summary = {
        "git_commit": git_commit(),
        "random_scores": random_scores,
        "random_mean": float(random_arr.mean()),
        "random_std": float(random_arr.std()),
        "student_scores": student_scores,
        "student_mean": float(student_arr.mean()) if len(student_arr) else None,
        "student_std": float(student_arr.std()) if len(student_arr) else None,
        "gap_mean": (float(student_arr.mean() - random_arr.mean())
                     if len(student_arr) else None),
        "test_untouched": "Test TICs never loaded; validation-only.",
    }
    with open(os.path.join(ART_DIR, "results.json"), "w") as handle:
        json.dump(summary, handle, indent=2)

    lines = ["# seed_distribution_screen (validation-only)", "",
             f"git commit: {summary['git_commit']}", "",
             f"random S4D  (n={len(random_arr)}): "
             f"{random_arr.mean():.4f} +/- {random_arr.std():.4f}   "
             f"[{random_arr.min():.4f} .. {random_arr.max():.4f}]"]
    if len(student_arr):
        lines += [f"MLP-JEPA students (n={len(student_arr)}): "
                  f"{student_arr.mean():.4f} +/- {student_arr.std():.4f}   "
                  f"[{student_arr.min():.4f} .. {student_arr.max():.4f}]", "",
                  f"gap (student mean - random mean): "
                  f"{summary['gap_mean']:+.4f}",
                  f"random spread (2 std): {2 * random_arr.std():.4f}", "",
                  ("VERDICT: student mean clears random mean by more than "
                   "2x random std -- real separation"
                   if summary['gap_mean'] > 2 * random_arr.std()
                   else "VERDICT: student mean is within the random spread -- "
                        "no reliable separation")]
    lines += ["", summary["test_untouched"], ""]
    md_path = os.path.join(ART_DIR, "final_summary.md")
    with open(md_path, "w") as handle:
        handle.write("\n".join(lines))
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
