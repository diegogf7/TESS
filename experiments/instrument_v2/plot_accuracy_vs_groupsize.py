# All this code is from Claude
"""One clean figure for a talk: chip-identification accuracy vs group size.

For each K, build K-star area fingerprints (median + log-MAD on the shared
grid), flatten them, and train a linear probe to name the chip (16-way
camera x CCD). Plot balanced accuracy vs K. Single star (K=1) sits near the
bottom; accuracy climbs as stars are pooled -> "instrument identity is a
collective property."

Star-disjoint by construction: within each area the stars are split into two
halves; train fingerprints come from half A, test fingerprints from half B.
TRAIN TICs only; test TICs never loaded.

    python -m src.instrument_v2.plot_accuracy_vs_groupsize
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd

from src.instrument_v2.area_commonmode_dataset import (
    Sector14GroupStatDataset,
    area_to_chip,
    ensure_area_column,
    group_statistics,
)
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_group_level_jepa import fast_probe
from src.instrument_v2.train_sector14_jepa import git_commit

K_VALUES = [int(k) for k in os.environ.get("K_VALUES", "1,2,4,8,16,32").split(",")]
N_PER_CHIP = int(os.environ.get("N_PER_CHIP", "120"))
REPEATS = int(os.environ.get("REPEATS", "5"))
SEED = int(os.environ.get("SEED", "0"))
ART_DIR = os.environ.get(
    "AVK_ART_DIR",
    os.path.join("artifacts", "instrument_v2", "accuracy_vs_groupsize"))
S14_DATA = os.environ.get(
    "S14_DATA",
    "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get(
    "SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
BASE_ART_DIR = os.environ.get(
    "BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))
CHANCE = 1.0 / 16


def sample_fingerprints(X, M, rows_by_area, chip_to_areas, k, rng):
    mv = max(1, math.ceil(k / 2))
    feats, labels = [], []
    for chip, areas in chip_to_areas.items():
        usable = [a for a in areas if len(rows_by_area[a]) >= k]
        if not usable:
            continue
        for _ in range(N_PER_CHIP):
            area = usable[rng.integers(len(usable))]
            rows = rng.choice(rows_by_area[area], size=k, replace=False)
            median, log_mad, _, _ = group_statistics(X[rows], M[rows], mv)
            feats.append(np.concatenate([median, log_mad]))
            labels.append(chip)
    return np.asarray(feats), np.asarray(labels)


def main():
    os.makedirs(ART_DIR, exist_ok=True)
    print(f"git commit: {git_commit()}", flush=True)

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    df = ensure_area_column(df)
    train_tics, _, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, train_tics)

    base = Sector14GroupStatDataset(df, train_tics, t_range, "area", k=1)
    assert not set(base.tics) & test_tics, "test TIC leaked into figure"
    X, M, labels = base.X, base.M, base.group_labels

    # split each area's stars into two disjoint halves (train / test)
    split_rng = np.random.default_rng(SEED)
    rows_A, rows_B, chip_to_areas = {}, {}, {}
    for area in np.unique(labels):
        rows = np.flatnonzero(labels == area)
        split_rng.shuffle(rows)
        half = len(rows) // 2
        if half < 1:
            continue
        rows_A[int(area)] = rows[:half]
        rows_B[int(area)] = rows[half:]
        chip_to_areas.setdefault(area_to_chip(area), []).append(int(area))

    results = []
    for k in K_VALUES:
        scores = []
        for r in range(REPEATS):
            rng = np.random.default_rng(SEED + 100 * r + k)
            usable_A = {a: rows_A[a] for a in rows_A if len(rows_A[a]) >= k}
            usable_B = {a: rows_B[a] for a in rows_B if len(rows_B[a]) >= k}
            chips = {c: [a for a in areas if a in usable_A and a in usable_B]
                     for c, areas in chip_to_areas.items()}
            chips = {c: a for c, a in chips.items() if a}
            if len(chips) < 2:
                continue
            Xtr, ytr = sample_fingerprints(X, M, usable_A, chips, k, rng)
            Xte, yte = sample_fingerprints(X, M, usable_B, chips, k, rng)
            scores.append(fast_probe(Xtr, ytr, Xte, yte))
        if not scores:
            print(f"K={k}: too few stars, skipped", flush=True)
            continue
        mean, std = float(np.mean(scores)), float(np.std(scores))
        results.append({"k": k, "mean": mean, "std": std, "n_repeats": len(scores)})
        print(f"K={k:2d}: camccd bacc = {mean:.3f} +/- {std:.3f}", flush=True)

    with open(os.path.join(ART_DIR, "accuracy_vs_groupsize.json"), "w") as h:
        json.dump({"git_commit": git_commit(), "chance": CHANCE,
                   "results": results,
                   "test_untouched": "TRAIN stars only; test TICs never loaded."},
                  h, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import ScalarFormatter
    except Exception as exc:
        print(f"matplotlib unavailable ({exc}); JSON written", flush=True)
        return

    ks = [r["k"] for r in results]
    means = np.array([r["mean"] for r in results])
    stds = np.array([r["std"] for r in results])

    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.axhline(CHANCE, color="0.6", ls="--", lw=1)
    ax.plot(ks, means, "-o", color="#1d6a96", lw=2.2, ms=7)
    ax.fill_between(ks, means - stds, means + stds, color="#1d6a96", alpha=0.18)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xlabel("stars pooled together")
    ax.set_ylabel("chip identification accuracy")
    ax.set_title("accuracy with logistic regression with star grouping")
    ax.margins(x=0.04)
    fig.tight_layout()
    out = os.path.join(ART_DIR, "accuracy_vs_groupsize.png")
    fig.savefig(out, dpi=150)
    print(f"plot -> {out}", flush=True)


if __name__ == "__main__":
    main()
