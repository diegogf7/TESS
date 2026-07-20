# All this code is from Claude
"""Two-panel explanatory figure: individual stars -> group fingerprint.

Top:    K individual normalized light curves from ONE detector area, on the
        shared Sector-14 grid (each faint, each different, gaps left blank).
Bottom: the group fingerprint = per-cadence MEDIAN of those K stars, with a
        shaded band = robust scatter (MAD). Averaging suppresses each star's
        own noise (~1/sqrt(K)) and leaves the shared instrument shape.

TRAIN stars only; test TICs never loaded.

    python -m src.instrument_v2.plot_fingerprint_construction
Env: AREA (default = most-populated area), K (default 8), SEED,
     FPC_ART_DIR, S14_DATA, ...
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd

from src.instrument_v2.area_commonmode_dataset import (
    Sector14GroupStatDataset,
    ensure_area_column,
    min_valid_stars,
)
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_sector14_jepa import git_commit

K = int(os.environ.get("K", "8"))
SEED = int(os.environ.get("SEED", "0"))
AREA_ENV = os.environ.get("AREA", "")
ART_DIR = os.environ.get(
    "FPC_ART_DIR",
    os.path.join("artifacts", "instrument_v2", "fingerprint_construction"))
S14_DATA = os.environ.get(
    "S14_DATA",
    "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get(
    "SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
BASE_ART_DIR = os.environ.get(
    "BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))


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

    areas, counts = np.unique(labels, return_counts=True)
    area = int(AREA_ENV) if AREA_ENV else int(areas[counts.argmax()])
    rows_all = np.flatnonzero(labels == area)
    rng = np.random.default_rng(SEED)
    rows = rng.choice(rows_all, size=min(K, len(rows_all)), replace=False)
    camera, ccd, ring = area // 100, (area // 10) % 10, area % 10
    print(f"area {area} (camera {camera}, CCD {ccd}, ring {ring}): "
          f"{len(rows_all)} stars, showing {len(rows)}", flush=True)

    grid = np.arange(X.shape[1])
    curves = np.where(M[rows] > 0, X[rows], np.nan)     # gaps -> blank

    vals = np.where(M[rows] > 0, X[rows], np.nan)
    median = np.nanmedian(vals, axis=0)
    mad = 1.4826 * np.nanmedian(np.abs(vals - median[None, :]), axis=0)
    n_obs = (M[rows] > 0).sum(axis=0)
    valid = n_obs >= min_valid_stars(len(rows))
    median = np.where(valid, median, np.nan)
    mad = np.where(valid, mad, np.nan)

    with open(os.path.join(ART_DIR, "fingerprint_construction.json"), "w") as h:
        json.dump({"git_commit": git_commit(), "area": area, "k": len(rows),
                   "n_stars_in_area": int(len(rows_all)),
                   "median_scatter_typical": float(np.nanmedian(mad)),
                   "single_star_scatter_typical":
                       float(np.nanmedian(np.nanstd(vals, axis=1))),
                   "test_untouched": "TRAIN stars only; test never loaded."},
                  h, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"matplotlib unavailable ({exc}); JSON written", flush=True)
        return

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={"height_ratios": [1.5, 1]})

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(rows)))
    for curve, color in zip(curves, colors):
        ax_top.plot(grid, curve, lw=0.8, alpha=0.7, color=color)
    ax_top.axhline(0, color="0.7", lw=0.6)
    ax_top.set_ylabel("normalized flux")
    ax_top.set_title(f"Sector 14, one detector region (camera {camera}, "
                     f"CCD {ccd}, ring {ring}): {len(rows)} individual stars")

    ax_bot.plot(grid, median, color="#0b3d5c", lw=1.6)
    ax_bot.axhline(0, color="0.7", lw=0.6)
    ax_bot.set_ylabel("normalized flux")
    ax_bot.set_xlabel("cadence (shared grid index)")
    ax_bot.set_title(f"median of the {len(rows)} stars")

    fig.tight_layout()
    out = os.path.join(ART_DIR, "fingerprint_construction.png")
    fig.savefig(out, dpi=150)
    print(f"plot -> {out}", flush=True)


if __name__ == "__main__":
    main()
