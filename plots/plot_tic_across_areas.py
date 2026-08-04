"""Plot one star's light curve across the different AREAS it lands in.

The same TIC, observed in multiple TESS sectors, falls on a different
camera / CCD / ring each time -> a different area (= camera*100 + ccd*10 + ring).
Each observation is resampled to the shared 1024-point grid with the SAME
functions the training pipeline uses (src.data.data.resample_to_grid / normalize)
and the sectors are stacked, so the per-area instrument signature stands out
against the (fixed) astrophysical signal.

Run from the repo root, on the cluster (the parquet lives there):

    python plots/plot_tic_across_areas.py            # auto-pick a multi-sector star
    TIC=426342065 python plots/plot_tic_across_areas.py

Env: ALL_PARQUET, TIC, MAX_PANELS, OUT.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.data import resample_to_grid, normalize   # the 1024-grid we always use
from src.regions.areas import add_area                  # ra/dec -> area label

ALL_PARQUET = os.environ.get(
    "ALL_PARQUET",
    "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_all.parquet")
MAX_PANELS = int(os.environ.get("MAX_PANELS", "6"))
OUT = os.environ.get("OUT", os.path.join(os.path.dirname(__file__), "tic_across_areas.png"))


def grid_curve(time, flux, n=1024):
    """One raw curve -> normalized 1024-grid, gaps as NaN so they read as breaks."""
    time = np.asarray(time, float)
    flux = np.asarray(flux, float)
    good = np.isfinite(time) & np.isfinite(flux)
    time, flux = time[good], flux[good]
    time, idx = np.unique(time, return_index=True)       # sorted + de-duplicated for interp1d
    flux = normalize(flux[idx])
    grid_flux, mask = resample_to_grid(time, flux, n)
    return np.where(mask > 0, grid_flux, np.nan)


def main():
    # 1) pick the TIC (cheap pass: only TIC+sector, no flux arrays loaded)
    tic = os.environ.get("TIC")
    if tic is None:
        key = pd.read_parquet(ALL_PARQUET, columns=["TIC", "sector"])
        n_sectors = key.astype({"TIC": str}).groupby("TIC")["sector"].nunique()
        tic = str(n_sectors.idxmax())
        print(f"auto-picked TIC {tic} ({int(n_sectors.max())} sectors)")

    # 2) load just this star's rows (predicate pushdown -> no full-file scan)
    star = pd.read_parquet(ALL_PARQUET, filters=[("TIC", "==", str(tic))])
    if star.empty:
        raise SystemExit(f"TIC {tic} not found in {ALL_PARQUET}")
    star = add_area(star).sort_values("sector")
    star = star.drop_duplicates(["sector", "area"]).head(MAX_PANELS)

    # 3) one stacked panel per (sector, area), all on the same 1024-grid x-axis
    n = len(star)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.8 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (_, row) in zip(axes, star.iterrows()):
        ax.plot(grid_curve(row["time"], row["flux"]), lw=0.6)
        ax.axhline(0, color="k", lw=0.3, alpha=0.4)
        ax.set_ylabel(f"s{int(row['sector'])}\narea {int(row['area'])}", fontsize=8)
    axes[-1].set_xlabel("shared 1024-point grid index")
    fig.suptitle(f"TIC {tic} — same star across {n} areas (normalized, 1024-grid)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
