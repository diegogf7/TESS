"""Plot ONE star (same TIC) across the different sectors/areas it was observed in.

A TIC gets one light curve per sector, on a different camera/CCD/ring each time
(area = camera*100 + ccd*10 + ring). We pull every sector for one TIC, resample
each to the shared 1024-point grid with the pipeline's own functions
(src.data.data.resample_to_grid / normalize), and stack them so the per-area
instrument signature stands out against the fixed astrophysical signal.

Area labels are best-effort: if the parquet lacks ra/dec they're merged from
tglc_positions.parquet; if that fails too, panels are labeled by sector only.

    python plots/plot_tic_across_areas.py            # auto-pick a many-sector star
    TIC=229774966 python plots/plot_tic_across_areas.py

Env: ALL_PARQUET, POSITIONS, TIC, MAX_PANELS, OUT.
"""

import os
import sys

# make `src` importable even when run as `python plots/plot_tic_across_areas.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.data import resample_to_grid, normalize   # the 1024-grid we always use
from src.regions.areas import add_area                  # ra/dec -> area label

ALL_PARQUET = os.environ.get(
    "ALL_PARQUET",
    "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_pretrain_train.parquet")
POSITIONS = os.environ.get(
    "POSITIONS",
    "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_positions.parquet")
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


def attach_area(star):
    """Best-effort `area` column; leaves -1 if positions can't be resolved."""
    need = ["ra", "dec", "camera", "ccd"]
    missing = [c for c in need if c not in star.columns]
    if missing and "GAIADR3" in star.columns:
        pos = pd.read_parquet(POSITIONS, columns=["GAIADR3", "sector"] + missing)
        star = star.merge(pos, on=["GAIADR3", "sector"], how="left")
    if all(c in star.columns for c in need) and star["ra"].notna().any():
        try:
            return add_area(star)
        except Exception as e:
            print("add_area failed, labeling by sector only:", e)
    star["area"] = -1
    return star


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
    star = attach_area(star)
    star = star.drop_duplicates("sector").sort_values("sector").head(MAX_PANELS)

    # 3) one stacked panel per sector, all on the same 1024-grid x-axis
    n = len(star)
    fig, axes = plt.subplots(n, 1, figsize=(11, 1.8 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (_, row) in zip(axes, star.iterrows()):
        ax.plot(grid_curve(row["time"], row["flux"]), lw=0.6)
        ax.axhline(0, color="k", lw=0.3, alpha=0.4)
        area = int(row["area"])
        label = f"s{int(row['sector'])}" + (f"\narea {area}" if area != -1 else "")
        ax.set_ylabel(label, fontsize=8)
    axes[-1].set_xlabel("shared 1024-point grid index")
    fig.suptitle(f"TIC {tic} — same star across {n} sectors (normalized, 1024-grid)")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
