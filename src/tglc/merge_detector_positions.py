"""Add PHYSICAL detector DETECTOR_X / DETECTOR_Y (TESS CCD column/row pixels) to the
Sector-14 dense parquet for detector_nearest grouping -- WITHOUT re-splitting (the
train/val/test split is TIC-keyed).

The coordinates are the OFFICIAL TESS detector column/row for each star's (RA, Dec) in
its sector/camera/CCD, computed with tess-point (tess_stars2px). This replaces the
earlier WRONG aperture-local formula (CUT_X*150 + STAR_X), which produced a fake tile
grid. We keep only the sector-14 solution whose camera/CCD matches the parquet, so the
coordinates are camera/CCD-consistent by construction. RA/Dec are never used for
grouping -- only to derive the physical detector position.

    IN_PARQUET=.../..._dense_v2.parquet  OUT_PARQUET=.../..._dense_v2_xy.parquet \
    python -m src.tglc.merge_detector_positions
Env: IN_PARQUET, OUT_PARQUET, PLOT, MIN_COVERAGE, SECTOR, PLOT_ONLY.
"""
import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tess_stars2px import tess_stars2px_function_entry
from src.instrument_v2.area_commonmode_dataset import ensure_area_column

ROOT = "/orcd/scratch/orcd/006/diegogon/tglc_primary"
IN_PARQUET = os.environ.get("IN_PARQUET", f"{ROOT}/tglc_raw_cadence_s14_dense_v2.parquet")
OUT_PARQUET = os.environ.get("OUT_PARQUET", f"{ROOT}/tglc_raw_cadence_s14_dense_v2_xy.parquet")
PLOT = os.environ.get("PLOT", "artifacts/shared_s4d/correction_v1/detector_xy.png")
MIN_COVERAGE = float(os.environ.get("MIN_COVERAGE", "0.98"))
SECTOR = int(os.environ.get("SECTOR", "14"))
# TESS CCD is 2048x2048 science pixels; tess-point col/row include small collateral/overscan
# margins, so accept a slightly wider documented window and report the actual range.
COL_MAX, ROW_MAX = 2136.0, 2078.0


def radec_to_detector(ra, dec, sector):
    """(oid, camera, ccd, DETECTOR_X, DETECTOR_Y) rows for the given sector via tess-point."""
    ids = np.arange(len(ra))
    oid, _, _, sec, cam, ccd, col, row, _ = tess_stars2px_function_entry(
        ids, np.asarray(ra, float), np.asarray(dec, float), trySector=sector)
    keep = np.asarray(sec) == sector
    return pd.DataFrame({"oid": np.asarray(oid)[keep], "camera": np.asarray(cam)[keep].astype(int),
                         "ccd": np.asarray(ccd)[keep].astype(int),
                         "DETECTOR_X": np.asarray(col)[keep], "DETECTOR_Y": np.asarray(row)[keep]})


def validate(chk):
    """Return (coverage, in_bounds, locality_ok) and print stats. chk has camera/ccd/DETECTOR_X/Y + ra/dec."""
    have = chk["DETECTOR_X"].notna()
    coverage = float(have.mean())
    x = chk.loc[have, "DETECTOR_X"].to_numpy(); y = chk.loc[have, "DETECTOR_Y"].to_numpy()
    in_bounds = bool(np.isfinite(x).all() and np.isfinite(y).all()
                     and (x >= -2).all() and (x <= COL_MAX).all() and (y >= -2).all() and (y <= ROW_MAX).all())
    print(f"coverage {coverage:.4f} | DETECTOR_X [{x.min():.1f},{x.max():.1f}] "
          f"DETECTOR_Y [{y.min():.1f},{y.max():.1f}] | in_bounds {in_bounds}", flush=True)
    # RA/Dec locality: within one CCD, close sky pairs should have small detector distance
    sub = chk[have]
    locality_ok = True
    for (cam, ccd), g in sub.groupby(["camera", "ccd"]):
        if len(g) < 50:
            continue
        g = g.sample(min(len(g), 400), random_state=0)
        rd = np.c_[np.radians(g["ra"]), np.radians(g["dec"])]; xy = g[["DETECTOR_X", "DETECTOR_Y"]].to_numpy()
        i = np.random.default_rng(0).integers(len(g))
        sky = np.degrees(np.sqrt(((rd - rd[i]) ** 2).sum(1))) * 60          # arcmin
        det = np.sqrt(((xy - xy[i]) ** 2).sum(1))                           # px
        near = sky < 5                                                      # <5' on sky
        if near.sum() >= 5 and np.median(det[near]) > 200:                 # should be < ~200 px
            locality_ok = False
        break
    print(f"RA/Dec locality within CCD: {'OK' if locality_ok else 'SUSPECT'}", flush=True)
    return coverage, in_bounds, locality_ok


def plot_per_ccd(chk):
    sub = ensure_area_column(chk.dropna(subset=["DETECTOR_X"]).copy())
    fig, axes = plt.subplots(4, 4, figsize=(14, 14)); axes = axes.ravel()
    for k, ((cam, ccd), g) in enumerate(sorted(sub.groupby(["camera", "ccd"]))):
        if k >= 16:
            break
        ax = axes[k]
        ax.scatter(g["DETECTOR_X"], g["DETECTOR_Y"], s=2, c=(g["area"] % 20), cmap="tab20")
        ax.set_title(f"cam{cam} ccd{ccd} (n={len(g)})", fontsize=8)
        ax.set_xlim(0, COL_MAX); ax.set_ylim(0, ROW_MAX)
    fig.suptitle("physical detector X/Y per CCD (smooth coverage = no artificial tile grid)")
    os.makedirs(os.path.dirname(PLOT), exist_ok=True)
    fig.tight_layout(); fig.savefig(PLOT, dpi=110); plt.close(fig)
    print(f"wrote {PLOT}", flush=True)


def main():
    if os.environ.get("PLOT_ONLY") == "1":
        d = pd.read_parquet(OUT_PARQUET, columns=["TIC", "sector", "camera", "ccd", "ra", "dec",
                                                  "DETECTOR_X", "DETECTOR_Y"])
        plot_per_ccd(d.dropna(subset=["DETECTOR_X"]))
        return
    if os.path.exists(OUT_PARQUET):
        raise SystemExit(f"refusing to overwrite existing {OUT_PARQUET}")

    meta = pd.read_parquet(IN_PARQUET, columns=["TIC", "sector", "camera", "ccd", "ra", "dec"])
    meta["TIC"] = meta["TIC"].astype(str)
    meta = meta.reset_index(drop=True)
    print(f"parquet: {len(meta)} rows; converting RA/Dec -> detector col/row (sector {SECTOR}) via tess-point",
          flush=True)

    res = radec_to_detector(meta["ra"].to_numpy(), meta["dec"].to_numpy(), SECTOR)
    res = res.drop_duplicates(["oid", "camera", "ccd"])          # a boundary star can hit 2 CCDs
    meta2 = meta.reset_index().rename(columns={"index": "oid"})  # keep parquet camera/ccd as the truth
    chk = meta2.merge(res, on=["oid", "camera", "ccd"], how="left")
    assert len(chk) == len(meta), "row count changed -- bad merge"

    coverage, in_bounds, locality_ok = validate(chk)
    if coverage < MIN_COVERAGE:
        raise SystemExit(f"FATAL: DETECTOR_X/Y coverage {coverage:.4f} < {MIN_COVERAGE}")
    if not in_bounds:
        raise SystemExit("FATAL: detector coordinates out of documented CCD bounds")
    if not locality_ok:
        raise SystemExit("FATAL: nearby RA/Dec do not map to nearby detector positions -- bad conversion")

    df = pd.read_parquet(IN_PARQUET)                             # full read once, at write
    df["TIC"] = df["TIC"].astype(str)
    xy = chk[["TIC", "camera", "ccd", "DETECTOR_X", "DETECTOR_Y"]]
    merged = df.merge(xy, on=["TIC", "camera", "ccd"], how="left")
    assert len(merged) == len(df), "row count changed on full merge"
    merged.to_parquet(OUT_PARQUET)
    print(f"wrote {OUT_PARQUET} ({len(merged)} rows, +DETECTOR_X/DETECTOR_Y, coverage {coverage:.4f})", flush=True)
    plot_per_ccd(chk)


if __name__ == "__main__":
    main()
