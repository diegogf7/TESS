"""Add ABSOLUTE detector STAR_X / STAR_Y (CCD pixels) to the Sector-14 dense parquet
for detector_nearest grouping -- WITHOUT unpacking the FITS archive and WITHOUT
re-splitting (the train/val/test split is TIC-keyed).

Detector position is in the TGLC LC FITS PRIMARY header: STAR_X / STAR_Y are the
star's position WITHIN its CUTSIZE-px FFI cutout (a few px) and CUT_X / CUT_Y are the
cutout's tile index, so the absolute CCD pixel is  CUT_* * CUTSIZE + STAR_* . We first
INSPECT several real FITS to confirm those fields exist (checking header keys AND HDU1
table columns), then stream headers straight from s0014_fits.tar. FITS members are
filtered by the Gaia DR3 id parsed from the filename (falling back to the header) so
only the ~parquet stars are opened. The join is validated against each curve's
camera/CCD. If the fields genuinely do not exist we STOP and report what we inspected.

The dry run (N>0) reads only scalar columns (TIC/GAIADR3/camera/ccd) so it is fast; the
FULL cadence parquet is read once, only when actually writing.

    TAR=.../s0014_fits.tar  IN_PARQUET=.../..._dense_v2.parquet \
    OUT_PARQUET=.../..._dense_v2_xy.parquet  python -m src.tglc.merge_detector_positions
Env: TAR, IN_PARQUET, OUT_PARQUET, PLOT, MIN_COVERAGE, N (dry-run cap on members scanned).
"""
import os
import re
import sys
import tarfile

import numpy as np
import pandas as pd
from astropy.io import fits

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.instrument_v2.area_commonmode_dataset import ensure_area_column

ROOT = "/orcd/scratch/orcd/006/diegogon/tglc_primary"
TAR = os.environ.get("TAR", f"{ROOT}/s0014_fits.tar")
IN_PARQUET = os.environ.get("IN_PARQUET", f"{ROOT}/tglc_raw_cadence_s14_dense_v2.parquet")
OUT_PARQUET = os.environ.get("OUT_PARQUET", f"{ROOT}/tglc_raw_cadence_s14_dense_v2_xy.parquet")
PLOT = os.environ.get("PLOT", "artifacts/shared_s4d/correction_v1/detector_neighbors.png")
MIN_COVERAGE = float(os.environ.get("MIN_COVERAGE", "0.98"))
N_CAP = int(os.environ.get("N", "0"))                 # >0 = stop after N .fits members (dry run)
N_INSPECT = 5

GAIA_RE = re.compile(r"gaiaid[-_]?(\d+)", re.IGNORECASE)
POS_FIELDS = ("STAR_X", "STAR_Y", "CUT_X", "CUT_Y", "CUTSIZE")   # required header cards
ID_FIELDS = ("TICID", "GAIADR3", "CAMERA", "CCD")


def parse_gaiaid(name):
    m = GAIA_RE.search(name)
    return int(m.group(1)) if m else None


def absolute_xy(hdr):
    cs = float(hdr["CUTSIZE"])
    return (float(hdr["CUT_X"]) * cs + float(hdr["STAR_X"]),      # CCD column pixel
            float(hdr["CUT_Y"]) * cs + float(hdr["STAR_Y"]))      # CCD row pixel


def _inspect(hdul, name, log):
    hdr = hdul[0].header
    cols = [c.name for c in hdul[1].columns]
    log.append(f"  {name.split('/')[-1]}")
    log.append(f"    header POS/ID fields present : {[k for k in POS_FIELDS + ID_FIELDS if k in hdr]}")
    log.append(f"    HDU1 table columns          : {cols}")
    return set(k for k in POS_FIELDS if k in hdr)


def scan_tar(need_tic, need_gaia):
    """Stream FITS headers from the tar; return a TIC->STAR_X/STAR_Y frame (+ hdr cam/ccd)."""
    rows, seen, opened, inspected, log = [], 0, 0, 0, []
    with tarfile.open(TAR, "r") as tf:
        for m in tf:
            if not m.name.endswith(".fits"):
                continue
            seen += 1
            g = parse_gaiaid(m.name)
            if need_gaia and g is not None and g not in need_gaia:
                continue                                          # cheap skip without opening
            f = tf.extractfile(m)
            if f is None:
                continue
            opened += 1
            with fits.open(f, memmap=False) as hdul:
                if inspected < N_INSPECT:
                    present = _inspect(hdul, m.name, log)
                    inspected += 1
                    if inspected == 1 and not set(POS_FIELDS) <= present:
                        print("=== FITS field inspection ===\n" + "\n".join(log), flush=True)
                        print(f"FATAL: required detector-position header fields missing "
                              f"{sorted(set(POS_FIELDS) - present)}. Not fabricating coordinates, "
                              f"not substituting RA/Dec.")
                        sys.exit(1)
                hdr = hdul[0].header
                tic = str(hdr.get("TICID", "")); gg = int(hdr.get("GAIADR3", -1))
                if tic not in need_tic and gg not in need_gaia:
                    continue
                x, y = absolute_xy(hdr)
                rows.append((tic, x, y, int(hdr.get("CAMERA", -1)), int(hdr.get("CCD", -1))))
            if opened % 20000 == 0:
                print(f"  {seen} fits scanned, {opened} opened, {len(rows)} matched", flush=True)
            if N_CAP and seen >= N_CAP:
                break
    print(f"=== FITS field inspection (first {inspected}) ===\n" + "\n".join(log), flush=True)
    xy = pd.DataFrame(rows, columns=["TIC", "STAR_X", "STAR_Y", "camera", "ccd"])
    cross = int(xy["TIC"].duplicated().sum()) - int(xy.duplicated(["TIC", "camera", "ccd"]).sum())
    xy = xy.drop_duplicates(["TIC", "camera", "ccd"])
    if cross:
        print(f"note: {cross} TICs appear under >1 camera/ccd -- resolved by the (TIC,camera,ccd) join", flush=True)
    print(f"scanned {seen} fits, opened {opened}, matched {len(xy)} (TIC,camera,ccd) keys", flush=True)
    if len(xy):
        print(f"abs STAR_X range [{xy.STAR_X.min():.0f}, {xy.STAR_X.max():.0f}]  "
              f"STAR_Y [{xy.STAR_Y.min():.0f}, {xy.STAR_Y.max():.0f}]  (expect ~[0, 2048])", flush=True)
    return xy


def main():
    if os.path.exists(OUT_PARQUET):
        raise SystemExit(f"refusing to overwrite existing {OUT_PARQUET}")

    # FAST: scalar columns only for the filter + join validation (no cadence arrays)
    meta = pd.read_parquet(IN_PARQUET, columns=["TIC", "GAIADR3", "camera", "ccd"])
    meta["TIC"] = meta["TIC"].astype(str)
    need_tic = set(meta["TIC"])
    need_gaia = set(int(g) for g in meta["GAIADR3"].unique() if pd.notna(g) and int(g) > 0)
    print(f"parquet: {len(meta)} rows, {len(need_tic)} TICs, {len(need_gaia)} gaia ids", flush=True)

    xy = scan_tar(need_tic, need_gaia)

    chk = meta.merge(xy, on=["TIC", "camera", "ccd"], how="left")   # safe key: TIC + camera + CCD
    assert len(chk) == len(meta), "row count changed -- bad merge key"
    have = chk["STAR_X"].notna()
    coverage = float(have.mean())
    finite = bool(np.isfinite(chk.loc[have, ["STAR_X", "STAR_Y"]].to_numpy()).all())
    print(f"coverage {coverage:.4f} | finite {finite} | joined on (TIC,camera,ccd) "
          f"-> coords are camera/CCD-consistent by construction", flush=True)

    if N_CAP:
        print(f"dry run (N={N_CAP}) -- inspection + range + partial coverage shown, NOT writing", flush=True)
        return
    if coverage < MIN_COVERAGE:
        raise SystemExit(f"FATAL: STAR_X/STAR_Y coverage {coverage:.4f} < {MIN_COVERAGE} -- insufficient")
    if not finite:
        raise SystemExit("FATAL: non-finite coordinates -- refusing to write")

    # Real write: read the FULL cadence parquet once and add the two columns
    df = pd.read_parquet(IN_PARQUET)
    df["TIC"] = df["TIC"].astype(str)
    merged = df.merge(xy[["TIC", "camera", "ccd", "STAR_X", "STAR_Y"]], on=["TIC", "camera", "ccd"], how="left")
    assert len(merged) == len(df), "row count changed on full merge"
    merged.to_parquet(OUT_PARQUET)
    print(f"wrote {OUT_PARQUET} ({len(merged)} rows, +STAR_X/STAR_Y, coverage {coverage:.4f})", flush=True)

    _distance_stats_and_plot(chk[have][["TIC", "camera", "ccd", "STAR_X", "STAR_Y"]].copy())


def _distance_stats_and_plot(df, gs=16, n_panels=4):
    """Detector within-group radius stats + one diagnostic plot of anchors + 15 neighbors."""
    df = ensure_area_column(df).drop_duplicates("TIC")
    xy = df[["STAR_X", "STAR_Y"]].to_numpy(float)
    area = df["area"].to_numpy()
    rng = np.random.default_rng(0)
    uareas = [a for a in np.unique(area) if (area == a).sum() >= gs]
    radii = []
    for a in uareas:
        idx = np.where(area == a)[0]
        anch = idx if len(idx) <= 200 else rng.choice(idx, 200, replace=False)
        for i in anch:
            d = np.sqrt(((xy[idx] - xy[i]) ** 2).sum(1))
            radii.append(np.sort(d)[:gs][-1])
    radii = np.asarray(radii)
    print(f"detector within-group radius (px): median {np.median(radii):.1f}  "
          f"p90 {np.percentile(radii, 90):.1f}  max {radii.max():.1f}  (over {len(radii)} anchors)", flush=True)

    pick = rng.choice(uareas, min(n_panels, len(uareas)), replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10)); axes = axes.ravel()
    for ax, a in zip(axes, pick):
        idx = np.where(area == a)[0]; p = xy[idx]; anchor = rng.integers(len(idx))
        near = np.argsort(np.sqrt(((p - p[anchor]) ** 2).sum(1)))[:gs]
        ax.scatter(p[:, 0], p[:, 1], s=6, c="0.8", label="area stars")
        ax.scatter(p[near, 0], p[near, 1], s=28, c="tab:blue", label="15 neighbors")
        ax.scatter(p[anchor, 0], p[anchor, 1], s=90, marker="*", c="tab:red", label="anchor")
        ax.set_title(f"area {int(a)} (cam{int(a)//100} ccd{(int(a)//10)%10})", fontsize=9)
        ax.set_xlabel("STAR_X (px)"); ax.set_ylabel("STAR_Y (px)")
    axes[0].legend(fontsize=7)
    fig.suptitle("detector_nearest: anchor + 15 closest by STAR_X/STAR_Y")
    os.makedirs(os.path.dirname(PLOT), exist_ok=True)
    fig.tight_layout(); fig.savefig(PLOT, dpi=130); plt.close(fig)
    print(f"wrote {PLOT}", flush=True)


if __name__ == "__main__":
    main()
