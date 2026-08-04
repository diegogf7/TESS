"""Add ABSOLUTE detector STAR_X / STAR_Y (CCD pixels) to the Sector-14 dense parquet
for detector_nearest grouping -- WITHOUT unpacking the FITS archive and WITHOUT
re-splitting (the train/val/test split is TIC-keyed).

Detector position is in the TGLC LC FITS PRIMARY header: STAR_X / STAR_Y are the
star's position WITHIN its CUTSIZE-px FFI cutout (a few px) and CUT_X / CUT_Y are the
cutout's tile index, so the absolute CCD pixel is  CUT_* * CUTSIZE + STAR_* . We first
INSPECT several real FITS to confirm those fields exist (checking header keys AND HDU1
table columns), then stream headers straight from s0014_fits.tar. FITS members are
filtered by the Gaia DR3 id parsed from the filename (falling back to reading the
header) so only the ~parquet TICs are opened. Join is validated against each curve's
camera/CCD. If the fields genuinely do not exist we STOP and report what we inspected.

    TAR=.../s0014_fits.tar  IN_PARQUET=.../..._dense_v2.parquet \
    OUT_PARQUET=.../..._dense_v2_xy.parquet  python -m src.tglc.merge_detector_positions
Env: TAR, IN_PARQUET, OUT_PARQUET, PLOT, MIN_COVERAGE, N (dry-run cap on members read).
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


def _inspect_and_validate_fields(hdul, name, log):
    """Print header/table fields of one FITS; return the set of POS_FIELDS actually present."""
    hdr = hdul[0].header
    cols = [c.name for c in hdul[1].columns]
    have_hdr = [k for k in POS_FIELDS + ID_FIELDS if k in hdr]
    log.append(f"  {name.split('/')[-1]}")
    log.append(f"    header POS/ID fields present : {have_hdr}")
    log.append(f"    HDU1 table columns          : {cols}")
    return set(k for k in POS_FIELDS if k in hdr)


def main():
    if os.path.exists(OUT_PARQUET):
        raise SystemExit(f"refusing to overwrite existing {OUT_PARQUET}")

    df = pd.read_parquet(IN_PARQUET)
    df["TIC"] = df["TIC"].astype(str)
    need_tic = set(df["TIC"])
    have_gaia = "GAIADR3" in df.columns
    need_gaia = set(int(g) for g in df["GAIADR3"].unique() if pd.notna(g) and int(g) > 0) if have_gaia else set()
    print(f"parquet: {len(df)} rows, {len(need_tic)} unique TICs"
          f"{f', {len(need_gaia)} gaia ids for filename filter' if need_gaia else ''}", flush=True)

    rows, seen, opened, inspected, inspect_log, pos_ok = [], 0, 0, 0, [], None
    with tarfile.open(TAR, "r") as tf:
        for m in tf:                                             # single streaming pass
            if not m.name.endswith(".fits"):
                continue
            seen += 1
            g = parse_gaiaid(m.name)
            if need_gaia and g is not None and g not in need_gaia:
                continue                                         # cheap skip: not one of our stars
            f = tf.extractfile(m)
            if f is None:
                continue
            opened += 1
            with fits.open(f, memmap=False) as hdul:
                if inspected < N_INSPECT:                        # inspect the first few real FITS
                    present = _inspect_and_validate_fields(hdul, m.name, inspect_log)
                    inspected += 1
                    pos_ok = present if pos_ok is None else (pos_ok & present)
                    if inspected == 1 and not set(POS_FIELDS) <= present:
                        print("=== FITS field inspection ===\n" + "\n".join(inspect_log), flush=True)
                        print(f"FATAL: required detector-position header fields missing "
                              f"{sorted(set(POS_FIELDS) - present)}. Inspected header keys + HDU1 "
                              f"columns above. Not fabricating coordinates, not substituting RA/Dec.")
                        sys.exit(1)
                hdr = hdul[0].header
                tic = str(hdr.get("TICID", ""))
                gg = int(hdr.get("GAIADR3", -1))
                if tic not in need_tic and gg not in need_gaia:  # header confirms not needed
                    continue
                x, y = absolute_xy(hdr)
                rows.append((tic, gg, x, y, int(hdr.get("CAMERA", -1)), int(hdr.get("CCD", -1))))
            if opened % 20000 == 0:
                print(f"  {seen} fits scanned, {opened} opened, {len(rows)} matched", flush=True)
            if N_CAP and seen >= N_CAP:
                break

    print("=== FITS field inspection (first %d) ===\n%s" % (inspected, "\n".join(inspect_log)), flush=True)
    xy = pd.DataFrame(rows, columns=["TIC", "GAIADR3", "STAR_X", "STAR_Y", "cam_hdr", "ccd_hdr"])
    dup = int(xy["TIC"].duplicated().sum())
    if dup:
        print(f"WARNING: {dup} duplicate TICs in FITS matches -- keeping first", flush=True)
    xy = xy.drop_duplicates("TIC")
    print(f"scanned {seen} fits, opened {opened}, matched {len(xy)}/{len(need_tic)} TICs", flush=True)
    if len(xy):
        print(f"abs STAR_X range [{xy.STAR_X.min():.0f}, {xy.STAR_X.max():.0f}]  "
              f"STAR_Y [{xy.STAR_Y.min():.0f}, {xy.STAR_Y.max():.0f}]  (expect ~[0, 2048])", flush=True)

    # ---- join (TIC key; camera/CCD used to VALIDATE the join) --------------------------------
    merged = df.merge(xy[["TIC", "STAR_X", "STAR_Y", "cam_hdr", "ccd_hdr"]], on="TIC", how="left")
    assert len(merged) == len(df), f"row count changed {len(df)} -> {len(merged)} (bad merge key)"

    coverage = float(merged["STAR_X"].notna().mean())
    have = merged["STAR_X"].notna()
    cam_ok = bool((merged.loc[have, "cam_hdr"] == merged.loc[have, "camera"]).all())
    ccd_ok = bool((merged.loc[have, "ccd_hdr"] == merged.loc[have, "ccd"]).all())
    finite = bool(np.isfinite(merged.loc[have, ["STAR_X", "STAR_Y"]].to_numpy()).all())
    print(f"coverage {coverage:.4f} | finite {finite} | camera-consistent {cam_ok} | ccd-consistent {ccd_ok}",
          flush=True)

    if N_CAP:
        print(f"dry run (N={N_CAP}) -- inspection + range shown, NOT writing", flush=True)
        return
    if coverage < MIN_COVERAGE:
        raise SystemExit(f"FATAL: STAR_X/STAR_Y coverage {coverage:.4f} < {MIN_COVERAGE} -- insufficient")
    if not (finite and cam_ok and ccd_ok):
        raise SystemExit("FATAL: non-finite coords or camera/CCD mismatch -- refusing to write")

    out = merged.drop(columns=["cam_hdr", "ccd_hdr"])
    out.to_parquet(OUT_PARQUET)
    print(f"wrote {OUT_PARQUET} ({len(out)} rows, +STAR_X/STAR_Y, coverage {coverage:.4f})", flush=True)

    _distance_stats_and_plot(out)


def _distance_stats_and_plot(df, gs=16, n_areas=6, n_panels=4):
    """Detector-nearest within-group radius stats + one diagnostic plot of anchors+neighbors."""
    df = ensure_area_column(df.copy()).drop_duplicates("TIC")
    df = df[df["STAR_X"].notna()]
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
            radii.append(np.sort(d)[:gs][-1])                    # anchor -> farthest of 15 nearest
    radii = np.asarray(radii)
    print(f"detector within-group radius (px): median {np.median(radii):.1f}  "
          f"p90 {np.percentile(radii, 90):.1f}  max {radii.max():.1f}  (over {len(radii)} anchors)", flush=True)

    pick = rng.choice(uareas, min(n_panels, len(uareas)), replace=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10)); axes = axes.ravel()
    for ax, a in zip(axes, pick):
        idx = np.where(area == a)[0]
        p = xy[idx]
        anchor = rng.integers(len(idx))
        d = np.sqrt(((p - p[anchor]) ** 2).sum(1)); near = np.argsort(d)[:gs]
        ax.scatter(p[:, 0], p[:, 1], s=6, c="0.8", label="area stars")
        ax.scatter(p[near, 0], p[near, 1], s=28, c="tab:blue", label="15 neighbors")
        ax.scatter(p[anchor, 0], p[anchor, 1], s=90, marker="*", c="tab:red", label="anchor")
        ax.set_title(f"area {int(a)}  (cam{int(a)//100} ccd{(int(a)//10)%10})", fontsize=9)
        ax.set_xlabel("STAR_X (px)"); ax.set_ylabel("STAR_Y (px)")
    axes[0].legend(fontsize=7)
    fig.suptitle("detector_nearest: anchor + 15 closest by STAR_X/STAR_Y")
    os.makedirs(os.path.dirname(PLOT), exist_ok=True)
    fig.tight_layout(); fig.savefig(PLOT, dpi=130); plt.close(fig)
    print(f"wrote {PLOT}", flush=True)


if __name__ == "__main__":
    main()
