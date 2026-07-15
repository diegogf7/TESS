# All this code is from Claude
"""FITS -> parquet extraction retaining cadence metadata.

Same as extract_raw_parquet.py (kept untouched) plus four columns needed by
the cadence-aligned pipeline: cadence_num, TESS_flags, TGLC_flags,
background. Columns missing from a FITS file are stored as None for that row
and reported at the end, so a schema drift is loud instead of silent.

Writes ONE parquet (no row-level train/val split -- the old random 2% split
was the star-leakage mistake; make star-disjoint splits downstream).
Does not overwrite any existing parquet.

    python -m src.tglc.extract_raw_parquet_cadence
Env: FITS_ROOT, OUT_PATH, N_WORKERS.
"""

import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from astropy.io import fits

FITS_ROOT = os.environ.get("FITS_ROOT", "/orcd/scratch/orcd/006/diegogon/tglc_primary")
OUT_PATH = os.environ.get("OUT_PATH", os.path.join(FITS_ROOT, "tglc_raw_cadence_all.parquet"))
MIN_POINTS = 200
N_WORKERS = int(os.environ.get("N_WORKERS", "16"))

EXTRA_COLUMNS = ("cadence_num", "TESS_flags", "TGLC_flags", "background")

if os.path.exists(OUT_PATH):
    raise SystemExit(f"refusing to overwrite existing {OUT_PATH}")


def extract_one(path):
    try:
        with fits.open(path, memmap=False) as hdul:
            data = hdul[1].data
            head = hdul[0].header
            available = set(data.columns.names)

            time = np.asarray(data["time"], dtype=np.float64)
            raw = np.asarray(data["aperture_flux"], dtype=np.float64)
            calibration = np.asarray(data["cal_aper_flux"], dtype=np.float64)

            good = np.isfinite(time) & np.isfinite(raw)
            if good.sum() < MIN_POINTS:
                return None

            row = {
                "time": time[good],
                "flux": raw[good],
                "flux_cal": calibration[good],
                "TIC": str(head.get("TICID", "")),
                "GAIADR3": int(head.get("GAIADR3", -1)),
                "sector": int(head.get("SECTOR", -1)),
                "camera": int(head.get("CAMERA", -1)),
                "ccd": int(head.get("CCD", -1)),
            }
            for col in EXTRA_COLUMNS:
                if col in available:
                    values = np.asarray(data[col])
                    row[col] = values[good]
                else:
                    row[col] = None
            return row
    except Exception:
        return None


if __name__ == "__main__":
    files = sorted(glob.glob(os.path.join(FITS_ROOT, "s*", "**", "*.fits"), recursive=True))
    print(f"{len(files)} FITS files found under {FITS_ROOT}")
    if not files:
        raise SystemExit("no FITS files -- if they were tarred/deleted after the "
                         "original extraction, restore them before running this")

    rows = []
    with Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(extract_one, files, chunksize=64)):
            if r is not None:
                rows.append(r)
            if i % 500 == 0:
                print(f"{i} / {len(files)} processed")

    df = pd.DataFrame(rows)
    for col in EXTRA_COLUMNS:
        n_missing = int(df[col].isna().sum()) if col in df.columns else len(df)
        if n_missing:
            print(f"WARNING: column {col} missing for {n_missing}/{len(df)} rows")

    df.to_parquet(OUT_PATH)
    print(f"wrote {OUT_PATH} ({len(df)} rows)")
