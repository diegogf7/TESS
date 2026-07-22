import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from astropy.io import fits

FITS_ROOT = "/orcd/scratch/orcd/006/diegogon/tglc_primary"
OUT_TRAIN = os.path.join(FITS_ROOT, "tglc_pretrain_train.parquet")
OUT_VALIDATION = os.path.join(FITS_ROOT, "tglc_pretrain_val.parquet")

FLUX_COL = "cal_aper_flux"
MIN_COVERAGE = 0.0
MIN_POINTS = 200
VAL_FRACTION = 0.02
N_WORKERS = 16


#basically all this code is from Claude to try to deal with this data
def extract_one(path):

    try:
        with fits.open(path, memmap = False) as hdul:
            data = hdul[1].data
            head = hdul[0].header

            time = np.asarray(data["time"], dtype = np.float64)
            flux = np.asarray(data[FLUX_COL], dtype = np.float32)

            good = (np.asarray(data["TESS_flags"]) == 0) & (np.asarray(data["TGLC_flags"]) ==0)

            good &= np.isfinite(time) & np.isfinite(flux)

            if good.mean() < MIN_COVERAGE or good.sum() < MIN_POINTS:
                return None
            
            return {
                "time": time[good],
                "flux": flux[good],
                "TIC": str(head.get("TICID", "")),
                "GAIADR3": int(head.get("GAIADR3", -1)),
                "sector": int(head.get("SECTOR", -1)),
                "camera": int(head.get("CAMERA", -1)),
                "ccd": int(head.get("CCD", -1)),
                "coverage": float(good.mean())

            }
    except Exception:
        return None
    
if __name__=="__main__":
    files = sorted(glob.glob(os.path.join(FITS_ROOT, "s*", "**", "*.fits"), recursive = True))

    print(f"{len(files)} FITS files found")

    rows = []
    with Pool(N_WORKERS) as pool:

        for i, r in enumerate(pool.imap_unordered(extract_one, files, chunksize = 64)):
            if r is not None:

                rows.append(r)
            
            if i % 500 == 0:
                print(f"{i} / {len(files)} processed | {len(rows)} kept", flush = True)
    
    df = pd.DataFrame(rows)

    print(f"kept {len(df)} / {len(files)} curves ({100 * len(df) / len(files):.2f})")

    print(f"curves per sector: {df['sector'].value_counts().sort_index().to_string()}")

    random_number = np.random.default_rng(0)
    validation_mask = random_number.random(len(df)) < VAL_FRACTION
    df[~validation_mask].to_parquet(OUT_TRAIN)
    df[validation_mask].to_parquet(OUT_VALIDATION)
    print(f"\nwrote {OUT_TRAIN} ({(~validation_mask).sum()} rows)")
    print(f"wrote {OUT_VALIDATION} ({validation_mask.sum()} rows)")

