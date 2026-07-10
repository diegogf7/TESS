import glob

import os
from multiprocessing import Pool

import numpy as np
import pandas as pd

from astropy.io import fits

FITS_ROOT = "/orcd/scratch/orcd/006/diegogon/tglc_primary"
OUT_TRAIN = os.path.join(FITS_ROOT, "tglc_raw_train.parquet")
OUT_VALIDATION = os.path.join(FITS_ROOT, "tglc_raw_val.parquet")

MIN_POINTS = 200
VAL_FRACTION = 0.02
N_WORKERS = 16

def extract_one(path):

    try:
        with fits.open(path, memmap = False) as hdul:
            data = hdul[1].data

            head = hdul[0].header

            time = np.asarray(data["time"], dtype = np.float64)
            raw = np.asarray(data["aperture_flux"], dtype = np.float64)
            calibration = np.asarray(data["cal_aper_flux"], dtype = np.float64)

            good = np.isfinite(time) & np.isfinite(raw)

            if good.sum() < MIN_POINTS:

                return None
            
            return {
                "time": time[good],
                "flux": raw[good],
                "flux_cal": calibration[good],
                "TIC": str(head.get("TICID", "")),
                "GAIADR3": int(head.get("GAIADR3", -1)),
                "sector": int(head.get("SECTOR", -1)),
                "camera": int(head.get("CAMERA", -1)),
                "ccd": int(head.get("CCD", -1)),

            }
    
    except Exception:
        return None
    
if __name__ == "__main__":

    files = sorted(glob.glob(os.path.join(FITS_ROOT, "s*", "**", "*.fits"), recursive = True))

    print(f"{len(files)} FITS files found")

    rows = []

    with Pool(N_WORKERS) as pool:
        for i, r in enumerate(pool.imap_unordered(extract_one, files, chunksize = 64)):
            if r is not None:

                rows.append(r)
            if i % 500 == 0:
                print(f"{i} / {len(files)} processed")

    df = pd.DataFrame(rows)

    random_number = np.random.default_rng(0)
    validation_mask = random_number.random(len(df)) < VAL_FRACTION
    df[~validation_mask].to_parquet(OUT_TRAIN)

    df[validation_mask].to_parquet(OUT_VALIDATION)

    print(f"\n wrote {OUT_TRAIN} ({(~validation_mask).sum()} rows)")
    print(f"wrote {OUT_VALIDATION} ({validation_mask.sum()} rows)")