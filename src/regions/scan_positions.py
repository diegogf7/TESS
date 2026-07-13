import glob, os
from multiprocessing import Pool

import pandas as pd
from astropy.io import fits

FITS_ROOT = "/orcd/scratch/orcd/006/diegogon/tglc_primary"
OUT = os.path.join(FITS_ROOT, "tglc_positions.parquet")
N_WORKERS = 16

def read_header(path):
    try:

        with fits.open(path, memmap = False) as hdul:

            h = hdul[0].header
            return {

                "GAIADR3": int(h.get("GAIADR3", -1)),
                "sector": int(h.get("SECTOR", -1)), 
                "camera": int(h.get("CAMERA", -1)),
                "ccd": int(h.get("CCD", -1)),
                "ra": float(h.get("RA_OBJ", float("nan"))),
                "dec": float(h.get("DEC_OBJ", float("nan"))),
                "tessmag": float(h.get("TESSMAG", float("nan"))),
            }
    
    except Exception:
        return None
    
if __name__ == "__main__":

    files = sorted(glob.glob(os.path.join(FITS_ROOT, "s*", "**", "*.fits"), recursive=True))

    print(f"{len(files)} FITS for files found")
    rows = []

    with Pool(N_WORKERS) as pool:

        for i, r in enumerate(pool.imap_unordered(read_header, files, chunksize = 256)):
            if r is not None:

                rows.append(r)
            if i % 400 == 0:
                print(f"{i} / {len(files)}")

        df = pd.DataFrame(rows).reset_index(drop = True)
        df.to_parquet(OUT)

        print(f"Wrote {OUT} with {len(files)} rows")

        