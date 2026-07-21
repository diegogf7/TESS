import os 
import collections

import numpy as np
import pandas as pd

from tess_stars2px import tess_stars2px_function_entry

POSITIONS = os.environ.get(
    "POSITIONS",
    "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_positions.parquet",
)

OUT = os.environ.get(
    "MISSING_OUT",
    "/orcd/scratch/orcd/006/diegogon/tglc_primary/missing_sectors.parquet",
)

#first I need to load the stars
df = pd.read_parquet(POSITIONS, columns = ["GAIADR3", "sector", "ra", "dec"])
present = df.groupby("GAIADR3")["sector"].agg(set).to_dict()

#Just need to work with curves that I have downloaded

scope = {int(s) for s in df["sector"].unique() if s > 0}

#need to see each row per star
proper_stars = df.drop_duplicates("GAIADR3")
#need a filter of having more than 0 stars + proper coordinates in space system
proper_stars = proper_stars[(proper_stars["GAIADR3"] >0) & proper_stars["ra"].notna() & proper_stars["dec"].notna()]

gaia_id = proper_stars["GAIADR3"].to_numpy()

ra = proper_stars["ra"].to_numpy()
dec = proper_stars["dec"].to_numpy()

print(f"{len(gaia_id)} stars with scope = sectors {sorted(scope)}")

index = np.arange(len(gaia_id))

#keep getting errors as to how many inputs comes out of function
out_ID, _, _, out_Sector, _, _, _, _, _ = tess_stars2px_function_entry(index, ra, dec)

predicted = collections.defaultdict(set)
for i, sector in zip(out_ID, out_Sector):

    if sector >0:
        predicted[int(i)].add(int(sector))

    
#now for report

rows = []

for i, sid in enumerate(gaia_id):
    sid = int(sid)

    presents = present.get(sid, set())
    prediction = predicted.get(i, set()) & scope
    missing = prediction - presents

    rows.append({
        "GAIADR3": sid,
        "n_present": len(presents),
        "n_predicted": len(prediction),
        "n_missing": len(missing)
        "missing_sectors": sorted(missing),
    })


final = (pd.DataFrame(rows).sort_values("n_missing", ascending = False).reset_index(drop = True))


star_number = len(final)
number_missing = int((final["n_missing"] > 0).sum())

