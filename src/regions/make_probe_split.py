import pandas as pd

FULL = "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_train_full_area.parquet"
TRAIN = "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_train_40k_area.parquet"
OUT = "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_probe20k_area.parquet"

train_ids = set(pd.read_parquet(TRAIN, columns = ["GAIADR3"])["GAIADR3"])

pool = pd.read_parquet(FULL)
pool = pool[~pool["GAIADR3"].isin(train_ids)]
pool = pool.drop_duplicates("GAIADR3")

#this line is completely from claude code I kept failing my tests 
probe = (pool.groupby(["sector", "camera", "ccd"], group_keys=False).apply(lambda g: g.sample(min(len(g), 48), random_state=0)))

counts = probe.groupby(["sector", "camera", "ccd"]).size()

print(f"{len(probe)} curves, {probe["GAIADR3"].nunique()} stars, min/max per chip class: {counts.min()} / {counts.max()}")
probe.to_parquet(OUT)