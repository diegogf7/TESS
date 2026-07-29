import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astroquery.mast import Catalogs

HF_REPO = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split"

FILES = ["tess_classification_train_30min.parquet",
         "tess_classification_val_30min.parquet",
         "tess_classification_test_30min.parquet"]

OUT_PNG = "phyts_sector14_tmag_hist.png"
BATCH = 200


frames= []
print(pd.read_parquet(f"{HF_REPO}/{FILES[0]}").columns.tolist())

for f in FILES:
    d = pd.read_parquet(f"{HF_REPO}/{f}", columns = None)
    frames.append(d[["TIC", "sector"]] if {"TIC", "sector"} <= set(d.columns) else d)

df = pd.concat(frames, ignore_index = True)
df = df[df["sector"] == 14].copy()
df["TIC"] = df["TIC"].astype(int)

df = df.drop_duplicates("TIC").reset_index(drop = True)
print(f"Sector-14 unique TICs: {len(df)}", flush = True)

ids = df["TIC"].tolist()
tmag_by_tic = {}
for k in range(0, len(ids), BATCH):

    chunk = ids[k:k + BATCH]
    try:

        t = Catalogs.query_criteria(catalog = "Tic", ID = chunk)
        tab = t["ID", "Tmag"].to_pandas()
        tab = tab[np.isfinite(tab["Tmag"])]

        tmag_by_tic.update(dict(zip(tab["ID"].astype(int), tab["Tmag"].astype(float))))

    except Exception as e:
        print(f"Chunk {k}-{k + len(chunk)} failed: {e}", flush = True)
    
    print(f"Queried {min(k + BATCH, len(ids))}/{len(ids)}", flush = True)

df["Tmag"] = df["TIC"].map(tmag_by_tic)

tm = df["Tmag"].dropna().to_numpy()
plt.figure(figsize = (7, 5))
plt.hist(tm, bins = 30, edgecolor = "black")
plt.xlabel("Magnitude")

plt.ylabel("Number of stars")
plt.title(f"PhyTS sector 14 Magnitude")
plt.tight_layout()
plt.savefig(OUT_PNG, dpi= 130)
plt.close()

matched = int(df["Tmag"].notna().sum())
missing = int(df["Tmag"].isna().sum())

print(f"Matched: {matched}")
print(f"Missing: {missing}")

if matched:
    print(f"Min magnitude: {np.min(tm):.5f}")
    print(f"median magnitude: {np.median(tm):.5f}")
    print(f"max magnitude: {np.max(tm):.5f}")

print(f"Wrote {OUT_PNG}", flush = True)