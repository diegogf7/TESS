# All this code is from Claude
"""Fresh star-disjoint 85/15 train/test split over the group-32 v2 dense
parquet, then report per-area FINAL train counts (after the 10% val carve that
ensure_splits applies) so we confirm every area clears 256 BEFORE submitting.

    python -m src.tglc.make_dense_v2_split
"""
import os

import numpy as np
import pandas as pd

from src.instrument_v2.sector14_dataset import carve_validation

ROOT = "/orcd/scratch/orcd/006/diegogon/tglc_primary"
FINAL = f"{ROOT}/tglc_raw_cadence_s14_dense_v2.parquet"
SPLIT_DIR = "artifacts/instrument_v2/dense_v2_split"
TEST_FRAC = 0.15
SEED = 14

os.makedirs(SPLIT_DIR, exist_ok=True)
d = pd.read_parquet(FINAL, columns=["TIC", "area"]).drop_duplicates("TIC")
d["TIC"] = d["TIC"].astype(str)
tics = np.sort(d["TIC"].values)

perm = np.random.default_rng(SEED).permutation(len(tics))
n_test = int(round(TEST_FRAC * len(tics)))
test = set(tics[perm[:n_test]].tolist())
train_full = set(tics[perm[n_test:]].tolist())

with open(f"{SPLIT_DIR}/split_train_tics.txt", "w") as fh:
    fh.write("\n".join(sorted(train_full)))
with open(f"{SPLIT_DIR}/split_test_tics.txt", "w") as fh:
    fh.write("\n".join(sorted(test)))

# real train = train_full minus the 10% val carve (seed 43) that ensure_splits does
train_final, val = carve_validation(train_full)
area = d.set_index("TIC")["area"].astype(int)
tr = area[area.index.isin(train_final)]
g = tr.groupby(tr.values).size()
short = g[g < 256]

print(f"split_dir: {SPLIT_DIR}")
print(f"tics {len(tics)} | train {len(train_final)} val {len(val)} test {len(test)}")
print(f"areas {len(g)} | train/area  min {g.min()}  median {int(g.median())}  max {g.max()}")
print(f"areas < 256 train: {len(short)}")
if len(short):
    print("STILL SHORT (area: train_count):", {int(a): int(n) for a, n in short.items()})
else:
    print(">>> ALL AREAS >= 256 TRAIN -- ready to submit <<<")
