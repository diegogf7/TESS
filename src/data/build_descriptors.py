import numpy as np
import pandas as pd
import tessvectors

BASE = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split"
OUTDIR = "/orcd/scratch/orcd/006/diegogon/phyts/TESS"

def curve_descriptor(time, vec):
    midtime = vec["MidTime"].to_numpy()

    jitter = np.sqrt(vec["Quat1_StdDev"]**2 + vec["Quat2_StdDev"]**2 + vec["Quat3_StdDev"]**2).to_numpy()
    earth = vec["Earth_Distance"].to_numpy()

    moon = vec["Moon_Distance"].to_numpy()

    t = np.asarray(time)
    idx = np.searchsorted(midtime, t)
    idx = np.clip(idx, 1, len(midtime) - 1)

    pick_left = (t - midtime[idx - 1]) <= (midtime[idx] - t)

    idx = np.where(pick_left, idx - 1, idx)
    j, e, m = jitter[idx], earth[idx], moon[idx]

    return np.array([np.nanmedian(j), np.nanpercentile(j, 95), np.nanmin(e), np.nanmedian(e), np.nanmin(m), np.nanmedian(m)])

vec_cache = {}
def get_vec(s):
    s = int(s)

    if s not in vec_cache:

        vec_cache[s] = tessvectors.getvector(("FFI", s, 1))
    
    return vec_cache[s]

for split in ["train", "val"]:
    df = pd.read_parquet(f"{BASE}/tess_classification_{split}.parquet")
    times = df["time"].to_numpy()

    secs = df["sector"].to_numpy()

    D = np.zeros((len(df), 6))

    for i in range(len(df)):
        D[i] = curve_descriptor(times[i], get_vec(secs[i]))
        if i % 2000 == 0:
            print(split, i, "/", len(df))

    np.save(f"{OUTDIR}/descriptors_{split}.npy", D)
    print("saved", split, D.shape)

    overall = D.std(axis=0)
    uniq = np.unique(secs)
    within = np.array([
        np.average([D[secs == s][:, k].std() for s in uniq],
                   weights=[(secs == s).sum() for s in uniq])
        for k in range(6)
    ])
    print(split, "within/overall per descriptor:", np.round(within / overall, 3))