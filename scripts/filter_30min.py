import numpy as np
import pandas as pd

def cadence_minutes(time):

    time = np.asarray(time)
    return float(np.median(np.diff(time))) * 1440.0 

def filter_30min(in_path, out_path, low = 26.0, high = 34.0):
    df = pd.read_parquet(in_path)

    df["cadence_min"] = df["time"].apply(cadence_minutes)
    keep = df["cadence_min"].between(low, high) & (df["label"] != "INSTRUMENT/JUNK")

    print(f"{in_path}: keeping {keep.sum()}/{len(df)} "f"({100*keep.mean():.1f}%)")
    print(df["cadence_min"].round(1).value_counts().head()) #just a smoke test I'm putting in
    df[keep].reset_index(drop=True).to_parquet(out_path)

if __name__ == "__main__":

    base = "/orcd/scratch/orcd/006/diegogon/phyts/TESS/TESS/split"
    for split in ["train", "val"]:

        filter_30min(f"{base}/tess_classification_{split}.parquet", f"{base}/tess_classification_{split}_30min.parquet")
