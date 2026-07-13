import pandas as pd

from src.regions.areas import add_area

ROOT = "/orcd/scratch/orcd/006/diegogon/tglc_primary"
positions = pd.read_parquet(f"{ROOT}/tglc_positions.parquet")
positions = positions[["GAIADR3", "sector", "ra", "dec", "tessmag"]]

for name in ["tglc_raw_train_full", "tglc_raw_train_40k", "tglc_raw_val"]:

    df = pd.read_parquet(f"{ROOT}/{name}.parquet")
    merged = df.merge(positions, on = ["GAIADR3", "sector"], how = "left", validate = "many_to_one")

    merged = add_area(merged).reset_index(drop = True)

    out = f"{ROOT}/{name}_area.parquet"
    merged.to_parquet(out)

    print(f"{out}: {len(merged)} rows, unmatched ra: {merged['ra'].isna().sum()}, area: {(merged['area'] == -1).sum()}")

    