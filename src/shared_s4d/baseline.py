from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.instrument_v2.area_commonmode_dataset import Sector14GroupStatDataset, ensure_area_column
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.shared_s4d.ae_dataset import AreaGroupAEDataset
from src.shared_s4d.correction_losses import mean_abs_pairwise_corr
from src.shared_s4d.train_correction import (SEED, GROUP_SIZE, N_STARS, MIN_OVERLAP,
                                             S14_DATA, SPLIT_DIR, BASE_ART_DIR)


def _roll_group(Xg, Mg, rng):

    Xr, Mr = Xg.clone(), Mg.clone()
    L = Xg.shape[1]

    for i in range(Xg.shape[0]):
        s = int(rng.integers(1, L))

        Xr[i] = torch.roll(Xg[i], s); Mr[i] = torch.roll(Mg[i], s)
    
    return Xr, Mr

def common_mode_fraction(Xg, Mg):

    m = (Mg > 0).float()
    cm = (Xg * m).sum(0) / m.sum(0).clamp(min = 1.0)

    valid = m.sum(0) >= 4
    if int(valid.sum()) < 2:
        return np.nan
    
    cm_rms = float(torch.sqrt((cm[valid] ** 2).mean()))

    per = torch.sqrt((Xg ** 2 * m).sum(1) / m.sum(1).clamp(min = 1.0))

    return cm_rms / float(per.mean().clamp(min = 1e-8))

def main():
    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    df = ensure_area_column(df)
    tr, va, te = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, tr)
    base_va = Sector14GroupStatDataset(df, va, t_range, "area", GROUP_SIZE, min_valid=16)
    val_ds = AreaGroupAEDataset(base_va.X, base_va.M, base_va.areas, base_va.tics,
                                n_stars=N_STARS, group_size=GROUP_SIZE, seed=SEED,
                                require_full=False, resample=False)
    rng = np.random.default_rng(SEED)

    same, shuf, cmf = [], [], []
    for k in range(len(val_ds)):
        Xg, Mg = val_ds[k]
        same.append(mean_abs_pairwise_corr(Xg, Mg, MIN_OVERLAP))
        Xr, Mr = _roll_group(Xg, Mg, rng)
        shuf.append(mean_abs_pairwise_corr(Xr, Mr, MIN_OVERLAP))
        cmf.append(common_mode_fraction(Xg, Mg))

    X = torch.tensor(base_va.X); M = torch.tensor(base_va.M); areas = np.asarray(base_va.areas)
    uniq = np.unique(areas)
    cross = []
    for _ in range(len(val_ds)):                            # 32 stars, each a different area
        chosen = rng.choice(uniq, size=min(GROUP_SIZE, len(uniq)), replace=False)
        rows = [int(rng.choice(np.where(areas == a)[0])) for a in chosen]
        cross.append(mean_abs_pairwise_corr(X[rows], M[rows], MIN_OVERLAP))

    f = lambda v: float(np.nanmean(v))
    floor = min(f(cross), f(shuf))
    print(f"same-area        mean|corr| : {f(same):.3f}   (starting point)")
    print(f"cross-area       mean|corr| : {f(cross):.3f}   (different areas)")
    print(f"cadence-shuffled mean|corr| : {f(shuf):.3f}   (instrument alignment destroyed)")
    print(f"FLOOR                       : {floor:.3f}")
    print(f"common-mode RMS fraction    : {f(cmf):.3f}   (target c/x for FULL removal)")
    print(f"max legit reduction         : {100 * (f(same) - floor) / f(same):.0f}%")


if __name__ == "__main__":
    main()
