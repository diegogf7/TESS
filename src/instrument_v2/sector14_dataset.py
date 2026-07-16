# All this code is from Claude
"""Sector-14 chip-pair dataset + frozen splits for the instrument JEPA v2.

Data contract (see README.md):
  - RAW `flux` column only (TGLC aperture_flux). `flux_cal` is never read.
  - Every finite raw-flux cadence is retained -- quality flags do NOT remove
    observations in the primary experiment.
  - Median/MAD normalization per curve from its observed flux.
  - Two grid arms: shared_sector_1024 (one global Sector-14 time range for
    every star) and legacy_local_1024 (per-curve local grid).
  - Unobserved bins hold normalized zero; the observed mask is returned and
    passed to the encoder and loss. No random iid gap infilling.

Splits:
  - train/test TIC lists are FROZEN artifacts of the Step-1 diagnostic and are
    loaded, never re-drawn.
  - 10% of train TICs become validation (seed 43), saved next to the
    experiment artifacts. Train/val/test are asserted mutually disjoint.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from src.instrument_v2.diagnose_chip_common_signal import (
    GRID_1024,
    chip_index,
    legacy_local_grid,
    normalize_median_mad,
)

VAL_SEED = 43
VAL_FRAC = 0.10


# ----------------------------------------------------------------- splits
def load_frozen_split(split_dir):
    """Load the frozen train/test TIC lists written by the Step-1 diagnostic."""
    def read(name):
        path = os.path.join(split_dir, name)
        if not os.path.exists(path):
            raise RuntimeError(f"frozen split file missing: {path}")
        with open(path) as fh:
            return {line.strip() for line in fh if line.strip()}
    train = read("split_train_tics.txt")
    test = read("split_test_tics.txt")
    if train & test:
        raise RuntimeError("frozen split has overlapping train/test TICs")
    return train, test


def carve_validation(train_tics, seed=VAL_SEED, frac=VAL_FRAC):
    """Deterministically move `frac` of the TRAIN TICs into a validation set."""
    ordered = np.sort(np.asarray(sorted(train_tics), dtype=str))
    perm = np.random.default_rng(seed).permutation(len(ordered))
    n_val = max(1, int(round(frac * len(ordered))))
    val = set(ordered[perm[:n_val]].tolist())
    remaining = set(ordered.tolist()) - val
    return remaining, val


def ensure_splits(split_dir, art_dir):
    """Load frozen train/test, carve (or reload) validation, assert disjoint.

    Returns (train, val, test) TIC sets. The validation list is saved to
    art_dir on first creation and reloaded identically afterwards.
    """
    train_full, test = load_frozen_split(split_dir)
    os.makedirs(art_dir, exist_ok=True)
    val_path = os.path.join(art_dir, "split_val_tics.txt")
    if os.path.exists(val_path):
        with open(val_path) as fh:
            val = {line.strip() for line in fh if line.strip()}
        if not val <= train_full:
            raise RuntimeError("saved validation TICs are not a subset of frozen train")
        train = train_full - val
    else:
        train, val = carve_validation(train_full)
        with open(val_path, "w") as fh:
            fh.write("\n".join(sorted(val)))
    assert not (train & val) and not (train & test) and not (val & test), \
        "train/val/test TIC sets overlap"
    return train, val, test


# ------------------------------------------------------------------ grids
def shared_grid_bin(time, t0, t1, grid_length=GRID_1024):
    """Absolute time -> shared bin index; identical for every star by
    construction (same t0/t1/grid_length)."""
    span = max(t1 - t0, 1e-9)
    return np.clip(((np.asarray(time, dtype=np.float64) - t0) / span
                    * grid_length).astype(np.int64), 0, grid_length - 1)


def grid_curve_shared(time, flux, t0, t1, grid_length=GRID_1024):
    """One curve onto the shared Sector-14 grid; bin-average duplicates."""
    b = shared_grid_bin(time, t0, t1, grid_length)
    total = np.zeros(grid_length)
    count = np.zeros(grid_length)
    np.add.at(total, b, np.asarray(flux, dtype=np.float64))
    np.add.at(count, b, 1.0)
    hit = count > 0
    x = np.zeros(grid_length, dtype=np.float32)
    x[hit] = (total[hit] / count[hit]).astype(np.float32)
    return x, hit.astype(np.float32)


def grid_curve_legacy(time, flux, grid_length=GRID_1024):
    """Per-curve local grid (mirrors the legacy pipeline)."""
    return legacy_local_grid(time, flux, grid_length)


def train_time_range(df, train_tics):
    """Global time range from TRAIN stars only -- test TICs never touched."""
    is_train = df["TIC"].astype(str).isin(train_tics).to_numpy()
    if not is_train.any():
        raise RuntimeError("no train TICs found in dataframe")
    t0 = min(float(np.min(t)) for t in df["time"][is_train])
    t1 = max(float(np.max(t)) for t in df["time"][is_train])
    return t0, t1


def ensure_time_range(art_dir, df, train_tics):
    """Persist the shared-grid time range so every arm/seed/eval agrees."""
    path = os.path.join(art_dir, "grid_range.json")
    if os.path.exists(path):
        with open(path) as fh:
            saved = json.load(fh)
        return float(saved["t0"]), float(saved["t1"])
    t0, t1 = train_time_range(df, train_tics)
    os.makedirs(art_dir, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"t0": t0, "t1": t1}, fh)
    return t0, t1


def grid_frame(df, arm, t_range, grid_length=GRID_1024):
    """Normalize (median/MAD) and grid every row of df. RAW flux only.

    Retains every finite raw-flux cadence -- quality flags are ignored here
    by design (primary experiment; see README).
    """
    n = len(df)
    X = np.zeros((n, grid_length), dtype=np.float32)
    M = np.zeros_like(X)
    for i in range(n):
        time = np.asarray(df["time"].iloc[i], dtype=np.float64)
        flux = np.asarray(df["flux"].iloc[i], dtype=np.float64)   # RAW only
        finite = np.isfinite(time) & np.isfinite(flux)
        time, flux = time[finite], flux[finite]
        normed, _, _ = normalize_median_mad(flux)
        if arm == "shared":
            X[i], M[i] = grid_curve_shared(time, normed, *t_range, grid_length)
        elif arm == "legacy":
            X[i], M[i] = grid_curve_legacy(time, normed, grid_length)
        else:
            raise ValueError(f"unknown arm {arm!r} (use 'shared' or 'legacy')")
    return X, M


# ---------------------------------------------------------------- dataset
class Sector14ChipPairDataset(Dataset):
    """Positive pairs: two DIFFERENT TICs, same camera, same CCD, Sector 14.

    Chip groups are sampled UNIFORMLY first, then two distinct stars within
    the chip -- row-uniform sampling would let camera 1 dominate. Curves are
    pre-gridded at init (raw flux, median/MAD normalized, mask preserved).
    """

    def __init__(self, df, allowed_tics, arm, t_range, grid_length=GRID_1024,
                 pairs_per_epoch=None, sector=14, return_chip=False):
        self.return_chip = return_chip
        df = df[df["sector"] == sector]
        df = df[df["TIC"].astype(str).isin(set(allowed_tics))].reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError("no rows left after TIC filtering")
        self.tics = df["TIC"].astype(str).to_numpy()
        self.chips = np.array([chip_index(c, d)
                               for c, d in zip(df["camera"], df["ccd"])])
        self.X, self.M = grid_frame(df, arm, t_range, grid_length)

        self.chip_rows = {}
        for chip in np.unique(self.chips):
            rows = np.flatnonzero(self.chips == chip)
            if len(rows) >= 2:                     # need two distinct stars
                self.chip_rows[int(chip)] = rows
        dropped = sorted(set(np.unique(self.chips).tolist())
                         - set(self.chip_rows.keys()))
        if dropped:
            print(f"WARNING: chips {dropped} have <2 stars and are excluded from pairing")
        if not self.chip_rows:
            raise RuntimeError("no chip has >= 2 stars")
        self.chip_list = sorted(self.chip_rows.keys())
        self.pairs_per_epoch = pairs_per_epoch or len(df)

    def __len__(self):
        return self.pairs_per_epoch

    def _sample_pair(self):
        """(row_a, row_b, chip): chip drawn uniformly, then two distinct stars."""
        chip = self.chip_list[np.random.randint(len(self.chip_list))]
        a, b = np.random.choice(self.chip_rows[chip], size=2, replace=False)
        return int(a), int(b), int(chip)

    def __getitem__(self, idx):
        a, b, chip = self._sample_pair()
        items = (torch.tensor(self.X[a]), torch.tensor(self.M[a]),
                 torch.tensor(self.X[b]), torch.tensor(self.M[b]))
        if self.return_chip:
            return items + (torch.tensor(chip, dtype=torch.int64),)
        return items
