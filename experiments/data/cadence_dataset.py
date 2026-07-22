# All this code is from Claude
"""Cadence-aligned dataset: same FFI cadence -> same tensor index, every star.

The legacy resample_to_grid() warps each curve onto its OWN
linspace(time[0], time[-1], 1024) grid, so a shared instrument event (a
momentum dump, a scattered-light ramp) lands at a different index for every
star. This dataset instead builds one common grid per sector from
cadence_num: index = cadence_num - sector_min_cadence. No interpolation, no
per-curve warping.

Requires a parquet with a cadence_num column (see
src/tglc/extract_raw_parquet_cadence.py). Returns three tensors per curve:
  flux      (L,) normalized flux, 0.0 where unobserved
  observed  (L,) 1 = cadence present in this curve's data
  quality   (L,) 1 = present AND all quality flags clean

Tests: python -m src.tests.test_cadence_dataset
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.data import normalize

QUALITY_COLUMNS = ("TESS_flags", "TGLC_flags")


def sector_cadence_grid(cadence_arrays, pad_to_multiple=16):
    """Common grid for one sector: (first_cadence, grid_length).

    grid_length spans min..max cadence over all curves, padded up to a
    multiple of pad_to_multiple so the S4 tokenizer can reshape it.
    """
    first = min(int(np.min(c)) for c in cadence_arrays)
    last = max(int(np.max(c)) for c in cadence_arrays)
    length = last - first + 1
    if pad_to_multiple:
        remainder = length % pad_to_multiple
        if remainder:
            length += pad_to_multiple - remainder
    return first, length


def place_on_grid(values, cadence_num, first_cadence, grid_length):
    """Scatter per-cadence values onto the common grid. Pure numpy, no interp."""
    grid = np.zeros(grid_length, dtype=np.float32)
    idx = np.asarray(cadence_num, dtype=np.int64) - first_cadence
    if idx.min() < 0 or idx.max() >= grid_length:
        raise ValueError("cadence_num outside the sector grid")
    grid[idx] = values
    return grid, idx


class CadenceAlignedDataset(Dataset):
    """One sector per instance -- cadence grids are sector-specific by design.

    df_or_parquet: DataFrame or parquet path with columns
        time, flux, cadence_num, sector (+ optional TESS_flags/TGLC_flags).
    """

    def __init__(self, df_or_parquet, sector, pad_to_multiple=16):
        if isinstance(df_or_parquet, pd.DataFrame):
            df = df_or_parquet
        else:
            df = pd.read_parquet(df_or_parquet)
        if "cadence_num" not in df.columns:
            raise ValueError("cadence_num column required -- re-extract with "
                             "src/tglc/extract_raw_parquet_cadence.py")
        self.df = df[df["sector"] == sector].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"no rows for sector {sector}")
        self.sector = sector
        self.first_cadence, self.grid_length = sector_cadence_grid(
            [np.asarray(c) for c in self.df["cadence_num"]], pad_to_multiple)
        self.quality_columns = [c for c in QUALITY_COLUMNS if c in self.df.columns]

    def index_of_cadence(self, cadence_num):
        """Tensor index of an FFI cadence -- identical for every star."""
        return int(cadence_num) - self.first_cadence

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cadence = np.asarray(row["cadence_num"], dtype=np.int64)
        flux = normalize(np.asarray(row["flux"], dtype=np.float64))

        grid_flux, grid_idx = place_on_grid(flux, cadence,
                                            self.first_cadence, self.grid_length)
        observed = np.zeros(self.grid_length, dtype=np.float32)
        observed[grid_idx] = 1.0

        clean = np.ones(len(cadence), dtype=bool)
        for col in self.quality_columns:
            flags = np.asarray(row[col])
            clean &= (flags == 0)
        quality = np.zeros(self.grid_length, dtype=np.float32)
        quality[grid_idx[clean]] = 1.0

        return (torch.tensor(grid_flux, dtype=torch.float32),
                torch.tensor(observed, dtype=torch.float32),
                torch.tensor(quality, dtype=torch.float32))
