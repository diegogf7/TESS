"""Balanced same-chip set sampling for group-level instrument JEPA.

This module deliberately reuses the frozen TIC splits and raw shared-grid
preprocessing from :mod:`sector14_dataset`.  The only experimental change is
that one item contains two disjoint sets of stars from the same camera/CCD.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from src.instrument_v2.sector14_dataset import Sector14ChipPairDataset


class Sector14ChipGroupDataset(Sector14ChipPairDataset):
    """Two disjoint sets of ``group_size`` TICs from one uniformly drawn chip.

    ``pairs_per_epoch`` defaults to ``ceil(n_stars / group_size)``.  Together
    with a batch size scaled as ``256 / group_size``, this keeps the number of
    S4D curve encodes and optimizer steps close to the original pair-JEPA.
    """

    def __init__(self, *args, group_size=8, pairs_per_epoch=None, **kwargs):
        if group_size < 2:
            raise ValueError("group_size must be >= 2")
        self.group_size = int(group_size)
        super().__init__(*args, pairs_per_epoch=pairs_per_epoch, **kwargs)

        minimum = 2 * self.group_size
        self.chip_rows = {
            chip: rows for chip, rows in self.chip_rows.items() if len(rows) >= minimum
        }
        if not self.chip_rows:
            raise RuntimeError(
                f"no chip has >= {minimum} stars required for two disjoint groups"
            )
        self.chip_list = sorted(self.chip_rows)
        if pairs_per_epoch is None:
            self.pairs_per_epoch = max(1, math.ceil(len(self.tics) / self.group_size))

    def _sample_groups(self):
        chip = self.chip_list[np.random.randint(len(self.chip_list))]
        rows = np.random.choice(
            self.chip_rows[chip], size=2 * self.group_size, replace=False
        )
        return rows[: self.group_size], rows[self.group_size :], int(chip)

    def __getitem__(self, idx):
        context_rows, target_rows, chip = self._sample_groups()
        items = (
            torch.from_numpy(self.X[context_rows]),
            torch.from_numpy(self.M[context_rows]),
            torch.from_numpy(self.X[target_rows]),
            torch.from_numpy(self.M[target_rows]),
        )
        if self.return_chip:
            return items + (torch.tensor(chip, dtype=torch.int64),)
        return items
