# All this code is from Claude
"""Group-statistic (common-mode) dataset for the area-conditioned JEPA.

One item = one CONTEXT star plus the median / robust-MAD curves of K OTHER
stars from the same group, computed cadence-by-cadence on the shared absolute
Sector-14 grid. Two matched grouping arms:

  chip: (sector, camera, CCD)          -- the control
  area: (sector, area)                 -- area = camera*100 + ccd*10 + ring,
                                          assigned by src/regions/areas.py
                                          (REUSED, never re-derived here)

Contracts:
  - Frozen TIC splits and the persisted shared-grid time range are reused
    from sector14_dataset; test TICs never enter.
  - RAW aperture flux, median/MAD per-curve normalization, observed masks,
    NO gap infilling (all inherited from Sector14ChipPairDataset).
  - Group statistics use only OBSERVED stars at each cadence; zero-filled
    bins never contribute. A target cadence is valid only when at least
    max(4, ceil(K/2)) target stars are observed there.
  - Groups are sampled uniformly; the context TIC is excluded from the
    target set by sampling K+1 distinct rows without replacement.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
import torch

from src.instrument_v2.diagnose_chip_common_signal import chip_index
from src.instrument_v2.sector14_dataset import Sector14ChipPairDataset

AREA_SOURCE = os.environ.get(
    "AREA_SOURCE",
    "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_train_full_area.parquet",
)
LOG_MAD_EPS = 1e-6


# ------------------------------------------------------------------- areas
def area_to_chip(area):
    """Existing area code -> 0..15 chip index (412 = cam 4, ccd 1, ring 2)."""
    area = int(area)
    return chip_index(area // 100, (area // 10) % 10)


def ensure_area_column(df, area_source=None):
    """Attach the EXISTING area assignment; never re-derive it.

    If df already carries `area`, it is used as-is. Otherwise areas are
    merged from the *_area parquet produced by src/regions/merge_areas.py,
    keyed on (TIC, sector). Rows without a valid area are dropped loudly;
    coverage below 90% raises rather than silently shrinking the sample.
    """
    if "area" not in df.columns:
        source = pd.read_parquet(area_source or AREA_SOURCE,
                                 columns=["TIC", "sector", "area"])
        source = source.drop_duplicates(["TIC", "sector"])
        source["TIC"] = source["TIC"].astype(str)
        df = df.copy()
        df["TIC"] = df["TIC"].astype(str)
        df = df.merge(source, on=["TIC", "sector"], how="left",
                      validate="many_to_one")
    valid = df["area"].notna() & (df["area"] != -1)
    n_bad = int((~valid).sum())
    if n_bad:
        print(f"WARNING: dropping {n_bad}/{len(df)} rows without a valid area",
              flush=True)
    if valid.sum() < 0.9 * len(df):
        raise RuntimeError(
            f"area coverage too low ({int(valid.sum())}/{len(df)}); check "
            f"AREA_SOURCE={area_source or AREA_SOURCE}")
    df = df[valid].reset_index(drop=True)
    df["area"] = df["area"].astype(int)
    # consistency with the existing chip assignment
    mismatch = int((df["area"] // 100 != df["camera"]).sum()
                   + ((df["area"] // 10) % 10 != df["ccd"]).sum())
    if mismatch:
        raise RuntimeError(f"{mismatch} rows where area code contradicts camera/ccd")
    return df


# ---------------------------------------------------------------- statistics
def group_statistics(flux, mask, min_valid):
    """Robust per-cadence statistics over the K target stars.

    flux, mask: (K, L) arrays; mask 1 = observed. Statistics use ONLY the
    observed stars at each cadence -- zero-filled bins never contribute.

    Returns (median, log_mad, valid, n_observed), each (L,) float32; median
    and log_mad are zero outside valid cadences.
    """
    flux = np.asarray(flux, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    values = np.where(mask > 0, flux, np.nan)
    n_observed = mask.sum(axis=0)
    valid = n_observed >= min_valid

    import warnings
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cadences -> masked below
        median = np.nanmedian(values, axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(values - median[None, :]), axis=0)
    median = np.where(valid, median, 0.0)
    log_mad = np.where(valid, np.log(np.maximum(mad, 0.0) + LOG_MAD_EPS), 0.0)
    return (median.astype(np.float32), log_mad.astype(np.float32),
            valid.astype(np.float32), n_observed.astype(np.float32))


def min_valid_stars(k):
    return max(4, math.ceil(k / 2))


# ------------------------------------------------------------------ dataset
class Sector14GroupStatDataset(Sector14ChipPairDataset):
    """Context star + same-group median/MAD target curves, shared grid only."""

    def __init__(self, df, allowed_tics, t_range, grouping, k,
                 grid_length=1024, items_per_epoch=None, sector=14,
                 min_valid=None):
        if grouping not in ("chip", "area"):
            raise ValueError(f"grouping must be 'chip' or 'area', got {grouping!r}")
        if grouping == "area" and "area" not in df.columns:
            raise ValueError("df lacks 'area'; call ensure_area_column first")
        self.grouping = grouping
        self.k = int(k)                       # group size (stars per group), NOT the CBV rank
        # min_valid is EXPLICIT. It is a cadence-validity threshold and MUST NOT be
        # derived from the CBV rank. Legacy callers that omit it keep the old
        # group-size heuristic; the group-CBV pipeline passes MIN_VALID_STARS.
        self.min_valid = int(min_valid) if min_valid is not None else min_valid_stars(self.k)
        if not (1 <= self.min_valid <= self.k):
            raise ValueError(
                f"min_valid must satisfy 1 <= min_valid <= group_size; got "
                f"min_valid={self.min_valid}, group_size={self.k}")

        # Mirror the parent's row filtering so per-row areas stay aligned.
        filtered = df[df["sector"] == sector]
        filtered = filtered[filtered["TIC"].astype(str).isin(set(allowed_tics))]
        filtered = filtered.reset_index(drop=True)
        super().__init__(filtered, set(filtered["TIC"].astype(str)), "shared",
                         t_range, grid_length=grid_length, sector=sector)
        self.areas = (filtered["area"].to_numpy(dtype=np.int64)
                      if "area" in filtered.columns else None)
        self.group_labels = self.chips if grouping == "chip" else self.areas

        self.group_rows = {}
        for group in np.unique(self.group_labels):
            rows = np.flatnonzero(self.group_labels == group)
            if len(rows) >= self.k + 1:      # context + K disjoint targets
                self.group_rows[int(group)] = rows
        if not self.group_rows:
            raise RuntimeError(
                f"no {grouping} group has >= {self.k + 1} stars (K={self.k})")
        self.group_list = sorted(self.group_rows)
        self.items_per_epoch = (items_per_epoch
                                or max(1, math.ceil(2 * len(self.tics) / (self.k + 1))))

    # -------------------------------------------------------------- summary
    def group_count_table(self):
        """(group, parent_chip, n_stars) rows for the pre-training diagnostic."""
        rows = []
        for group in np.unique(self.group_labels):
            n = int((self.group_labels == group).sum())
            chip = (int(group) if self.grouping == "chip"
                    else area_to_chip(group))
            rows.append({"grouping": self.grouping, "group": int(group),
                         "parent_chip": chip, "n_stars": n,
                         "usable": n >= self.k + 1})
        return rows

    def __len__(self):
        return self.items_per_epoch

    # -------------------------------------------------------------- sampling
    def _sample_item(self):
        group = self.group_list[np.random.randint(len(self.group_list))]
        rows = np.random.choice(self.group_rows[group], size=self.k + 1,
                                replace=False)
        return int(rows[0]), rows[1:], int(group)

    def __getitem__(self, idx):
        context, targets, group = self._sample_item()
        median, log_mad, valid, n_observed = group_statistics(
            self.X[targets], self.M[targets], self.min_valid)
        return (torch.tensor(self.X[context]), torch.tensor(self.M[context]),
                torch.tensor(median), torch.tensor(log_mad),
                torch.tensor(valid), torch.tensor(n_observed),
                torch.tensor(group, dtype=torch.int64))

    # ------------------------------------------------- cosine-metric samplers
    def sample_disjoint_same_group(self):
        """Two disjoint K-star target sets from one group (needs >= 2K stars)."""
        eligible = [g for g in self.group_list
                    if len(self.group_rows[g]) >= 2 * self.k]
        if not eligible:
            return None
        group = eligible[np.random.randint(len(eligible))]
        rows = np.random.choice(self.group_rows[group], size=2 * self.k,
                                replace=False)
        return rows[:self.k], rows[self.k:], group

    def sample_cross_group(self):
        """Two K-star sets from DIFFERENT groups sharing a parent.

        area arm: same camera x CCD, different area (the gate comparison).
        chip arm: same camera, different CCD (the analogous control).
        """
        parents = {}
        for group in self.group_list:
            parent = (area_to_chip(group) if self.grouping == "area"
                      else int(group) // 4)          # chip -> camera index
            parents.setdefault(parent, []).append(group)
        eligible = [siblings for siblings in parents.values() if len(siblings) >= 2]
        if not eligible:
            return None
        siblings = eligible[np.random.randint(len(eligible))]
        g1, g2 = np.random.choice(len(siblings), size=2, replace=False)
        g1, g2 = siblings[g1], siblings[g2]
        rows1 = np.random.choice(self.group_rows[g1], size=self.k, replace=False)
        rows2 = np.random.choice(self.group_rows[g2], size=self.k, replace=False)
        return rows1, rows2, (g1, g2)

    def group_target_tensors(self, rows):
        """Median/MAD/valid tensors for an explicit row set (metric helper)."""
        median, log_mad, valid, _ = group_statistics(
            self.X[rows], self.M[rows], self.min_valid)
        return (torch.tensor(median), torch.tensor(log_mad), torch.tensor(valid))


def valid_k_values(dataset_or_counts, candidates=(8, 16, 32), min_groups=8):
    """Which K from `candidates` have >= min_groups groups with K+1 stars."""
    if isinstance(dataset_or_counts, dict):
        sizes = list(dataset_or_counts.values())
    else:
        labels = dataset_or_counts.group_labels
        sizes = [int((labels == g).sum()) for g in np.unique(labels)]
    out = []
    for k in candidates:
        if sum(1 for n in sizes if n >= k + 1) >= min_groups:
            out.append(int(k))
    return out
