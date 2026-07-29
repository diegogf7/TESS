from __future__ import annotations
"""Part-A (regional teacher) dynamic pair sampler, scaled.

Per epoch, for each area (which must have a CBV basis and >= n_stars training
stars): from a FIXED per-area pool of exactly n_stars stars, generate n_pairs
Group-A/Group-B pairs. Each pair draws 2*group_size distinct same-area stars
without replacement -> first group_size = Group A, remaining group_size = Group
B (A and B disjoint). Pairs are regenerated every epoch from a deterministic
seed [seed, epoch, area, pair_index]; groups may overlap across pairs.

Each group's 2-channel fingerprint is (K=8 CBV ridge reconstruction of the group
median, log-MAD) -- identical to the existing AreaGroupCBVPairDataset, so the
trained teacher stays compatible with the existing Part-B student. Requires >=
min_valid observed stars per cadence.

If require_full and ANY area with a basis has < n_stars training stars, __init__
RAISES with a per-area count report -- it never duplicates stars.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.instrument_v2.area_commonmode_dataset import area_to_chip
from src.instrument_v2.regional_cbv import cbv_fingerprint


class DynamicAreaPairDataset(Dataset):
    def __init__(self, X, M, areas, tics, area_bases, group_size, min_valid, ridge_lambda,
                 n_stars=1000, n_pairs=1000, seed=0, resample=True, require_full=True):
        self.X, self.M = X, M
        self.areas = np.asarray(areas, dtype=np.int64)
        self.tics = np.asarray(tics, dtype=str)
        self.area_bases = area_bases
        self.group_size = int(group_size)
        self.min_valid = int(min_valid)
        self.ridge_lambda = float(ridge_lambda)
        self.n_stars = int(n_stars)
        self.n_pairs = int(n_pairs)
        self.base_seed = int(seed)
        self.resample = bool(resample)
        need_pair = 2 * self.group_size                       # 64: enough for one A/B pair

        rows_by_area = {}
        for i, a in enumerate(self.areas):
            rows_by_area.setdefault(int(a), []).append(i)
        rows_by_area = {a: np.asarray(sorted(r), dtype=np.int64) for a, r in rows_by_area.items()}
        self.area_counts = {int(a): int(len(r)) for a, r in rows_by_area.items()}
        with_basis = [a for a in rows_by_area if a in area_bases]

        if require_full:
            short = {a: len(rows_by_area[a]) for a in with_basis if len(rows_by_area[a]) < self.n_stars}
            if short:
                report = "\n".join(f"    area {a}: {n} train stars (< {self.n_stars})"
                                   for a, n in sorted(short.items()))
                raise RuntimeError(
                    f"AREA-COUNT FAILURE: {len(short)}/{len(with_basis)} areas have fewer than "
                    f"{self.n_stars} training stars. Refusing to duplicate stars.\n{report}")
            # exactly n_stars per area, deterministic per (seed, area)
            self.pool = {}
            for a in with_basis:
                rng = np.random.default_rng([self.base_seed, int(a)])
                self.pool[a] = np.sort(rng.choice(rows_by_area[a], size=self.n_stars, replace=False))
            self.eligible = sorted(with_basis)
        else:                                                 # validation: use all available, need >= 64
            self.pool = {a: rows_by_area[a] for a in with_basis if len(rows_by_area[a]) >= need_pair}
            self.eligible = sorted(self.pool)
        if not self.eligible:
            raise RuntimeError("no area has a CBV basis and enough stars for a pair")
        self._build(0)

    def _build(self, epoch):
        e = epoch if self.resample else 0
        items = []
        for a in self.eligible:
            pool = self.pool[a]
            for p in range(self.n_pairs):
                rng = np.random.default_rng([self.base_seed, e, int(a), p])   # per (seed,epoch,area,pair)
                sel = rng.choice(pool, size=2 * self.group_size, replace=False)  # 64 distinct
                items.append((sel[:self.group_size].astype(np.int64),         # Group A
                              sel[self.group_size:].astype(np.int64),         # Group B (disjoint)
                              int(a)))
        self.items = items

    def set_epoch(self, epoch):
        if self.resample:
            self._build(int(epoch))

    def assert_contracts(self):
        per_area = {}
        for A, B, a in self.items:
            per_area[a] = per_area.get(a, 0) + 1
            assert len(A) == self.group_size and len(np.unique(A)) == self.group_size, "Group A not 32 unique"
            assert len(B) == self.group_size and len(np.unique(B)) == self.group_size, "Group B not 32 unique"
            assert not (set(A.tolist()) & set(B.tolist())), "Group A and B overlap"
        for a in self.eligible:
            assert per_area.get(a, 0) == self.n_pairs, (a, per_area.get(a, 0))   # exactly n_pairs
        return True

    def group_input(self, rows, area):
        instrument, log_mad, valid = cbv_fingerprint(
            self.X, self.M, rows, self.area_bases[int(area)], self.min_valid, self.ridge_lambda)
        stats = torch.stack([torch.tensor(instrument), torch.tensor(log_mad)], dim=-1)
        return stats, torch.tensor(valid)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        A, B, a = self.items[idx]
        stats_a, valid_a = self.group_input(A, a)
        stats_b, valid_b = self.group_input(B, a)
        return (stats_a, valid_a, stats_b, valid_b,
                torch.tensor(a, dtype=torch.int64),
                torch.tensor(area_to_chip(a), dtype=torch.int64))
