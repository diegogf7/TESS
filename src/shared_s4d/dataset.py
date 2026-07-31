from __future__ import annotations
"""Per-epoch group sampler + leave-one-out systematics targets.

Per area, a FIXED deterministic pool of exactly n_stars training stars. Each
epoch the pool is reshuffled (seed, epoch, area); the first floor(pool/32)*32
stars are partitioned into disjoint 32-star groups (31 groups for 1000 stars,
leaving 8 that rotate next epoch). For every curve i in a group the target is the
leave-one-out median of the OTHER 31 curves, valid where >=4 others are observed
(reuses the existing group_statistics). No group combos are precomputed.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.instrument_v2.area_commonmode_dataset import group_statistics


class AreaGroupLOODataset(Dataset):
    def __init__(self, X, M, areas, tics, n_stars=1000, group_size=32,
                 target_min_valid=4, seed=0, require_full=True, resample=True):
        self.X, self.M = X, M
        self.areas = np.asarray(areas, dtype=np.int64)
        self.tics = np.asarray(tics, dtype=str)
        self.group_size = int(group_size)
        self.n_stars = int(n_stars)
        self.target_min_valid = int(target_min_valid)
        self.base_seed = int(seed)
        self.resample = bool(resample)

        rows_by_area = {}
        for i, a in enumerate(self.areas):
            rows_by_area.setdefault(int(a), []).append(i)
        rows_by_area = {a: np.asarray(sorted(r), dtype=np.int64) for a, r in rows_by_area.items()}
        self.area_counts = {a: int(len(r)) for a, r in rows_by_area.items()}

        if require_full:
            short = {a: len(r) for a, r in rows_by_area.items() if len(r) < self.n_stars}
            if short:
                rep = "\n".join(f"    area {a}: {n} train stars (< {self.n_stars})"
                                for a, n in sorted(short.items()))
                raise RuntimeError(f"AREA-COUNT FAILURE: {len(short)} areas below "
                                   f"{self.n_stars} stars. Refusing to duplicate.\n{rep}")
            self.pool = {a: np.sort(np.random.default_rng([self.base_seed, a]).choice(
                r, size=self.n_stars, replace=False)) for a, r in rows_by_area.items()}
        else:                                            # val: use all available (>= one group)
            self.pool = {a: r for a, r in rows_by_area.items() if len(r) >= self.group_size}
        self.eligible = sorted(self.pool)
        if not self.eligible:
            raise RuntimeError("no area has enough stars for a group")
        self._build(0)

    def _build(self, epoch):
        e = epoch if self.resample else 0
        items = []
        for a in self.eligible:
            pool = self.pool[a]
            order = np.random.default_rng([self.base_seed, e, a]).permutation(len(pool))
            n_groups = len(pool) // self.group_size      # 31 for 1000; leftover rotates
            for g in range(n_groups):
                sel = order[g * self.group_size:(g + 1) * self.group_size]
                items.append((pool[sel].astype(np.int64), int(a)))
        self.items = items

    def set_epoch(self, epoch):
        if self.resample:
            self._build(int(epoch))

    def assert_contracts(self):
        per_area = {}
        for rows, a in self.items:
            per_area[a] = per_area.get(a, 0) + 1
            assert len(rows) == self.group_size == len(np.unique(rows)), "group not 32 unique"
        for a in self.eligible:                          # each area: floor(pool/32) disjoint groups
            assert per_area.get(a, 0) == len(self.pool[a]) // self.group_size, (a, per_area.get(a, 0))
        return True

    def loo_targets(self, rows):
        """(target, valid) each (32, L): leave-one-out median of the other 31."""
        gs = self.group_size
        T = np.zeros((gs, self.X.shape[1]), np.float32)
        V = np.zeros((gs, self.X.shape[1]), np.float32)
        Xg, Mg = self.X[rows], self.M[rows]
        for i in range(gs):
            others = np.delete(np.arange(gs), i)
            med, _log_mad, valid, _n = group_statistics(Xg[others], Mg[others], self.target_min_valid)
            T[i] = med
            V[i] = valid.astype(np.float32)
        return T, V

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rows, _a = self.items[idx]
        T, V = self.loo_targets(rows)
        return (torch.tensor(self.X[rows]), torch.tensor(self.M[rows]),
                torch.tensor(T), torch.tensor(V))
