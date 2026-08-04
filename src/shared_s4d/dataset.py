from __future__ import annotations


import numpy as np
import torch
from torch.utils.data import Dataset

from src.instrument_v2.area_commonmode_dataset import group_statistics


class AreaGroupLOODataset(Dataset):
    def __init__(self, X, M, areas, tics, n_stars=1000, group_size=32,
                 target_min_valid=4, seed=0, require_full=True, resample=True,
                 grouping_mode="random", radec=None):
        self.X, self.M = X, M
        self.areas = np.asarray(areas, dtype=np.int64)
        self.tics = np.asarray(tics, dtype=str)
        self.group_size = int(group_size)
        self.n_stars = int(n_stars)
        self.target_min_valid = int(target_min_valid)
        self.base_seed = int(seed)
        self.resample = bool(resample)
        self.grouping_mode = str(grouping_mode)
        if self.grouping_mode not in ("random", "nearest"):
            raise ValueError(f"grouping_mode must be random|nearest, got {grouping_mode!r}")
        # RA/Dec aligned to X rows; REQUIRED for nearest, never fall back to random silently
        self.radec = None if radec is None else np.asarray(radec, dtype=np.float64)
        if self.grouping_mode == "nearest" and self.radec is None:
            raise RuntimeError("GROUPING_MODE=nearest requires RA/Dec; none provided "
                               "-- refusing to silently use random grouping")

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
        self._nearest = self._build_nearest_groups() if self.grouping_mode == "nearest" else None
        self._build(0)

    def _build_nearest_groups(self):
        """Per area: one anchor-centered group per pool star = the group_size nearest
        stars (incl. the anchor) by RA/Dec angular distance, restricted to the same
        area (=> same sector/camera/ccd). Deterministic; groups overlap (not disjoint)."""
        gs = self.group_size
        groups = {}
        for a in self.eligible:
            pool = self.pool[a]
            if len(pool) < gs:
                continue
            rd = self.radec[pool]
            ra = np.radians(rd[:, 0]); dec = np.radians(rd[:, 1])
            cosd = np.cos(np.clip(dec, -np.pi / 2, np.pi / 2))
            dra = (ra[:, None] - ra[None, :]) * ((cosd[:, None] + cosd[None, :]) / 2)   # small-field
            d2 = dra ** 2 + (dec[:, None] - dec[None, :]) ** 2                          # ang. dist^2
            order = np.argsort(d2, axis=1, kind="stable")[:, :gs]                        # nearest gs incl self
            n_anchor = min(len(pool), self.n_stars)          # cap: up to ~n_stars anchor groups/area
            if n_anchor < len(pool):
                anchors = np.random.default_rng([self.base_seed, a]).choice(len(pool), size=n_anchor, replace=False)
            else:
                anchors = np.arange(len(pool))
            groups[a] = [pool[order[i]].astype(np.int64) for i in anchors]
        return groups

    def _build(self, epoch):
        if self.grouping_mode == "nearest":                  # anchor-centered, epoch-independent
            self.items = [(grp, int(a)) for a in self.eligible for grp in self._nearest.get(a, [])]
            return
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
        for rows, a in self.items:
            assert len(rows) == self.group_size == len(np.unique(rows)), "group wrong size / has dupes"
            assert set(self.areas[rows].tolist()) == {a}, "group spans areas"
        if self.grouping_mode == "random":               # disjoint partition only for random
            per_area = {}
            for rows, a in self.items:
                per_area[a] = per_area.get(a, 0) + 1
            for a in self.eligible:
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
