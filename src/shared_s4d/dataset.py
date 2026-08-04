from __future__ import annotations


import numpy as np
import torch
from torch.utils.data import Dataset

from src.instrument_v2.area_commonmode_dataset import group_statistics


class AreaGroupLOODataset(Dataset):
    def __init__(self, X, M, areas, tics, n_stars=1000, group_size=32,
                 target_min_valid=4, seed=0, require_full=True, resample=True,
                 grouping_mode="random", radec=None, detxy=None, groups_per_area=None):
        self.X, self.M = X, M
        self.areas = np.asarray(areas, dtype=np.int64)
        self.tics = np.asarray(tics, dtype=str)
        self.group_size = int(group_size)
        self.n_stars = int(n_stars)
        self.target_min_valid = int(target_min_valid)
        self.base_seed = int(seed)
        self.resample = bool(resample)
        self.grouping_mode = str(grouping_mode)
        if self.grouping_mode not in ("random", "nearest", "detector_nearest"):
            raise ValueError(f"grouping_mode must be random|nearest|detector_nearest, got {grouping_mode!r}")
        # Coordinates aligned to X rows; REQUIRED for the local modes, never fall back silently.
        # nearest      -> radec  (RA/Dec, small-field angular distance)
        # detector_nearest -> detxy (STAR_X, STAR_Y detector pixels, Euclidean distance)
        self.radec = None if radec is None else np.asarray(radec, dtype=np.float64)
        self.detxy = None if detxy is None else np.asarray(detxy, dtype=np.float64)
        self.groups_per_area = None if groups_per_area is None else int(groups_per_area)
        if self.grouping_mode == "nearest" and self.radec is None:
            raise RuntimeError("GROUPING_MODE=nearest requires RA/Dec; none provided "
                               "-- refusing to silently use random grouping")
        if self.grouping_mode == "detector_nearest" and self.detxy is None:
            raise RuntimeError(
                "GROUPING_MODE=detector_nearest requires physical DETECTOR_X/DETECTOR_Y; none "
                "provided. Regenerate the parquet with src/tglc/merge_detector_positions.py "
                "(RA/Dec -> TESS detector col/row via tess-point). Refusing to fall back to RA/Dec.")

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
        else:                                            # cap each area at n_stars DETERMINISTICALLY
            self.pool = {}
            for a, r in rows_by_area.items():
                if len(r) < self.group_size:
                    continue
                if len(r) > self.n_stars:                 # never let a candidate pool exceed n_stars
                    r = np.sort(np.random.default_rng([self.base_seed, a]).choice(
                        r, size=self.n_stars, replace=False))
                self.pool[a] = r
        self.eligible = sorted(self.pool)
        if not self.eligible:
            raise RuntimeError("no area has enough stars for a group")
        self._nearest = (self._build_local_groups()
                         if self.grouping_mode in ("nearest", "detector_nearest") else None)
        self._build(0)

    def _pool_pairwise_d2(self, pool):
        """Per-pool squared-distance matrix for the active local grouping mode.
        detector_nearest -> Euclidean^2 in STAR_X/STAR_Y; nearest -> small-field RA/Dec."""
        if self.grouping_mode == "detector_nearest":
            xy = self.detxy[pool]                                                  # (n, 2) DETECTOR_X/Y px
            if not np.isfinite(xy).all():
                raise RuntimeError("detector_nearest: non-finite DETECTOR_X/DETECTOR_Y in pool -- fix the data")
            diff = xy[:, None, :] - xy[None, :, :]
            return (diff ** 2).sum(-1)                                             # Euclidean^2
        rd = self.radec[pool]
        if not np.isfinite(rd).all():
            raise RuntimeError("nearest: non-finite RA/Dec in pool -- fix the data")
        ra = np.radians(rd[:, 0]); dec = np.radians(rd[:, 1])
        cosd = np.cos(np.clip(dec, -np.pi / 2, np.pi / 2))
        dra = (ra[:, None] - ra[None, :]) * ((cosd[:, None] + cosd[None, :]) / 2)  # small-field
        return dra ** 2 + (dec[:, None] - dec[None, :]) ** 2                       # ang. dist^2

    def _build_local_groups(self):
        """Per area: one anchor-centered group per pool star = the group_size nearest stars
        (incl. the anchor) by the mode's distance, restricted to the same area (=> same
        sector/camera/ccd since area = camera*100+ccd*10+bin) and the same split (per-split
        dataset). The pool is already capped at n_stars. Groups overlap but EXACT-duplicate
        groups are removed. Records per-area stats in self.group_stats."""
        gs = self.group_size
        groups, self.group_stats = {}, {}
        for a in self.eligible:
            pool = self.pool[a]
            assert len(pool) <= self.n_stars, f"area {a} pool {len(pool)} > n_stars {self.n_stars}"
            if len(pool) < gs:
                continue
            d2 = self._pool_pairwise_d2(pool)
            order = np.argsort(d2, axis=1, kind="stable")[:, :gs]                  # nearest gs incl self
            seen, glist, nd = set(), [], []
            for i in range(len(pool)):                        # every pool star anchors one group
                sel = pool[order[i]].astype(np.int64)
                key = frozenset(int(x) for x in sel)
                if key in seen:                               # drop exact-duplicate group
                    continue
                seen.add(key); glist.append(sel)
                nd.append(np.sqrt(d2[i, order[i, 1:]]))       # anchor -> its 15 neighbor distances
            groups[a] = glist
            nd = np.concatenate(nd) if nd else np.zeros(0)
            self.group_stats[a] = {"pool": int(len(pool)), "groups": int(len(glist)),
                                   "med_dist": float(np.median(nd)) if nd.size else float("nan"),
                                   "max_dist": float(nd.max()) if nd.size else float("nan")}
        return groups

    def _build(self, epoch):
        if self.grouping_mode in ("nearest", "detector_nearest"):
            # Deterministically sample up to groups_per_area of the EXISTING nearest groups per
            # area (fresh subset each epoch when resample=True). The nearest groups themselves
            # (self._nearest) are never rebuilt or changed.
            e = epoch if self.resample else 0
            items = []
            for a in self.eligible:
                groups = self._nearest.get(a, [])
                if self.groups_per_area and len(groups) > self.groups_per_area:
                    idx = np.random.default_rng([self.base_seed, e, a]).choice(
                        len(groups), size=self.groups_per_area, replace=False)
                    groups = [groups[i] for i in idx]
                items.extend((grp, int(a)) for grp in groups)
            self.items = items
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
