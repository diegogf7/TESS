from __future__ import annotations
"""Dynamic per-area group sampler for the Sector-14 instrument Stage-B student.

Every epoch, for each ELIGIBLE area (>= n_context training stars and a CBV
basis): draw n_context DISTINCT context stars; for each context star draw
group_size OTHER stars from the SAME area (context excluded, all distinct).
Groups may overlap between context examples. The whole assignment is
regenerated each epoch from a deterministic seed (SEED, epoch, area), so it is
reproducible yet changes between epochs.

Teacher target = the group's K=8 CBV ridge reconstruction of the group median
(identical construction to the existing Sector14GroupCBVReconDataset), so this
is a drop-in replacement for the Stage-B dataset -- only the SAMPLING changes.
Preprocessing, CBVs, min-valid, and the returned tuple are unchanged.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.instrument_v2.area_commonmode_dataset import group_statistics
from src.instrument_v2.regional_cbv import ridge_reconstruct


class DynamicAreaGroupDataset(Dataset):
    def __init__(self, X, M, areas, tics, area_bases, group_size, min_valid,
                 ridge_lambda, n_context=1000, seed=0, resample=True):
        self.X, self.M = X, M
        self.areas = np.asarray(areas, dtype=np.int64)
        self.tics = np.asarray(tics, dtype=str)
        self.area_bases = area_bases
        self.group_size = int(group_size)
        self.min_valid = int(min_valid)
        self.ridge_lambda = float(ridge_lambda)
        self.n_context = None if n_context is None else int(n_context)
        self.base_seed = int(seed)
        self.resample = bool(resample)

        # area -> deterministic sorted row indices into X/M
        rows_by_area = {}
        for i, a in enumerate(self.areas):
            rows_by_area.setdefault(int(a), []).append(i)
        self.area_rows = {a: np.asarray(sorted(r), dtype=np.int64)
                          for a, r in rows_by_area.items()}

        # an area is eligible only if it has a CBV basis AND enough stars: at
        # least n_context distinct contexts (train) or group_size+1 (val).
        need = (self.group_size + 1 if self.n_context is None
                else max(self.n_context, self.group_size + 1))
        self.eligible = sorted(a for a, r in self.area_rows.items()
                               if a in area_bases and len(r) >= need)
        self.excluded = {int(a): int(len(r)) for a, r in self.area_rows.items()
                         if a not in self.eligible}
        if not self.eligible:
            raise RuntimeError(f"no area has a CBV basis and >= {need} stars")
        self._build(0)

    # ------------------------------------------------------------------ build
    def _build(self, epoch):
        """(re)generate the (context, 32-star group, area) items for this epoch."""
        e = epoch if self.resample else 0
        items = []
        for a in self.eligible:
            rows = self.area_rows[a]
            rng = np.random.default_rng([self.base_seed, e, int(a)])   # deterministic per area/epoch
            if self.n_context is None:
                ctx = rows                                             # all stars, fixed order (val)
            else:
                ctx = rng.choice(rows, size=self.n_context, replace=False)   # 1000 distinct contexts
            for c in ctx:
                others = rows[rows != c]                              # context excluded
                g = rng.choice(others, size=self.group_size, replace=False)  # 32 distinct others
                items.append((int(c), g.astype(np.int64), int(a)))
        self.items = items

    def set_epoch(self, epoch):
        if self.resample:
            self._build(int(epoch))

    # -------------------------------------------------------------- contracts
    def assert_contracts(self):
        """Item-9 sampling guarantees for the CURRENT epoch's items."""
        per_area = {}
        for c, g, a in self.items:
            per_area.setdefault(a, []).append(c)
            assert len(g) == self.group_size, (a, len(g))          # 32 stars
            assert len(np.unique(g)) == self.group_size, "group has repeats"
            assert c not in set(g.tolist()), "context star inside its own group"
        for a in self.eligible:
            ctx = per_area.get(a, [])
            if self.n_context is not None:
                assert len(ctx) == self.n_context, (a, len(ctx))   # exactly 1000
            assert len(set(ctx)) == len(ctx), "duplicate context star in area"
        return True

    # ------------------------------------------------------------------ items
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        c, g, a = self.items[idx]
        median, log_mad, valid, n_obs = group_statistics(self.X[g], self.M[g], self.min_valid)
        instrument = ridge_reconstruct(median, valid, self.area_bases[a], self.ridge_lambda)
        return (torch.tensor(self.X[c]), torch.tensor(self.M[c]),
                torch.tensor(instrument), torch.tensor(log_mad),
                torch.tensor(valid), torch.tensor(n_obs),
                torch.tensor(a, dtype=torch.int64))
