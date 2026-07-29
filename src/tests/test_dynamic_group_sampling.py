# All this code is from Claude
"""Contracts for the dynamic per-area group sampler (Stage-B retrain). Synthetic
only -- no cluster, no real checkpoints, no test TICs.

Run: python -m src.tests.test_dynamic_group_sampling
"""

import numpy as np

from src.instrument_v2.dynamic_group_dataset import DynamicAreaGroupDataset

GROUP, MINV, RIDGE, RANK = 32, 16, 1e-2, 8


def _synth(n_per_area, areas=(111, 233), n_context=50):
    """Two areas with n_per_area stars each; random gridded curves + CBV bases."""
    rng = np.random.default_rng(0)
    n = n_per_area * len(areas)
    X = rng.normal(size=(n, 1024)).astype(np.float32)
    M = np.ones((n, 1024), np.float32)
    area_ids = np.repeat(np.array(areas, dtype=np.int64), n_per_area)
    tics = np.array([f"T{i}" for i in range(n)])
    bases = {int(a): rng.normal(size=(1024, RANK)).astype(np.float32) for a in areas}
    return X, M, area_ids, tics, bases


def _items_by_area(ds):
    out = {}
    for c, g, a in ds.items:
        out.setdefault(a, []).append((c, g))
    return out


# 1) exactly n_context context examples per eligible area per epoch
def test_exactly_n_context_per_area():
    X, M, a, t, bases = _synth(200)
    ds = DynamicAreaGroupDataset(X, M, a, t, bases, GROUP, MINV, RIDGE, n_context=50, seed=0, resample=True)
    assert set(ds.eligible) == {111, 233}
    for epoch in (1, 2, 3):
        ds.set_epoch(epoch)
        by = _items_by_area(ds)
        for area in ds.eligible:
            assert len(by[area]) == 50, (area, len(by[area]))     # exactly 50 (=n_context)
        assert len(ds) == 50 * len(ds.eligible)
        ds.assert_contracts()


# 2) every target group has group_size distinct stars, context excluded
def test_group_distinct_and_context_excluded():
    X, M, a, t, bases = _synth(200)
    ds = DynamicAreaGroupDataset(X, M, a, t, bases, GROUP, MINV, RIDGE, n_context=50, seed=0)
    ds.set_epoch(1)
    for c, g, area in ds.items:
        assert len(g) == GROUP                                   # 32 stars
        assert len(np.unique(g)) == GROUP                        # distinct
        assert c not in set(g.tolist())                          # context excluded
        assert all(int(area_id) == area for area_id in a[g])     # all from the same area


# 3) distinct context stars within an area (1,000-different-stars rule)
def test_context_stars_distinct():
    X, M, a, t, bases = _synth(200)
    ds = DynamicAreaGroupDataset(X, M, a, t, bases, GROUP, MINV, RIDGE, n_context=50, seed=0)
    ds.set_epoch(1)
    by = _items_by_area(ds)
    for area, items in by.items():
        ctx = [c for c, _ in items]
        assert len(set(ctx)) == len(ctx)                         # no repeated context star


# 4) reproducible within an epoch, changes between epochs
def test_reproducible_but_changes_between_epochs():
    X, M, a, t, bases = _synth(200)
    d1 = DynamicAreaGroupDataset(X, M, a, t, bases, GROUP, MINV, RIDGE, n_context=50, seed=0)
    d2 = DynamicAreaGroupDataset(X, M, a, t, bases, GROUP, MINV, RIDGE, n_context=50, seed=0)
    d1.set_epoch(5); d2.set_epoch(5)
    same = all(c1 == c2 and np.array_equal(g1, g2) and a1 == a2
               for (c1, g1, a1), (c2, g2, a2) in zip(d1.items, d2.items))
    assert same                                                  # reproducible for a given (seed, epoch)
    d1.set_epoch(5); items5 = list(d1.items)
    d1.set_epoch(6); items6 = list(d1.items)
    changed = any(c5 != c6 or not np.array_equal(g5, g6)
                  for (c5, g5, _), (c6, g6, _) in zip(items5, items6))
    assert changed                                              # different epoch -> different sampling


# 5) areas with < n_context stars are excluded (never crash, reported)
def test_small_areas_excluded():
    # area 111 has 200 stars, area 233 has only 40 (< n_context=50)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(240, 1024)).astype(np.float32)
    M = np.ones((240, 1024), np.float32)
    a = np.array([111] * 200 + [233] * 40, dtype=np.int64)
    t = np.array([f"T{i}" for i in range(240)])
    bases = {111: rng.normal(size=(1024, RANK)).astype(np.float32),
             233: rng.normal(size=(1024, RANK)).astype(np.float32)}
    ds = DynamicAreaGroupDataset(X, M, a, t, bases, GROUP, MINV, RIDGE, n_context=50, seed=0)
    assert ds.eligible == [111]                                 # 233 excluded (only 40 < 50)
    assert ds.excluded == {233: 40}
    ds.set_epoch(1); ds.assert_contracts()


# 6) fixed validation set: n_context=None uses every star once, no resampling
def test_val_fixed_all_stars():
    X, M, a, t, bases = _synth(60)                               # 60 per area
    ds = DynamicAreaGroupDataset(X, M, a, t, bases, GROUP, MINV, RIDGE, n_context=None, seed=0, resample=False)
    by = _items_by_area(ds)
    for area in ds.eligible:
        assert len(by[area]) == 60                              # all stars used as context once
    before = [(c, g.copy()) for c, g, _ in ds.items]
    ds.set_epoch(7)                                            # resample=False -> no change
    after = [(c, g) for c, g, _ in ds.items]
    assert all(c1 == c2 and np.array_equal(g1, g2) for (c1, g1), (c2, g2) in zip(before, after))


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_")]
    for t_ in tests:
        t_(); print(f"PASS {t_.__name__}")
    print(f"ALL {len(tests)} DYNAMIC-GROUP-SAMPLING TESTS PASSED")
