# All this code is from Claude
"""Contracts for the scaled Part-A regional-teacher pair sampler + Part-B
handoff. Synthetic only -- no cluster, no real checkpoints, no test TICs.

Run: python -m src.tests.test_dynamic_pair_sampling
"""

import json
import os
import tempfile

import numpy as np
import torch

from src.instrument_v2.dynamic_pair_dataset import DynamicAreaPairDataset

GROUP, MINV, RIDGE, RANK = 32, 16, 1e-2, 8


def _synth(n_per_area, areas=(111, 233)):
    rng = np.random.default_rng(0)
    n = n_per_area * len(areas)
    X = rng.normal(size=(n, 1024)).astype(np.float32)
    M = np.ones((n, 1024), np.float32)
    a = np.repeat(np.array(areas, dtype=np.int64), n_per_area)
    t = np.array([f"T{i}" for i in range(n)])
    bases = {int(x): rng.normal(size=(1024, RANK)).astype(np.float32) for x in areas}
    return X, M, a, t, bases


def _raises(fn):
    try:
        fn(); return False
    except RuntimeError:
        return True


# 1) exactly n_pairs per area per epoch; groups 32-unique; A/B disjoint
def test_pairs_per_area_and_group_contracts():
    X, M, a, t, bases = _synth(120)
    ds = DynamicAreaPairDataset(X, M, a, t, bases, GROUP, MINV, RIDGE,
                                n_stars=100, n_pairs=50, seed=0, resample=True)
    for epoch in (1, 2):
        ds.set_epoch(epoch)
        per = {}
        for A, B, area in ds.items:
            per[area] = per.get(area, 0) + 1
            assert len(A) == 32 and len(np.unique(A)) == 32          # Group A 32 unique
            assert len(B) == 32 and len(np.unique(B)) == 32          # Group B 32 unique
            assert not (set(A.tolist()) & set(B.tolist()))           # disjoint
        for area in ds.eligible:
            assert per[area] == 50                                   # exactly n_pairs
        ds.assert_contracts()


# 2) exactly n_stars per area pool; drawn only from that area
def test_pool_exactly_n_stars_same_area():
    X, M, a, t, bases = _synth(300)
    ds = DynamicAreaPairDataset(X, M, a, t, bases, GROUP, MINV, RIDGE,
                                n_stars=200, n_pairs=10, seed=0)
    for area in ds.eligible:
        assert len(ds.pool[area]) == 200                            # exactly n_stars
        assert all(int(a[r]) == area for r in ds.pool[area])        # all same-area
    ds.set_epoch(1)
    for A, B, area in ds.items:
        for r in np.concatenate([A, B]):
            assert r in set(ds.pool[area].tolist())                 # pairs drawn from the pool only


# 3) reproducible for a given seed, changes between epochs
def test_reproducible_but_changes():
    X, M, a, t, bases = _synth(120)
    d1 = DynamicAreaPairDataset(X, M, a, t, bases, GROUP, MINV, RIDGE, n_stars=100, n_pairs=50, seed=0)
    d2 = DynamicAreaPairDataset(X, M, a, t, bases, GROUP, MINV, RIDGE, n_stars=100, n_pairs=50, seed=0)
    d1.set_epoch(3); d2.set_epoch(3)
    assert all(np.array_equal(A1, A2) and np.array_equal(B1, B2) and a1 == a2
               for (A1, B1, a1), (A2, B2, a2) in zip(d1.items, d2.items))      # reproducible
    d1.set_epoch(3); e3 = [(A.copy(), B.copy()) for A, B, _ in d1.items]
    d1.set_epoch(4); e4 = [(A, B) for A, B, _ in d1.items]
    assert any(not np.array_equal(a3, a4) or not np.array_equal(b3, b4)
               for (a3, b3), (a4, b4) in zip(e3, e4))                          # changes per epoch


# 4) fewer than n_stars in an area -> hard fail (no silent duplication)
def test_fail_on_short_area():
    # area 111 has 300, area 233 has only 40
    rng = np.random.default_rng(1)
    X = rng.normal(size=(340, 1024)).astype(np.float32); M = np.ones((340, 1024), np.float32)
    a = np.array([111] * 300 + [233] * 40, dtype=np.int64); t = np.array([f"T{i}" for i in range(340)])
    bases = {111: rng.normal(size=(1024, RANK)).astype(np.float32),
             233: rng.normal(size=(1024, RANK)).astype(np.float32)}
    assert _raises(lambda: DynamicAreaPairDataset(X, M, a, t, bases, GROUP, MINV, RIDGE,
                                                  n_stars=100, n_pairs=10, seed=0, require_full=True))
    # require_full=False (val) tolerates it: 233 kept if >= 64, else dropped
    ds = DynamicAreaPairDataset(X, M, a, t, bases, GROUP, MINV, RIDGE,
                                n_stars=100, n_pairs=10, seed=0, resample=False, require_full=False)
    assert ds.eligible == [111]                                     # 233 (40 < 64) dropped


# 5) validation set is fixed (resample=False -> unchanged across set_epoch)
def test_val_fixed():
    X, M, a, t, bases = _synth(120)
    ds = DynamicAreaPairDataset(X, M, a, t, bases, GROUP, MINV, RIDGE,
                                n_stars=100, n_pairs=30, seed=0, resample=False, require_full=False)
    before = [(A.copy(), B.copy()) for A, B, _ in ds.items]
    ds.set_epoch(9)
    after = [(A, B) for A, B, _ in ds.items]
    assert all(np.array_equal(a1, a2) and np.array_equal(b1, b2)
               for (a1, b1), (a2, b2) in zip(before, after))


# 6) __getitem__ returns the 2-channel fingerprint tuple the teacher expects
def test_getitem_tuple_shapes():
    X, M, a, t, bases = _synth(120)
    ds = DynamicAreaPairDataset(X, M, a, t, bases, GROUP, MINV, RIDGE, n_stars=100, n_pairs=10, seed=0)
    ds.set_epoch(1)
    sa, va, sb, vb, area, chip = ds[0]
    assert sa.shape == (1024, 2) and sb.shape == (1024, 2)          # (median/CBV, log-MAD)
    assert va.shape == (1024,) and vb.shape == (1024,)


# 7) Part-B handoff: saved EMA loads via load_frozen_teacher and hash matches
def test_ema_loads_into_partb_and_hash():
    os.environ["JEPA_DMODEL"] = "32"; os.environ["JEPA_NLAYERS"] = "1"   # tiny for the unit test
    from src.instrument_v2.regional_group_teacher import build_regional_teacher, state_hash
    from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA, load_frozen_teacher
    torch.manual_seed(0)
    teacher = build_regional_teacher()                               # RegionalGroupTeacher (online+ema+predictor)
    ema_hash = state_hash(teacher.ema_encoder)
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "regteacher_g32_n1000_s0_best.pth")
        torch.save(teacher.state_dict(), ckpt)                       # full model -> has ema_encoder.* keys
        sel = os.path.join(d, "selection.json")
        json.dump({"tag": "regteacher_g32_n1000_s0", "checkpoint": ckpt}, open(sel, "w"))
        student = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=32, n_layers=1,
                                             readout="mean", predictor_type="mlp")
        load_frozen_teacher(student, sel)                            # Part-B loader path
        assert student.teacher_hash() == ema_hash                    # frozen teacher == saved EMA encoder
        assert all(not p.requires_grad for p in student.teacher.parameters())
    os.environ.pop("JEPA_DMODEL"); os.environ.pop("JEPA_NLAYERS")


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_")]
    for t_ in tests:
        t_(); print(f"PASS {t_.__name__}")
    print(f"ALL {len(tests)} DYNAMIC-PAIR-SAMPLING TESTS PASSED")
