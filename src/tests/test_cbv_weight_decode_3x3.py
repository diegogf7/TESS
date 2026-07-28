# All this code is from Claude
"""Contracts for the K=8 CBV-weight decoder + matched 3x3 physics-JEPA
comparison. Synthetic only -- no cluster, no real checkpoints, no TGLC/PhyTS.

Run: python -m src.tests.test_cbv_weight_decode_3x3
"""

import json
import os
import tempfile

import numpy as np
import torch

from sklearn.model_selection import GroupShuffleSplit

from src.instrument_v2.decode_single_star_k8 import build_decoder, sha256_file
from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.regional_group_teacher import state_hash
from src.instrument_v2.regional_cbv import (
    area_bases_cache_path,
    build_or_load_area_bases,
    load_area_bases,
    train_tic_hash,
)
from src.instrument_v2.eval_phyts_instrument_ab import physics_grid, DEVICE, GRID
from src.instrument_v2.eval_phyts_raw_tglc_ab import quality_filter
from src.instrument_v2.run_tglc_physics_jepa_ab import (
    build_arms, grid_for_instrument, subtract_native, _grid_times,
)
from src.instrument_v2.run_tglc_physics_jepa_cbv3x3 import (
    ARMS, CBV_RANK, CELLS,
    area_basis, assert_cbv_manifest_matches, batched_weights, build_cbv_arm,
    require_shared_init, require_valid_areas,
)
from src.worked_folder.physics.latent_jepa import build_latent_jepa

T0, T1 = 1683.0, 1710.0


def _tiny():
    torch.manual_seed(0)
    inst = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=32, n_layers=1,
                                      readout="mean", predictor_type="mlp").to(DEVICE).eval()
    wdec = build_decoder(CBV_RANK).to(DEVICE).eval()
    ddec = build_decoder(1024).to(DEVICE).eval()
    for p in (list(inst.parameters()) + list(wdec.parameters()) + list(ddec.parameters())):
        p.requires_grad_(False)
    return inst, wdec, ddec


def _curve(seed=0, n=400):
    rng = np.random.default_rng(seed)
    time = np.linspace(T0, T1, n)
    flux = 100.0 + 20.0 * np.sin(np.linspace(0, 9, n)) + rng.normal(0, 1, n)
    z = np.zeros(n, int)
    return quality_filter(time, flux, z, z)


def _bases(seed=1, areas=(111, 232)):
    rng = np.random.default_rng(seed)
    return {a: rng.normal(size=(GRID, CBV_RANK)) for a in areas}


def _raises(fn):
    try:
        fn()
        return False
    except RuntimeError:
        return True


# 1) decoder output is [batch, 8]
def test_weight_decoder_output_batch_8():
    w = build_decoder(CBV_RANK)(torch.randn(5, 16, 16))
    assert w.shape == (5, CBV_RANK)
    inst, wdec, _ = _tiny()
    curves = [_curve(0), _curve(1)]
    Xg = np.zeros((2, GRID), np.float32); Mg = np.zeros((2, GRID), np.float32)
    for i, (ft, ff) in enumerate(curves):
        Xg[i], Mg[i], _, _, _ = grid_for_instrument(ft, ff, T0, T1)
    W = batched_weights(inst, wdec, Xg, Mg)
    assert W.shape == (2, CBV_RANK)


# 2) CBV reconstruction is [batch, 1024] and equals basis @ weights
def test_cbv_reconstruction_batch_1024():
    w = build_decoder(CBV_RANK)(torch.randn(3, 16, 16))
    B = torch.randn(GRID, CBV_RANK)
    curve = w @ B.T                                   # decode() path
    assert curve.shape == (3, GRID)
    Bt = B[None].expand(3, -1, -1)
    ein = torch.einsum("bk,blk->bl", w, Bt)           # training-loop path
    assert torch.allclose(curve, ein, atol=1e-5)
    for i in range(3):                                # template = basis @ weights
        assert torch.allclose(curve[i], B @ w[i], atol=1e-5)


# 3) every curve receives the correct area basis; missing bases hard-fail
def test_each_curve_gets_its_area_basis():
    inst, wdec, _ = _tiny()
    bases = _bases()
    curves = [_curve(0), _curve(1), _curve(2)]
    times = [c[0] for c in curves]; fluxes = [c[1] for c in curves]
    areas = np.array([111, 232, 111])
    _, _, cbv_X, _, nfb = build_cbv_arm(times, fluxes, areas, inst, wdec, bases, T0, T1)
    assert nfb == 0
    grid_times = _grid_times(T0, T1)
    # decode the weights exactly as build_cbv_arm does: one batched pass
    Xg = np.zeros((3, GRID), np.float32); Mg = np.zeros((3, GRID), np.float32)
    scales, valids = np.zeros(3), []
    for i in range(3):
        Xg[i], Mg[i], _, scales[i], v = grid_for_instrument(times[i], fluxes[i], T0, T1)
        valids.append(v)
    W = batched_weights(inst, wdec, Xg, Mg)
    for i, a in enumerate(areas):
        own = area_basis(bases, a) @ W[i].astype(np.float64)
        other = area_basis(bases, 232 if a == 111 else 111) @ W[i].astype(np.float64)
        expect_X, _ = physics_grid(times[i], subtract_native(
            times[i], fluxes[i], own, valids[i], scales[i], grid_times))
        wrong_X, _ = physics_grid(times[i], subtract_native(
            times[i], fluxes[i], other, valids[i], scales[i], grid_times))
        # float32: build_cbv_arm stores its output arrays as float32
        assert np.array_equal(cbv_X[i], expect_X.astype(np.float32))   # its own area's basis
        assert not np.allclose(cbv_X[i], wrong_X.astype(np.float32))   # not any other basis
    # unknown area -> hard-fail, never a silent raw fallback
    assert _raises(lambda: build_cbv_arm(times, fluxes, np.array([111, 999, 111]),
                                         inst, wdec, bases, T0, T1))
    assert _raises(lambda: area_basis(bases, 999))
    # wrong-shaped basis also hard-fails
    assert _raises(lambda: area_basis({111: np.zeros((GRID, 3))}, 111))


# 4) CBV bases are built from training TICs only (provenance-pinned cache)
def test_bases_train_tics_only():
    rng = np.random.default_rng(0)
    n, L, gs, mv, k = 8, 64, 4, 2, 2
    X = rng.normal(size=(n, L)).astype(np.float32)
    M = np.ones((n, L), np.float32)
    areas = np.array([111] * n)
    train_tics = [f"T{i}" for i in range(n)]
    with tempfile.TemporaryDirectory() as d:
        bases = build_or_load_area_bases(X, M, areas, sorted(train_tics), k, d, gs, mv)
        npz = area_bases_cache_path(d, k, gs, mv, sorted(train_tics))
        assert os.path.exists(npz)                     # cache keyed by the train-TIC hash
        sidecar = json.load(open(os.path.splitext(npz)[0] + ".json"))
        assert sidecar["train_tic_hash"] == train_tic_hash(sorted(train_tics))
        assert sidecar["n_train_tics"] == n
        # a different TIC set (e.g. with val/test stars) can NEVER silently reuse it
        other = area_bases_cache_path(d, k, gs, mv, sorted(train_tics + ["VAL1"]))
        assert other != npz and not os.path.exists(other)
        # explicit loader round-trips and hard-fails on a missing file
        loaded = load_area_bases(npz)
        assert np.allclose(loaded[111], bases[111])
        assert _raises(lambda: load_area_bases(other))


# 5) instrument JEPA + weight decoder stay hash-identical through cleaning
def test_instrument_models_hash_identical():
    inst, wdec, _ = _tiny()
    before = (state_hash(inst.teacher), state_hash(inst.student),
              state_hash(inst.predictor), state_hash(wdec))
    curves = [_curve(0), _curve(1)]
    build_cbv_arm([c[0] for c in curves], [c[1] for c in curves],
                  np.array([111, 232]), inst, wdec, _bases(), T0, T1)
    after = (state_hash(inst.teacher), state_hash(inst.student),
             state_hash(inst.predictor), state_hash(wdec))
    assert after == before
    for p in list(inst.parameters()) + list(wdec.parameters()):
        assert p.requires_grad is False and p.grad is None


# 6) raw, direct-cleaned and CBV-cleaned share identical rows, masks, raw flux
def test_masks_identical_across_three_arms():
    inst, wdec, ddec = _tiny()
    curves = [_curve(3), _curve(4), _curve(5)]
    times = [c[0] for c in curves]; fluxes = [c[1] for c in curves]
    raw_X, raw_M, dir_X, dir_M = build_arms(times, fluxes, inst, ddec, T0, T1)
    raw_X2, raw_M2, cbv_X, cbv_M, _ = build_cbv_arm(
        times, fluxes, np.array([111, 232, 111]), inst, wdec, _bases(), T0, T1)
    assert np.array_equal(raw_X, raw_X2)               # bit-identical raw arm
    assert np.array_equal(raw_M, raw_M2)
    assert np.array_equal(raw_M, dir_M) and np.array_equal(raw_M, cbv_M)
    assert not np.allclose(dir_X, cbv_X)               # only the cleaning differs
    # a curve too sparse to clean (<8 valid grid bins) stays raw in the CBV arm
    # under the SAME rule as the direct arm, and is counted as a fallback
    st, sf = _curve(6, n=4)
    rX, rM, cX, cM, nfb = build_cbv_arm([st], [sf], np.array([111]),
                                        inst, wdec, _bases(), T0, T1)
    assert nfb == 1 and np.array_equal(rX, cX) and np.array_equal(rM, cM)


# 7) cleaning happens at native timestamps, before ONE physics resample
def test_native_cleaning_before_single_resample():
    inst, wdec, _ = _tiny()
    ft, ff = _curve(7)
    Xg, Mg, _, scale, valid = grid_for_instrument(ft, ff, T0, T1)
    w = batched_weights(inst, wdec, Xg[None], Mg[None])[0]
    template = area_basis(_bases(), 111) @ w.astype(np.float64)
    grid_times = _grid_times(T0, T1)
    cf = subtract_native(ft, ff, template, valid, scale, grid_times)
    assert cf.shape == ff.shape and len(ff) != GRID    # still native, not resampled
    dec_native = np.interp(ft, grid_times[valid], template[valid])
    assert np.allclose(cf, ff - dec_native * scale)    # flux units, fixed MAD scale
    X, M = physics_grid(ft, cf)                        # exactly one physics resample
    assert X.shape == (GRID,)


# 8) all physics JEPAs start from the same initialization; a lost init hard-fails
def test_shared_initialization():
    torch.manual_seed(11)
    init = build_latent_jepa().state_dict()
    models = [build_latent_jepa() for _ in range(3)]
    for m in models:
        m.load_state_dict(init, strict=True)
    states = [m.state_dict() for m in models]
    for s in states[1:]:
        assert all(torch.equal(states[0][k], s[k]) for k in states[0])
    with tempfile.TemporaryDirectory() as d:
        missing = os.path.join(d, "physics_jepa_init_s0.pth")
        assert _raises(lambda: require_shared_init(missing))       # never recreated
        torch.save(init, missing)
        sha = require_shared_init(missing)
        assert sha == sha256_file(missing)
        assert require_shared_init(missing, expected_sha=sha) == sha
        assert _raises(lambda: require_shared_init(missing, expected_sha="0" * 64))


# 9) all nine evaluation cells exist and share identical rows and indices
def test_eval_cells_identical_rows():
    assert set(CELLS) == {f"{jm}_jepa_on_{dm}" for jm in ARMS for dm in ARMS}
    assert len(CELLS) == 9 and set(ARMS) == {"raw", "direct", "cbv"}
    tics = np.array([f"T{i // 2}" for i in range(120)])
    y = np.array([i % 7 for i in range(120)])
    tr1, te1 = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(
        np.arange(120), y, groups=tics))
    tr2, te2 = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(
        np.arange(120), y, groups=tics))
    assert np.array_equal(tr1, tr2) and np.array_equal(te1, te2)   # one split, all 9 cells
    assert not (set(tics[tr1]) & set(tics[te1]))                   # TIC-disjoint


# 10) stale or mismatched artifacts hard-fail
def test_stale_or_mismatched_artifacts_hard_fail():
    current = {"pretrain_sig": "a", "eval_tglc_sig": "b", "inst_sig": "c",
               "decoder_sig": "d", "grid_sig": "e", "bad_tess_mask": 16437,
               "weight_decoder_sig": "w", "bases_sig": "s", "cbv_rank": 8}
    assert_cbv_manifest_matches(dict(current), current)            # identical -> ok
    for key in ("inst_sig", "decoder_sig", "weight_decoder_sig", "bases_sig", "cbv_rank"):
        stale = dict(current); stale[key] = "CHANGED"
        assert _raises(lambda: assert_cbv_manifest_matches(stale, current))
    missing = dict(current); missing.pop("bases_sig")
    assert _raises(lambda: assert_cbv_manifest_matches(missing, current))


# 11) rows without a valid area hard-fail instead of getting a wrong basis
def test_invalid_areas_hard_fail():
    import pandas as pd
    ok = pd.DataFrame({"area": [111, 232, 111]})
    assert np.array_equal(require_valid_areas(ok, "x"), [111, 232, 111])
    assert _raises(lambda: require_valid_areas(pd.DataFrame({"area": [111, -1]}), "x"))
    assert _raises(lambda: require_valid_areas(pd.DataFrame({"area": [111, np.nan]}), "x"))


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"ALL {len(tests)} CBV-WEIGHT-DECODE 3x3 TESTS PASSED")
