# All this code is from Claude
"""Contracts for the frozen 2x2 native-vs-grid cleaning diagnostic. Synthetic
only -- no cluster, no real checkpoints, no PhyTS/TGLC files.

Run: python -m src.tests.test_phyts_native_cleaning_controls
"""

import inspect

import numpy as np
import pandas as pd
import torch

from src.instrument_v2.eval_phyts_native_cleaning_controls import (
    strict_load, grid_curve_from_template, classify_arm,
)
from src.instrument_v2.finetune_phyts_raw_tglc_ab import (
    decode_native_template, cleaned_native_flux,
)
from src.instrument_v2.eval_phyts_instrument_ab import (
    physics_grid, instrument_cleaned_curve, matched_split, encode_physics, GRID, DEVICE,
)
from src.instrument_v2.eval_phyts_raw_tglc_ab import match_phyts_tglc, quality_filter
from src.instrument_v2.decode_single_star_k8 import build_decoder
from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.regional_group_teacher import state_hash
from src.instrument_v2.diagnose_chip_common_signal import normalize_median_mad
from src.instrument_v2.sector14_dataset import grid_curve_shared
from src.worked_folder.physics.latent_jepa import build_latent_jepa

T0, T1 = 1683.0, 1710.0


def _tiny_instrument():
    torch.manual_seed(0)
    m = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=32, n_layers=1,
                                   readout="mean", predictor_type="mlp").to(DEVICE).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    dec = build_decoder(1024).to(DEVICE).eval()
    for p in dec.parameters():
        p.requires_grad_(False)
    return m, dec


def _curve(seed=0):
    rng = np.random.default_rng(seed)
    time = np.linspace(T0, T1, 400)
    flux = 100.0 + 20.0 * np.sin(np.linspace(0, 9, 400)) + rng.normal(0, 1, 400)
    z = np.zeros(400, int)
    return quality_filter(time, flux, z, z)


def _build_all(curves, inst, dec):
    out = {a: [] for a in "ABCD"}
    for ft, ff in curves:
        tpl = decode_native_template(ft, ff, inst, dec, T0, T1)
        out["A"].append(physics_grid(ft, ff)[0])
        out["B"].append(physics_grid(ft, cleaned_native_flux(ft, ff, inst, dec, T0, T1, template=tpl))[0])
        out["C"].append(physics_grid(*grid_curve_from_template(tpl, ft, ff, subtract=False))[0])
        out["D"].append(physics_grid(*grid_curve_from_template(tpl, ft, ff, subtract=True))[0])
    return {a: np.array(v) for a, v in out.items()}


# 1) all arms share rows / ordering (reordering inputs permutes every arm alike)
def test_all_arms_identical_rows_and_order():
    inst, dec = _tiny_instrument()
    curves = [_curve(0), _curve(1), _curve(2)]
    a1 = _build_all(curves, inst, dec)
    assert all(a1[a].shape == (3, GRID) for a in "ABCD")
    a2 = _build_all([curves[1], curves[0], curves[2]], inst, dec)   # swap rows 0,1
    for a in "ABCD":
        assert np.allclose(a1[a][0], a2[a][1]) and np.allclose(a1[a][1], a2[a][0])


# 2) A and B: identical native timestamps -> identical final physics masks
def test_A_B_identical_native_timestamps_and_masks():
    inst, dec = _tiny_instrument()
    ft, ff = _curve()
    tpl = decode_native_template(ft, ff, inst, dec, T0, T1)
    A_X, A_M = physics_grid(ft, ff)
    B_X, B_M = physics_grid(ft, cleaned_native_flux(ft, ff, inst, dec, T0, T1, template=tpl))
    assert np.array_equal(A_M, B_M)                    # same native support
    assert not np.allclose(A_X, B_X)                   # only flux values differ


# 3) C and D: identical shared-grid timestamps -> identical final physics masks
def test_C_D_identical_grid_timestamps_and_masks():
    inst, dec = _tiny_instrument()
    ft, ff = _curve()
    tpl = decode_native_template(ft, ff, inst, dec, T0, T1)
    ct_c, cf_c = grid_curve_from_template(tpl, ft, ff, subtract=False)
    ct_d, cf_d = grid_curve_from_template(tpl, ft, ff, subtract=True)
    assert np.array_equal(ct_c, ct_d)                  # same valid grid timestamps
    _, C_M = physics_grid(ct_c, cf_c)
    _, D_M = physics_grid(ct_d, cf_d)
    assert np.array_equal(C_M, D_M)


# 4) Arm B is numerically identical to the fine-tuning cleaned_native_flux
def test_B_matches_cleaned_native_flux():
    inst, dec = _tiny_instrument()
    ft, ff = _curve(7)
    tpl = decode_native_template(ft, ff, inst, dec, T0, T1)
    b_via_template = cleaned_native_flux(ft, ff, inst, dec, T0, T1, template=tpl)
    b_standalone = cleaned_native_flux(ft, ff, inst, dec, T0, T1)
    assert np.array_equal(b_via_template, b_standalone)


# 5) Arm C performs no subtraction
def test_C_has_no_subtraction():
    inst, dec = _tiny_instrument()
    ft, ff = _curve(2)
    tpl = decode_native_template(ft, ff, inst, dec, T0, T1)
    _, cf_c = grid_curve_from_template(tpl, ft, ff, subtract=False)
    expected = (tpl["X"] * tpl["scale"] + tpl["med"])[tpl["valid"]]   # raw on grid, decoded NOT removed
    assert np.allclose(cf_c, expected)
    _, cf_d = grid_curve_from_template(tpl, ft, ff, subtract=True)
    assert not np.allclose(cf_c, cf_d)                 # D actually subtracts something


# 6) Arm D reproduces the existing instrument_cleaned_curve
def test_D_matches_instrument_cleaned_curve():
    inst, dec = _tiny_instrument()
    ft, ff = _curve(4)
    tpl = decode_native_template(ft, ff, inst, dec, T0, T1)
    ct_d, cf_d = grid_curve_from_template(tpl, ft, ff, subtract=True)
    ct_ref, cf_ref = instrument_cleaned_curve(ft, ff, inst, dec, T0, T1)
    assert np.array_equal(ct_d, ct_ref)
    assert np.allclose(cf_d, cf_ref, atol=1e-5)


# 7) no fitted scale / regression / label enters cleaning
def test_no_fit_or_label_in_cleaning():
    inst, dec = _tiny_instrument()
    ft, ff = _curve(5)
    tpl = decode_native_template(ft, ff, inst, dec, T0, T1)
    normed, med, mad = normalize_median_mad(ff)        # cleaning scale is fixed 1.4826*MAD, not fitted
    scale = 1.4826 * mad
    X, M = grid_curve_shared(ft, normed, T0, T1, GRID)
    valid = M > 0
    with torch.no_grad():
        z = inst.encode(torch.tensor(X, dtype=torch.float32, device=DEVICE)[None],
                        torch.tensor(M, dtype=torch.float32, device=DEVICE)[None], view="predicted")
        decoded = dec(z).squeeze(0).cpu().numpy()
    expected = ((X - decoded) * scale + med)[valid]
    _, cf_d = grid_curve_from_template(tpl, ft, ff, subtract=True)
    assert np.allclose(cf_d, expected, atol=1e-5)
    assert "y" not in inspect.signature(grid_curve_from_template).parameters      # no labels
    assert "label" not in inspect.signature(cleaned_native_flux).parameters


# 8) PhyTS flux is never used -- the curve comes from TGLC aperture_flux only
def test_phyts_flux_never_used():
    tglc = pd.DataFrame({"TIC": ["A"], "sector": [14], "GAIADR3": [100],
                         "time": [np.arange(5.0)], "aperture_flux": [np.arange(5.0) + 3],
                         "TESS_flags": [np.zeros(5, int)], "TGLC_flags": [np.zeros(5, int)]})
    phyts = pd.DataFrame({"TIC": ["A"], "sector": [14], "phyts_gaia": [100], "label": ["ECLIPSE"]})
    m, _ = match_phyts_tglc(phyts, tglc)
    assert "flux" not in m.columns
    assert np.array_equal(m["aperture_flux"].iloc[0], tglc["aperture_flux"].iloc[0])


# 9) train/test TICs disjoint and the split is deterministic (shared by all arms)
def test_split_disjoint_and_shared():
    tics = np.array([f"T{i // 3}" for i in range(90)])
    y = np.array([i % 5 for i in range(90)])
    tr, te = matched_split(tics, y)
    assert not (set(tics[tr]) & set(tics[te]))
    tr2, te2 = matched_split(tics, y)                  # deterministic -> every arm gets the same split
    assert np.array_equal(tr, tr2) and np.array_equal(te, te2)


# 10) scaler + KNN fitted on training indices only
def test_probe_fitted_on_train_only():
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    rng = np.random.default_rng(0)
    lat = rng.normal(size=(120, 8))
    y = np.array([i % 4 for i in range(120)])
    tics = np.array([f"T{i}" for i in range(120)])
    tr, te = matched_split(tics, y)
    present = np.unique(y)
    _, _, pred = classify_arm(lat, y, tr, te, present)
    sc = StandardScaler().fit(lat[tr])                 # replicate a strict train-only fit
    knn = KNeighborsClassifier(n_neighbors=20).fit(sc.transform(lat[tr]), y[tr])
    assert np.array_equal(pred, knn.predict(sc.transform(lat[te])))


# 11) physics/instrument/decoder: no gradients, hash-identical through inference
def test_models_frozen_and_hash_identical():
    inst, dec = _tiny_instrument()
    phys = build_latent_jepa().to(DEVICE).eval()
    for p in phys.parameters():
        p.requires_grad_(False)
    before = (state_hash(phys), state_hash(inst.teacher), state_hash(inst.student),
              state_hash(inst.predictor), state_hash(dec))
    a = _build_all([_curve(0), _curve(1)], inst, dec)
    encode_physics(phys, a["A"], np.ones_like(a["A"]))
    after = (state_hash(phys), state_hash(inst.teacher), state_hash(inst.student),
             state_hash(inst.predictor), state_hash(dec))
    assert after == before
    for p in list(phys.parameters()) + list(inst.parameters()) + list(dec.parameters()):
        assert p.requires_grad is False and p.grad is None


# 12) strict checkpoint loading fails on missing/unexpected keys
def test_strict_checkpoint_loading():
    orig = torch.load
    try:
        torch.load = lambda *a, **k: {}                # empty -> every key missing
        raised = False
        try:
            strict_load(build_decoder(1024), "x")
        except RuntimeError:
            raised = True
        assert raised
        good = build_decoder(1024).state_dict()
        torch.load = lambda *a, **k: good              # exact keys -> loads clean
        strict_load(build_decoder(1024), "x")
    finally:
        torch.load = orig


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"ALL {len(tests)} PHYTS-NATIVE-CLEANING-CONTROL TESTS PASSED")
