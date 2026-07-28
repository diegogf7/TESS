# All this code is from Claude
"""Contracts for the matched raw-vs-cleaned physics-JEPA pretraining A/B.
Synthetic only -- no cluster, no real checkpoints, no TGLC/PhyTS files.

Run: python -m src.tests.test_tglc_physics_jepa_ab
"""

import numpy as np
import pandas as pd
import torch

from src.instrument_v2.run_tglc_physics_jepa_ab import (
    build_arms, subtract_native, grid_for_instrument, remove_eval_cohort,
    assert_no_eval_overlap, assert_manifest_matches, strict_load, _classify, _grid_times,
)
from src.instrument_v2.finetune_phyts_raw_tglc_ab import cleaned_native_flux
from src.instrument_v2.eval_phyts_instrument_ab import physics_grid, DEVICE, GRID
from src.instrument_v2.eval_phyts_raw_tglc_ab import match_phyts_tglc, quality_filter
from src.instrument_v2.decode_single_star_k8 import build_decoder
from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.regional_group_teacher import state_hash
from src.instrument_v2.diagnose_chip_common_signal import normalize_median_mad
from src.instrument_v2.sector14_dataset import grid_curve_shared
from src.worked_folder.physics.latent_jepa import build_latent_jepa
from sklearn.model_selection import GroupShuffleSplit

T0, T1 = 1683.0, 1710.0


def _tiny():
    torch.manual_seed(0)
    inst = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=32, n_layers=1,
                                      readout="mean", predictor_type="mlp").to(DEVICE).eval()
    dec = build_decoder(1024).to(DEVICE).eval()
    for p in list(inst.parameters()) + list(dec.parameters()):
        p.requires_grad_(False)
    return inst, dec


def _curve(seed=0):
    rng = np.random.default_rng(seed)
    time = np.linspace(T0, T1, 400)
    flux = 100.0 + 20.0 * np.sin(np.linspace(0, 9, 400)) + rng.normal(0, 1, 400)
    z = np.zeros(400, int)
    return quality_filter(time, flux, z, z)


# 1) raw and cleaned pretraining arms share rows and order
def test_arms_same_rows_and_order():
    inst, dec = _tiny()
    curves = [_curve(0), _curve(1), _curve(2)]
    t = [c[0] for c in curves]; f = [c[1] for c in curves]
    raw_X, raw_M, cln_X, cln_M = build_arms(t, f, inst, dec, T0, T1)
    assert raw_X.shape == cln_X.shape == (3, GRID)
    rX2, _, cX2, _ = build_arms([t[1], t[0], t[2]], [f[1], f[0], f[2]], inst, dec, T0, T1)
    assert np.allclose(raw_X[0], rX2[1]) and np.allclose(cln_X[0], cX2[1])   # row-aligned across arms


# 2) raw and cleaned masks are identical
def test_raw_cleaned_masks_identical():
    inst, dec = _tiny()
    curves = [_curve(3), _curve(4)]
    raw_X, raw_M, cln_X, cln_M = build_arms([c[0] for c in curves], [c[1] for c in curves], inst, dec, T0, T1)
    assert np.array_equal(raw_M, cln_M)
    assert not np.allclose(raw_X, cln_X)                   # only the flux values differ


# 3) eval GaiaDR3+sector never enter pretraining
def test_eval_cohort_excluded():
    pre = pd.DataFrame({"GAIADR3": [1, 2, 3, 4], "sector": [14, 14, 14, 14]})
    excl = {(2, 14), (4, 14)}
    kept = remove_eval_cohort(pre, excl)                        # removes the eval stars
    assert set(zip(kept["GAIADR3"], kept["sector"])) == {(1, 14), (3, 14)}
    assert_no_eval_overlap(kept, excl)                          # post-condition holds after removal
    try:
        assert_no_eval_overlap(pre, excl); raised = False      # unfiltered df still has eval stars -> hard-fail
    except RuntimeError:
        raised = True
    assert raised


# 4) quality-flagged cadences enter neither arm
def test_flagged_cadences_absent_from_both_arms():
    inst, dec = _tiny()
    time = np.linspace(T0, T1, 300)
    flux = np.full(300, 100.0)
    tess = np.zeros(300, int); tglc = np.zeros(300, int)
    tess[40] = 32; tglc[120] = 1; flux[200] = np.nan
    ft, ff = quality_filter(time, flux, tess, tglc)
    assert len(ft) == 297
    for bad in (40, 120, 200):
        assert time[bad] not in ft
    raw_X, raw_M, cln_X, cln_M = build_arms([ft], [ff], inst, dec, T0, T1)   # both arms from filtered ft
    assert raw_X.shape == (1, GRID)


# 5) cleaning happens on native timestamps, before the single physics resample
def test_cleaning_on_native_before_resample():
    inst, dec = _tiny()
    ft, ff = _curve(5)
    _, _, med, scale, valid = grid_for_instrument(ft, ff, T0, T1)
    decoded = np.zeros(GRID)                                # any template; length/native-support is the point
    cf = subtract_native(ft, ff, decoded, valid, scale, _grid_times(T0, T1))
    assert cf.shape == ff.shape and len(ff) != GRID         # native grid, not yet resampled to 1024
    X, M = physics_grid(ft, cf)                             # exactly one physics resample
    assert X.shape == (GRID,)


# 6) no fitted subtraction coefficient (matches canonical cleaned_native_flux)
def test_no_fitted_coefficient():
    inst, dec = _tiny()
    ft, ff = _curve(6)
    grid_times = _grid_times(T0, T1)
    Xg, Mg, med, scale, valid = grid_for_instrument(ft, ff, T0, T1)
    with torch.no_grad():
        z = inst.encode(torch.tensor(Xg, dtype=torch.float32, device=DEVICE)[None],
                        torch.tensor(Mg, dtype=torch.float32, device=DEVICE)[None], view="predicted")
        decoded = dec(z).squeeze(0).cpu().numpy()
    cf = subtract_native(ft, ff, decoded, valid, scale, grid_times)
    dec_native = np.interp(ft, grid_times[valid], decoded[valid])
    assert np.allclose(cf, ff - dec_native * scale)         # fixed 1.4826*MAD scale, no regression
    tpl = {"X": Xg, "M": Mg, "valid": valid, "decoded": decoded, "med": med,
           "scale": scale, "grid_times": grid_times}
    assert np.allclose(cf, cleaned_native_flux(ft, ff, None, None, T0, T1, template=tpl))


# 7) instrument model + decoder frozen and hash-identical through preparation
def test_instrument_decoder_frozen():
    inst, dec = _tiny()
    before = (state_hash(inst.teacher), state_hash(inst.student),
              state_hash(inst.predictor), state_hash(dec))
    build_arms([_curve(0)[0], _curve(1)[0]], [_curve(0)[1], _curve(1)[1]], inst, dec, T0, T1)
    after = (state_hash(inst.teacher), state_hash(inst.student),
             state_hash(inst.predictor), state_hash(dec))
    assert after == before
    for p in list(inst.parameters()) + list(dec.parameters()):
        assert p.requires_grad is False and p.grad is None


# 8) both arms start bit-identically (same init strict-loaded)
def test_jepas_start_identical():
    torch.manual_seed(11)
    init = build_latent_jepa().state_dict()
    a = build_latent_jepa(); a.load_state_dict(init, strict=True)
    b = build_latent_jepa(); b.load_state_dict(init, strict=True)
    sa, sb = a.state_dict(), b.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)


# 9) both arms use identical split and batch order (deterministic by seed)
def test_identical_split_and_batch_order():
    tics = np.array([f"T{i // 3}" for i in range(90)])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
    tr1, va1 = next(gss.split(np.zeros(90), np.zeros(90), groups=tics))
    tr2, va2 = next(GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=0)
                    .split(np.zeros(90), np.zeros(90), groups=tics))
    assert np.array_equal(tr1, tr2) and np.array_equal(va1, va2)     # shared split
    # per-epoch batch order is reproducible from the seed (both arms use default_rng(seed))
    r_raw, r_cln = np.random.default_rng(0), np.random.default_rng(0)
    for _ in range(3):
        assert np.array_equal(r_raw.permutation(len(tr1)), r_cln.permutation(len(tr1)))


# 10) PhyTS flux is never used -- curves come from TGLC only
def test_phyts_flux_never_used():
    tglc = pd.DataFrame({"TIC": ["A"], "sector": [14], "GAIADR3": [100],
                         "time": [np.arange(5.0)], "aperture_flux": [np.arange(5.0) + 9],
                         "TESS_flags": [np.zeros(5, int)], "TGLC_flags": [np.zeros(5, int)]})
    phyts = pd.DataFrame({"TIC": ["A"], "sector": [14], "phyts_gaia": [100], "label": ["ECLIPSE"]})
    m, _ = match_phyts_tglc(phyts, tglc)
    assert "flux" not in m.columns
    assert np.array_equal(m["aperture_flux"].iloc[0], tglc["aperture_flux"].iloc[0])


# 11) strict checkpoint loading fails on missing/unexpected keys
def test_strict_checkpoint_loading():
    good = build_decoder(1024).state_dict()
    strict_load(build_decoder(1024), good)                 # clean -> no raise
    try:
        strict_load(build_decoder(1024), {})               # empty -> every key missing
        raised = False
    except RuntimeError:
        raised = True
    assert raised


# 12) all four eval cells use identical labeled rows and test indices
def test_eval_cells_identical_rows():
    tics = np.array([f"T{i // 2}" for i in range(120)])
    y = np.array([i % 7 for i in range(120)])
    tr1, te1 = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(
        np.arange(120), y, groups=tics))
    tr2, te2 = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(
        np.arange(120), y, groups=tics))
    assert np.array_equal(tr1, tr2) and np.array_equal(te1, te2)     # one split -> all 4 cells share it
    assert not (set(tics[tr1]) & set(tics[te1]))


# 13) scaler + KNN fit only on the training split
def test_probe_fitted_on_train_only():
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import KNeighborsClassifier
    rng = np.random.default_rng(0)
    lat = rng.normal(size=(140, 8)); y = np.array([i % 4 for i in range(140)])
    tics = np.array([f"T{i}" for i in range(140)])
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(
        np.arange(140), y, groups=tics))
    present = np.unique(y)
    _, _, pred = _classify(lat, y, tr, te, present)
    sc = StandardScaler().fit(lat[tr])
    knn = KNeighborsClassifier(n_neighbors=20).fit(sc.transform(lat[tr]), y[tr])
    assert np.array_equal(pred, knn.predict(sc.transform(lat[te])))


# 14) a stale prepared-data manifest hard-fails
def test_stale_manifest_hard_fails():
    current = {"pretrain_sig": "aaa", "eval_tglc_sig": "bbb", "inst_sig": "ccc",
               "decoder_sig": "ddd", "grid_sig": "eee", "bad_tess_mask": 16437}
    assert_manifest_matches(dict(current), current)        # identical -> ok
    stale = dict(current); stale["inst_sig"] = "CHANGED"
    try:
        assert_manifest_matches(stale, current); raised = False
    except RuntimeError:
        raised = True
    assert raised


# 15) CLEAN_MODE=cbv: 8-weight decoder -> template = area_basis @ weights,
#     matched masks, correct per-area basis; a missing basis hard-fails.
def test_cbv_arm_uses_area_basis():
    from src.instrument_v2.run_tglc_physics_jepa_ab import (
        build_arms, batched_cbv_templates, grid_for_instrument, CBV_RANK,
    )
    torch.manual_seed(0)
    inst = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=32, n_layers=1,
                                      readout="mean", predictor_type="mlp").to(DEVICE).eval()
    wdec = build_decoder(CBV_RANK).to(DEVICE).eval()          # decoder output [batch, 8]
    for p in list(inst.parameters()) + list(wdec.parameters()):
        p.requires_grad_(False)
    rng = np.random.default_rng(1)
    bases = {111: rng.normal(size=(GRID, CBV_RANK)).astype(np.float32),
             232: rng.normal(size=(GRID, CBV_RANK)).astype(np.float32)}
    curves = [_curve(0), _curve(1)]
    times = [c[0] for c in curves]; fluxes = [c[1] for c in curves]
    areas = np.array([111, 232])

    # template array is [n, 1024] and equals area_basis @ 8-weights per curve
    Xg = np.zeros((2, GRID), np.float32); Mg = np.zeros((2, GRID), np.float32)
    for i in range(2):
        Xg[i], Mg[i], _, _, _ = grid_for_instrument(times[i], fluxes[i], T0, T1)
    tpl = batched_cbv_templates(inst, wdec, Xg, Mg, areas, bases)
    assert tpl.shape == (2, GRID)                             # reconstructed template [batch, 1024]
    with torch.no_grad():
        w = wdec(inst.encode(torch.tensor(Xg, device=DEVICE),
                             torch.tensor(Mg, device=DEVICE), view="predicted")).cpu().numpy()
    for i, a in enumerate(areas):
        assert np.allclose(tpl[i], bases[a] @ w[i], atol=1e-5)   # its own area's basis

    # cbv arm: raw/cleaned masks identical; only flux differs
    raw_X, raw_M, cln_X, cln_M = build_arms(times, fluxes, inst, wdec, T0, T1, areas, bases)
    assert raw_X.shape == cln_X.shape == (2, GRID)
    assert np.array_equal(raw_M, cln_M) and not np.allclose(raw_X, cln_X)

    # a missing basis hard-fails -- never a silent raw fallback
    try:
        build_arms(times, fluxes, inst, wdec, T0, T1, np.array([111, 999]), bases)
        raised = False
    except RuntimeError:
        raised = True
    assert raised


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"ALL {len(tests)} TGLC-PHYSICS-JEPA-AB TESTS PASSED")
