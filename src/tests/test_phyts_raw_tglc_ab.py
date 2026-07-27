# All this code is from Claude
"""Contracts for the raw-TGLC PhyTS A/B. Synthetic only; no cluster, real
checkpoints, or PhyTS test split.

Run: python -m src.tests.test_phyts_raw_tglc_ab
"""

import numpy as np
import pandas as pd
import torch

from src.instrument_v2.eval_phyts_raw_tglc_ab import (
    match_phyts_tglc,
    quality_filter,
)
from src.instrument_v2.eval_phyts_instrument_ab import (
    matched_split,
    instrument_cleaned_curve,
    physics_grid,
)
from src.instrument_v2.sector14_dataset import BAD_TESS_MASK, grid_curve_shared
from src.instrument_v2.diagnose_chip_common_signal import normalize_median_mad
from src.instrument_v2.decode_single_star_k8 import build_decoder
from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.regional_group_teacher import state_hash

T0, T1 = 1683.0, 1710.0


def _tiny():
    torch.manual_seed(0)
    m = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=32, n_layers=1,
                                   readout="mean", predictor_type="mlp").eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _tglc(tics, sectors):
    n = len(tics)
    return pd.DataFrame({
        "TIC": tics, "sector": sectors, "GAIADR3": list(range(1, n + 1)),
        "time": [np.arange(5.0)] * n, "aperture_flux": [np.ones(5)] * n,
        "TESS_flags": [np.zeros(5, int)] * n, "TGLC_flags": [np.zeros(5, int)] * n})


def test_matching_uses_exactly_tic_and_sector():
    phyts = pd.DataFrame({"TIC": ["A", "B"], "sector": [14, 14],
                          "label": ["ECLIPSE", "APERIODIC"]})
    tglc = _tglc(["A", "B"], [14, 13])                 # B is sector 13 -> should NOT match
    matched, unmatched = match_phyts_tglc(phyts, tglc)
    assert list(matched["TIC"]) == ["A"]
    assert list(unmatched["TIC"]) == ["B"]


def test_duplicate_raw_match_hard_fails():
    phyts = pd.DataFrame({"TIC": ["A"], "sector": [14], "label": ["ECLIPSE"]})
    tglc = _tglc(["A", "A"], [14, 14])                 # duplicate (TIC, sector)
    raised = False
    try:
        match_phyts_tglc(phyts, tglc)
    except RuntimeError:
        raised = True
    assert raised


def test_changing_phyts_flux_does_not_change_arms():
    tglc = _tglc(["A"], [14])
    # the eval keeps only TIC/sector/label from PhyTS, so PhyTS flux can't enter
    p1 = pd.DataFrame({"TIC": ["A"], "sector": [14], "label": ["ECLIPSE"]})
    p2 = p1.copy()
    m1, _ = match_phyts_tglc(p1, tglc)
    m2, _ = match_phyts_tglc(p2, tglc)
    assert np.array_equal(m1["aperture_flux"].iloc[0], m2["aperture_flux"].iloc[0])
    assert "flux" not in m1.columns                    # only raw aperture_flux is carried


def test_changing_raw_flux_changes_inputs():
    time = np.linspace(T0, T1, 200)
    ft, ff1 = quality_filter(time, np.full(200, 100.0), np.zeros(200, int), np.zeros(200, int))
    _, ff2 = quality_filter(time, 100.0 + np.sin(np.linspace(0, 10, 200)),
                            np.zeros(200, int), np.zeros(200, int))
    ax1, _ = physics_grid(ft, ff1)
    ax2, _ = physics_grid(ft, ff2)
    assert not np.allclose(ax1, ax2)


def test_same_filtered_arrays_enter_both_arms():
    model, decoder = _tiny(), build_decoder(1024)
    time = np.linspace(T0, T1, 300)
    flux = 100.0 + 30.0 * np.sin(np.linspace(0, 12, 300))
    z = np.zeros(300, int)
    ft, ff = quality_filter(time, flux, z, z)
    ft2, ff2 = quality_filter(time, flux, z, z)
    assert np.array_equal(ft, ft2) and np.array_equal(ff, ff2)      # deterministic
    ax, am = physics_grid(ft, ff)                                   # arm A consumes ft, ff
    ct, cf = instrument_cleaned_curve(ft, ff, model, decoder, T0, T1)  # arm B consumes SAME ft, ff
    assert ax.shape == (1024,) and len(ct) == len(cf)


def test_flagged_cadences_absent_from_both_arms():
    time = np.linspace(T0, T1, 300)
    flux = np.full(300, 100.0)
    tess = np.zeros(300, int); tglc = np.zeros(300, int)
    tess[50] = 32; tglc[100] = 1; flux[150] = np.nan               # momentum dump, tglc flag, nan
    ft, ff = quality_filter(time, flux, tess, tglc)
    assert len(ft) == 297
    for bad in (50, 100, 150):
        assert time[bad] not in ft
    assert np.isfinite(ff).all()


def test_train_test_tics_disjoint():
    tics = np.array([f"T{i // 3}" for i in range(60)])
    y = np.array([i % 4 for i in range(60)])
    tr, te = matched_split(tics, y)
    assert not (set(tics[tr]) & set(tics[te]))


def test_frozen_models_unchanged_during_eval():
    model, decoder = _tiny(), build_decoder(1024)
    for p in decoder.parameters():
        p.requires_grad_(False)
    before = (state_hash(model.teacher), state_hash(model.student),
              state_hash(model.predictor), state_hash(decoder))
    time = np.linspace(T0, T1, 300)
    instrument_cleaned_curve(time, np.random.default_rng(0).normal(100, 3, 300),
                             model, decoder, T0, T1)
    after = (state_hash(model.teacher), state_hash(model.student),
             state_hash(model.predictor), state_hash(decoder))
    assert after == before


def test_no_scale_fit_in_cleaned_arm():
    model, decoder = _tiny(), build_decoder(1024)
    time = np.linspace(T0, T1, 300)
    flux = 100.0 + 30.0 * np.sin(np.linspace(0, 12, 300))
    _, cf = instrument_cleaned_curve(time, flux, model, decoder, T0, T1)
    # reproduce as EXACTLY input-minus-decoded, back-scaled -- no fitted coefficient
    normed, med, mad = normalize_median_mad(flux)
    scale = 1.4826 * mad
    X, M = grid_curve_shared(time, normed, T0, T1, 1024)
    with torch.no_grad():
        z = model.encode(torch.tensor(X, dtype=torch.float32)[None],
                         torch.tensor(M, dtype=torch.float32)[None], view="predicted")
        decoded = decoder(z).squeeze(0).numpy()
    expected = ((X - decoded) * scale + med)[M > 0]
    assert np.allclose(cf, expected, atol=1e-4)


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"ALL {len(tests)} PHYTS-RAW-TGLC-AB TESTS PASSED")
