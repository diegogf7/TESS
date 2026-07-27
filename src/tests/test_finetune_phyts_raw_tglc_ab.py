# All this code is from Claude
"""Contracts for the matched supervised fine-tuning A/B. Synthetic only -- no
cluster, no real checkpoints, no PhyTS/TGLC files. The physics encoder is a
freshly-built (random) LatentJEPA target encoder; instrument/decoder are tiny.

Run: python -m src.tests.test_finetune_phyts_raw_tglc_ab
"""

import copy

import numpy as np
import pandas as pd
import torch

from src.instrument_v2.finetune_phyts_raw_tglc_ab import (
    cleaned_native_flux, FTModel, load_physics_backbone, DEVICE, GRID,
    HEAD_LR, BACKBONE_LR,
)
from src.instrument_v2.eval_phyts_instrument_ab import physics_grid, matched_split
from src.instrument_v2.eval_phyts_raw_tglc_ab import match_phyts_tglc, quality_filter
from src.instrument_v2.decode_single_star_k8 import build_decoder
from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.regional_group_teacher import state_hash
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


def _backbone():
    bb = build_latent_jepa().target_encoder.to(DEVICE)
    for p in bb.parameters():
        p.requires_grad_(True)
    return bb


def _ft_model(seed, n_classes=7):
    torch.manual_seed(seed)
    bb = _backbone()
    with torch.no_grad():
        feat = bb(torch.zeros(1, GRID, 1, device=DEVICE), torch.ones(1, GRID, device=DEVICE)).reshape(1, -1).shape[1]
    return FTModel(bb, feat, n_classes).to(DEVICE)


def _curve(seed=0):
    rng = np.random.default_rng(seed)
    time = np.linspace(T0, T1, 400)
    flux = 100.0 + 20.0 * np.sin(np.linspace(0, 9, 400)) + rng.normal(0, 1, 400)
    z = np.zeros(400, int)
    return quality_filter(time, flux, z, z)


# 1) both arms: identical timestamps + masks + order (arms differ only in flux values)
def test_both_arms_identical_rows_masks_order():
    inst, dec = _tiny_instrument()
    ft, ff = _curve()
    A_X, A_M = physics_grid(ft, ff)
    cf = cleaned_native_flux(ft, ff, inst, dec, T0, T1)
    B_X, B_M = physics_grid(ft, cf)
    assert np.array_equal(A_M, B_M)                    # identical time support / masks
    assert A_X.shape == B_X.shape == (GRID,)
    assert not np.allclose(A_X, B_X)                   # only the flux values differ


# 2) PhyTS flux is never used -- the curve comes from TGLC aperture_flux only
def test_phyts_flux_never_used():
    tglc = pd.DataFrame({"TIC": ["A"], "sector": [14], "GAIADR3": [100],
                         "time": [np.arange(5.0)], "aperture_flux": [np.arange(5.0) + 7],
                         "TESS_flags": [np.zeros(5, int)], "TGLC_flags": [np.zeros(5, int)]})
    phyts = pd.DataFrame({"TIC": ["A"], "sector": [14], "phyts_gaia": [100], "label": ["ECLIPSE"]})
    m, _ = match_phyts_tglc(phyts, tglc)
    assert "flux" not in m.columns                     # no PhyTS flux column survives
    assert np.array_equal(m["aperture_flux"].iloc[0], tglc["aperture_flux"].iloc[0])


# 3) flagged cadences absent from both arms (they share one filtered array)
def test_flagged_cadences_absent_from_both_arms():
    inst, dec = _tiny_instrument()
    time = np.linspace(T0, T1, 300)
    flux = np.full(300, 100.0)
    tess = np.zeros(300, int); tglc = np.zeros(300, int)
    tess[40] = 32; tglc[120] = 1; flux[200] = np.nan
    ft, ff = quality_filter(time, flux, tess, tglc)
    assert len(ft) == 297 and np.isfinite(ff).all()
    for bad in (40, 120, 200):
        assert time[bad] not in ft
    cf = cleaned_native_flux(ft, ff, inst, dec, T0, T1)
    assert len(cf) == len(ff)                          # cleaned arm inherits the same filtered support


# 4) both arms start bit-identical per seed (and re-seeding reproduces the init)
def test_arms_start_bit_identical_per_seed():
    base = _ft_model(seed=1)
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    sa, sb = a.state_dict(), b.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    again = _ft_model(seed=1).state_dict()             # same seed -> identical construction
    assert all(torch.equal(sa[k], again[k]) for k in sa)
    diff = _ft_model(seed=2).state_dict()              # different seed -> head differs
    assert any(not torch.equal(sa[k], diff[k]) for k in sa)


# 5) instrument + decoder never get grads and never change during cleaning
def test_instrument_decoder_frozen_no_grad():
    inst, dec = _tiny_instrument()
    before = (state_hash(inst.teacher), state_hash(inst.student),
              state_hash(inst.predictor), state_hash(dec))
    ft, ff = _curve(3)
    cleaned_native_flux(ft, ff, inst, dec, T0, T1)
    after = (state_hash(inst.teacher), state_hash(inst.student),
             state_hash(inst.predictor), state_hash(dec))
    assert after == before
    for p in list(inst.parameters()) + list(dec.parameters()):
        assert p.requires_grad is False and p.grad is None


# 6) train / val / test TICs are disjoint (matched_split + val carve)
def test_train_val_test_tics_disjoint():
    from sklearn.model_selection import GroupShuffleSplit
    tics = np.array([f"T{i // 3}" for i in range(90)])
    y = np.array([i % 5 for i in range(90)])
    train_full, test_idx = matched_split(tics, y)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    sub_tr, sub_val = next(gss.split(train_full, y[train_full], groups=tics[train_full]))
    tr, val = train_full[sub_tr], train_full[sub_val]
    assert not (set(tics[tr]) & set(tics[val]))
    assert not (set(tics[tr]) & set(tics[test_idx]))
    assert not (set(tics[val]) & set(tics[test_idx]))


# 7) cleaned arm subtracts on the native grid BEFORE the single physics preprocessing
def test_subtraction_before_single_physics_preprocessing():
    inst, dec = _tiny_instrument()
    ft, ff = _curve(5)
    cf = cleaned_native_flux(ft, ff, inst, dec, T0, T1)
    assert cf.shape == ff.shape                        # native cadence grid, NOT already gridded to 1024
    assert len(ff) != GRID                             # so the single physics_grid below is the only resample
    assert not np.allclose(cf, ff)                     # subtraction actually happened
    X, M = physics_grid(ft, cf)                        # one -- and only one -- physics preprocessing
    assert X.shape == (GRID,)


# 8) physics checkpoint loading is strict (abort on missing/unexpected keys)
def test_checkpoint_loading_is_strict():
    orig = torch.load
    try:
        torch.load = lambda *a, **k: {}                # empty -> every key missing
        raised = False
        try:
            load_physics_backbone()
        except RuntimeError:
            raised = True
        assert raised, "strict load must abort on missing keys"
        good = build_latent_jepa().state_dict()        # exact keys -> loads clean
        torch.load = lambda *a, **k: good
        load_physics_backbone()
    finally:
        torch.load = orig


# 9) smoke: the physics encoder actually changes under full fine-tuning
def test_physics_encoder_changes_under_finetuning():
    model = _ft_model(seed=0)
    before = copy.deepcopy(next(model.backbone.parameters()).detach())
    X = torch.randn(8, GRID, device=DEVICE)
    M = torch.ones(8, GRID, device=DEVICE)
    y = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0], device=DEVICE)
    opt = torch.optim.AdamW([{"params": model.head.parameters(), "lr": HEAD_LR},
                             {"params": model.backbone.parameters(), "lr": BACKBONE_LR}])
    ce = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(5):
        opt.zero_grad()
        ce(model(X, M), y).backward()
        opt.step()
    after = next(model.backbone.parameters()).detach()
    assert not torch.allclose(before, after), "backbone did not update during fine-tuning"


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"ALL {len(tests)} PHYTS-FT-AB TESTS PASSED")
