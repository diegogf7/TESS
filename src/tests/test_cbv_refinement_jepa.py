# All this code is from Claude
"""Contracts for the CBV-refinement pilot. Synthetic data only.

Run: python -m src.tests.test_cbv_refinement_jepa
"""

import numpy as np
import torch

from src.instrument_v2.diagnose_chip_common_signal import fit_chip_bases
from src.instrument_v2.fixed_teacher_instrument_jepa import fixed_teacher_loss
from src.instrument_v2.regional_group_teacher import state_hash
from src.instrument_v2.train_cbv_refinement_jepa import (
    Sector14CBVGroupStatDataset,
    build_model,
    reconstruct_curve,
    train_tic_hash,
)
from src.tests.test_area_commonmode_jepa import frame_with_areas, make_dataset


def cbv_dataset(k=4):
    frame = frame_with_areas(n_per_chip=16)
    t0 = min(float(np.min(t)) for t in frame["time"])
    t1 = max(float(np.max(t)) for t in frame["time"])
    base = make_dataset(grouping="area", k=k, frame=frame)
    bases, _ = fit_chip_bases(base.X, base.M, base.chips, k_max=8)
    ds = Sector14CBVGroupStatDataset(
        frame, set(frame["TIC"].astype(str)), (t0, t1), "area", k,
        grid_length=64, bases=bases, k_cbv=8)
    return ds, bases


# ------------------------------------------------- CBV reconstruction
def test_reconstruction_ignores_missing_cadences():
    rng = np.random.default_rng(0)
    mean = rng.normal(size=32)
    components = np.linalg.svd(rng.normal(size=(8, 32)),
                               full_matrices=False)[2][:4]
    x = mean + components[:2].sum(axis=0)
    m = np.ones(32)
    m[10:20] = 0.0                                   # unobserved block
    recon = reconstruct_curve(x, m, mean, components, k=4)
    assert recon.shape == (32,)
    assert np.isfinite(recon).all()
    # coefficients fit on observed cadences only -> observed region reconstructs
    obs = m > 0
    assert np.mean((recon[obs] - x[obs]) ** 2) < np.mean((mean[obs] - x[obs]) ** 2)


def test_reconstruction_falls_back_when_too_few_observed():
    mean = np.arange(16.0)
    components = np.zeros((4, 16))
    x = np.ones(16)
    m = np.zeros(16)
    m[:2] = 1.0                                      # fewer than k+1 observed
    recon = reconstruct_curve(x, m, mean, components, k=4)
    np.testing.assert_allclose(recon, mean.astype(np.float32))


def test_bases_use_only_provided_rows():
    ds, bases = cbv_dataset()
    # each basis mean length matches the grid; components rank <= k_cbv
    for chip, (mean, comp, n) in bases.items():
        assert mean.shape == (64,)
        assert comp.shape[0] <= 8
        assert n >= 4


def test_train_tic_hash_is_split_specific():
    assert train_tic_hash(["A", "B", "C"]) == train_tic_hash(["C", "B", "A"])
    assert train_tic_hash(["A", "B"]) != train_tic_hash(["A", "B", "C"])


# ----------------------------------------------------- dataset contract
def test_context_excluded_and_targets_same_area():
    ds, _ = cbv_dataset()
    np.random.seed(1)
    for _ in range(100):
        context, targets, group = ds._sample_item()
        assert context not in set(targets.tolist())
        assert np.all(ds.group_labels[targets] == group)


def test_item_uses_cbv_targets_and_raw_context():
    ds, _ = cbv_dataset()
    np.random.seed(2)
    ctx_f, ctx_m, median, log_mad, valid, n_obs, group = ds[0]
    assert ctx_f.shape == ctx_m.shape == (64,)
    assert median.shape == log_mad.shape == valid.shape == (64,)
    assert torch.isfinite(median).all() and torch.isfinite(log_mad).all()
    assert (valid.sum() > 0)


def test_no_test_tics_enter():
    frame = frame_with_areas(n_per_chip=16)
    all_tics = set(frame["TIC"].astype(str))
    held = {t for t in all_tics if t.endswith(("6", "7"))}
    allowed = all_tics - held
    sub = frame[frame["TIC"].isin(allowed)]
    t0 = min(float(np.min(t)) for t in sub["time"])
    t1 = max(float(np.max(t)) for t in sub["time"])
    base = make_dataset(grouping="area", k=4, frame=sub)
    bases, _ = fit_chip_bases(base.X, base.M, base.chips, k_max=8)
    ds = Sector14CBVGroupStatDataset(sub, allowed, (t0, t1), "area", 4,
                                     grid_length=64, bases=bases, k_cbv=8)
    assert not set(ds.tics) & held


# ----------------------------------------------------------- model / loss
def test_teacher_frozen_through_training_step():
    torch.manual_seed(0)
    model = build_model()
    before = model.teacher_hash()
    student_before = state_hash(model.student)
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2)
    for _ in range(3):
        star = torch.randn(4, 1024)
        stats = torch.randn(4, 1024, 2)
        valid = torch.ones(4, 1024)
        prediction, target, tokens = model(star, torch.ones_like(star),
                                            stats, valid)
        loss = fixed_teacher_loss(prediction, target, tokens, valid)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)
    assert model.teacher_hash() == before             # teacher untouched
    assert state_hash(model.student) != student_before  # student trained
    assert all(p.grad is None for p in model.teacher.parameters())


def test_predictor_absent_from_frozen_encoder_eval():
    torch.manual_seed(0)
    model = build_model().eval()
    star = torch.randn(3, 1024)
    with torch.no_grad():
        tokens = model.encode(star, torch.ones_like(star), view="online")
    assert tokens.shape == (3, 16, 16)                # encoder only, no predictor
    assert torch.isfinite(tokens).all()


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items())
             if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"ALL {len(tests)} CBV-REFINEMENT TESTS PASSED")
