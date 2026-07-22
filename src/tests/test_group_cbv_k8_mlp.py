# All this code is from Claude
"""Contracts for the two-stage group-CBV (K=8, MLP) experiment. Synthetic only;
no cluster, no checkpoints, no test TICs.

Run: python -m src.tests.test_group_cbv_k8_mlp
"""

import glob
import json
import os
import tempfile

import numpy as np
import torch

from src.instrument_v2.fixed_teacher_instrument_jepa import (
    FixedTeacherInstrumentJEPA,
    fixed_teacher_loss,
)
from src.instrument_v2.regional_cbv import (
    build_or_load_area_bases,
    ridge_reconstruct,
    train_tic_hash,
    uncentered_area_basis,
)
from src.instrument_v2.regional_group_teacher import build_regional_teacher, state_hash
from src.instrument_v2.train_group_cbv_k8_mlp import (
    AreaGroupCBVPairDataset,
    Sector14GroupCBVReconDataset,
    stack_stats,
)
from src.loss_function.gapblind_fix import gapblind_loss
from src.tests.test_area_commonmode_jepa import frame_with_areas, make_dataset

K = 8
GROUP_SIZE = 2      # small groups so synthetic areas can support 8 components


def _bases_and_base(n_per_chip=40):
    frame = frame_with_areas(n_per_chip=n_per_chip)
    base = make_dataset(grouping="area", k=K, frame=frame)
    tmp = tempfile.mkdtemp()
    bases = build_or_load_area_bases(base.X, base.M, base.areas, sorted(base.tics),
                                     K, tmp, GROUP_SIZE, 1)
    return base, bases, frame


def _pair_dataset():
    frame = frame_with_areas(n_per_chip=40)
    base = make_dataset(grouping="area", k=K, frame=frame)
    tmp = tempfile.mkdtemp()
    bases = build_or_load_area_bases(base.X, base.M, base.areas, sorted(base.tics),
                                     K, tmp, GROUP_SIZE, 1)
    return AreaGroupCBVPairDataset(base, bases, 1e-2)


def _recon_dataset():
    frame = frame_with_areas(n_per_chip=40)
    base = make_dataset(grouping="area", k=K, frame=frame)
    tmp = tempfile.mkdtemp()
    bases = build_or_load_area_bases(base.X, base.M, base.areas, sorted(base.tics),
                                     K, tmp, GROUP_SIZE, 1)
    t0 = min(float(np.min(t)) for t in frame["time"])
    t1 = max(float(np.max(t)) for t in frame["time"])
    ds = Sector14GroupCBVReconDataset(frame, set(frame["TIC"].astype(str)),
                                      (t0, t1), "area", K, grid_length=64,
                                      area_bases=bases)
    return ds


def test_regional_basis_shape_cadences_by_8():
    base, bases, _frame = _bases_and_base()
    L = base.X.shape[1]
    assert len(bases) > 0
    for B in bases.values():
        assert B.shape == (L, K)


def test_basis_fails_loudly_without_eight_components():
    rng = np.random.default_rng(0)
    raised = False
    try:
        uncentered_area_basis(rng.normal(size=(3, 64)), np.ones((3, 64)), K)
    except RuntimeError:
        raised = True
    assert raised


def test_no_val_or_test_tic_in_basis_metadata():
    frame = frame_with_areas(n_per_chip=40)
    all_t = set(frame["TIC"].astype(str))
    held = {t for t in all_t if t.endswith(("6", "7"))}
    train = sorted(all_t - held)
    sub = frame[frame["TIC"].astype(str).isin(set(train))]
    base = make_dataset(grouping="area", k=K, frame=sub)
    tmp = tempfile.mkdtemp()
    build_or_load_area_bases(base.X, base.M, base.areas, sorted(base.tics),
                             K, tmp, GROUP_SIZE, 1)
    meta = json.load(open(glob.glob(os.path.join(tmp, "*.json"))[0]))
    assert meta["train_tic_hash"] == train_tic_hash(train)
    assert meta["train_tic_hash"] != train_tic_hash(sorted(all_t))
    assert not set(base.tics) & held


def test_stage_a_groups_are_sixteen_unique_stars():
    pairs = _pair_dataset()
    np.random.seed(0)
    for _ in range(100):
        rows_a, rows_b, _area = pairs.draw_groups()
        assert len(rows_a) == K and len(rows_b) == K
        assert len(set(list(rows_a) + list(rows_b))) == 2 * K   # 16 unique, disjoint


def test_stage_b_context_excluded_from_teacher_group():
    ds = _recon_dataset()
    np.random.seed(1)
    for _ in range(100):
        context, targets, group = ds._sample_item()
        assert context not in set(targets.tolist())
        assert len(targets) == K


def test_masked_ridge_is_finite():
    _b, bases, _f = _bases_and_base()
    B = next(iter(bases.values()))
    rng = np.random.default_rng(0)
    median = rng.normal(size=B.shape[0]).astype(np.float32)
    valid = np.ones(B.shape[0])
    valid[10:40] = 0.0
    recon = ridge_reconstruct(median, valid, B, 1e-2)
    assert recon.shape == (B.shape[0],) and np.isfinite(recon).all()


def test_reconstruction_is_B_w_with_no_added_mean():
    _b, bases, _f = _bases_and_base()
    B = next(iter(bases.values()))
    valid = np.ones(B.shape[0])
    recon0 = ridge_reconstruct(np.zeros(B.shape[0], dtype=np.float32), valid, B, 1e-2)
    assert np.allclose(recon0, 0.0, atol=1e-6)                  # zero median -> exactly 0
    comp = B[:, 0].astype(np.float32)
    recon_c = ridge_reconstruct(comp, valid, B, 1e-6)
    assert np.mean((recon_c - comp) ** 2) < np.mean(comp ** 2)


def test_ema_updates_during_stage_a():
    torch.manual_seed(0)
    model = build_regional_teacher()
    before = state_hash(model.ema_encoder)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-2)
    stats_a, stats_b = torch.randn(4, 64, 2), torch.randn(4, 64, 2)
    valid = torch.ones(4, 64)
    pred, target, ctx = model(stats_a, valid, stats_b, valid)
    loss = gapblind_loss(pred, target, ctx, target_mask=valid)
    opt.zero_grad(); loss.backward(); opt.step(); model.update_target()
    assert state_hash(model.ema_encoder) != before             # EMA moved


def test_frozen_teacher_identical_while_online_changes_stage_b():
    torch.manual_seed(0)
    model = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=256,
                                       n_layers=4, readout="mean", predictor_type="mlp")
    before_teacher = model.teacher_hash()
    before_student = state_hash(model.student)
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-2)
    for _ in range(3):
        star = torch.randn(4, 1024)
        stats = torch.randn(4, 1024, 2)
        valid = torch.ones(4, 1024)
        pred, target, tokens = model(star, torch.ones_like(star), stats, valid)
        loss = fixed_teacher_loss(pred, target, tokens, valid)
        opt.zero_grad(); loss.backward(); opt.step()
        assert torch.isfinite(loss)
    assert model.teacher_hash() == before_teacher              # teacher bit-identical
    assert state_hash(model.student) != before_student         # online S4D changed
    assert all(p.grad is None for p in model.teacher.parameters())


def test_one_batch_smoke():
    ds = _recon_dataset()
    model = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=256,
                                       n_layers=4, readout="mean", predictor_type="mlp")
    items = [ds[i] for i in range(2)]          # batch of 2: cross-batch spread penalty needs >=2
    ctx_f = torch.stack([it[0] for it in items])
    ctx_m = torch.stack([it[1] for it in items])
    stats = stack_stats(torch.stack([it[2] for it in items]),
                        torch.stack([it[3] for it in items]))
    valid = torch.stack([it[4] for it in items])
    before = model.teacher_hash()
    pred, target, tokens = model(ctx_f, ctx_m, stats, valid)
    loss = fixed_teacher_loss(pred, target, tokens, valid)
    loss.backward()
    assert torch.isfinite(loss) and model.teacher_hash() == before


if __name__ == "__main__":
    tests = [v for name, v in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"ALL {len(tests)} GROUP-CBV-K8-MLP TESTS PASSED")
