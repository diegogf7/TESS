# All this code is from Claude
"""Model-level tests for the Instance-to-Subspace JEPA. Synthetic only.

Run: python -m src.tests.test_instance_subspace_jepa
"""

import numpy as np
import torch

from src.instrument_v2.group_level_dataset import Sector14ChipGroupDataset
from src.instrument_v2.instance_subspace_jepa import ARMS, build_instance_subspace
from src.tests.test_sector14_jepa import synthetic_frame

torch.manual_seed(0)


def _dataset(group_size=2, n_per_chip=8):
    df = synthetic_frame(n_per_chip=n_per_chip)
    tics = set(df["TIC"].astype(str))
    t0 = min(float(np.min(t)) for t in df["time"])
    t1 = max(float(np.max(t)) for t in df["time"])
    return Sector14ChipGroupDataset(df, tics, "shared", (t0, t1),
                                    group_size=group_size, return_chip=True)


def test_context_target_disjoint_same_chip():
    ds = _dataset(group_size=3)
    np.random.seed(0)
    for _ in range(200):
        ctx_rows, tgt_rows, chip = ds._sample_groups()
        assert not set(ctx_rows) & set(tgt_rows), "context/target sets overlap"
        for row in list(ctx_rows) + list(tgt_rows):
            assert ds.chips[row] == chip, "set member from a different chip"
        assert len(set(ds.tics[list(ctx_rows) + list(tgt_rows)])) == 6, "TIC repeated"


def test_group_statistics_permutation_invariant():
    z = torch.randn(3, 6, 256)
    perm = torch.randperm(6)
    for arm in ARMS:
        model = build_instance_subspace(arm)
        a = model.group_target(z)
        b = model.group_target(z[:, perm, :])
        assert torch.allclose(a, b, atol=1e-5), f"{arm} target not permutation invariant"


def test_cov_target_sensitive_to_orientation():
    """Same mean, same per-dim variance, different covariance orientation:
    instance_cov target must change; instance_mean_var target must not."""
    base = torch.randn(1, 8, 256)
    centered = base - base.mean(dim=1, keepdim=True)
    flipped = centered.clone()
    flipped[..., 0] = -flipped[..., 0]           # flip one dim around the mean
    set_a = centered
    set_b = flipped                              # same mean (0), same per-dim var
    cov_model = build_instance_subspace("instance_cov")
    mv_model = build_instance_subspace("instance_mean_var")
    assert not torch.allclose(cov_model.group_target(set_a),
                              cov_model.group_target(set_b), atol=1e-5), \
        "instance_cov target blind to covariance orientation"
    assert torch.allclose(mv_model.group_target(set_a),
                          mv_model.group_target(set_b), atol=1e-5), \
        "mean+var target should not see orientation changes"


def test_zero_sum_perturbation_distinguishes_losses():
    """Per-star deltas summing to zero leave the mean unchanged: mean_to_mean
    prediction loss is unchanged, instance loss must increase."""
    ctx = torch.randn(2, 4, 256)
    target_code = torch.randn(2, 256)
    delta = torch.randn(2, 4, 256)
    delta = delta - delta.mean(dim=1, keepdim=True)          # zero-sum across stars
    m2m = build_instance_subspace("mean_to_mean")
    inst = build_instance_subspace("instance_mean")
    inst.load_state_dict(m2m.state_dict())                   # identical weights
    l_m2m_a = m2m.prediction_loss_from_embeddings(ctx, target_code)
    l_m2m_b = m2m.prediction_loss_from_embeddings(ctx + 5 * delta, target_code)
    assert torch.allclose(l_m2m_a, l_m2m_b, atol=1e-4), \
        "mean_to_mean loss changed under zero-sum perturbation"
    l_inst_a = inst.prediction_loss_from_embeddings(ctx, target_code)
    l_inst_b = inst.prediction_loss_from_embeddings(ctx + 5 * delta, target_code)
    assert l_inst_b > l_inst_a + 1e-3, \
        "instance loss did not increase under zero-sum perturbation"


def test_gradients_flow_online_and_projector_not_ema():
    model = build_instance_subspace("instance_cov")
    ds = _dataset(group_size=2)
    ctx_f, ctx_m, tgt_f, tgt_m, _ = ds[0]
    out = model(ctx_f[None], ctx_m[None], tgt_f[None], tgt_m[None])
    out["loss"].backward()
    online_grads = [p.grad for p in model.context_encoder.parameters() if p.grad is not None]
    proj_grads = [p.grad for p in model.instrument_projector.parameters() if p.grad is not None]
    assert online_grads and any(g.abs().sum() > 0 for g in online_grads), \
        "online encoder got no gradient"
    assert proj_grads and any(g.abs().sum() > 0 for g in proj_grads), \
        "instrument projector got no gradient"
    assert all(p.grad is None for p in model.target_encoder.parameters()), \
        "EMA encoder received gradients"


def test_ema_update_moves_target():
    model = build_instance_subspace("instance_mean")
    with torch.no_grad():
        next(model.context_encoder.parameters()).add_(1.0)
    before = next(model.target_encoder.parameters()).clone()
    model.update_target()
    after = next(model.target_encoder.parameters())
    assert not torch.equal(before, after), "EMA update did not move the target"


def test_all_arms_smoke_two_batches_finite():
    ds = _dataset(group_size=2)
    losses = {}
    for arm in ARMS:
        torch.manual_seed(1)
        model = build_instance_subspace(arm)
        opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
        for _ in range(2):
            ctx_f, ctx_m, tgt_f, tgt_m, _ = ds[0]
            out = model(ctx_f[None], ctx_m[None], tgt_f[None], tgt_m[None])
            assert torch.isfinite(out["loss"]), f"{arm}: non-finite loss"
            opt.zero_grad(); out["loss"].backward(); opt.step(); model.update_target()
        losses[arm] = float(out["loss"])
    print("smoke losses:", {k: round(v, 5) for k, v in losses.items()})


ALL_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_")]

if __name__ == "__main__":
    for fn in ALL_TESTS:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(ALL_TESTS)}/{len(ALL_TESTS)} tests passed")
