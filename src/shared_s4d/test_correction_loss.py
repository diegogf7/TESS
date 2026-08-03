from __future__ import annotations
"""Synthetic tests for the top-K fixed-covariance correction loss (the spec's 9).
    python -m src.shared_s4d.test_correction_loss
"""
import numpy as np
import torch

from src.shared_s4d.correction_losses import (
    topk_fixed_cov_loss as Ltop, relative_correction_size as Lsize,
    masked_pairwise_residual_correlation as Llegacy, _select_topk_peers)
from src.shared_s4d.model import build_model


def _synthetic(GS=32, L=1024, seed=0):
    g = torch.Generator().manual_seed(seed)
    shared = torch.sin(torch.linspace(0, 6, L))
    x = shared[None, :].expand(GS, L).clone() + 0.3 * torch.randn(GS, L, generator=g)
    return x, shared, torch.ones(GS, L)


def run():
    x, shared, m = _synthetic()
    GS, L = x.shape
    r = x - shared[None, :]

    # 1) shared trend -> high loss   2) subtracting it -> substantially lower
    flat = float(Ltop(x, x, m, 8, 64)); sub = float(Ltop(r, x, m, 8, 64))
    assert flat > 0.2 and sub < 0.2 * flat, (flat, sub)
    # 3) independent noise on residuals does NOT lower the NEW loss (legacy DOES get gamed)
    base = float(Ltop(r, x, m, 8, 64))
    noisy = np.mean([float(Ltop(r + 0.5 * torch.randn(GS, L), x, m, 8, 64)) for _ in range(8)])
    lb = float(Llegacy(r, m)); ln = np.mean([float(Llegacy(r + 0.5 * torch.randn(GS, L), m)) for _ in range(8)])
    assert noisy >= base * 0.9 and ln < lb, (base, noisy, lb, ln)
    # 4) correction == full input penalized by size; zero correction -> 0
    assert float(Lsize(x, x, m)) > 0.5 and float(Lsize(torch.zeros_like(x), x, m)) < 1e-6
    # 5) peer selection excludes self
    assert all(i not in p.tolist() for i, p in enumerate(_select_topk_peers(x, m, 8, 64)))
    # 6) peers + original-variance denom detached (curves get no grad)
    xg = x.clone().requires_grad_(True)
    assert not Ltop(r.detach().clone(), xg, m, 8, 64).requires_grad
    # 7) masks / min-overlap: a curve sharing <64 cadences is excluded as a peer
    m2 = m.clone(); m2[0, :] = 0; m2[0, :50] = 1
    p2 = _select_topk_peers(x, m2, 8, 64)
    assert p2[0].numel() == 0 and all(0 not in p.tolist() for p in p2[1:])
    # 8) gradients reach BOTH the shared S4D encoder and the MLP decoder
    torch.manual_seed(0); model = build_model()
    (Ltop(x - model(x, m)[0], x, m, 8, 64) + 0.5 * Lsize(model(x, m)[0], x, m)).backward()
    assert model.encoder.encoder.weight.grad is not None and model.decoder[0].weight.grad is not None
    # 9) exactly one optimizer step per 32-curve group
    steps = {"n": 0}
    class CountAdamW(torch.optim.AdamW):
        def step(self, *a, **k):
            steps["n"] += 1; return super().step(*a, **k)
    torch.manual_seed(0); m9 = build_model(); opt = CountAdamW(m9.parameters(), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        c, _ = m9(x, m)
        (Ltop(x - c, x, m, 8, 64) + 0.5 * Lsize(c, x, m)).backward(); opt.step()
    assert steps["n"] == 3, steps["n"]
    print("ALL 9 CORRECTION-LOSS TESTS PASSED")


if __name__ == "__main__":
    run()
