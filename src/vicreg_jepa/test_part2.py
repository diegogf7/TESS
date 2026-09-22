"""ARCHITECTURE.md section 5 — correctness checks, written before training.

    python -m src.vicreg_jepa.test_part2
"""
import sys

import numpy as np
import torch

from .part2_losses import l_inv, l_var, l_cov, l_cor_sys, total_loss
from .part2_model import PhysicsJEPA, PhysicsEncoder, Predictor, block_mask
from .models import S4Encoder

PASS, FAIL = [], []


def check(name):
    def deco(fn):
        try:
            fn()
            PASS.append(name)
            print(f"  PASS  {name}")
        except AssertionError as e:
            FAIL.append((name, str(e)))
            print(f"  FAIL  {name}: {e}")
        return fn
    return deco


def frozen_part1(sys_dim=32):
    enc = S4Encoder(sys_dim, 4, 64, 16, 2, 0.0)
    enc.eval()
    enc.requires_grad_(False)
    return enc


def tiny(total_steps=100):
    return PhysicsJEPA(frozen_part1(), latent_dim=64, sys_dim=32,
                       d_model=32, n_layers=2, d_state=16, dropout=0.0,
                       total_steps=total_steps)


def batch(b=32, L=1024):
    flux = torch.randn(b, L)
    obs = torch.ones(b, L)
    mf, mc = block_mask(flux, obs)
    return flux, obs, mf, mc


# ---------------------------------------------------------------- shapes
@check("section 4 — every tensor has the specified shape")
def _():
    m = tiny()
    flux, obs, mf, mc = batch()
    zo, zp, zt, zs = m(flux, obs, mf, mc)
    assert zo.shape == (32, 64) and zp.shape == (32, 64)
    assert zt.shape == (32, 64) and zs.shape == (32, 32)
    b = zo.shape[0]
    zo_s = (zo - zo.mean(0)) / torch.sqrt(zo.var(0) + 1e-4)
    zs_s = (zs - zs.mean(0)) / torch.sqrt(zs.var(0) + 1e-4)
    assert ((zo_s.T @ zs_s) / b).shape == (64, 32), "R must be (64, 32)"
    zc = zo - zo.mean(0)
    assert ((zc.T @ zc) / (b - 1)).shape == (64, 64), "C must be (64, 64)"


# ---------------------------------------------------------------- masking
@check("section 2.2 — masking is contiguous blocks, not scattered cadences")
def _():
    flux, obs = torch.randn(8, 1024), torch.ones(8, 1024)
    _, mc = block_mask(flux, obs, 0.4, 16, 64)
    hidden = (obs - mc)[0].numpy()
    runs, cur = [], 0
    for v in hidden:
        if v > 0:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    assert runs, "nothing was masked"
    assert min(runs) >= 8, f"found a {min(runs)}-cadence run; blocks should be 16-64"
    assert np.median(runs) >= 16, f"median run {np.median(runs)} too short for blocks"


@check("section 2.2 — masked positions carry flux 0 and mask channel 0")
def _():
    flux, obs = torch.randn(8, 1024).abs() + 1.0, torch.ones(8, 1024)
    mf, mc = block_mask(flux, obs)
    assert ((mc == 0) & (mf != 0)).sum() == 0, "a masked position kept its flux"


@check("section 2.2 — a fresh mask is drawn every call")
def _():
    flux, obs = torch.randn(8, 1024), torch.ones(8, 1024)
    _, a = block_mask(flux, obs)
    _, b = block_mask(flux, obs)
    assert not torch.equal(a, b)


@check("masking hides roughly mask_ratio of the observed cadences")
def _():
    flux, obs = torch.randn(16, 1024), torch.ones(16, 1024)
    _, mc = block_mask(flux, obs, 0.4)
    frac = ((obs - mc).sum(1) / obs.sum(1)).mean().item()
    assert 0.3 <= frac <= 0.6, f"hid {frac:.2f}, expected ~0.4"


# ---------------------------------------------------------------- losses
@check("section 5.3 — L_var ~0 for std>=1, >0 for constant data")
def _():
    assert l_var(torch.randn(4096, 64)).item() < 0.02
    assert l_var(torch.zeros(64, 64)).item() > 0.9


@check("section 5.4 — L_cov ~0 for independent dims, large for duplicates")
def _():
    assert l_cov(torch.randn(8192, 64)).item() < 0.1
    dup = torch.randn(8192, 1).repeat(1, 64)
    assert l_cov(dup).item() > 1.0


@check("section 5.5 — L_cor_sys ~0 for independent, large when z_online copies z_sys")
def _():
    zs = torch.randn(4096, 32)
    indep = l_cor_sys(torch.randn(4096, 64), zs).item()
    copied = l_cor_sys(torch.cat([zs, zs], dim=1), zs).item()
    assert indep < 0.01, f"independent gave {indep:.4f}"
    assert copied > 10 * max(indep, 1e-6), f"copy {copied:.4f} vs indep {indep:.4f}"


@check("L_inv is zero when the prediction equals the target")
def _():
    z = torch.randn(32, 64)
    assert l_inv(z, z).item() < 1e-12


# ------------------------------------------------- gradient isolation
@check("section 5.1 — TargetEncoder and SystematicsEncoder get no gradient")
def _():
    m = tiny()
    flux, obs, mf, mc = batch()
    total_loss(*m(flux, obs, mf, mc))[0].backward()
    assert all(p.grad is None for p in m.target.parameters()), "gradient reached TargetEncoder"
    assert all(p.grad is None for p in m.sys.parameters()), "gradient reached SystematicsEncoder"


@check("section 2.6 — the optimiser steps only OnlineEncoder and Predictor")
def _():
    m = tiny()
    trainable = {n for n, p in m.named_parameters() if p.requires_grad}
    assert all(n.startswith(("online.", "predictor.")) for n in trainable), \
        sorted(n for n in trainable if not n.startswith(("online.", "predictor.")))[:3]
    flux, obs, mf, mc = batch()
    total_loss(*m(flux, obs, mf, mc))[0].backward()
    for mod, name in ((m.online, "OnlineEncoder"), (m.predictor, "Predictor")):
        g = [p.grad for p in mod.parameters() if p.grad is not None]
        assert g and max(x.abs().max().item() for x in g) > 0, f"no gradient into {name}"


@check("every regulariser contributes gradient to OnlineEncoder")
def _():
    m = tiny()
    flux, obs, mf, mc = batch()

    def g(fn):
        m.zero_grad(set_to_none=True)
        zo, zp, zt, zs = m(flux, obs, mf, mc)
        fn(zo, zp, zt, zs).backward()
        return torch.cat([p.grad.flatten() for p in m.online.parameters()
                          if p.grad is not None])

    for name, fn in (("L_var", lambda zo, zp, zt, zs: l_var(zo)),
                     ("L_cov", lambda zo, zp, zt, zs: l_cov(zo)),
                     ("L_cor_sys", lambda zo, zp, zt, zs: l_cor_sys(zo, zs))):
        assert g(fn).norm() > 0, f"{name} produced no gradient"


@check("section 6 — the same regularisers on z_target would be inert")
def _():
    m = tiny()
    flux, obs, mf, mc = batch()

    def g(fn):
        m.zero_grad(set_to_none=True)
        zo, zp, zt, zs = m(flux, obs, mf, mc)
        fn(zo, zp, zt, zs).backward()
        return torch.cat([p.grad.flatten() for p in m.online.parameters()
                          if p.grad is not None])

    base = g(lambda zo, zp, zt, zs: l_inv(zp, zt))
    wrong = g(lambda zo, zp, zt, zs: l_inv(zp, zt) + 25.0 * l_cor_sys(zt, zs)
              + l_var(zt) + 0.01 * l_cov(zt))
    assert torch.equal(base, wrong), "expected the z_target terms to add nothing"


# ---------------------------------------------------------------- frozen / EMA
@check("section 5.2 — SystematicsEncoder is bit-identical across a training step")
def _():
    m = tiny()
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
    flux, obs, mf, mc = batch()
    before = m.encode_sys(flux, obs).clone()
    snap = [p.detach().clone() for p in m.sys.parameters()]
    loss, _ = total_loss(*m(flux, obs, mf, mc))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    m.update_ema(0)
    after = m.encode_sys(flux, obs)
    assert torch.equal(before, after), "frozen Part 1 output changed"
    for a, b in zip(snap, m.sys.parameters()):
        assert torch.equal(a, b), "a frozen Part 1 parameter changed"


@check("section 5.6 — EMA lags: target moves slightly and is not equal to online")
def _():
    m = tiny(total_steps=1000)
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    flux, obs, mf, mc = batch()
    before = next(m.target.parameters()).clone()
    loss, _ = total_loss(*m(flux, obs, mf, mc))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    m.update_ema(0)
    moved = (next(m.target.parameters()) - before).abs().max().item()
    assert moved > 0, "EMA did not move"
    po, pt = next(m.online.parameters()), next(m.target.parameters())
    assert not torch.equal(po, pt), "target equals online -- it is not lagging"


@check("section 2.7 — tau ramps 0.996 -> 1.0 on a cosine schedule")
def _():
    m = tiny(total_steps=1000)
    t0, tmid, t1 = m.tau(0), m.tau(500), m.tau(1000)
    assert abs(t0 - 0.996) < 1e-6, t0
    assert abs(t1 - 1.0) < 1e-6, t1
    assert t0 < tmid < t1, (t0, tmid, t1)


@check("no NaN or inf in any loss term on a real forward pass")
def _():
    m = tiny()
    flux, obs, mf, mc = batch()
    out = m(flux, obs, mf, mc)
    assert all(torch.isfinite(t).all() for t in out)
    loss, parts = total_loss(*out)
    assert torch.isfinite(loss) and all(np.isfinite(v) for v in parts.values()), parts


if __name__ == "__main__":
    print()
    print(f"{len(PASS) + len(FAIL)} checks run")
    if FAIL:
        print(f"\n{len(FAIL)} FAILED:")
        for n, e in FAIL:
            print(f"  - {n}: {e}")
        sys.exit(1)
    print(f"ALL {len(PASS)} PART 2 CHECKS PASSED")
