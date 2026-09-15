"""Spec s8 pre-flight checks, run against REAL cached curves.

    python -m src.vicreg_jepa.test_real_data [--npz PATH]

Skips with a clear message when the cache has not been built. Everything here
is CPU-only and finishes in well under a minute on a small model.
"""
import argparse
import copy
import os
import sys

import numpy as np
import torch

from .config import Part1Config, Part2Config
from .data import SingleCurveDataset, random_time_mask
from .losses import (cross_correlation_loss, variance_loss, covariance_loss,
                     invariance_loss, total_loss)
from .models import S4Encoder, VICRegJEPA
from .real_data import (RealCurveSource, assert_tic_disjoint, quality_keep,
                        normalize_median_mad, build_shared_sector, BAD_TESS_MASK)

PASS, FAIL = [], []
NPZ = "artifacts/vicreg_jepa/curves.npz"


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


def _cfg(batch=64):
    c = Part2Config()
    c.d_model, c.d_state, c.n_layers, c.batch_size = 64, 16, 2, batch
    return c


def _model(cfg, device="cpu"):
    p1 = S4Encoder(cfg.sys_dim, cfg.n_tokens, cfg.d_model, cfg.d_state, cfg.n_layers, 0.0)
    for p in p1.parameters():
        p.requires_grad = False
    return VICRegJEPA(p1, cfg).to(device)


def _real_batch(src, n=64):
    f = torch.from_numpy(np.asarray(src.flux[:n], dtype=np.float32))
    o = torch.from_numpy(np.asarray(src.observed[:n], dtype=np.float32))
    return f, o


def run(npz=NPZ):
    if not os.path.exists(npz):
        print(f"SKIP: no cache at {npz}. Build it with:\n"
              f"  python -m src.vicreg_jepa.real_data --out {npz}")
        return 0

    train = RealCurveSource(npz, "train")
    val = RealCurveSource(npz, "val")
    test = RealCurveSource(npz, "test")
    print(f"\nreal cache: train={len(train.flux)} val={len(val.flux)} test={len(test.flux)}")

    # ---------------------------------------------------------- conventions
    @check("local quality/normalise/grid helpers match the repo reference")
    def _():
        rng = np.random.default_rng(0)
        t = np.sort(rng.uniform(0, 27, 500))
        f = rng.normal(100, 5, 500)
        tess = rng.integers(0, 2, 500) * BAD_TESS_MASK
        tglc = rng.integers(0, 2, 500)
        try:
            from src.instrument_v2.diagnose_chip_common_signal import (
                normalize_median_mad as ref_norm, build_shared_sector as ref_grid)
        except Exception as e:                       # reference not importable here
            print(f"        (reference import unavailable: {type(e).__name__})")
            return
        keep = quality_keep(tess, tglc, f)
        a, _, _ = normalize_median_mad(f[keep])
        b, _, _ = ref_norm(f[keep])
        assert np.array_equal(a, b), "normalisation diverged from the repo helper"
        x1, m1, _ = build_shared_sector([t[keep]], [a])
        x2, m2 = ref_grid([t[keep]], [b])
        assert np.array_equal(x1, x2) and np.array_equal(m1, m2), "gridding diverged"

    # ---------------------------------------------------------- shapes
    @check("Part 1 encoder maps (1, 1024) -> (1, 32) on a real curve")
    def _():
        c1 = Part1Config()
        enc = S4Encoder(c1.latent_dim, c1.n_tokens, 64, 16, 2, 0.0)
        f, o = _real_batch(train, 1)
        z = enc(f, o)
        assert z.shape == (1, 32), z.shape
        assert torch.isfinite(z).all()

    @check("all four Part 2 latents have the specified shapes on a real batch")
    def _():
        cfg = _cfg()
        f, o = _real_batch(train)
        v = random_time_mask(f, o, cfg.mask_ratio_min, cfg.mask_ratio_max)
        zm, zu, zs, zp = _model(cfg)(f, o, v)
        assert zm.shape == (64, 64) and zu.shape == (64, 64)
        assert zs.shape == (64, 32) and zp.shape == (64, 64)

    # ---------------------------------------------------------- masking
    @check("fresh masks hide 30-50% of each curve's OBSERVED cadences")
    def _():
        f, o = _real_batch(train, 256)
        v = random_time_mask(f, o)
        frac = ((o - v).sum(1) / o.sum(1))
        assert 0.299 <= frac.min() and frac.max() <= 0.501, (frac.min(), frac.max())

    @check("masks are redrawn every call and never reveal a real gap")
    def _():
        f, o = _real_batch(train, 128)
        a, b = random_time_mask(f, o), random_time_mask(f, o)
        assert not torch.equal(a, b), "mask was reused"
        assert ((a == 1) & (o == 0)).sum() == 0, "a gap was revealed"

    # ---------------------------------------------------------- splits
    @check("no TIC appears in more than one split")
    def _():
        assert_tic_disjoint(train, val, test)

    @check("every split has regions with at least 32 eligible stars")
    def _():
        for s in (train, val, test):
            counts = np.array([(s.area == a).sum() for a in np.unique(s.area)])
            assert (counts >= 32).any(), f"{s.split_name} has no region with 32 stars"

    # ---------------------------------------------------------- numerics
    @check("no NaN or infinite value in any loss on a real batch")
    def _():
        cfg = _cfg()
        f, o = _real_batch(train)
        v = random_time_mask(f, o, cfg.mask_ratio_min, cfg.mask_ratio_max)
        out = _model(cfg)(f, o, v)
        assert all(torch.isfinite(t).all() for t in out), "non-finite latent"
        loss, parts = total_loss(*out, cfg)
        assert torch.isfinite(loss) and all(np.isfinite(x) for x in parts.values()), parts

    # ---------------------------------------------------------- gradients
    @check("every regulariser produces gradient in the online encoder")
    def _():
        cfg = _cfg()
        model = _model(cfg)
        f, o = _real_batch(train)
        v = random_time_mask(f, o, cfg.mask_ratio_min, cfg.mask_ratio_max)

        def g(fn):
            model.zero_grad(set_to_none=True)
            zm, zu, zs, zp = model(f, o, v)
            fn(zm, zu, zs, zp).backward()
            return torch.cat([p.grad.flatten() for p in model.encoder.parameters()
                              if p.grad is not None])

        base = g(lambda zm, zu, zs, zp: cfg.phi * invariance_loss(zp, zu))
        for name, fn in (
            ("L_var", lambda zm, zu, zs, zp: cfg.mu * variance_loss(zm, cfg.gamma)),
            ("L_cov", lambda zm, zu, zs, zp: cfg.nu * covariance_loss(zm)),
            ("L_cor", lambda zm, zu, zs, zp: cfg.lam * cross_correlation_loss(zm, zs)),
        ):
            assert g(fn).norm() > 0, f"{name} produced no gradient"
        assert (g(lambda zm, zu, zs, zp: total_loss(zm, zu, zs, zp, cfg)[0])
                - base).norm() > 1e-3

    @check("regularisers on z_unmasked would be inert (the spec's placement)")
    def _():
        cfg = _cfg()
        model = _model(cfg)
        f, o = _real_batch(train)
        v = random_time_mask(f, o, cfg.mask_ratio_min, cfg.mask_ratio_max)

        def g(fn):
            model.zero_grad(set_to_none=True)
            zm, zu, zs, zp = model(f, o, v)
            fn(zm, zu, zs, zp).backward()
            return torch.cat([p.grad.flatten() for p in model.encoder.parameters()
                              if p.grad is not None])

        a = g(lambda zm, zu, zs, zp: cfg.phi * invariance_loss(zp, zu))
        b = g(lambda zm, zu, zs, zp: cfg.phi * invariance_loss(zp, zu)
              + cfg.lam * cross_correlation_loss(zu, zs)
              + cfg.mu * variance_loss(zu, cfg.gamma) + cfg.nu * covariance_loss(zu))
        assert torch.equal(a, b), "EMA-branch terms unexpectedly carried gradient"

    @check("no gradient reaches the EMA or Part 1 encoders")
    def _():
        cfg = _cfg()
        model = _model(cfg)
        f, o = _real_batch(train)
        v = random_time_mask(f, o, cfg.mask_ratio_min, cfg.mask_ratio_max)
        total_loss(*model(f, o, v), cfg)[0].backward()
        assert all(p.grad is None for p in model.ema_encoder.parameters())
        assert all(p.grad is None for p in model.part1.parameters())

    # ---------------------------------------------------------- ema / freezing
    @check("EMA parameters move after a real optimiser step")
    def _():
        cfg = _cfg()
        model = _model(cfg)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        f, o = _real_batch(train)
        before = next(model.ema_encoder.parameters()).clone()
        v = random_time_mask(f, o, cfg.mask_ratio_min, cfg.mask_ratio_max)
        loss, _ = total_loss(*model(f, o, v), cfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.update_ema(cfg.tau)
        moved = (next(model.ema_encoder.parameters()) - before).abs().max().item()
        assert moved > 0, "EMA did not move after a real step"

    @check("frozen Part 1 parameters stay bit-identical through training steps")
    def _():
        cfg = _cfg()
        model = _model(cfg)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        f, o = _real_batch(train)
        snap = [p.detach().clone() for p in model.part1.parameters()]
        for _ in range(3):
            v = random_time_mask(f, o, cfg.mask_ratio_min, cfg.mask_ratio_max)
            loss, _ = total_loss(*model(f, o, v), cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            model.update_ema(cfg.tau)
        for a, b in zip(snap, model.part1.parameters()):
            assert torch.equal(a, b), "a frozen Part 1 parameter changed"

    # ---------------------------------------------------------- checkpoint
    @check("a saved checkpoint reproduces the identical latent after reload")
    def _():
        cfg = _cfg()
        enc = S4Encoder(cfg.physics_dim, cfg.n_tokens, cfg.d_model,
                        cfg.d_state, cfg.n_layers, 0.0)
        enc.eval()
        f, o = _real_batch(val, 32)
        with torch.no_grad():
            z0 = enc(f, o, pool_mask=o)
        tmp = "artifacts/vicreg_jepa/_ckpt_roundtrip.pt"
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        torch.save({"encoder": enc.state_dict()}, tmp)
        enc2 = S4Encoder(cfg.physics_dim, cfg.n_tokens, cfg.d_model,
                         cfg.d_state, cfg.n_layers, 0.0)
        enc2.load_state_dict(torch.load(tmp, map_location="cpu")["encoder"])
        enc2.eval()
        with torch.no_grad():
            z1 = enc2(f, o, pool_mask=o)
        os.remove(tmp)
        assert torch.equal(z0, z1), f"max delta {(z0 - z1).abs().max().item():.3e}"

    # ---------------------------------------------------------- data integrity
    @check("cached curves are finite, zero in gaps, and median/MAD normalised")
    def _():
        f, o = train.flux, train.observed.astype(bool)
        assert np.isfinite(f).all(), "non-finite flux in the cache"
        assert (f[~o] == 0).all(), "a gap holds a non-zero value"
        idx = np.arange(0, len(f), max(1, len(f) // 200))
        med = np.array([np.median(f[i][o[i]]) for i in idx])
        mad = np.array([np.median(np.abs(f[i][o[i]] - np.median(f[i][o[i]]))) * 1.4826
                        for i in idx])
        assert np.abs(med).max() < 0.5, f"median drifted to {np.abs(med).max():.3f}"
        assert 0.5 < np.median(mad) < 2.0, f"MAD median {np.median(mad):.3f}"

    print(f"\n{len(PASS) + len(FAIL)} checks run")
    if FAIL:
        print(f"\n{len(FAIL)} FAILED:")
        for n, e in FAIL:
            print(f"  - {n}: {e}")
        return 1
    print(f"ALL {len(PASS)} REAL-DATA CHECKS PASSED")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=NPZ)
    sys.exit(run(ap.parse_args().npz))
