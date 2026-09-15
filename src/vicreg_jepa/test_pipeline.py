"""Pre-flight checks for the VICReg-JEPA pipeline.

Run before any long training job:
    python -m src.vicreg_jepa.test_pipeline
Every check is CPU-only and the whole file finishes in well under a minute.
"""
import copy
import numpy as np
import torch

from .config import Part1Config, Part2Config
from .data import (SyntheticCurveSource, SingleCurveDataset, RegionGroupDataset,
                   random_time_mask, normalise)
from .losses import (cross_correlation_loss, variance_loss, covariance_loss,
                     invariance_loss, leave_one_out_mean, masked_mse,
                     common_mode_loss, total_loss)
from .models import S4Encoder, CommonModeDecoder, VICRegJEPA

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


def _tiny_cfg2(batch=64):
    cfg = Part2Config()
    cfg.d_model, cfg.d_state, cfg.n_layers, cfg.batch_size = 64, 16, 2, batch
    return cfg


def _tiny_model(cfg):
    p1 = S4Encoder(cfg.sys_dim, cfg.n_tokens, cfg.d_model, cfg.d_state, cfg.n_layers, 0.0)
    for p in p1.parameters():
        p.requires_grad = False
    return VICRegJEPA(p1, cfg)


# ---------------------------------------------------------------- shapes
@check("Instruction 1.1 -- Part 1 encoder maps (B,1024) -> (B,32)")
def _():
    z = S4Encoder(32, 4, 64, 16, 2)(torch.randn(8, 1024), torch.ones(8, 1024))
    assert z.shape == (8, 32), z.shape
    assert torch.isfinite(z).all()


@check("Instruction 2.3 -- all four latents have the specified shapes")
def _():
    cfg = _tiny_cfg2()
    zm, zu, zs, zp = _tiny_model(cfg)(
        torch.randn(64, 1024), torch.ones(64, 1024), torch.ones(64, 1024))
    assert zm.shape == (64, 64) and zu.shape == (64, 64)
    assert zs.shape == (64, 32) and zp.shape == (64, 64)


@check("a fully-missing time block pools to ~0, never NaN")
def _():
    from .models import masked_token_pool
    p = masked_token_pool(torch.randn(2, 1024, 8), torch.zeros(2, 1024), 4)
    assert torch.isfinite(p).all() and p.abs().max() < 1e-2


# ---------------------------------------------------------------- masking
@check("Instruction 2.2 -- mask ratio stays in [0.30, 0.50]")
def _():
    r = 1 - random_time_mask(torch.zeros(512, 1024), torch.ones(512, 1024)).mean(1)
    assert 0.295 <= r.min() and r.max() <= 0.505, (r.min().item(), r.max().item())


@check("Instruction 2.2 -- a fresh mask is drawn every call")
def _():
    a = random_time_mask(torch.zeros(256, 1024), torch.ones(256, 1024))
    b = random_time_mask(torch.zeros(256, 1024), torch.ones(256, 1024))
    assert not torch.equal(a, b)


@check("masking never reveals a real gap")
def _():
    obs = (torch.rand(64, 1024) > 0.1).float()
    vis = random_time_mask(torch.zeros(64, 1024), obs)
    assert ((vis == 1) & (obs == 0)).sum() == 0


# ---------------------------------------------------------------- losses
@check("Instruction 2.5 -- L_cor_sys matches numpy corrcoef")
def _():
    zp, zs = torch.randn(512, 64), torch.randn(512, 32)
    ref = np.abs(np.corrcoef(zp.T.numpy(), zs.T.numpy())[:64, 64:]).mean()
    assert abs(cross_correlation_loss(zp, zs).item() - ref) < 1e-5


@check("L_cor_sys is 1.0 for identical signals and sign-invariant")
def _():
    x = torch.randn(4096, 1)
    assert abs(cross_correlation_loss(x, x).item() - 1.0) < 1e-4
    assert abs(cross_correlation_loss(x, -3 * x).item() - 1.0) < 1e-4


@check("Instruction 2.6 -- L_var is ~0 at std=1 and ~gamma when collapsed")
def _():
    assert variance_loss(torch.randn(4096, 64)).item() < 0.02
    assert variance_loss(torch.zeros(64, 64)).item() > 0.98


@check("Instruction 2.7 -- L_cov is ~0 for iid dims, large for duplicated dims")
def _():
    assert covariance_loss(torch.randn(8192, 64)).item() < 0.05
    assert covariance_loss(torch.randn(8192, 1).repeat(1, 64)).item() > 1.0


@check("leave-one-out mean matches an explicit per-curve loop")
def _():
    z = torch.randn(3, 32, 16)
    brute = torch.stack([torch.stack([torch.cat([z[b, :i], z[b, i + 1:]]).mean(0)
                                      for i in range(32)]) for b in range(3)])
    assert torch.allclose(leave_one_out_mean(z), brute, atol=1e-5)


@check("masked_mse ignores gap cadences")
def _():
    m = torch.zeros(4, 10)
    m[:, :5] = 1
    assert abs(masked_mse(torch.zeros(4, 10), torch.ones(4, 10), m).item() - 1.0) < 1e-6


@check("no loss term returns NaN on a real batch")
def _():
    cfg = _tiny_cfg2()
    out = _tiny_model(cfg)(torch.randn(64, 1024), torch.ones(64, 1024),
                           random_time_mask(torch.zeros(64, 1024), torch.ones(64, 1024)))
    loss, parts = total_loss(*out, cfg)
    assert torch.isfinite(loss) and all(np.isfinite(v) for v in parts.values()), parts


# ------------------------------------------------- gradient routing (critical)
@check("regularisers on z_unmasked would add ZERO gradient (spec placement)")
def _():
    cfg = _tiny_cfg2()
    model = _tiny_model(cfg)
    flux, obs = torch.randn(64, 1024), torch.ones(64, 1024)
    vis = random_time_mask(flux, obs)

    def enc_grad(fn):
        model.zero_grad(set_to_none=True)
        zm, zu, zs, zp = model(flux, obs, vis)
        fn(zm, zu, zs, zp).backward()
        return torch.cat([p.grad.flatten() for p in model.encoder.parameters()
                          if p.grad is not None])

    g_inv = enc_grad(lambda zm, zu, zs, zp: cfg.phi * invariance_loss(zp, zu))
    g_spec = enc_grad(lambda zm, zu, zs, zp: cfg.phi * invariance_loss(zp, zu)
                      + cfg.lam * cross_correlation_loss(zu, zs)
                      + cfg.mu * variance_loss(zu, cfg.gamma)
                      + cfg.nu * covariance_loss(zu))
    assert torch.equal(g_spec, g_inv), "expected the EMA-branch terms to be inert"


@check("all four terms carry gradient into the online encoder (as implemented)")
def _():
    cfg = _tiny_cfg2()
    model = _tiny_model(cfg)
    flux, obs = torch.randn(64, 1024), torch.ones(64, 1024)
    vis = random_time_mask(flux, obs)

    def enc_grad(fn):
        model.zero_grad(set_to_none=True)
        zm, zu, zs, zp = model(flux, obs, vis)
        fn(zm, zu, zs, zp).backward()
        return torch.cat([p.grad.flatten() for p in model.encoder.parameters()
                          if p.grad is not None])

    g_inv = enc_grad(lambda zm, zu, zs, zp: cfg.phi * invariance_loss(zp, zu))
    g_all = enc_grad(lambda zm, zu, zs, zp: total_loss(zm, zu, zs, zp, cfg)[0])
    assert (g_all - g_inv).norm() > 1e-3


# ---------------------------------------------------------------- freezing / EMA
@check("Part 1 encoder and EMA encoder are frozen and never receive gradient")
def _():
    cfg = _tiny_cfg2()
    model = _tiny_model(cfg)
    assert not any(p.requires_grad for p in model.part1.parameters())
    assert not any(p.requires_grad for p in model.ema_encoder.parameters())
    assert not model.part1.training
    model.train()
    assert not model.part1.training, "model.train() must not un-eval Part 1"
    out = model(torch.randn(64, 1024), torch.ones(64, 1024), torch.ones(64, 1024))
    total_loss(*out, cfg)[0].backward()
    assert all(p.grad is None for p in model.ema_encoder.parameters())
    assert all(p.grad is None for p in model.part1.parameters())


@check("Instruction 2.9 -- EMA update equals tau*ema + (1-tau)*online")
def _():
    cfg = _tiny_cfg2()
    model = _tiny_model(cfg)
    delta = 1e-2
    ref = next(model.ema_encoder.parameters()).clone()
    online_ref = copy.deepcopy(model.encoder.state_dict())
    with torch.no_grad():
        for p in model.encoder.parameters():
            p.add_(torch.full_like(p, delta))
    model.update_ema(cfg.tau)
    moved = (next(model.ema_encoder.parameters()) - ref).abs().max().item()
    expected = (1 - cfg.tau) * delta
    assert abs(moved - expected) <= 0.05 * expected, (moved, expected)
    for k, v in model.encoder.state_dict().items():
        assert torch.equal(v, online_ref[k] + delta), "update_ema mutated the online encoder"


# ---------------------------------------------------------------- data
@check("per-curve normalisation gives ~zero median, ~unit MAD over observed only")
def _():
    src = SyntheticCurveSource(n_regions=2, per_region=8)
    f = normalise(src.flux, src.observed)
    m = src.observed.astype(bool)
    meds = np.array([np.median(f[i][m[i]]) for i in range(len(f))])
    mads = np.array([np.median(np.abs(f[i][m[i]] - np.median(f[i][m[i]]))) * 1.4826
                     for i in range(len(f))])
    assert np.abs(meds).max() < 1e-4 and np.abs(mads - 1).max() < 1e-3


@check("Instruction 1.3 -- region groups are 32 curves from a single region")
def _():
    src = SyntheticCurveSource(n_regions=4, per_region=40)
    ds = RegionGroupDataset(src, group_size=32)
    flux, obs = ds[0]
    assert flux.shape == (32, 1024) and obs.shape == (32, 1024)
    for r, idx in ds.by_region.items():
        assert len(np.unique(src.region[idx])) == 1


# ---------------------------------------------------------------- end to end
@check("one Part 1 step reduces the common-mode loss")
def _():
    torch.manual_seed(0)
    c1 = Part1Config()
    c1.d_model, c1.d_state, c1.n_layers = 64, 16, 2
    src = SyntheticCurveSource(n_regions=4, per_region=40)
    ds = RegionGroupDataset(src, c1.group_size)
    enc = S4Encoder(c1.latent_dim, c1.n_tokens, c1.d_model, c1.d_state, c1.n_layers, 0.0)
    dec = CommonModeDecoder(c1.latent_dim, c1.seq_len)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    flux, obs = ds[0]
    flux, obs = flux.unsqueeze(0), obs.unsqueeze(0)
    first = None
    for _ in range(6):
        z = enc(flux[0], obs[0]).unsqueeze(0)
        loss = common_mode_loss(dec, z, flux, obs)
        first = first if first is not None else loss.item()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert loss.item() < first, (first, loss.item())


@check("one Part 2 step reduces L_inv and moves the EMA encoder")
def _():
    torch.manual_seed(0)
    cfg = _tiny_cfg2()
    model = _tiny_model(cfg)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    src = SyntheticCurveSource(n_regions=4, per_region=24)
    ds = SingleCurveDataset(src)
    flux = torch.stack([ds[i][0] for i in range(64)])
    obs = torch.stack([ds[i][1] for i in range(64)])
    ema_before = next(model.ema_encoder.parameters()).clone()
    first = None
    for _ in range(6):
        vis = random_time_mask(flux, obs, cfg.mask_ratio_min, cfg.mask_ratio_max)
        out = model(flux, obs, vis)
        loss, parts = total_loss(*out, cfg)
        first = first if first is not None else parts["inv"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        model.update_ema(cfg.tau)
    assert parts["inv"] < first, (first, parts["inv"])
    assert (next(model.ema_encoder.parameters()) - ema_before).abs().max() > 0


def loss_weight_balance(steps=80):
    """Not a pass/fail check -- prints how the latent scale evolves under
    different (lambda, mu). Reproduces the note in config.py."""
    src = SyntheticCurveSource(n_regions=8, per_region=40)
    ds = SingleCurveDataset(src)
    from torch.utils.data import DataLoader
    print("\n  latent scale vs loss weights (target gamma = 1.0)")
    print(f"  {'lambda':>7} {'mu':>5} {'med std':>9} {'cor_sys':>9} {'inv':>9}")
    for lam, mu in ((25.0, 1.0), (5.0, 1.0), (25.0, 25.0)):
        torch.manual_seed(0)
        cfg = _tiny_cfg2()
        cfg.lam, cfg.mu = lam, mu
        model = _tiny_model(cfg)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr)
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
        step = 0
        while step < steps:
            for flux, obs in dl:
                vis = random_time_mask(flux, obs, cfg.mask_ratio_min, cfg.mask_ratio_max)
                zm, zu, zs, zp = model(flux, obs, vis)
                loss, parts = total_loss(zm, zu, zs, zp, cfg)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                model.update_ema(cfg.tau)
                step += 1
                if step >= steps:
                    break
        print(f"  {lam:>7} {mu:>5} {zm.std(dim=0).median():>9.4f} "
              f"{parts['cor_sys']:>9.4f} {parts['inv']:>9.4f}")


if __name__ == "__main__":
    import sys
    print(f"\n{len(PASS)+len(FAIL)} checks run")
    if FAIL:
        print(f"\n{len(FAIL)} FAILED:")
        for n, e in FAIL:
            print(f"  - {n}: {e}")
        sys.exit(1)
    print(f"ALL {len(PASS)} CHECKS PASSED")
    if "--weights" in sys.argv:
        loss_weight_balance()
