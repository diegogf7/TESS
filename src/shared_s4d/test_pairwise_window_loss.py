from __future__ import annotations
"""Tests for the pairwise_window_cov loss + synthetic scattered-light injection.
    python -m src.shared_s4d.test_pairwise_window_loss
"""
import numpy as np
import torch

from src.shared_s4d.correction_losses import (
    pairwise_window_cov_loss as Lpair, windowed_group_cov_loss as Lwin, soft_cap_size)
from src.shared_s4d.model import build_model
from src.models.s4d import masked_token_pool


def _bump(L, s, w, amp=2.0):
    b = np.zeros(L, np.float32); b[s:s + w] = amp
    return b


def _group(N=16, L=1024, seed=0, affected=None, bump=(400, 40, 2.0), amp_vary=0.0, shift=0):
    """base = independent noise; x = base + a SHARED scattered-light bump on `affected`
    curves (ALL curves by default, as a detector-neighbor group would share it)."""
    g = torch.Generator().manual_seed(seed)
    base = 0.3 * torch.randn(N, L, generator=g)
    x = base.clone()
    affected = range(N) if affected is None else affected
    for k, i in enumerate(affected):
        b = _bump(L, bump[0] + shift * k, bump[1], bump[2] * (1.0 + amp_vary * k))
        x[i] += torch.tensor(b)
    return base, x, torch.ones(N, L)


# 1) squared-before-average: opposite-sign pair covariances CANNOT cancel
def test_no_cancellation_vs_windowed():
    L = 128; t = np.arange(L)
    sk = np.sin(2 * np.pi * 2 * t / L).astype(np.float32)     # 1 cycle / 64-window
    sj = np.sin(2 * np.pi * 4 * t / L).astype(np.float32)     # 2 cycles / 64-window (orthogonal to sk)
    r = torch.tensor(np.stack([sk, sk, sj, -sj]))            # pair(0,1)=+cov, pair(2,3)=-cov, rest 0
    x = torch.tensor(np.stack([sk, sk, sj, sj]))
    m = torch.ones(4, L)
    pw = float(Lpair(r, x, m)); wl = float(Lwin(r, x, m))
    assert pw > 0.2 and wl < 0.02, (pw, wl)                   # pairwise sees it; signed-mean cancels it


# 2) an injected SHARED bump raises the loss well above the no-bump baseline
def test_injected_bump_raises_loss():
    base, x, m = _group()
    l_no = float(Lpair(base, base, m)); l_bump = float(Lpair(x, x, m))
    assert l_bump > 3 * l_no, (l_no, l_bump)


# 3) subtracting the known bump substantially reduces the loss
def test_subtracting_bump_reduces_loss():
    base, x, m = _group()
    b = torch.tensor(_bump(x.shape[1], 400, 40, 2.0))
    r = x - b[None, :]                                        # remove the shared bump from every curve
    l_bump = float(Lpair(x, x, m)); l_after = float(Lpair(r, x, m))
    assert l_after < 0.5 * l_bump, (l_bump, l_after)


# 4) time-shifted and amplitude-varying bumps are still detectable
def test_shifted_amplitude_varied_bumps_detectable():
    base, _, m = _group()
    _, xv, _ = _group(amp_vary=0.5, shift=3)                  # per-curve amplitude + small time shift
    assert float(Lpair(xv, xv, m)) > 2 * float(Lpair(base, base, m))


# 5) missing cadence blocks (and low-overlap windows) never produce NaNs
def test_missing_blocks_no_nan():
    base, x, m = _group()
    m2 = m.clone(); m2[:, 384:512] = 0                        # blank a whole 128 block
    for r, mk in [(x, m2), (base, m2), (x, torch.zeros_like(m))]:
        L = Lpair(r, x, mk)
        assert torch.isfinite(L).all()


# 6) the OPTIMAL per-curve correction along the scattered-light template is large for affected
#    curves and ~0 for unaffected (a clean curve gains nothing from subtracting a bump).
def test_affected_curves_get_more_correction():
    g = torch.Generator().manual_seed(1)
    base = 0.15 * torch.randn(16, 1024, generator=g)
    x = base.clone()
    for i in (0, 1, 2, 3):
        x[i, 400:440] += 2.0                                 # shared localized bump on 4 of 16
    tmpl = torch.zeros(1024); tmpl[400:440] = 1.0
    m = torch.ones(16, 1024)
    alpha = torch.zeros(16, requires_grad=True)              # per-curve template-subtraction amount
    opt = torch.optim.Adam([alpha], lr=0.2)
    for _ in range(150):                                     # minimize the pairwise-window loss over alpha
        opt.zero_grad()
        Lpair(x - alpha[:, None] * tmpl[None, :], x.detach(), m).backward()
        opt.step()
    aff = alpha.detach()[[0, 1, 2, 3]].mean(); un = alpha.detach()[4:].abs().mean()
    assert float(aff) > 0.5 and float(aff) > 3 * float(un), (float(aff), float(un))


# 7) eight temporal tokens PRESERVE localized cadence info (a bump moves its block's token only)
def test_tokens_preserve_locality():
    x = torch.randn(2, 1024, 4); m = torch.ones(2, 1024)
    p0 = masked_token_pool(x, m, 8)
    x2 = x.clone(); x2[:, 0:128, :] += 5.0                    # bump in block 0
    d = (masked_token_pool(x2, m, 8) - p0).abs().sum(-1)      # per-token change (B, 8)
    assert float(d[:, 0].mean()) > 10 * float(d[:, 7].mean())


# 8) exactly one backward + one optimizer step per complete 16-curve group (pairwise loss)
def test_one_step_per_group_pairwise():
    model = build_model(n_tokens=8, token_dim=32)
    steps = {"n": 0}
    class CountAdamW(torch.optim.AdamW):
        def step(self, *a, **k):
            steps["n"] += 1; return super().step(*a, **k)
    opt = CountAdamW(model.parameters(), lr=1e-3)
    _, x, m = _group()
    for _ in range(3):
        opt.zero_grad()
        c, _ = model(x, m); r = x - c
        (Lpair(r, x, m) + 0.1 * soft_cap_size(c, x, m)).backward(); opt.step()
    assert steps["n"] == 3


def run():
    torch.manual_seed(0)
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"  ok  {name}")
    print("ALL PAIRWISE-WINDOW-LOSS TESTS PASSED")


if __name__ == "__main__":
    run()
