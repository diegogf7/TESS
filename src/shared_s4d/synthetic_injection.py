from __future__ import annotations
"""Synthetic scattered-light injection demo for the pairwise_window_cov + 8-token model.

Builds a 16-curve group of independent stellar noise, injects a SHARED localized
scattered-light bump into most curves (with per-curve amplitude/width variation) plus
one UNAFFECTED clean curve, trains the real shared-weight model for a few hundred steps
with the pairwise-window loss + soft cap, and saves original(grey)/correction(red)/
cleaned(blue) plots. Prints the pairwise-window loss reduction and the correction RMS
for affected vs unaffected curves -- affected should be corrected, the clean one left alone.

    python -m src.shared_s4d.synthetic_injection
Env: OUT (png path), STEPS, LR, SEED, N, LAMBDA_SIZE.
"""
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.shared_s4d.model import build_model
from src.shared_s4d.correction_losses import pairwise_window_cov_loss, soft_cap_size, relative_correction_size

OUT = os.environ.get("OUT", "artifacts/shared_s4d/correction_v1/synthetic_injection.png")
STEPS = int(os.environ.get("STEPS", "400"))
LR = float(os.environ.get("LR", "3e-3"))
SEED = int(os.environ.get("SEED", "0"))
N = int(os.environ.get("N", "16"))
LAMBDA_SIZE = float(os.environ.get("LAMBDA_SIZE", "0.1"))
L = 1024


def make_group():
    g = torch.Generator().manual_seed(SEED)
    base = 0.3 * torch.randn(N, L, generator=g)               # independent stellar noise
    # give each curve its own smooth stellar variability (must be PRESERVED, not removed)
    t = torch.linspace(0, 6.28, L)
    for i in range(N):
        base[i] += 0.5 * torch.sin(t * (1 + 0.3 * i) + i)
    x = base.clone()
    affected = list(range(N - 1))                             # last curve is UNAFFECTED (clean)
    for k, i in enumerate(affected):
        s = 400 + 2 * k                                       # small per-curve time shift
        w = 40 + (k % 5)                                      # width variation
        amp = 2.0 * (1.0 + 0.25 * (k % 4))                    # amplitude variation
        x[i, s:s + w] += amp
    return base, x, torch.ones(N, L), affected


def main():
    torch.manual_seed(SEED)
    base, x, m, affected = make_group()
    unaffected = [N - 1]
    model = build_model(n_tokens=8, token_dim=32)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    pw0 = float(pairwise_window_cov_loss(x, x, m))
    for step in range(STEPS):
        opt.zero_grad()
        c, _ = model(x, m); r = x - c
        loss = pairwise_window_cov_loss(r, x, m) + LAMBDA_SIZE * soft_cap_size(c, x, m)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        c, _ = model(x, m); r = x - c
    pw1 = float(pairwise_window_cov_loss(r, x, m))
    reduction = 100.0 * (pw0 - pw1) / pw0
    crms = torch.sqrt((c ** 2).mean(1))
    print(f"pairwise-window loss {pw0:.4f} -> {pw1:.4f}  ({reduction:.1f}% reduction)", flush=True)
    print(f"correction RMS: affected median {crms[affected].median():.3f} | "
          f"unaffected {crms[unaffected].mean():.3f}  (want affected >> unaffected)", flush=True)
    print(f"energy ratio (mean c^2 / var x): {float(relative_correction_size(c, x, m)):.3f}", flush=True)
    finite = bool(torch.isfinite(c).all())
    print(f"finite corrections: {finite}", flush=True)

    show = affected[:4] + unaffected                          # 4 affected + the clean one
    fig, axes = plt.subplots(len(show), 1, figsize=(12, 2.1 * len(show)), sharex=True)
    g = np.arange(L)
    for ax, i in zip(axes, show):
        lab = "UNAFFECTED (clean)" if i in unaffected else "affected"
        ax.plot(g, x[i].numpy(), ".", ms=2, color="0.6", label="original")
        ax.plot(g, c[i].numpy(), color="tab:red", lw=0.9, label="correction c")
        ax.plot(g, r[i].numpy(), ".", ms=2, color="tab:blue", label="cleaned x-c")
        ax.set_ylabel(f"curve {i}\n{lab}", fontsize=7)
    axes[0].legend(fontsize=7, ncol=3, loc="upper right")
    fig.suptitle(f"synthetic scattered-light injection: pairwise_window_cov + 8 tokens "
                 f"({reduction:.0f}% pw reduction)")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.tight_layout(); fig.savefig(OUT, dpi=130); plt.close(fig)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
