"""Part 2 — physics encoder. Built to ARCHITECTURE.md section 2.

OnlineEncoder sees the masked curve and is the model kept at the end.
TargetEncoder sees the full curve, is an exact copy at init, and is updated
only by EMA. Predictor maps online -> target. SystematicsEncoder comes from
Part 1, frozen, and never receives a gradient.
"""
import copy
import math

import torch
import torch.nn as nn

from .models import S4D, S4Encoder


# ------------------------------------------------------------------ encoder
class PhysicsEncoder(nn.Module):
    """S4D backbone, 2-channel input, masked-mean pool, linear head. Section 2.1."""

    def __init__(self, latent_dim=64, d_model=128, n_layers=4, d_state=64,
                 dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.inp = nn.Linear(2, d_model)
        self.layers = nn.ModuleList(
            [S4D(d_model, d_state, dropout) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.head = nn.Linear(d_model, latent_dim)

    def forward(self, flux, observed_mask):
        """flux, observed_mask: (B, L). Returns (B, latent_dim).

        At masked positions the caller has already set flux to 0 and the mask
        channel to 0, per section 2.2.
        """
        x = torch.stack([flux, observed_mask], dim=-1)      # (B, L, 2)
        x = self.inp(x).transpose(-1, -2)                   # (B, d_model, L)
        for layer, norm in zip(self.layers, self.norms):
            x = norm((layer(x) + x).transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(-1, -2)                             # (B, L, d_model)
        w = observed_mask.unsqueeze(-1)                     # masked mean over valid steps
        pooled = (x * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        return self.head(pooled)


class Predictor(nn.Module):
    """64 -> 256 -> 64, GELU, LayerNorm on the hidden layer. Section 2.1."""

    def __init__(self, dim=64, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, dim))

    def forward(self, z):
        return self.net(z)


# ------------------------------------------------------------------ masking
def block_mask(flux, observed, mask_ratio=0.4, block_min=16, block_max=64,
               generator=None):
    """Section 2.2 — hide contiguous blocks, resampled fresh every call.

    Blocks, not scattered cadences: isolated points are trivially recovered by
    local interpolation, which teaches the model nothing.

    Returns (masked_flux, masked_channel). Masked positions carry flux 0 and
    mask channel 0; real gaps stay 0 in the mask channel as well.
    """
    B, L = flux.shape
    dev = flux.device
    keep = observed.clone()
    g = generator
    for b in range(B):
        n_obs = int(observed[b].sum().item())
        target = int(mask_ratio * n_obs)
        hidden = 0
        guard = 0
        while hidden < target and guard < 512:
            guard += 1
            blk = int(torch.randint(block_min, block_max + 1, (1,),
                                    generator=g).item())
            start = int(torch.randint(0, max(L - blk, 1), (1,), generator=g).item())
            seg = keep[b, start:start + blk]
            hidden += int(seg.sum().item())
            keep[b, start:start + blk] = 0.0
    return flux * keep, keep


# ------------------------------------------------------------------ model
class PhysicsJEPA(nn.Module):
    """Section 2.1 / 2.3 / 2.7."""

    def __init__(self, systematics_encoder, latent_dim=64, sys_dim=32,
                 d_model=128, n_layers=4, d_state=64, dropout=0.1,
                 tau_base=0.996, total_steps=None):
        super().__init__()
        self.latent_dim, self.sys_dim = latent_dim, sys_dim
        self.online = PhysicsEncoder(latent_dim, d_model, n_layers, d_state, dropout)
        self.target = copy.deepcopy(self.online)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(latent_dim)

        self.sys = systematics_encoder                      # Part 1, frozen
        self.sys.eval()
        self.sys.requires_grad_(False)

        self.tau_base = float(tau_base)
        self.total_steps = total_steps

    def train(self, mode=True):
        super().train(mode)
        self.sys.eval()          # frozen extractor never returns to train mode
        self.target.eval()
        return self

    def tau(self, step):
        """Cosine ramp 0.996 -> 1.0 across training. Section 2.7."""
        if not self.total_steps:
            return self.tau_base
        p = min(max(step / float(self.total_steps), 0.0), 1.0)
        return 1.0 - (1.0 - self.tau_base) * (math.cos(math.pi * p) + 1.0) / 2.0

    def encode_sys(self, full_flux, observed):
        """Part 1's contract: full curve -> 32-D, always under no_grad."""
        with torch.no_grad():
            if isinstance(self.sys, S4Encoder):
                return self.sys(full_flux, observed, pool_mask=observed)
            return self.sys(full_flux, observed)

    def forward(self, full_flux, observed, masked_flux, masked_channel):
        # Section 2.3, in order.
        z_online = self.online(masked_flux, masked_channel)
        z_pred = self.predictor(z_online)
        with torch.no_grad():
            z_target = self.target(full_flux, observed)
            z_sys = self.encode_sys(full_flux, observed)

        B = full_flux.shape[0]
        assert z_online.shape == (B, self.latent_dim)
        assert z_target.shape == (B, self.latent_dim)
        assert z_pred.shape == (B, self.latent_dim)
        assert z_sys.shape == (B, self.sys_dim)
        return z_online, z_pred, z_target, z_sys

    @torch.no_grad()
    def update_ema(self, step=0):
        t = self.tau(step)
        for pt, po in zip(self.target.parameters(), self.online.parameters()):
            pt.mul_(t).add_(po.detach(), alpha=1.0 - t)
        for bt, bo in zip(self.target.buffers(), self.online.buffers()):
            bt.copy_(bo)
        return t
