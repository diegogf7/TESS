import math
import copy
import torch
import torch.nn as nn


class S4DKernel(nn.Module):
    """Diagonal state-space kernel."""

    def __init__(self, d_model, N=64, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        H, half = d_model, N // 2
        log_dt = torch.rand(H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        C = torch.randn(H, half, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.log_dt = nn.Parameter(log_dt)
        self.log_A_real = nn.Parameter(torch.log(0.5 * torch.ones(H, half)))
        self.A_imag = nn.Parameter(
            math.pi * torch.arange(half).float().unsqueeze(0).repeat(H, 1))

    def forward(self, L):
        dt = torch.exp(self.log_dt)                                  # (H,)
        C = torch.view_as_complex(self.C)                            # (H, N/2)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag           # (H, N/2)
        dtA = A * dt.unsqueeze(-1)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)
        C = C * (torch.exp(dtA) - 1.0) / A
        return 2 * torch.einsum("hn,hnl->hl", C, torch.exp(K)).real


class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0):
        super().__init__()
        self.h = d_model
        self.D = nn.Parameter(torch.randn(d_model))
        self.kernel = S4DKernel(d_model, N=d_state)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.output_linear = nn.Sequential(
            nn.Conv1d(d_model, 2 * d_model, 1), nn.GLU(dim=-2))

    def forward(self, u):                       # u: (B, H, L)
        L = u.size(-1)
        k = self.kernel(L)
        k_f = torch.fft.rfft(k, n=2 * L)
        u_f = torch.fft.rfft(u, n=2 * L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]
        y = y + u * self.D.unsqueeze(-1)
        return self.output_linear(self.dropout(self.activation(y)))


def masked_token_pool(x, mask, n_tokens, with_std=True):
    """(B, L, D) -> (B, n_tokens, D or 2D). Split the time axis into n_tokens
    consecutive blocks; take the mask-weighted mean (and std) inside each.
    A fully-missing block yields zeros, never NaN."""
    B, L, D = x.shape
    xr = x.reshape(B, n_tokens, L // n_tokens, D)
    mf = mask.reshape(B, n_tokens, L // n_tokens, 1).to(x.dtype)
    denom = mf.sum(dim=2).clamp(min=1.0)
    mean = (xr * mf).sum(dim=2) / denom
    if not with_std:
        return mean
    var = (((xr - mean.unsqueeze(2)) ** 2) * mf).sum(dim=2) / denom
    return torch.cat([mean, (var + 1e-6).sqrt()], dim=-1)


class S4Encoder(nn.Module):
    """Light curve -> flat latent. Input is 2-channel: (value, visible)."""

    def __init__(self, latent_dim, n_tokens=4, d_model=256, d_state=64,
                 n_layers=4, dropout=0.0):
        super().__init__()
        assert latent_dim % n_tokens == 0
        self.n_tokens = n_tokens
        self.latent_dim = latent_dim
        self.inp = nn.Linear(2, d_model)
        self.layers = nn.ModuleList(
            [S4D(d_model, d_state, dropout) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.head = nn.Linear(2 * d_model, latent_dim // n_tokens)
        self.mask_token = nn.Parameter(torch.zeros(1))

    def forward(self, flux, visible, pool_mask=None):
        """flux/visible: (B, L). pool_mask defaults to `visible`."""
        v = visible.unsqueeze(-1)
        value = flux.unsqueeze(-1) * v + self.mask_token * (1 - v)
        x = self.inp(torch.cat([value, v], dim=-1))          # (B, L, d_model)
        B = x.shape[0]
        x = x.transpose(-1, -2)
        for layer, norm in zip(self.layers, self.norms):
            x = norm((layer(x) + x).transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(-1, -2)                              # (B, L, d_model)
        pooled = masked_token_pool(
            x, visible if pool_mask is None else pool_mask, self.n_tokens)
        return self.head(pooled).reshape(B, self.latent_dim)


class CommonModeDecoder(nn.Module):
    """Part 1 head: group latent -> the held-out peer's curve.
    A Linear here is literally a learned CBV basis; the hidden layer just
    lets the mixing be non-linear."""

    def __init__(self, latent_dim=32, seq_len=1024, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.GELU(), nn.Linear(hidden, seq_len))

    def forward(self, z):
        return self.net(z)


class MLPPredictor(nn.Module):
    """Instruction 2.1 #3."""

    def __init__(self, dim=64, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Linear(hidden, dim))

    def forward(self, z):
        return self.net(z)


class VICRegJEPA(nn.Module):
    """Instruction 2.1: online encoder, EMA encoder, predictor, frozen Part 1."""

    def __init__(self, part1_encoder, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = S4Encoder(cfg.physics_dim, cfg.n_tokens, cfg.d_model,
                                 cfg.d_state, cfg.n_layers, cfg.dropout)
        self.ema_encoder = copy.deepcopy(self.encoder)
        for p in self.ema_encoder.parameters():
            p.requires_grad = False
        self.predictor = MLPPredictor(cfg.physics_dim)

        self.part1 = part1_encoder                       # Instruction 1.3: frozen
        self.part1.eval()
        for p in self.part1.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.part1.eval()          # frozen extractor never goes back to train mode
        return self

    def forward(self, flux, observed, visible):
        # Instruction 2.3, in order.
        z_masked = self.encoder(flux, visible, pool_mask=observed)
        with torch.no_grad():
            z_unmasked = self.ema_encoder(flux, observed, pool_mask=observed)
            z_systematics = self.part1(flux, observed, pool_mask=observed)
        z_pred = self.predictor(z_masked)

        B = flux.shape[0]
        assert z_masked.shape == (B, self.cfg.physics_dim)
        assert z_unmasked.shape == (B, self.cfg.physics_dim)
        assert z_systematics.shape == (B, self.cfg.sys_dim)
        assert z_pred.shape == (B, self.cfg.physics_dim)
        return z_masked, z_unmasked, z_systematics, z_pred

    @torch.no_grad()
    def update_ema(self, tau):
        """Instruction 2.9: theta_ema = tau * theta_ema + (1 - tau) * theta."""
        for pe, p in zip(self.ema_encoder.parameters(), self.encoder.parameters()):
            pe.mul_(tau).add_(p.detach(), alpha=1 - tau)
        for be, b in zip(self.ema_encoder.buffers(), self.encoder.buffers()):
            be.copy_(b)
