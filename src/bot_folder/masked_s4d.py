"""
Masked light-curve modelling with the S4D encoder.

This replaces the BYOL/"physics_jepa" two-view objective (which had no real
prediction target and an L2-normalised 4-dim loss that is collapse-prone) with
a genuine masked-reconstruction pretext, exactly in the spirit of "take the
masking we already do for S4D and hand the hidden part to the predictor":

    1. patchify the curve into N contiguous segments of length P = L // N,
    2. randomly HIDE a fraction of the segments (zero their flux input),
    3. run the shared S4D encoder over the (partially blanked) curve -- the SSM
       convolution propagates information from the visible context into the
       hidden segments,
    4. pool each segment to a token and let a small MLP decoder reconstruct the
       *flux* of the hidden segments,
    5. loss = MSE on the hidden segments, on observed points only.

The reconstruction target is the real flux (not a moving network output), so
there is nothing to collapse onto -- training is stable, and to fill the gaps
the encoder is forced to learn the temporal/physical structure of the curve.

Stays inside the existing framework: encoder is the shared `S4Model`, the
decoder is a small MLP (same role the old `Predictor` played).
"""

import torch
import torch.nn as nn

from src.models.s4d import S4Model


class MaskedS4D(nn.Module):
    def __init__(
        self,
        grid_length=1024,
        n_tokens=16,
        token_dim=16,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        mask_ratio=0.5,
        decoder_hidden=256,
    ):
        super().__init__()

        assert grid_length % n_tokens == 0, "grid_length must be divisible by n_tokens"
        self.grid_length = grid_length
        self.n_tokens = n_tokens
        self.patch = grid_length // n_tokens
        self.token_dim = token_dim
        self.mask_ratio = mask_ratio

        # shared S4D encoder -> (B, n_tokens, token_dim) per-segment tokens
        self.encoder = S4Model(
            d_input=1,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            n_tokens=n_tokens,
            token_dim=token_dim,
        )

        # decoder: reconstruct each segment's P flux values from its token.
        # a learned per-segment position embedding tells the shared decoder
        # which segment it is reconstructing.
        self.pos_embed = nn.Parameter(torch.randn(1, n_tokens, token_dim) * 0.02)
        self.decoder = nn.Sequential(
            nn.Linear(token_dim, decoder_hidden),
            nn.GELU(),
            nn.Linear(decoder_hidden, self.patch),
        )

    def sample_segment_mask(self, batch_size, device):
        """1 = hidden segment (to be predicted), 0 = visible. Fixed count per row."""
        n_hidden = max(1, int(round(self.mask_ratio * self.n_tokens)))
        noise = torch.rand(batch_size, self.n_tokens, device=device)
        # the n_hidden smallest-noise positions are hidden -> exactly n_hidden per row
        hidden_idx = noise.argsort(dim=1)[:, :n_hidden]
        seg_mask = torch.zeros(batch_size, self.n_tokens, device=device)
        seg_mask.scatter_(1, hidden_idx, 1.0)
        return seg_mask

    def encode(self, flux, obs_mask=None):
        """Full-curve representation used by the probes (no masking)."""
        tokens = self.encoder(flux.unsqueeze(-1), obs_mask)  # (B, N, token_dim)
        return tokens

    def forward(self, flux, obs_mask=None):
        B, L = flux.shape
        device = flux.device

        seg_mask = self.sample_segment_mask(B, device)            # (B, N) 1=hidden
        keep_time = (1.0 - seg_mask).repeat_interleave(self.patch, dim=1)[:, :L]
        enc_in = (flux * keep_time).unsqueeze(-1)                 # blank hidden segments

        # plain-mean pooling (mask=None): hidden-segment tokens are non-zero
        # because the S4D convolution carries context into them.
        tokens = self.encoder(enc_in, mask=None)                 # (B, N, token_dim)
        recon = self.decoder(tokens + self.pos_embed)            # (B, N, P)
        recon = recon.reshape(B, self.n_tokens * self.patch)[:, :L]
        return recon, seg_mask


def build_masked_s4d():
    """Single source of truth for the real-data architecture, shared by the
    training and eval scripts so the checkpoint always loads cleanly."""
    return MaskedS4D(
        grid_length=1024,
        n_tokens=16,      # patch = 64 points ~ 1.3 days at 30-min cadence
        token_dim=16,     # -> 256-dim latent for the probe
        d_model=256,
        n_layers=4,
        dropout=0.2,
        mask_ratio=0.5,
    )


def masked_recon_loss(recon, flux, seg_mask, patch, obs_mask=None):
    """MSE on hidden segments only (and observed points only, if a mask is given)."""
    B, L = flux.shape
    weight = seg_mask.repeat_interleave(patch, dim=1)[:, :L]      # 1 on hidden segments
    if obs_mask is not None:
        weight = weight * obs_mask                               # ignore real gaps
    diff = (recon - flux) ** 2 * weight
    return diff.sum() / weight.sum().clamp(min=1.0)
