"""
LatentJEPA -- a JEPA that predicts in REPRESENTATION space (not data space).

This is the "work with JEPA in latent space" model. Unlike data-space
reconstruction (masked_s4d.py), here the predictor's output is a LATENT and the
loss is computed in latent space -- the defining property of JEPA.

It is the *fixed* version of the old physics_jepa, which collapsed. The four
changes that keep it from collapsing:

  1. MASKED-TARGET prediction (real I-JEPA), not two-augmented-views matching.
     The predictor must predict the target latent of each HIDDEN segment AT ITS
     POSITION -- so it has to output something different per position; it can't
     win by emitting one constant vector.
  2. TARGET LAYER-NORM (the I-JEPA / data2vec trick): targets are LayerNorm'd
     over the feature dim before the loss. This is NOT the old, broken
     "L2-normalise a 4-D vector" -- it standardises each token and removes the
     trivial constant solution.
  3. WIDER latent: token_dim 16 (=256-D), not 4.
  4. Optional variance regulariser (off by default) as a safety net.

Flow:  context curve (hidden segments zeroed) -> context encoder -> predictor
       -> predicted latent ;  full curve -> EMA target encoder -> target latent.
       loss = smooth-L1(predicted, target) on hidden segments only.
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.s4d import S4Model


class LatentPredictor(nn.Module):
    #going to change this to a transformer architecture

    def __init__(self, n_tokens=16, token_dimension=16, predictive_dimension = 128, depth =2, n_heads = 4, mlp_ratio = 4):
        super().__init__()
        

        self.in_projection = nn.Linear(token_dimension, predictive_dimension)
        self.position_embed = nn.Parameter(torch.randn(1, n_tokens, predictive_dimension) * 0.02)

        layer = nn.TransformerEncoderLayer(

            d_model = predictive_dimension,
            nhead = n_heads,
            dim_feedforward = predictive_dimension * mlp_ratio,
            dropout = 0.0,
            activation = "gelu",
            batch_first = True,
            norm_first = True,
    
        )

        self.transformer = nn.TransformerEncoder(layer, num_layers = depth)
        self.out_projection = nn.Linear(predictive_dimension, token_dimension)

    def forward(self, ctx_tokens):
        x = self.in_projection(ctx_tokens) + self.position_embed
        x = self.transformer(x)
        
        return self.out_projection(x)



class LatentJEPA(nn.Module):
    def __init__(
        self,
        grid_length=1024,
        n_tokens=16,
        token_dim=16,
        d_model=256,
        n_layers=4,
        dropout=0.2,
        mask_ratio=0.5,
        momentum=0.996,
        readout="mean",
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.patch = grid_length // n_tokens
        self.mask_ratio = mask_ratio
        self.momentum = momentum

        self.context_encoder = S4Model(
            d_input=1, d_model=d_model, n_layers=n_layers, dropout=dropout,
            n_tokens=n_tokens, token_dim=token_dim, readout=readout,
        )
        self.target_encoder = S4Model(
            d_input=1, d_model=d_model, n_layers=n_layers, dropout=dropout,
            n_tokens=n_tokens, token_dim=token_dim, readout=readout,
        )
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = LatentPredictor(n_tokens, token_dim)

    def sample_segment_mask(self, batch_size, device):
        """1 = hidden segment (a prediction target), 0 = visible context."""
        n_hidden = max(1, int(round(self.mask_ratio * self.n_tokens)))
        noise = torch.rand(batch_size, self.n_tokens, device=device)
        hidden_idx = noise.argsort(dim=1)[:, :n_hidden]
        seg_mask = torch.zeros(batch_size, self.n_tokens, device=device)
        seg_mask.scatter_(1, hidden_idx, 1.0)
        return seg_mask

    @torch.no_grad()
    def encode(self, flux, obs_mask=None):
        """Downstream representation = EMA target encoder on the full curve."""
        return self.target_encoder(flux.unsqueeze(-1), obs_mask)

    def forward(self, flux, obs_mask=None):
        B, L = flux.shape
        device = flux.device

        seg_mask = self.sample_segment_mask(B, device)         
        keep_time = (1.0 - seg_mask).repeat_interleave(self.patch, dim=1)[:, :L]
        ctx_in = (flux * keep_time).unsqueeze(-1)       

        ctx_tokens = self.context_encoder(ctx_in, mask=None)  
        pred = self.predictor(ctx_tokens) 

        with torch.no_grad():
            tgt = self.target_encoder(flux.unsqueeze(-1), obs_mask)
            tgt = F.layer_norm(tgt, (tgt.shape[-1],))

        return pred, tgt, seg_mask

    @torch.no_grad()
    def update_target(self):
        m = self.momentum
        for online_p, target_p in zip(self.context_encoder.parameters(),
                                      self.target_encoder.parameters()):
            target_p.data = m * target_p.data + (1.0 - m) * online_p.data


def jepa_latent_loss(pred, tgt, seg_mask, var_weight=0.0):
    """smooth-L1 in latent space, on hidden segments only (+ optional variance reg)."""
    per_token = F.smooth_l1_loss(pred, tgt, reduction="none").mean(dim=-1)  # (B, N)
    loss = (per_token * seg_mask).sum() / seg_mask.sum().clamp(min=1.0)

    if var_weight > 0.0:
        # discourage collapse: per-feature std across the batch should stay >~1
        flat = pred.reshape(-1, pred.shape[-1])
        std = flat.std(dim=0)
        loss = loss + var_weight * torch.relu(1.0 - std).mean()
    return loss


def build_latent_jepa():
    """Shared architecture for train + eval so the checkpoint can't drift.

    Config is read from env vars so the cluster runner can sweep configs without
    editing source (and train/eval always agree). Defaults reproduce the original
    MLP/transformer model exactly:
        JEPA_NTOKENS    (default 16)   more tokens = smaller segments = resolves
                                        faster oscillations (class signal)
        JEPA_TOKENDIM   (default 16)
        JEPA_READOUT    (default mean) "mean_std" keeps per-segment std so
                                        sub-segment class amplitude survives pooling
        JEPA_MASK_RATIO (default 0.5)
    """
    return LatentJEPA(
        grid_length=1024,
        n_tokens=int(os.environ.get("JEPA_NTOKENS", "16")),
        token_dim=int(os.environ.get("JEPA_TOKENDIM", "16")),
        d_model=256, n_layers=4, dropout=0.2,
        mask_ratio=float(os.environ.get("JEPA_MASK_RATIO", "0.5")),
        momentum=0.996,
        readout=os.environ.get("JEPA_READOUT", "mean"),
    )
