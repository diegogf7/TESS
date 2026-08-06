"""Dual-latent model: shared physics S4D + shared instrument S4D + one shared MLP.

Branch contract (masked same-sector):
  physics    sees the anchor's OWN curve -- same TIC, same sector, same cadence grid --
             with ~25% of its valid cadences hidden by contiguous windows; at inference
             the four complementary masks tile the whole curve;
  instrument sees ONLY other TICs on the anchor's sector/camera/CCD and the same
             absolute cadence grid, ordered nearest-to-farthest on the detector;
  decoder    sees the physics latent concatenated with the ordered peer latents.

The loss is scored only on hidden cadences, so the physics branch cannot copy the
values it must predict; the instrument branch has the right times but the wrong stars.

Both encoders are the repository's `S4Model` (src/models/s4d.py). That encoder takes
(B, L, d_input) and its masked token pooling drops missing cadences, so a curve
enters as `raw.unsqueeze(-1)` -- one channel of length 1024, i.e. the [B, 1, 1024]
single-channel contract with the channel axis where this S4D wants it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.s4d import S4Model

CURVE_LENGTH = 1024
N_PEERS = 8
N_TOKENS = 32
TOKEN_DIM = 16
LATENT_SIZE = N_TOKENS * TOKEN_DIM              # 512 per curve


def build_encoder(d_model=128, n_layers=4, dropout=0.0, n_tokens=N_TOKENS,
                  token_dim=TOKEN_DIM):
    """n_tokens/token_dim are hard-set here, never inherited from an environment
    default (the instrument_v2 models default to 16 tokens, which is not this
    contract)."""
    return S4Model(d_input=1, d_model=d_model, n_layers=n_layers, dropout=dropout,
                   n_tokens=int(n_tokens), token_dim=int(token_dim), readout="mean")


class SharedDecoder(nn.Module):
    """One MLP for every anchor -- no per-anchor/peer/area/camera/CCD/sector head.

    (The repository's existing shared MLP, `LatentCBVDecoder`, predicts coefficients
    against a per-chip CBV basis rather than a curve, so it cannot meet the
    [B, 4608] -> [B, 1024] contract; this is the compact fallback from the spec.)
    """

    def __init__(self, in_dim=LATENT_SIZE + N_PEERS * LATENT_SIZE,
                 curve_length=CURVE_LENGTH):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, curve_length),
            nn.GELU(),
            nn.LayerNorm(curve_length),
            nn.Linear(curve_length, curve_length),
        )

    def forward(self, decoder_input):
        return self.net(decoder_input)


class DisentangleModel(nn.Module):
    def __init__(self, d_model=128, n_layers=4, dropout=0.0, n_peers=N_PEERS,
                 n_tokens=N_TOKENS, token_dim=TOKEN_DIM, curve_length=CURVE_LENGTH):
        super().__init__()
        self.n_peers, self.n_tokens, self.token_dim = int(n_peers), int(n_tokens), int(token_dim)
        self.curve_length = int(curve_length)
        self.latent_size = self.n_tokens * self.token_dim

        self.physics_encoder = build_encoder(d_model, n_layers, dropout, n_tokens, token_dim)
        self.instrument_encoder = build_encoder(d_model, n_layers, dropout, n_tokens, token_dim)
        self.decoder = SharedDecoder(self.latent_size * (1 + self.n_peers), self.curve_length)

    # ------------------------------------------------------------------ branches
    def encode_physics(self, curve, mask):
        """[B, 1024] raw + [B, 1024] mask -> [B, 32, 16] tokens."""
        return self.physics_encoder(curve.unsqueeze(-1), mask)

    def encode_peers(self, peer_raw, peer_mask):
        """[B, 8, 1024] -> tokens [B, 8, 32, 16] and ordered context [B, 4096].

        Every peer goes through the SAME instrument weights in one flattened call,
        and the peers stay in nearest-to-farthest order (never averaged)."""
        B, P, L = peer_raw.shape
        tokens = self.instrument_encoder(peer_raw.reshape(B * P, L).unsqueeze(-1),
                                         peer_mask.reshape(B * P, L))
        tokens = tokens.reshape(B, P, self.n_tokens, self.token_dim)
        return tokens, tokens.reshape(B, P * self.latent_size)

    @staticmethod
    def global_physics(tokens):
        """Mean-pool over the token/time axis, then L2-normalize: [B, T, D] -> [B, D]."""
        return F.normalize(tokens.mean(dim=1), dim=-1)

    # ------------------------------------------------------------------- forward
    def forward(self, masked_anchor_raw, physics_input_mask, peer_raw, peer_mask,
                hidden_mask):
        """Masked same-sector physics + detector-peer instrument context.

        ONE curve per star: the physics encoder sees the anchor's own sector with ~25%
        of its valid cadences hidden, so it carries that star's local timing but cannot
        copy the withheld values it is asked to predict.
        """
        current_physics_tokens = self.encode_physics(masked_anchor_raw, physics_input_mask)
        peer_instrument_tokens, instrument_context = self.encode_peers(peer_raw, peer_mask)

        physics_latent = current_physics_tokens.flatten(1)                      # [B, 512]
        decoder_input = torch.cat([physics_latent, instrument_context], dim=-1)  # [B, 4608]
        outputs = {
            "predicted_raw_anchor": self.decoder(decoder_input),                # [B, 1024]
            "current_physics_tokens": current_physics_tokens,                   # [B, 32, 16]
            "current_physics_latent": physics_latent,
            "current_global_physics": self.global_physics(current_physics_tokens),
            "peer_instrument_tokens": peer_instrument_tokens,                   # [B, 8, 32, 16]
            "instrument_context": instrument_context,                           # [B, 4096]
            "decoder_input": decoder_input,
            "hidden_mask": hidden_mask,
        }
        return outputs

    def parameter_count(self):
        groups = {"physics_s4d": self.physics_encoder, "instrument_s4d": self.instrument_encoder,
                  "decoder": self.decoder}
        counts = {k: sum(p.numel() for p in m.parameters()) for k, m in groups.items()}
        counts["total"] = sum(p.numel() for p in self.parameters())
        return counts
