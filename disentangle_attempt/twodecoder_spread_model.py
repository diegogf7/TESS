"""Two-decoder p64/i4 additive model for the isolated spread-peer experiment.

The learned routes are deliberately disjoint: the physics route receives only a
masked anchor, while the instrument route receives only eight different-TIC peers.
The two learned decoder outputs are added for the sole training prediction.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.s4d import S4Model


class CurveDecoder(nn.Module):
    """The exact requested latent-to-1024 MLP, with no shared parameters."""

    def __init__(self, input_dim: int, curve_length: int = 1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), 256),
            nn.GELU(),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, int(curve_length)),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.net(latent)


class TwoDecoderSpreadModel(nn.Module):
    """Masked-anchor physics decoder plus peer-only instrument decoder."""

    def __init__(
        self,
        curve_length: int = 1024,
        n_peers: int = 8,
        n_tokens: int = 32,
        token_dim: int = 16,
        physics_latent_dim: int = 64,
        instrument_token_dim: int = 4,
        d_model: int = 128,
        n_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.curve_length = int(curve_length)
        self.n_peers = int(n_peers)
        self.n_tokens = int(n_tokens)
        self.token_dim = int(token_dim)
        self.internal_dim = self.n_tokens * self.token_dim
        self.physics_latent_dim = int(physics_latent_dim)
        self.instrument_token_dim = int(instrument_token_dim)
        self.instrument_context_dim = self.n_peers * self.instrument_token_dim

        encoder_kwargs = {
            "d_input": 1,
            "d_model": int(d_model),
            "n_layers": int(n_layers),
            "dropout": float(dropout),
            "n_tokens": self.n_tokens,
            "token_dim": self.token_dim,
            "readout": "mean",
        }
        self.physics_encoder = S4Model(**encoder_kwargs)
        self.instrument_encoder = S4Model(**encoder_kwargs)
        self.physics_projection = nn.Linear(
            self.internal_dim, self.physics_latent_dim
        )
        self.instrument_projection = nn.Linear(
            self.internal_dim, self.instrument_token_dim
        )
        self.physics_decoder = CurveDecoder(
            self.physics_latent_dim, self.curve_length
        )
        self.instrument_decoder = CurveDecoder(
            self.instrument_context_dim, self.curve_length
        )

    def encode_physics(
        self, masked_anchor: torch.Tensor, physics_visible_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if masked_anchor.shape != physics_visible_mask.shape:
            raise ValueError("masked anchor and physics mask must have the same shape")
        if masked_anchor.ndim != 2 or masked_anchor.shape[1] != self.curve_length:
            raise ValueError(f"physics input must be [B,{self.curve_length}]")
        visible = physics_visible_mask.to(dtype=torch.bool)
        safe_anchor = masked_anchor.masked_fill(~visible, 0.0)
        tokens = self.physics_encoder(safe_anchor.unsqueeze(-1), visible)
        latent = self.physics_projection(tokens.flatten(1))
        return tokens, latent

    def encode_instrument(
        self, peer_raw: torch.Tensor, peer_valid_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        expected = (self.n_peers, self.curve_length)
        if peer_raw.ndim != 3 or tuple(peer_raw.shape[1:]) != expected:
            raise ValueError(f"peer input must be [B,{expected[0]},{expected[1]}]")
        if peer_raw.shape != peer_valid_mask.shape:
            raise ValueError("peer flux and validity must have the same shape")

        # The module and projection are shared, but the eight peers deliberately make
        # eight separate forward calls. No peer aggregation occurs before concat.
        token_rows, latent_rows = [], []
        for peer_index in range(self.n_peers):
            valid = peer_valid_mask[:, peer_index].to(dtype=torch.bool)
            curve = peer_raw[:, peer_index].masked_fill(~valid, 0.0)
            tokens = self.instrument_encoder(curve.unsqueeze(-1), valid)
            token_rows.append(tokens)
            latent_rows.append(self.instrument_projection(tokens.flatten(1)))
        encoded_tokens = torch.stack(token_rows, dim=1)
        peer_tokens = torch.stack(latent_rows, dim=1)
        context = peer_tokens.flatten(1)
        return encoded_tokens, peer_tokens, context

    def forward(
        self,
        masked_anchor: torch.Tensor,
        physics_visible_mask: torch.Tensor,
        peer_raw: torch.Tensor,
        peer_valid_mask: torch.Tensor,
        hidden_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        physics_tokens, physics_latent = self.encode_physics(
            masked_anchor, physics_visible_mask
        )
        instrument_tokens, peer_tokens, instrument_context = self.encode_instrument(
            peer_raw, peer_valid_mask
        )
        physics_curve = self.physics_decoder(physics_latent)
        instrument_correction = self.instrument_decoder(instrument_context)
        reconstructed_raw = physics_curve + instrument_correction
        outputs = {
            "physics_encoder_tokens": physics_tokens,
            "physics_latent": physics_latent,
            "instrument_encoder_tokens": instrument_tokens,
            "peer_tokens": peer_tokens,
            "instrument_context": instrument_context,
            "physics_curve": physics_curve,
            "instrument_correction": instrument_correction,
            "reconstructed_raw": reconstructed_raw,
        }
        if hidden_mask is not None:
            outputs["hidden_mask"] = hidden_mask
        return outputs

    @staticmethod
    def apply_output_equations(
        raw_anchor: torch.Tensor, outputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Apply public equations without feeding the anchor into either decoder."""
        enriched = dict(outputs)
        enriched["cleaned_curve"] = raw_anchor - outputs["instrument_correction"]
        enriched["residual"] = raw_anchor - outputs["reconstructed_raw"]
        return enriched

    def parameter_count(self) -> dict[str, int]:
        modules = {
            "physics_s4d": self.physics_encoder,
            "instrument_s4d": self.instrument_encoder,
            "physics_projection": self.physics_projection,
            "instrument_projection": self.instrument_projection,
            "physics_decoder": self.physics_decoder,
            "instrument_decoder": self.instrument_decoder,
        }
        counts = {
            name: sum(parameter.numel() for parameter in module.parameters())
            for name, module in modules.items()
        }
        counts["total"] = sum(parameter.numel() for parameter in self.parameters())
        return counts


def build_twodecoder_model(config: dict) -> TwoDecoderSpreadModel:
    return TwoDecoderSpreadModel(
        curve_length=config["curve_length"],
        n_peers=config["n_peers"],
        n_tokens=config["n_tokens"],
        token_dim=config["token_dim"],
        physics_latent_dim=config["physics_latent_dim"],
        instrument_token_dim=config["instrument_token_dim"],
        d_model=config["d_model"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
    )
