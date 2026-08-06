"""Frozen encoders + two additive MLP heads: does the latent space split additively?

    predicted_raw = physics_head(physics_latent) + instrument_head(instrument_latent)

The physics and instrument S4D weights are taken from an existing checkpoint and never
updated -- eval mode, requires_grad_(False), and their hashes are compared before and
after training. The original shared decoder is NOT loaded or used.

The question this asks is narrow on purpose: given latent spaces that were learned by a
CONCATENATED decoder, can two independent heads reproduce the anchor by adding their
outputs? Nothing constrains what each head puts in its curve, so a collapse (one head
explaining everything, the other near-constant) is a real possible answer and is
reported rather than designed away.
"""

import hashlib

import torch
import torch.nn as nn

from disentangle_attempt.model import DisentangleModel

LATENT_SIZE = 512
HEAD_HIDDEN = 512


def state_hash(module):
    """Stable digest of a module's weights, for before/after comparison."""
    digest = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()[:16]


class PhysicsHead(nn.Module):
    """[B, 512] -> [B, 1024]. No final activation: the output is a flux curve."""

    def __init__(self, in_dim=LATENT_SIZE, hidden=HEAD_HIDDEN, curve_length=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, curve_length),
        )

    def forward(self, latent):
        return self.net(latent)


class InstrumentHead(nn.Module):
    """[B, 4096] -> [B, 1024]. No final activation."""

    def __init__(self, in_dim=8 * LATENT_SIZE, hidden=HEAD_HIDDEN, curve_length=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, curve_length),
        )

    def forward(self, representation):
        return self.net(representation)


class AdditiveHeadsModel(nn.Module):
    """Frozen encoders from a source checkpoint + two trainable additive heads."""

    def __init__(self, checkpoint_path, map_location="cpu"):
        super().__init__()
        state = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        config = state["config"]
        source = DisentangleModel(
            d_model=config.get("d_model", 128), n_layers=config.get("n_layers", 4),
            dropout=0.0, n_peers=config["n_peers"], n_tokens=config["n_tokens"],
            token_dim=config["token_dim"], curve_length=config["curve_length"])
        source.load_state_dict(state["model"])

        # The shared decoder is deliberately dropped: only the two encoders come across.
        self.physics_encoder = source.physics_encoder
        self.instrument_encoder = source.instrument_encoder
        self.source_config = config
        self.curve_length = int(config["curve_length"])
        self.n_peers = int(config["n_peers"])
        self.latent_size = int(config["n_tokens"]) * int(config["token_dim"])

        for encoder in (self.physics_encoder, self.instrument_encoder):
            encoder.eval()
            encoder.requires_grad_(False)
        self.frozen_hashes = {"physics_s4d": state_hash(self.physics_encoder),
                              "instrument_s4d": state_hash(self.instrument_encoder)}

        self.physics_head = PhysicsHead(self.latent_size, HEAD_HIDDEN, self.curve_length)
        self.instrument_head = InstrumentHead(self.n_peers * self.latent_size,
                                              HEAD_HIDDEN, self.curve_length)

    def train(self, mode=True):
        """Keep the frozen encoders in eval mode even when the model is training."""
        super().train(mode)
        self.physics_encoder.eval()
        self.instrument_encoder.eval()
        return self

    def trainable_parameters(self):
        return list(self.physics_head.parameters()) + list(self.instrument_head.parameters())

    def encoders_unchanged(self):
        return {name: state_hash(getattr(self, name.replace("_s4d", "_encoder")))
                == value for name, value in self.frozen_hashes.items()}

    # ------------------------------------------------------------------ encoding
    @torch.no_grad()
    def physics_latent(self, masked_curve, visible_mask):
        """[B, 1024] -> [B, 512], detached: no gradient reaches the encoder."""
        tokens = self.physics_encoder(masked_curve.unsqueeze(-1), visible_mask)
        return tokens.flatten(1).detach()

    @torch.no_grad()
    def instrument_representation(self, peer_raw, peer_mask):
        """[B, 8, 1024] -> ([B, 8, 512] per-peer, [B, 4096] concatenated), detached.

        Peers keep their nearest-to-farthest order; the eight latents are concatenated,
        never averaged.
        """
        B, P, L = peer_raw.shape
        tokens = self.instrument_encoder(peer_raw.reshape(B * P, L).unsqueeze(-1),
                                         peer_mask.reshape(B * P, L))
        per_peer = tokens.reshape(B, P, self.latent_size).detach()
        return per_peer, per_peer.reshape(B, P * self.latent_size)

    # ------------------------------------------------------------------- forward
    def forward(self, physics_latent, instrument_representation, hidden_mask=None):
        """Latents in, additive reconstruction out. Gradients reach only the heads."""
        predicted_physics = self.physics_head(physics_latent)
        predicted_instrument = self.instrument_head(instrument_representation)
        return {
            "predicted_physics": predicted_physics,               # [B, 1024]
            "predicted_instrument": predicted_instrument,         # [B, 1024]
            "predicted_raw_anchor": predicted_physics + predicted_instrument,
            "hidden_mask": hidden_mask,
        }

    def parameter_count(self):
        counts = {
            "physics_s4d_frozen": sum(p.numel() for p in self.physics_encoder.parameters()),
            "instrument_s4d_frozen": sum(p.numel() for p in self.instrument_encoder.parameters()),
            "physics_head": sum(p.numel() for p in self.physics_head.parameters()),
            "instrument_head": sum(p.numel() for p in self.instrument_head.parameters()),
        }
        counts["trainable"] = counts["physics_head"] + counts["instrument_head"]
        return counts
