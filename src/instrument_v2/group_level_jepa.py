"""Group-mean instrument JEPA with an otherwise unchanged S4D backbone.

Each curve is encoded independently.  JEPA prediction happens between the
mean latent of two disjoint star sets from the same camera/CCD.  Consequently
the individual S4D encoder remains directly usable for frozen probing and
matched fine-tuning; no group information is required at downstream time.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.s4d import S4Model
from src.worked_folder.physics.latent_jepa import LatentPredictor


class GroupMeanInstrumentJEPA(nn.Module):
    def __init__(
        self,
        n_tokens=16,
        token_dim=16,
        d_model=256,
        n_layers=4,
        dropout=0.0,
        momentum=0.996,
        readout="mean",
    ):
        super().__init__()
        self.momentum = float(momentum)
        encoder_kwargs = dict(
            d_input=1,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            n_tokens=n_tokens,
            token_dim=token_dim,
            readout=readout,
        )
        self.context_encoder = S4Model(**encoder_kwargs)
        self.target_encoder = S4Model(**encoder_kwargs)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False
        self.predictor = LatentPredictor(
            n_tokens=n_tokens, token_dimension=token_dim
        )

    @staticmethod
    def _encode_set(encoder, flux, observed_mask):
        if flux.ndim != 3:
            raise ValueError(f"expected flux (batch,set,time), got {tuple(flux.shape)}")
        batch, group_size, length = flux.shape
        flat_flux = flux.reshape(batch * group_size, length)
        flat_mask = observed_mask.reshape(batch * group_size, length)
        per_star = encoder(flat_flux.unsqueeze(-1), flat_mask)
        per_star = per_star.reshape(
            batch, group_size, per_star.shape[-2], per_star.shape[-1]
        )
        # Every sampled curve is real.  Masks govern cadence pooling inside S4D;
        # the group operation itself is the simple arithmetic mean being tested.
        return per_star.mean(dim=1), per_star

    def forward(self, context_flux, context_mask, target_flux, target_mask):
        context_group, context_per_star = self._encode_set(
            self.context_encoder, context_flux, context_mask
        )
        prediction = self.predictor(context_group)
        with torch.no_grad():
            target_group, _ = self._encode_set(
                self.target_encoder, target_flux, target_mask
            )
            target_group = F.layer_norm(target_group, (target_group.shape[-1],))
        return prediction, target_group, context_group, context_per_star

    @torch.no_grad()
    def encode(self, flux, observed_mask=None, view="online"):
        """Encode individual curves for a matched downstream comparison."""
        if view not in {"online", "ema"}:
            raise ValueError("view must be 'online' or 'ema'")
        encoder = self.context_encoder if view == "online" else self.target_encoder
        return encoder(flux.unsqueeze(-1), observed_mask)

    @torch.no_grad()
    def encode_group(self, flux, observed_mask, view="online"):
        encoder = self.context_encoder if view == "online" else self.target_encoder
        group, _ = self._encode_set(encoder, flux, observed_mask)
        return group

    @torch.no_grad()
    def update_target(self):
        momentum = self.momentum
        for online, target in zip(
            self.context_encoder.parameters(), self.target_encoder.parameters()
        ):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)


def build_groupmean_jepa():
    return GroupMeanInstrumentJEPA(
        n_tokens=int(os.environ.get("JEPA_NTOKENS", "16")),
        token_dim=int(os.environ.get("JEPA_TOKENDIM", "16")),
        d_model=int(os.environ.get("JEPA_DMODEL", "256")),
        n_layers=int(os.environ.get("JEPA_NLAYERS", "4")),
        dropout=0.0,
        momentum=float(os.environ.get("JEPA_MOMENTUM", "0.996")),
        readout=os.environ.get("JEPA_READOUT", "mean"),
    )
