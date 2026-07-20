# All this code is from Claude
"""Stage A: regional group teacher -- an S4D that reads GROUP fingerprints.

The v2 raw diagnostic proved area-specific common-mode signal exists in the
group median / log-MAD curves (val same-cross r = +0.117 [+0.101, +0.133]).
This teacher consumes exactly that representation: TWO input channels
(median, log-MAD) of an 8-star same-area group on the shared Sector-14 grid.

Teacher pretraining is a group-to-group JEPA: two DISJOINT same-area groups,
online encoder + predictor on group A, EMA target on group B, existing
masked smooth-L1 + spread penalty. After selection the EMA encoder is
FROZEN and handed to Stage B; it never trains again.
"""

from __future__ import annotations

import hashlib
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.instrument_v2.area_commonmode_dataset import (
    Sector14GroupStatDataset,
    area_to_chip,
)
from src.models.s4d import S4Model
from src.worked_folder.physics.latent_jepa import LatentPredictor


def state_hash(module):
    """Deterministic sha256 over a module's state dict (order-stable)."""
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class AreaGroupPairDataset(Dataset):
    """One item = two disjoint K-star groups from one uniformly drawn area,
    each reduced to its (median, log-MAD) 2-channel fingerprint."""

    def __init__(self, base: Sector14GroupStatDataset, items_per_epoch=None):
        if base.grouping != "area":
            raise ValueError("teacher pretraining is area-grouped by design")
        self.base = base
        eligible = [g for g in base.group_list
                    if len(base.group_rows[g]) >= 2 * base.k]
        if not eligible:
            raise RuntimeError(f"no area has >= {2 * base.k} stars")
        self.eligible = eligible
        self.items_per_epoch = (items_per_epoch or
                                max(1, math.ceil(len(base.tics) / base.k)))

    def __len__(self):
        return self.items_per_epoch

    @staticmethod
    def stack_channels(median, log_mad):
        return torch.stack([median, log_mad], dim=-1)   # (L, 2)

    def group_input(self, rows):
        median, log_mad, valid = self.base.group_target_tensors(rows)
        return self.stack_channels(median, log_mad), valid

    def __getitem__(self, idx):
        area = self.eligible[np.random.randint(len(self.eligible))]
        rows = np.random.choice(self.base.group_rows[area],
                                size=2 * self.base.k, replace=False)
        stats_a, valid_a = self.group_input(rows[:self.base.k])
        stats_b, valid_b = self.group_input(rows[self.base.k:])
        return (stats_a, valid_a, stats_b, valid_b,
                torch.tensor(int(area), dtype=torch.int64),
                torch.tensor(area_to_chip(area), dtype=torch.int64))


class RegionalGroupTeacher(nn.Module):
    """Two-channel S4D over group fingerprints; standard online/EMA JEPA."""

    def __init__(self, n_tokens=16, token_dim=16, d_model=256, n_layers=4,
                 momentum=0.996, readout="mean"):
        super().__init__()
        self.momentum = float(momentum)
        encoder_kwargs = dict(d_input=2, d_model=d_model, n_layers=n_layers,
                              dropout=0.0, n_tokens=n_tokens,
                              token_dim=token_dim, readout=readout)
        self.online_encoder = S4Model(**encoder_kwargs)
        self.ema_encoder = S4Model(**encoder_kwargs)
        self.ema_encoder.load_state_dict(self.online_encoder.state_dict())
        for parameter in self.ema_encoder.parameters():
            parameter.requires_grad = False
        self.predictor = LatentPredictor(n_tokens=n_tokens,
                                         token_dimension=token_dim)

    def forward(self, stats_a, valid_a, stats_b, valid_b):
        context_tokens = self.online_encoder(stats_a, valid_a)
        prediction = self.predictor(context_tokens)
        with torch.no_grad():
            target = self.ema_encoder(stats_b, valid_b)
            target = F.layer_norm(target, (target.shape[-1],))
        return prediction, target, context_tokens

    @torch.no_grad()
    def encode(self, stats, valid, view="ema"):
        encoder = self.ema_encoder if view == "ema" else self.online_encoder
        return encoder(stats, valid)

    @torch.no_grad()
    def update_target(self):
        momentum = self.momentum
        for online, target in zip(self.online_encoder.parameters(),
                                  self.ema_encoder.parameters()):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)


def build_regional_teacher():
    return RegionalGroupTeacher(
        n_tokens=int(os.environ.get("JEPA_NTOKENS", "16")),
        token_dim=int(os.environ.get("JEPA_TOKENDIM", "16")),
        d_model=int(os.environ.get("JEPA_DMODEL", "256")),
        n_layers=int(os.environ.get("JEPA_NLAYERS", "4")),
        momentum=float(os.environ.get("JEPA_MOMENTUM", "0.996")),
        readout=os.environ.get("JEPA_READOUT", "mean"),
    )
