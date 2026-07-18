# All this code is from Claude
"""Area/chip common-mode JEPA: predict group median (and robust log-MAD)
curves' latents from ONE individual star.

Architecture is the unchanged S4D + LatentPredictor stack. Differences from
the group-mean JEPA:
  - the EMA teacher encodes the group median curve (and, in the median_mad
    arm, the log-MAD curve, through the SAME one-channel teacher);
  - two fresh predictor heads (median, MAD) sit on the online tokens;
  - anti-collapse spread penalty applies to the INDIVIDUAL context-star
    tokens (and the median prediction), never only to a group average;
  - the online encoder warm-starts from the selected K=8 group-JEPA online
    encoder; the EMA teacher starts as an EXACT copy of that online encoder.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss_function.gapblind_fix import spread_penalty
from src.models.s4d import S4Model
from src.worked_folder.physics.latent_jepa import LatentPredictor


class AreaCommonModeJEPA(nn.Module):
    def __init__(self, n_tokens=16, token_dim=16, d_model=256, n_layers=4,
                 dropout=0.0, momentum=0.996, readout="mean"):
        super().__init__()
        if dropout != 0.0:
            raise ValueError("dropout must stay 0 (spread-penalty cheat channel)")
        self.momentum = float(momentum)
        encoder_kwargs = dict(d_input=1, d_model=d_model, n_layers=n_layers,
                              dropout=dropout, n_tokens=n_tokens,
                              token_dim=token_dim, readout=readout)
        self.context_encoder = S4Model(**encoder_kwargs)
        self.target_encoder = S4Model(**encoder_kwargs)
        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad = False
        self.predictor_median = LatentPredictor(n_tokens=n_tokens,
                                                token_dimension=token_dim)
        self.predictor_mad = LatentPredictor(n_tokens=n_tokens,
                                             token_dimension=token_dim)

    def _teach(self, curve, valid_mask):
        with torch.no_grad():
            tokens = self.target_encoder(curve.unsqueeze(-1), valid_mask)
            return F.layer_norm(tokens, (tokens.shape[-1],))

    def forward(self, context_flux, context_mask, median_curve, logmad_curve,
                valid_mask, target="median_mad"):
        context_tokens = self.context_encoder(
            context_flux.unsqueeze(-1), context_mask)
        pred_median = self.predictor_median(context_tokens)
        target_median = self._teach(median_curve, valid_mask)
        if target == "median":
            return pred_median, target_median, None, None, context_tokens
        pred_mad = self.predictor_mad(context_tokens)
        target_mad = self._teach(logmad_curve, valid_mask)
        return pred_median, target_median, pred_mad, target_mad, context_tokens

    @torch.no_grad()
    def encode(self, flux, observed_mask=None, view="online"):
        """Individual-star representation used by every downstream consumer."""
        if view not in {"online", "ema"}:
            raise ValueError("view must be 'online' or 'ema'")
        encoder = self.context_encoder if view == "online" else self.target_encoder
        return encoder(flux.unsqueeze(-1), observed_mask)

    @torch.no_grad()
    def update_target(self):
        momentum = self.momentum
        for online, target in zip(self.context_encoder.parameters(),
                                  self.target_encoder.parameters()):
            target.data.mul_(momentum).add_(online.data, alpha=1.0 - momentum)


def masked_prediction_loss(prediction, target, valid_mask):
    """Masked smooth-L1 (same core as gapblind_loss) weighted by the fraction
    of VALID cadences under each token."""
    per_token = F.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)
    batch, n_tokens = per_token.shape
    weight = valid_mask.reshape(batch, n_tokens, -1).mean(dim=2)
    return (per_token * weight).sum() / weight.sum().clamp(min=1e-6)


def covariance_penalty(context_tokens):
    """VICReg-style off-diagonal covariance penalty on the INDIVIDUAL context
    representation. The spread penalty gives every dimension variance but
    lets dimensions be redundant copies; this decorrelates them (the v1
    screen collapsed at effective rank ~8-10). Representation decorrelation
    only -- no covariance targets, teacher untouched."""
    z = context_tokens.flatten(1)
    z = z - z.mean(dim=0)
    cov = z.T @ z / max(z.shape[0] - 1, 1)
    offdiag = cov - torch.diag(torch.diag(cov))
    return offdiag.pow(2).sum() / z.shape[1]


def commonmode_loss(pred_median, target_median, pred_mad, target_mad,
                    context_tokens, valid_mask, mad_weight=0.25,
                    var_weight=0.5, cov_weight=0.0):
    """median loss + MAD_WEIGHT * mad loss + spread penalty on the INDIVIDUAL
    context tokens and the median prediction + COV_WEIGHT * off-diagonal
    covariance penalty. cov_weight=0 reproduces the v1 loss exactly.
    Returns (total, parts)."""
    median_loss = masked_prediction_loss(pred_median, target_median, valid_mask)
    mad_loss = (masked_prediction_loss(pred_mad, target_mad, valid_mask)
                if pred_mad is not None else torch.zeros_like(median_loss))
    var_loss = spread_penalty(context_tokens) + spread_penalty(pred_median)
    cov_loss = covariance_penalty(context_tokens)
    total = median_loss + mad_weight * mad_loss
    if var_weight > 0.0:
        total = total + var_weight * var_loss
    if cov_weight > 0.0:
        total = total + cov_weight * cov_loss
    return total, {"median_loss": float(median_loss.detach()),
                   "mad_loss": float(mad_loss.detach()),
                   "var_loss": float(var_loss.detach()),
                   "cov_loss": float(cov_loss.detach())}


def load_group_jepa_warmstart(model, selection_path):
    """Warm-start from the selected group-JEPA: online encoder copied into
    BOTH the online context encoder and the EMA teacher (exact copy);
    predictor heads stay freshly initialized."""
    with open(selection_path) as handle:
        selection = json.load(handle)
    state = torch.load(selection["checkpoint"], map_location="cpu")
    prefix = "context_encoder."
    encoder_state = {key[len(prefix):]: value for key, value in state.items()
                     if key.startswith(prefix)}
    if not encoder_state:
        raise RuntimeError(f"no {prefix} keys in {selection['checkpoint']}")
    model.context_encoder.load_state_dict(encoder_state, strict=True)
    model.target_encoder.load_state_dict(encoder_state, strict=True)
    return selection


def build_area_commonmode_jepa():
    return AreaCommonModeJEPA(
        n_tokens=int(os.environ.get("JEPA_NTOKENS", "16")),
        token_dim=int(os.environ.get("JEPA_TOKENDIM", "16")),
        d_model=int(os.environ.get("JEPA_DMODEL", "256")),
        n_layers=int(os.environ.get("JEPA_NLAYERS", "4")),
        dropout=0.0,
        momentum=float(os.environ.get("JEPA_MOMENTUM", "0.996")),
        readout=os.environ.get("JEPA_READOUT", "mean"),
    )
