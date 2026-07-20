# All this code is from Claude
"""Stage B model: single-star student distilled toward a FROZEN regional
teacher.

Student (online, trains):  one raw star -> 1-channel S4D -> MLP predictor
Teacher (frozen, Stage A): 8-star same-area median+logMAD -> 2-channel S4D

The teacher differs from area_commonmode_v1/v2 in the critical way: it is
trained FIRST (train_regional_group_teacher), selected on validation, then
loaded here with requires_grad=False, kept in eval mode permanently,
excluded from the optimizer, and NEVER EMA-updated. state_hash() lets every
epoch prove the teacher did not move.
"""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.instrument_v2.area_commonmode_jepa import masked_prediction_loss
from src.instrument_v2.regional_group_teacher import state_hash
from src.loss_function.gapblind_fix import spread_penalty
from src.models.s4d import S4Model
from src.worked_folder.physics.latent_jepa import LatentPredictor


class TransformerPredictor(nn.Module):
    """Drop-in predictor: 16x16 tokens -> project to 64 -> learned positional
    embeddings -> 2 pre-norm Transformer layers -> project back -> residual
    from the ORIGINAL student tokens. Input and output are both
    (batch, n_tokens, token_dim). Invalid tokens go into attention as
    src_key_padding_mask."""

    def __init__(self, n_tokens=16, token_dim=16, d_model=64, nhead=4,
                 dim_feedforward=256, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(token_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, token_dim)

    def forward(self, tokens, padding_mask=None):
        x = self.input_proj(tokens) + self.pos_embedding
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        return tokens + self.output_proj(x)


class FixedTeacherInstrumentJEPA(nn.Module):
    def __init__(self, n_tokens=16, token_dim=16, d_model=256, n_layers=4,
                 readout="mean", predictor_type="mlp"):
        super().__init__()
        if predictor_type not in ("mlp", "transformer"):
            raise ValueError(f"bad predictor_type {predictor_type!r}")
        self.predictor_type = predictor_type
        self.n_tokens = n_tokens
        self.student = S4Model(d_input=1, d_model=d_model, n_layers=n_layers,
                               dropout=0.0, n_tokens=n_tokens,
                               token_dim=token_dim, readout=readout)
        self.teacher = S4Model(d_input=2, d_model=d_model, n_layers=n_layers,
                               dropout=0.0, n_tokens=n_tokens,
                               token_dim=token_dim, readout=readout)
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.teacher.eval()
        if predictor_type == "transformer":
            self.predictor = TransformerPredictor(n_tokens=n_tokens,
                                                  token_dim=token_dim)
        else:
            self.predictor = LatentPredictor(n_tokens=n_tokens,
                                             token_dimension=token_dim)

    def _token_padding(self, observed_mask):
        """Cadence mask (B, L) -> attention padding mask (B, n_tokens),
        True = token has zero observed cadences (invalid)."""
        if observed_mask is None:
            return None
        batch = observed_mask.shape[0]
        fraction = observed_mask.reshape(batch, self.n_tokens, -1).mean(dim=2)
        return fraction == 0

    def _predict(self, tokens, observed_mask=None):
        if self.predictor_type == "transformer":
            return self.predictor(tokens, self._token_padding(observed_mask))
        return self.predictor(tokens)

    def train(self, mode=True):
        """Student/predictor follow `mode`; the frozen teacher NEVER leaves
        eval mode."""
        super().train(mode)
        self.teacher.eval()
        return self

    def trainable_parameters(self):
        """Optimizer input: student + predictor only, never the teacher."""
        yield from self.student.parameters()
        yield from self.predictor.parameters()

    def forward(self, star_flux, star_mask, group_stats, valid_mask):
        student_tokens = self.student(star_flux.unsqueeze(-1), star_mask)
        prediction = self._predict(student_tokens, star_mask)
        with torch.no_grad():
            target = self.teacher(group_stats, valid_mask)
            target = F.layer_norm(target, (target.shape[-1],))
        return prediction, target, student_tokens

    @torch.no_grad()
    def encode(self, flux, observed_mask=None, view="online"):
        """Individual-star representation. view='online' = encoder tokens;
        view='predicted' = predictor output tokens."""
        tokens = self.student(flux.unsqueeze(-1), observed_mask)
        if view == "predicted":
            return self._predict(tokens, observed_mask)
        return tokens

    def teacher_hash(self):
        return state_hash(self.teacher)


def fixed_teacher_loss(prediction, target, student_tokens, valid_mask,
                       var_weight=0.5):
    """Masked smooth-L1 to the frozen-teacher target + individual-token
    spread penalty. No covariance targets, no second latent."""
    loss = masked_prediction_loss(prediction, target, valid_mask)
    if var_weight > 0.0:
        loss = loss + var_weight * (spread_penalty(student_tokens)
                                    + spread_penalty(prediction))
    return loss


def load_frozen_teacher(model, teacher_selection_path):
    """Load the SELECTED Stage-A EMA encoder as the frozen teacher.

    Refuses anything that is not a Stage-A regional-teacher selection, so
    the failed area_commonmode_v1/v2 students can never sneak in."""
    with open(teacher_selection_path) as handle:
        selection = json.load(handle)
    if selection.get("checkpoint") is None:
        raise RuntimeError(f"{teacher_selection_path} has no checkpoint")
    if not str(selection.get("tag", "")).startswith("regteacher"):
        raise RuntimeError("refusing non-regional-teacher checkpoint as teacher")
    state = torch.load(selection["checkpoint"], map_location="cpu")
    prefix = "ema_encoder."
    teacher_state = {key[len(prefix):]: value for key, value in state.items()
                     if key.startswith(prefix)}
    if not teacher_state:
        raise RuntimeError(f"no {prefix} keys in {selection['checkpoint']}")
    model.teacher.load_state_dict(teacher_state, strict=True)
    for parameter in model.teacher.parameters():
        parameter.requires_grad = False
    model.teacher.eval()
    return selection


def load_student_warmstart(model, group_selection_path):
    """Student initialization = best existing K=8 group-JEPA ONLINE encoder."""
    with open(group_selection_path) as handle:
        selection = json.load(handle)
    state = torch.load(selection["checkpoint"], map_location="cpu")
    prefix = "context_encoder."
    encoder_state = {key[len(prefix):]: value for key, value in state.items()
                     if key.startswith(prefix)}
    if not encoder_state:
        raise RuntimeError(f"no {prefix} keys in {selection['checkpoint']}")
    model.student.load_state_dict(encoder_state, strict=True)
    return selection


def build_fixed_teacher_jepa():
    return FixedTeacherInstrumentJEPA(
        n_tokens=int(os.environ.get("JEPA_NTOKENS", "16")),
        token_dim=int(os.environ.get("JEPA_TOKENDIM", "16")),
        d_model=int(os.environ.get("JEPA_DMODEL", "256")),
        n_layers=int(os.environ.get("JEPA_NLAYERS", "4")),
        readout=os.environ.get("JEPA_READOUT", "mean"),
        predictor_type=os.environ.get("PREDICTOR", "mlp"),
    )
