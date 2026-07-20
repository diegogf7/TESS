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


class FixedTeacherInstrumentJEPA(nn.Module):
    def __init__(self, n_tokens=16, token_dim=16, d_model=256, n_layers=4,
                 readout="mean"):
        super().__init__()
        self.student = S4Model(d_input=1, d_model=d_model, n_layers=n_layers,
                               dropout=0.0, n_tokens=n_tokens,
                               token_dim=token_dim, readout=readout)
        self.teacher = S4Model(d_input=2, d_model=d_model, n_layers=n_layers,
                               dropout=0.0, n_tokens=n_tokens,
                               token_dim=token_dim, readout=readout)
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.teacher.eval()
        self.predictor = LatentPredictor(n_tokens=n_tokens,
                                         token_dimension=token_dim)

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
        prediction = self.predictor(student_tokens)
        with torch.no_grad():
            target = self.teacher(group_stats, valid_mask)
            target = F.layer_norm(target, (target.shape[-1],))
        return prediction, target, student_tokens

    @torch.no_grad()
    def encode(self, flux, observed_mask=None, view="online"):
        """Individual-star representation (the downstream interface)."""
        return self.student(flux.unsqueeze(-1), observed_mask)

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
    )
