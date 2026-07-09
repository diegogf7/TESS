"""Fix for the raw-latent collapse in the first gap-blind run (2026-07-09).

Diagnosis: with gap placement hidden, nothing holds the encoder's RAW output
spread open. The loss LayerNorms its targets (scale erased), and the old
var_weight regularizer watches the PREDICTOR's output -- which chases
unit-scale LayerNormed targets, so it is satisfied for free. Raw latent std
fell to ~0.006 while the loss happily improved; gap-blind sector probe 0.399.

Fix, same architecture: expose the context encoder's raw tokens and put the
variance penalty on THEM, measured per token position ACROSS the batch --
the cross-curve spread, which is exactly what collapsed. The EMA target
encoder follows the online weights, so its spread recovers too. No modules,
weights, or shapes change; checkpoints stay loadable by the plain class.
"""

import os

import torch
import torch.nn.functional as Functional

from src.worked_folder.instrument.instrument_jepa import InstrumentJEPA


class GapBlindInstrumentJEPA(InstrumentJEPA):
    """Identical to InstrumentJEPA except forward ALSO returns the context
    encoder's raw tokens so the loss can regularize them. state_dict keys are
    unchanged, so gapblind checkpoints load into the plain class and the
    probe keeps working as-is."""

    def forward(self, context_flux, context_mask, target_flux, target_mask):
        context_tokens = self.context_encoder(context_flux.unsqueeze(-1), context_mask)
        prediction = self.predictor(context_tokens)

        with torch.no_grad():
            target = self.target_encoder(target_flux.unsqueeze(-1), target_mask)
            target = Functional.layer_norm(target, (target.shape[-1],))

        return prediction, target, context_tokens


def spread_penalty(tokens, gamma=1.0):
    """relu(-log(std / gamma)) per (token, feature), std taken ACROSS THE BATCH.

    Two deliberate choices, both measured (2026-07-09 mini-trains):
    - std(dim=0) is the cross-curve spread at each token position. The old
      regularizer flattened batch and token dims together, so position-to-
      position variation could mask a cross-curve collapse.
    - The hinge is on LOG-std, not std. The collapse is ~200x deep in scale;
      a linear hinge gives constant absolute pressure and recovered only
      x2 per 600 steps, while the log hinge gives constant MULTIPLICATIVE
      pressure and restored std to ~1.0 within 450 steps. Zero once
      std >= gamma, so healthy runs pay nothing.
    """
    std = tokens.std(dim=0)  # (N, D)
    return torch.relu(-torch.log(std / gamma + 1e-8)).mean()


def gapblind_loss(prediction, target, context_tokens, target_mask=None, var_weight=0.5):
    """Masked smooth-L1 (identical core to instrument_jepa_loss) + spread
    penalty on the RAW context tokens and on the prediction."""
    per_token = Functional.smooth_l1_loss(prediction, target, reduction="none").mean(dim=-1)  # (B, N)

    if target_mask is None:
        loss = per_token.mean()
    else:
        B, N = per_token.shape
        observed_fraction = target_mask.reshape(B, N, -1).mean(dim=2)  # (B, N), 1 = fully real token
        loss = (per_token * observed_fraction).sum() / observed_fraction.sum().clamp(min=1e-6)

    if var_weight > 0.0:
        loss = loss + var_weight * (spread_penalty(context_tokens) + spread_penalty(prediction))

    return loss


def build_gapblind_jepa():
    return GapBlindInstrumentJEPA(
        n_tokens=int(os.environ.get("JEPA_NTOKENS", "16")),
        token_dimension=int(os.environ.get("JEPA_TOKENDIM", "16")),
        d_model=256, n_layers=4, dropout=0.2, momentum=0.996,
        readout=os.environ.get("JEPA_READOUT", "mean"),
    )
