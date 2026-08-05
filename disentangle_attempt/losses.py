"""Loss: masked L1 between the predicted and the true anchor curve. Nothing else.

The prediction is scored on every VALID anchor cadence. Quality-filtered cadences are
gaps and are excluded; there is no artificial masking to exclude.
"""

import torch


def masked_l1(prediction, target, valid_mask):
    """(|pred - target| * mask).sum() / mask.sum().clamp_min(1)."""
    mask = valid_mask.to(prediction.dtype)
    return ((prediction - target).abs() * mask).sum() / mask.sum().clamp_min(1)


def total_loss(outputs, anchor_raw, anchor_valid_mask):
    """Averaged over the whole batch; the caller backwards this once."""
    reconstruction = masked_l1(outputs["predicted_raw_anchor"], anchor_raw,
                               anchor_valid_mask)
    return reconstruction, {"reconstruction": float(reconstruction.detach()),
                            "total": float(reconstruction.detach()),
                            "n_loss_cadences": int(anchor_valid_mask.sum())}
