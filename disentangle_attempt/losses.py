"""Losses: masked reconstruction of the HIDDEN anchor cadences + cross-sector
global-physics consistency. Nothing else -- no covariance, correction-energy, CBV,
correlation, adversarial, classification or flow-matching term.

Scoring only the hidden cadences is what stops the decoder from copying: the physics
encoder never saw the values it is asked to predict.
"""

import torch
import torch.nn.functional as F


def masked_smooth_l1(prediction, target, mask, beta=1.0):
    """Smooth-L1 averaged over `mask` only. Returns a zero-grad 0.0 for an empty mask."""
    mask = mask.to(prediction.dtype)
    denom = mask.sum()
    if float(denom) == 0.0:
        return prediction.sum() * 0.0
    elementwise = F.smooth_l1_loss(prediction, target, beta=beta, reduction="none")
    return (elementwise * mask).sum() / denom


def sector_consistency_loss(current_global_physics, other_sector_global_physics):
    """1 - cosine between the two L2-normalized global physics vectors.

    Only a GLOBAL quantity is compared: the two sectors' cadences are different
    absolute times, so nothing is aligned, interpolated or compared per cadence.
    """
    cosine = F.cosine_similarity(current_global_physics,
                                 other_sector_global_physics, dim=-1)
    return (1.0 - cosine).mean()


def total_loss(outputs, anchor_raw, anchor_valid_mask, physics_consistency_weight=0.05,
               beta=1.0):
    """Averages over all anchors in the step; the caller backwards this once."""
    loss_mask = outputs["hidden_mask"] & anchor_valid_mask
    reconstruction = masked_smooth_l1(outputs["predicted_raw_anchor"], anchor_raw,
                                      loss_mask, beta=beta)
    consistency = sector_consistency_loss(outputs["current_global_physics"],
                                          outputs["other_sector_global_physics"])
    total = reconstruction + physics_consistency_weight * consistency
    return total, {"reconstruction": float(reconstruction.detach()),
                   "sector_consistency": float(consistency.detach()),
                   "total": float(total.detach()),
                   "n_loss_cadences": int(loss_mask.sum())}


@torch.no_grad()
def visible_reconstruction(outputs, anchor_raw, anchor_valid_mask, beta=1.0):
    """Diagnostic only: error on the cadences the physics encoder could see."""
    visible = anchor_valid_mask & ~outputs["hidden_mask"]
    return float(masked_smooth_l1(outputs["predicted_raw_anchor"], anchor_raw,
                                  visible, beta=beta))
