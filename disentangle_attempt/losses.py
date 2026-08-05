"""Loss: masked reconstruction of the HIDDEN anchor cadences. That is the whole loss.

The cross-sector consistency term was removed after it measured as useless: swapping in
a wrong star's partner curve moved the metric by -0.0007 (noise), and it never reached
the decoder. One curve per star now -- the anchor, and its masked copy.

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


def total_loss(outputs, anchor_raw, anchor_valid_mask, beta=1.0):
    """Averages over all anchors in the step; the caller backwards this once."""
    loss_mask = outputs["hidden_mask"] & anchor_valid_mask
    reconstruction = masked_smooth_l1(outputs["predicted_raw_anchor"], anchor_raw,
                                      loss_mask, beta=beta)
    return reconstruction, {"reconstruction": float(reconstruction.detach()),
                            "total": float(reconstruction.detach()),
                            "n_loss_cadences": int(loss_mask.sum())}


@torch.no_grad()
def visible_reconstruction(outputs, anchor_raw, anchor_valid_mask, beta=1.0):
    """Diagnostic only: error on the cadences the physics encoder could see."""
    visible = anchor_valid_mask & ~outputs["hidden_mask"]
    return float(masked_smooth_l1(outputs["predicted_raw_anchor"], anchor_raw,
                                  visible, beta=beta))
