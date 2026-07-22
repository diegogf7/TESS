# All this code is from Claude
"""Supervised contrastive loss (Khosla et al. 2020) over chip labels.

Embeddings are the flattened 16x16 context tokens (256-D), L2-normalized.
Same camera x CCD label = positive, different = negative, self excluded.
Anchors with no positive in the batch are ignored (never NaN). The two
different same-chip stars of a pair enter as two views: concatenate both
stars' embeddings and duplicate the chip labels before calling.
"""

import torch
import torch.nn.functional as F


def supcon_loss(embeddings, labels, temperature=0.1):
    """embeddings (N, ...) flattened to (N, D); labels (N,) int. Returns scalar.

    loss_i = -1/|P(i)| * sum_{p in P(i)} log( exp(z_i.z_p/T) / sum_{a != i} exp(z_i.z_a/T) )
    averaged over anchors that have at least one positive.
    """
    z = embeddings.reshape(embeddings.shape[0], -1)
    z = F.normalize(z, dim=1)
    labels = labels.reshape(-1)
    n = z.shape[0]
    if n < 2:
        raise ValueError("supcon_loss needs at least 2 embeddings")

    sim = z @ z.T / temperature                             # (N, N)
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()  # numerical stability

    exp_sim = torch.exp(sim).masked_fill(self_mask, 0.0)
    log_denominator = torch.log(exp_sim.sum(dim=1).clamp(min=1e-12))
    log_prob = sim - log_denominator[:, None]               # log p(j | i)

    positive = (labels[:, None] == labels[None, :]) & ~self_mask
    n_pos = positive.sum(dim=1)
    has_pos = n_pos > 0
    if not has_pos.any():
        # no anchor has a positive -- defined as zero, not NaN
        return z.sum() * 0.0

    per_anchor = -(log_prob * positive).sum(dim=1)[has_pos] / n_pos[has_pos]
    loss = per_anchor.mean()
    assert torch.isfinite(loss), "supcon_loss is not finite"
    return loss
