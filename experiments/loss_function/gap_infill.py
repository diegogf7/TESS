"""Gap infill: hide WHERE the gaps are so the encoder can only use flux.

The zero-flux control showed the instrument encoder reads sector identity
entirely from gap placement (flux adds ~0.000). The masked loss removed the
training REWARD for gaps, but not their VISIBILITY: the S4Model's masked
pooling stamps gap layout into the latents deterministically, no matter how
the model is trained.

This module removes the channel at the input instead. Every unobserved point
is replaced by a value resampled (with replacement) from the SAME curve's
observed points, and the encoder is handed an all-ones mask. The curve keeps
its own noise level and amplitude, but gap locations are no longer
detectable, so any sector signal the encoder extracts must be flux-borne.

The observed mask should STILL be used to weight the loss (see
instrument_jepa_loss) so the fabricated regions earn no credit.
"""

import torch


def infill_gaps(flux, mask):
    """flux, mask: (B, L) float tensors. mask 1 = real data, 0 = gap placeholder.

    Returns (filled_flux, all_ones_mask). Observed points are untouched; gap
    points are replaced by draws from the same curve's observed points. A
    curve with no observed points at all is returned unchanged.
    """
    weights = mask.to(flux.dtype).clone()

    empty = weights.sum(dim=1) == 0
    if empty.any():
        weights[empty] = 1.0  # degenerate all-gap curve: resamples from itself

    L = flux.shape[1]
    donor_idx = torch.multinomial(weights, L, replacement=True)  # (B, L) indices of observed points
    donors = torch.gather(flux, 1, donor_idx)
    filled = torch.where(mask > 0, flux, donors)

    return filled, torch.ones_like(mask)
