"""Cadence masking for the anchor physics view.

Training uses random contiguous windows (isolated single-cadence holes are trivially
interpolated, so they would not stop the physics encoder from copying its input).
Inference uses four deterministic complementary masks whose hidden regions tile the
whole 1024-cadence curve, so every output cadence is predicted while hidden.
"""

import torch

CURVE_LENGTH = 1024


def contiguous_hidden_mask(valid_mask, hidden_fraction=0.25, n_windows=(3, 6),
                           generator=None):
    """Hide ~`hidden_fraction` of each row's VALID cadences with contiguous windows.

    valid_mask: [B, L] bool. Returns hidden_mask [B, L] bool, always a subset of
    valid_mask. Windows are placed over the full index range and then intersected
    with validity, so a window landing in a data gap hides fewer real cadences and
    the loop keeps placing windows until the per-row target is met.
    """
    B, L = valid_mask.shape
    device = valid_mask.device
    hidden = torch.zeros_like(valid_mask)
    n_valid = valid_mask.sum(dim=1)
    target = (hidden_fraction * n_valid.to(torch.float32)).round().to(torch.long)

    low, high = int(n_windows[0]), int(n_windows[1])
    counts = torch.randint(low, high + 1, (B,), device=device, generator=generator)
    for row in range(B):
        want = int(target[row].item())
        if want <= 0:
            continue
        k = int(counts[row].item())
        # Nominal window length; the placement loop tops up whatever validity eats.
        span = max(int(round(want / max(k, 1))), 1)
        for _ in range(4 * k):                       # bounded retries, never infinite
            if int((hidden[row] & valid_mask[row]).sum().item()) >= want:
                break
            length = int(torch.randint(max(span // 2, 1), max(span * 2, 2), (1,),
                                       device=device, generator=generator).item())
            length = min(length, L)
            start = int(torch.randint(0, L - length + 1, (1,), device=device,
                                      generator=generator).item())
            hidden[row, start:start + length] = True
    return hidden & valid_mask


def complementary_masks(length=CURVE_LENGTH, n_masks=4, block=64):
    """`n_masks` deterministic masks whose union covers every cadence exactly once.

    The curve is cut into contiguous blocks of `block` cadences dealt round-robin to
    the masks, so each mask hides contiguous windows (matching the training style)
    and the four together tile the full curve.
    """
    if length % block != 0:
        raise ValueError(f"block {block} does not divide length {length}")
    masks = torch.zeros(n_masks, length, dtype=torch.bool)
    for b in range(length // block):
        masks[b % n_masks, b * block:(b + 1) * block] = True
    assert bool((masks.sum(dim=0) == 1).all()), "complementary masks must tile the curve"
    return masks


def apply_mask(raw, hidden_mask):
    """Hidden cadences are replaced by normalized zero (the same value an unobserved
    cadence carries), so the encoder cannot tell a hidden cadence from a real gap."""
    return raw.masked_fill(hidden_mask, 0.0)


def mask_views(anchor_raw, anchor_valid_mask, hidden_fraction=0.25, generator=None,
               hidden_mask=None):
    """Contract of the mask generator: raw + validity -> masked input and both masks."""
    if hidden_mask is None:
        hidden_mask = contiguous_hidden_mask(anchor_valid_mask, hidden_fraction,
                                             generator=generator)
    hidden_mask = hidden_mask & anchor_valid_mask
    visible_mask = anchor_valid_mask & ~hidden_mask
    return apply_mask(anchor_raw, hidden_mask), hidden_mask, visible_mask
