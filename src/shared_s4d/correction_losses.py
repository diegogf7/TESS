from __future__ import annotations

import torch

def _pairwise_corr(curves, masks, min_overlap):

    N, _ = curves.shape
    m = (masks >0).to(curves.dtype)

    ii, jj = torch.triu_indices(N, N, offset =1, device = curves.device)
    a, b = curves[ii], curves[jj]

    shared = m[ii] * m[jj]
    n = shared.sum(1)
    denom = n.clamp(min = 1.0)

    am = (a * shared).sum(1) / denom
    bm = (b * shared).sum(1) / denom

    ac = (a - am[:, None]) * shared
    bc = (b - bm[:, None]) * shared
    sa = torch.sqrt((ac * ac).sum(1) + 1e-8)
    sb = torch.sqrt((bc * bc).sum(1) + 1e-8)

    corr = (ac * bc).sum(1) / (sa * sb)
    return corr, n >= min_overlap


def masked_pairwise_residual_correlation(residuals, masks, min_overlap = 64):

    corr, valid = _pairwise_corr(residuals, masks, min_overlap)
    corr2 = (corr ** 2)[valid]
    if corr2.numel() == 0:
        return residuals.sum() * 0.0
    
    return corr2.mean()


def normalized_correction_energy(corrections, curves, masks, eps=1e-6):

    m = (masks > 0).to(curves.dtype)

    num = (m * corrections ** 2).sum(1)
    den = (m * curves ** 2).sum(1) + eps

    return (num / den).mean()


@torch.no_grad()

def mean_abs_pairwise_corr(curves, masks, min_overlap = 64):

    corr, valid = _pairwise_corr(curves, masks, min_overlap)
    c = corr[valid].abs()

    return float(c.mean()) if c.numel() else float("nan")