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

@torch.no_grad()

def _select_topk_peers(curves, masks, topk, min_overlap):

    N = curves.shape[0]
    corr, valid = _pairwise_corr(curves, masks, min_overlap)
    ii, jj = torch.triu_indices(N, N, offset = 1, device = curves.device)
    C = torch.full((N, N), float("-inf"), device = curves.device, dtype = curves.dtype)

    cv = torch.where(valid, corr, torch.full_like(corr, float("-inf")))

    C[ii, jj] = cv; C[jj, ii] = cv
    peers = []

    for i in range(N):
        pos = torch.where(C[i] > 0)[0]
        if pos.numel() == 0:
            peers.append(torch.empty(0, dtype=torch.long, device=curves.device))
        else:
            k = min(int(topk), int(pos.numel()))
            peers.append(pos[torch.topk(C[i][pos], k).indices])
    return peers


def topk_fixed_cov_loss(residuals, curves, masks, topk=8, min_overlap=64, eps=1e-6):
    """Per-curve top-K, fixed-denominator covariance loss.
      q_ij = Cov(r_i, r_j) / (sqrt(Var(x_i) Var(x_j)) + eps)
      L_i  = mean_{j in topK(i)} q_ij^2      L = mean over curves with >=1 peer
    Peer selection and the original-variance denominator are DETACHED, so lowering
    the loss requires shrinking residual COVARIANCE -- not inflating residual
    variance, and not flattening high-systematics curves (each curve is averaged
    separately, so a hard case can't hide inside 496 pairs)."""
    N, L = residuals.shape
    m = (masks > 0).to(residuals.dtype)
    # fixed per-curve ORIGINAL variance (detached), over each curve's own valid cadences
    x = curves.detach()
    xmean = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
    xc = (x - xmean[:, None]) * m
    xvar = (xc * xc).sum(1) / m.sum(1).clamp(min=1.0)                 # (N,) detached

    peers = _select_topk_peers(curves, masks, topk, min_overlap)     # detached
    ii_list, jj_list = [], []
    for i in range(N):
        p = peers[i]
        if p.numel():
            ii_list.append(torch.full((p.numel(),), i, dtype=torch.long, device=residuals.device))
            jj_list.append(p)
    if not ii_list:
        return residuals.sum() * 0.0                                 # no curve has a positive peer

    ii = torch.cat(ii_list); jj = torch.cat(jj_list)                 # (P,)
    ri, rj = residuals[ii], residuals[jj]                            # (P, L); GRADIENT flows here
    shared = m[ii] * m[jj]
    n = shared.sum(1).clamp(min=1.0)
    rim = (ri * shared).sum(1) / n; rjm = (rj * shared).sum(1) / n
    ric = (ri - rim[:, None]) * shared; rjc = (rj - rjm[:, None]) * shared
    cov = (ric * rjc).sum(1) / n                                     # Cov over shared valid cadences
    denom = torch.sqrt(xvar[ii] * xvar[jj]) + eps                    # FIXED original-scale denom (detached)
    q2 = (cov / denom) ** 2                                          # (P,)

    per_curve = torch.zeros(N, device=residuals.device).scatter_add(0, ii, q2)
    counts = torch.zeros(N, device=residuals.device).scatter_add(0, ii, torch.ones_like(q2))
    keep = counts > 0
    return (per_curve[keep] / counts[keep]).mean()


def relative_correction_size(corrections, curves, masks, eps=1e-6):
    """Per-curve L_size = mean(m*c^2) / (mean(m*(x-xbar)^2) + eps). Numerator uses the
    FULL correction mean-square (constant offsets penalized too); denominator is the
    CENTERED original variance. Averaged over curves."""
    m = (masks > 0).to(curves.dtype)
    num = (m * corrections ** 2).sum(1) / m.sum(1).clamp(min=1.0)
    xmean = (curves * m).sum(1) / m.sum(1).clamp(min=1.0)
    xc = (curves - xmean[:, None]) * m
    den = (xc * xc).sum(1) / m.sum(1).clamp(min=1.0) + eps
    return (num / den).mean()
