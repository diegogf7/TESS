from __future__ import annotations

import math

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


def _window_var(curves, masks, s, e):
    """Masked per-curve variance over window [s,e] -> (N,)."""
    xw = curves[:, s:e]; mw = (masks[:, s:e] > 0).to(curves.dtype)
    n = mw.sum(1).clamp(min=1.0)
    xm = (xw * mw).sum(1) / n
    return (((xw - xm[:, None]) * mw) ** 2).sum(1) / n


def _window_pair_cov(a, masks, s, e, ii, jj, overlap, norm_var=None):
    """Signed masked covariance per pair over window [s,e]. If norm_var (N,) given,
    normalize each pair by sqrt(var_i var_j)+eps. Returns (cov (P,), valid (P,))."""
    aw = a[:, s:e]; mw = (masks[:, s:e] > 0).to(a.dtype)
    ai, aj = aw[ii], aw[jj]; sh = mw[ii] * mw[jj]
    n = sh.sum(1); valid = n >= (overlap * (e - s))          # >=50% window overlap
    d = n.clamp(min=1.0)
    am = (ai * sh).sum(1) / d; bm = (aj * sh).sum(1) / d
    cov = (((ai - am[:, None]) * sh) * ((aj - bm[:, None]) * sh)).sum(1) / d
    if norm_var is not None:
        cov = cov / (torch.sqrt(norm_var[ii] * norm_var[jj]) + 1e-6)
    return cov, valid


def windowed_group_cov_loss(residuals, curves, masks, scales=(64, 128), overlap=0.5,
                            top_frac=0.25, group_frac=0.75, eps=1e-6):
    """Multi-scale, consensus-selected group-covariance loss over ALL curves in a
    group. Per scale: split into 50%-overlap windows; a window is usable only if at
    least ceil(group_frac*N) curves are >=50% observed within it AND at least
    C(ceil(group_frac*N),2) of the pairs formed from THOSE curves overlap -- so the
    validity requirement scales with group size instead of a fixed min_pairs. Rank
    usable windows by average SIGNED pairwise covariance of the ORIGINALS (over the
    sufficiently-valid pairs, so independent stellar behaviour averages out), keep
    the top 25% (detached), and on those minimize the SQUARED normalized group
    covariance of the CLEANED residuals (denominators = detached original window
    variances)."""
    N, L = residuals.shape
    ii, jj = torch.triu_indices(N, N, offset=1, device=residuals.device)
    min_curves = math.ceil(group_frac * N)
    min_pairs = math.comb(min_curves, 2)
    losses = []
    for W in scales:
        stride = max(1, int(W * overlap))
        starts = list(range(0, L - W + 1, stride))
        cons, keep = [], []
        with torch.no_grad():                                # selection detached
            for s in starts:
                mw = (masks[:, s:s + W] > 0).to(curves.dtype)
                curve_ok = mw.mean(1) >= 0.5                  # each curve >=50% of window observed
                if int(curve_ok.sum()) < min_curves:         # need enough sufficiently-valid curves
                    cons.append(float("-inf")); keep.append(None); continue
                pair_ok = curve_ok[ii] & curve_ok[jj]        # pairs only from those curves
                cov0, ov = _window_pair_cov(curves, masks, s, s + W, ii, jj, overlap)
                valid0 = ov & pair_ok
                if int(valid0.sum()) < min_pairs:            # scaled pair floor
                    cons.append(float("-inf")); keep.append(None); continue
                cons.append(float(cov0[valid0].mean())); keep.append((s, s + W, valid0))
        vw = [w for w in range(len(starts)) if keep[w] is not None]
        if not vw:
            continue
        k = max(1, int(round(top_frac * len(vw))))
        for w in sorted(vw, key=lambda w: -cons[w])[:k]:     # highest-consensus 25%
            s, e, valid0 = keep[w]
            var = _window_var(curves.detach(), masks, s, e)  # fixed detached denom
            covr, _ = _window_pair_cov(residuals, masks, s, e, ii, jj, overlap, norm_var=var)
            group_cov = covr[valid0].mean()                  # signed -> stellar averages out
            losses.append(group_cov ** 2)                    # minimize squared cleaned group cov
    if not losses:
        return residuals.sum() * 0.0
    return torch.stack(losses).mean()


def soft_cap_size(corrections, curves, masks, cap=0.5, eps=1e-6):
    """ratio_i = masked_mean(c_i^2) / (masked_var(x_i)+eps); size = mean(relu(ratio-0.5)^2).
    A soft cap: corrections up to half the original variance are free, beyond that penalized."""
    m = (masks > 0).to(curves.dtype)
    num = (m * corrections ** 2).sum(1) / m.sum(1).clamp(min=1.0)
    xmean = (curves * m).sum(1) / m.sum(1).clamp(min=1.0)
    var = (((curves - xmean[:, None]) * m) ** 2).sum(1) / m.sum(1).clamp(min=1.0) + eps
    return (torch.relu(num / var - cap) ** 2).mean()
