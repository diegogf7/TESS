import torch
import torch.nn.functional as F


def masked_mse(pred, target, mask):
    """Mean squared error over observed cadences only."""
    d = ((pred - target) ** 2) * mask
    return d.sum() / mask.sum().clamp(min=1.0)


def masked_smooth_l1(pred, target, mask, beta=1.0):
    """Huber / smooth-L1 over observed cadences only.

    Normalised TGLC curves carry extreme outliers that a squared error cannot
    cope with: measured on the Sector-1 cache, the worst 0.01% of cadences hold
    90.7% of the total squared flux, and peak |flux| reaches 2842 MAD. Under MSE
    the gradient is almost entirely those few points, so the encoder never
    learns the common mode. Beyond `beta` this loss grows linearly, so a 2842-MAD
    spike contributes ~2842 rather than ~8e6. Matches the convention already used
    in disentangle_attempt/losses.py.
    """
    d = torch.abs(pred - target)
    quad = torch.where(d < beta, 0.5 * d * d / beta, d - 0.5 * beta)
    return (quad * mask).sum() / mask.sum().clamp(min=1.0)


def leave_one_out_mean(z):
    """(B, G, D) -> (B, G, D): for each curve, the mean of its G-1 peers."""
    G = z.shape[1]
    return (z.sum(dim=1, keepdim=True) - z) / (G - 1)


def common_mode_loss(decoder, z, flux, observed, robust=True, beta=1.0):
    """Part 1. Predict each curve from its peers' latents, so z_i is trained to
    carry exactly the part of curve i that its neighbours share: systematics.

    `robust=True` uses smooth-L1. Plain MSE is left available for comparison but
    is not a sane default on these curves -- see masked_smooth_l1.
    """
    B, G, D = z.shape
    g = leave_one_out_mean(z).reshape(B * G, D)
    pred = decoder(g).reshape(B, G, -1)
    if robust:
        return masked_smooth_l1(pred, flux, observed, beta)
    return masked_mse(pred, flux, observed)


def zero_baseline_loss(flux, observed, robust=True, beta=1.0):
    """The score for predicting nothing. Part 1 must beat THIS, not just the
    region median -- on real curves the region median is barely better than zero.
    """
    zero = torch.zeros_like(flux)
    if robust:
        return masked_smooth_l1(zero, flux, observed, beta)
    return masked_mse(zero, flux, observed)


def variance_loss(z, gamma=1.0):
    """Instruction 2.6. Hinge on per-dimension std across the batch."""
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return torch.relu(gamma - std).mean()


def covariance_loss(z):
    """Instruction 2.7. Off-diagonal covariance, normalised by dim (not dim^2),
    matching the stated (1/D) * sum_{i!=j} formula."""
    B, D = z.shape
    zc = z - z.mean(dim=0)
    cov = (zc.T @ zc) / (B - 1)
    off = cov - torch.diag(torch.diag(cov))
    return off.pow(2).sum() / D


def cross_correlation_loss(z_physics, z_systematics):
    """Instruction 2.5. Mean |Pearson r| over the physics x systematics matrix."""
    zp = z_physics - z_physics.mean(dim=0)
    zs = z_systematics - z_systematics.mean(dim=0)
    zp = zp / (zp.norm(dim=0, keepdim=True) + 1e-8)
    zs = zs / (zs.norm(dim=0, keepdim=True) + 1e-8)
    return (zp.T @ zs).abs().mean()


def invariance_loss(z_pred, z_unmasked):
    """Instruction 2.4. Target is detached -- it comes from the EMA encoder."""
    return F.mse_loss(z_pred, z_unmasked.detach())


def total_loss(z_masked, z_unmasked, z_systematics, z_pred, cfg):
    """Instruction 2.8.

    The three regularisers act on z_masked, the ONLINE encoder output. On
    z_unmasked they would carry no gradient (EMA branch); on z_pred the
    predictor could satisfy them alone while the encoder collapsed.
    """
    l_inv = invariance_loss(z_pred, z_unmasked)
    l_cor = cross_correlation_loss(z_masked, z_systematics)
    l_var = variance_loss(z_masked, cfg.gamma)
    l_cov = covariance_loss(z_masked)
    total = cfg.phi * l_inv + cfg.lam * l_cor + cfg.mu * l_var + cfg.nu * l_cov
    return total, {"inv": l_inv.item(), "cor_sys": l_cor.item(),
                   "var": l_var.item(), "cov": l_cov.item(), "total": total.item()}
