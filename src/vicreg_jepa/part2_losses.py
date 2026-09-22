"""Part 2 losses — ARCHITECTURE.md section 2.4 / 2.5.

All four are minimised. eps = 1e-4 inside square roots. Batch statistics are
taken over the batch dimension.

L_var, L_cov and L_cor_sys are applied to z_online, never to z_target: the
target encoder runs under no_grad and is updated only by EMA, so a regulariser
on its output produces no gradient and silently does nothing.
"""
import torch
import torch.nn.functional as F

EPS = 1e-4


def off_diagonal(m):
    n = m.shape[0]
    return m.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def l_inv(z_pred, z_target):
    """JEPA prediction loss. The target is a constant."""
    return F.mse_loss(z_pred, z_target.detach())


def l_var(z, gamma=1.0):
    """VICReg variance — keeps every latent dimension alive."""
    std = torch.sqrt(z.var(dim=0) + EPS)
    return torch.relu(gamma - std).mean()


def l_cov(z):
    """VICReg covariance — decorrelates dimensions within the latent."""
    n, d = z.shape
    zc = z - z.mean(dim=0)
    c = (zc.T @ zc) / (n - 1)
    return off_diagonal(c).pow(2).sum() / d


def l_cor_sys(z_online, z_sys):
    """Disentanglement against the frozen systematics latent.

    Standardise each dimension across the batch, then R = Zo^T Zs / B and take
    mean(R^2). Squared, not absolute: section 2.4.
    """
    b = z_online.shape[0]
    zo = (z_online - z_online.mean(0)) / torch.sqrt(z_online.var(0) + EPS)
    zs = (z_sys - z_sys.mean(0)) / torch.sqrt(z_sys.var(0) + EPS)
    r = (zo.T @ zs.detach()) / b                       # (64, 32)
    return r.pow(2).mean()


def total_loss(z_online, z_pred, z_target, z_sys,
               phi=1.0, lam=25.0, mu=1.0, nu=0.01, gamma=1.0):
    """Section 2.5. Returns (total, parts) with every term logged separately."""
    inv = l_inv(z_pred, z_target)
    cor = l_cor_sys(z_online, z_sys)
    var = l_var(z_online, gamma)
    cov = l_cov(z_online)
    total = phi * inv + lam * cor + mu * var + nu * cov
    return total, {"inv": inv.item(), "cor_sys": cor.item(),
                   "var": var.item(), "cov": cov.item(), "total": total.item()}
