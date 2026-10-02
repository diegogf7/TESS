"""Part 1 with the correlation objective from ARCHITECTURE.md section 1.

    L_part1 = alpha * L_sys_corr + mu1 * L_var(Z) + nu1 * L_cov(Z)
    L_sys_corr = 1 - mean(off_diagonal(R))

R is the Pearson correlation matrix BETWEEN THE 32 CURVES of a group, computed
across the latent dimension. Minimising L_sys_corr maximises agreement between
same-region curves, which is what isolates the instrument. L_var and L_cov are
the VICReg terms on the pooled latents -- without them the trivial solution is
to map every curve to one constant vector, which has perfect correlation.

The objective has no decoder, so it yields a latent but no subtractable
template. `fit_decoder` trains one afterwards on the FROZEN encoder, mapping a
group's leave-one-out peer mean to the held-out curve, so this model can be
compared against the reconstruction-trained Part 1 on the same footing.
"""
import argparse, json, os, time
import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Part1Config
from .data import LocalGroupDataset
from .losses import leave_one_out_mean, masked_smooth_l1
from .models import S4Encoder, CommonModeDecoder
from .real_data import RealCurveSource, git_sha

DEVICE = os.environ.get("TESS_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-4


def pairwise_corr(Z):
    """Z: (B, G, D) -> mean off-diagonal Pearson r between the G curves."""
    Zc = Z - Z.mean(dim=2, keepdim=True)                 # centre each latent vector
    Zn = Zc / torch.sqrt((Zc * Zc).sum(dim=2, keepdim=True) + EPS)
    R = torch.bmm(Zn, Zn.transpose(1, 2))                # (B, G, G)
    G = R.shape[1]
    off = ~torch.eye(G, dtype=torch.bool, device=R.device)
    return R[:, off].mean()


def l_sys_corr(Z):
    return 1.0 - pairwise_corr(Z)


def l_var(Z, gamma=1.0):
    z = Z.reshape(-1, Z.shape[-1])
    return torch.relu(gamma - torch.sqrt(z.var(dim=0) + EPS)).mean()


def l_inv_vicreg(Z):
    """VICReg invariance: squared distance of each latent from its group mean.
    Equal, up to a constant, to the mean over all same-group pairs:
        mean_{i!=j} ||z_i - z_j||^2 = (2/(G-1)) * sum_i ||z_i - zbar||^2
    Unlike Pearson this is scale-sensitive -- it forces the latents to be equal,
    magnitude included, not merely parallel.
    """
    return ((Z - Z.mean(dim=1, keepdim=True)) ** 2).sum(-1).mean()


def l_cov(Z):
    z = Z.reshape(-1, Z.shape[-1])
    n, d = z.shape
    zc = z - z.mean(dim=0)
    C = (zc.T @ zc) / (n - 1)
    off = C - torch.diag(torch.diag(C))
    return off.pow(2).sum() / d


def train_encoder(src, cfg, steps, alpha, mu1, nu1, out, seed=0,
                  objective="correlation"):
    torch.manual_seed(seed); np.random.seed(seed)
    os.makedirs(out, exist_ok=True)
    enc = S4Encoder(cfg.latent_dim, cfg.n_tokens, cfg.d_model,
                    cfg.d_state, cfg.n_layers, 0.0).to(DEVICE)
    # the reconstruction objective needs a decoder during encoder training; it is
    # thrown away afterwards so every arm gets the SAME fresh frozen-decoder fit
    aux = CommonModeDecoder(cfg.latent_dim, cfg.seq_len).to(DEVICE) \
          if objective == "reconstruction" else None
    params = list(enc.parameters()) + (list(aux.parameters()) if aux else [])
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    ds = LocalGroupDataset(src, cfg.group_size, 4000, seed, cfg.group_radius_deg)
    print(f"[corr] groups: {ds.stats()}", flush=True)
    dl = DataLoader(ds, batch_size=cfg.batch_groups, num_workers=0, drop_last=True)
    step = 0
    while step < steps:
        for flux, obs in dl:
            flux, obs = flux.to(DEVICE), obs.to(DEVICE)
            B, G, L = flux.shape
            Z = enc(flux.reshape(B*G, L), obs.reshape(B*G, L)).reshape(B, G, -1)
            lv, lcov = l_var(Z, cfg.var_gamma), l_cov(Z)
            if objective == "vicreg":
                lc = l_inv_vicreg(Z)
            elif objective == "reconstruction":
                gmean = leave_one_out_mean(Z).reshape(B*G, -1)
                lc = masked_smooth_l1(aux(gmean).reshape(B, G, -1), flux, obs)
            else:
                lc = l_sys_corr(Z)
            loss = alpha*lc + mu1*lv + nu1*lcov
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            if step % 200 == 0:
                tag = {"vicreg":"vic ","reconstruction":"recon"}.get(objective,"corr")
                print(f"[{tag}] {step:5d} total={float(loss):.4f} "
                      f"corr_r={float(pairwise_corr(Z)):.4f} "
                      f"L_inv={float(lc):.4f} L_var={float(lv):.4f} L_cov={float(lcov):.4f}",
                      flush=True)
            step += 1
            if step >= steps: break
    torch.save({"encoder": enc.state_dict(), "cfg": cfg.__dict__, "git_sha": git_sha(),
                "objective": objective, "alpha": alpha, "mu1": mu1, "nu1": nu1},
               os.path.join(out, "corr_encoder.pt"))
    return enc


def fit_decoder(enc, src, cfg, steps, out, seed=0):
    """Decoder on the FROZEN correlation encoder, so templates are comparable."""
    enc.eval()
    for p in enc.parameters(): p.requires_grad_(False)
    dec = CommonModeDecoder(cfg.latent_dim, cfg.seq_len).to(DEVICE)
    opt = torch.optim.AdamW(dec.parameters(), lr=1e-3)
    ds = LocalGroupDataset(src, cfg.group_size, 4000, seed+5, cfg.group_radius_deg)
    dl = DataLoader(ds, batch_size=cfg.batch_groups, num_workers=0, drop_last=True)
    step = 0
    while step < steps:
        for flux, obs in dl:
            flux, obs = flux.to(DEVICE), obs.to(DEVICE)
            B, G, L = flux.shape
            with torch.no_grad():
                Z = enc(flux.reshape(B*G, L), obs.reshape(B*G, L)).reshape(B, G, -1)
            g = leave_one_out_mean(Z).reshape(B*G, -1)
            pred = dec(g).reshape(B, G, -1)
            loss = masked_smooth_l1(pred, flux, obs)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(dec.parameters(), cfg.grad_clip)
            opt.step()
            if step % 200 == 0:
                print(f"[dec ] {step:5d} recon={float(loss):.4f}", flush=True)
            step += 1
            if step >= steps: break
    torch.save({"encoder": enc.state_dict(), "decoder": dec.state_dict(),
                "cfg": cfg.__dict__, "git_sha": git_sha()},
               os.path.join(out, "corr_part1_best.pt"))
    print(f"[corr] saved {out}/corr_part1_best.pt", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="artifacts/vicreg_jepa/ccd3_fill.npz")
    p.add_argument("--out", default="artifacts/vicreg_jepa/corr_part1")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--dec-steps", type=int, default=3000)
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--group-radius", type=float, default=0.0583)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--mu1", type=float, default=1.0)
    p.add_argument("--nu1", type=float, default=0.04)
    p.add_argument("--objective",
                   choices=["correlation","vicreg","reconstruction"],
                   default="correlation")
    a = p.parse_args()
    cfg = Part1Config()
    cfg.group_size, cfg.group_radius_deg = a.group_size, a.group_radius
    cfg.d_model, cfg.n_layers = a.d_model, a.n_layers
    src = RealCurveSource(a.npz, "train")
    print(f"device={DEVICE} train={len(src.flux)}", flush=True)
    print(f"objective={a.objective}  alpha={a.alpha} mu1={a.mu1} nu1={a.nu1}", flush=True)
    enc = train_encoder(src, cfg, a.steps, a.alpha, a.mu1, a.nu1, a.out,
                        objective=a.objective)
    fit_decoder(enc, src, cfg, a.dec_steps, a.out)
    print("=== DONE ===", flush=True)
