"""Train Part 2 against Part 1's frozen output. ARCHITECTURE.md section 2.

    python -m src.vicreg_jepa.train_part2 \
        --npz artifacts/vicreg_jepa/s15_big.npz \
        --part1 artifacts/vicreg_jepa/part1_s15_big/part1/part1_best.pt \
        --steps 4000 --out artifacts/vicreg_jepa/part2_s15
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .collapse import CollapseThresholds, assess
from .data import SingleCurveDataset
from .models import S4Encoder
from .part2_losses import total_loss, l_inv, l_var, l_cov, l_cor_sys
from .part2_model import PhysicsJEPA, block_mask
from .gradnorm import GradNorm
from .real_data import RealCurveSource, git_sha

DEVICE = os.environ.get("TESS_DEVICE",
                        "cuda" if torch.cuda.is_available() else "cpu")


def balance_weights(model, flux, obs, mask_ratio, device, ref="inv"):
    """Set each weight to 1/||grad|| so every term starts with equal pull.

    VICReg's published weights (25 invariance / 25 variance / 1 covariance) were
    carried over here with the 25 landing on L_cor_sys instead of L_var, leaving
    the two anti-collapse terms with ~2% of the gradient. Measuring is safer than
    inheriting.
    """
    mf, mc = block_mask(flux, obs, mask_ratio)
    terms = {"inv": lambda zo, zp, zt, zs: l_inv(zp, zt),
             "cor_sys": lambda zo, zp, zt, zs: l_cor_sys(zo, zs),
             "var": lambda zo, zp, zt, zs: l_var(zo),
             "cov": lambda zo, zp, zt, zs: l_cov(zo)}
    g = {}
    for name, fn in terms.items():
        model.zero_grad(set_to_none=True)
        fn(*model(flux, obs, mf, mc)).backward()
        gs = [p.grad.flatten() for p in model.online.parameters() if p.grad is not None]
        g[name] = float(torch.cat(gs).norm()) if gs else 0.0
    model.zero_grad(set_to_none=True)
    base = g[ref] if g[ref] > 0 else 1.0
    return {k: (base / v if v > 0 else 0.0) for k, v in g.items()}, g


def load_part1(path, device):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    c = blob["cfg"]
    enc = S4Encoder(c["latent_dim"], c["n_tokens"], c["d_model"],
                    c["d_state"], c["n_layers"], 0.0)
    enc.load_state_dict(blob["encoder"])
    enc.eval()
    enc.requires_grad_(False)
    return enc.to(device), c["latent_dim"]


@torch.no_grad()
def validate(model, flux, obs, cfg, device, thresholds, streak):
    model.eval()
    lanes = {"masked_online": [], "full_online": [], "full_ema": []}
    sysl, inv_tot, nb = [], 0.0, 0
    g = torch.Generator().manual_seed(1234)
    for i in range(0, len(flux), cfg["batch_size"]):
        f = flux[i:i + cfg["batch_size"]].to(device)
        o = obs[i:i + cfg["batch_size"]].to(device)
        if len(f) < 8:
            break
        mf, mc = block_mask(f, o, cfg["mask_ratio"], generator=g)
        zo, zp, zt, zs = model(f, o, mf, mc)
        inv_tot += float(torch.nn.functional.mse_loss(zp, zt))
        lanes["masked_online"].append(zo.cpu().numpy())
        lanes["full_ema"].append(zt.cpu().numpy())
        lanes["full_online"].append(model.online(f, o).cpu().numpy())
        sysl.append(zs.cpu().numpy())
        nb += 1
    model.train()
    lanes = {k: np.concatenate(v) for k, v in lanes.items()}
    v = assess(lanes, thresholds, streak)
    ema = v["stats"]["full_ema"]
    return {"val_inv": inv_tot / max(nb, 1), "rejected": v["rejected"],
            "why": v["why"], "streak": v["online_fail_streak"],
            "effective_rank": ema["effective_rank"], "std_median": ema["std_median"],
            "offdiag_cov_ratio": ema["offdiag_cov_ratio"],
            "dup_pair_frac": ema["dup_pair_frac"],
            "meets_target_rank": v["meets_target_rank"]}


def main(a):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.out, exist_ok=True)
    thresholds = CollapseThresholds()

    part1, sys_dim = load_part1(a.part1, DEVICE)
    model = PhysicsJEPA(part1, latent_dim=a.latent_dim, sys_dim=sys_dim,
                        d_model=a.d_model, n_layers=a.n_layers,
                        d_state=a.d_state, dropout=a.dropout,
                        total_steps=a.steps).to(DEVICE)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=a.weight_decay)

    train_src = RealCurveSource(a.npz, "train")
    val_src = RealCurveSource(a.npz, "val")
    dl = DataLoader(SingleCurveDataset(train_src), batch_size=a.batch_size,
                    shuffle=True, drop_last=True)
    vidx = np.random.default_rng(1234).permutation(len(val_src.flux))[:a.val_n]
    vflux = torch.from_numpy(val_src.flux[np.sort(vidx)].astype(np.float32))
    vobs = torch.from_numpy(val_src.observed[np.sort(vidx)].astype(np.float32))

    cfg = {"latent_dim": a.latent_dim, "sys_dim": sys_dim, "d_model": a.d_model,
           "n_layers": a.n_layers, "d_state": a.d_state, "dropout": a.dropout,
           "phi": a.phi, "lam": a.lam, "mu": a.mu, "nu": a.nu, "gamma": a.gamma,
           "mask_ratio": a.mask_ratio, "batch_size": a.batch_size, "lr": a.lr,
           "weight_decay": a.weight_decay, "steps": a.steps, "seed": a.seed,
           "part1": a.part1, "npz": a.npz, "git_sha": git_sha(),
           "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)
    mpath = os.path.join(a.out, "metrics.jsonl")
    open(mpath, "w").close()
    print(f"device={DEVICE} train={len(train_src.flux)} val={len(vflux)} "
          f"sys_dim={sys_dim}", flush=True)

    # --- weighting -------------------------------------------------------
    phi, lam, mu, nu = a.phi, a.lam, a.mu, a.nu
    if a.auto_weights:
        fb = torch.from_numpy(train_src.flux[:a.batch_size].astype(np.float32)).to(DEVICE)
        ob = torch.from_numpy(train_src.observed[:a.batch_size].astype(np.float32)).to(DEVICE)
        w, gn = balance_weights(model, fb, ob, a.mask_ratio, DEVICE)
        phi, lam, mu, nu = w["inv"], w["cor_sys"] * a.lam_scale, w["var"], w["cov"]
        cfg["grad_norms_at_init"] = gn
        cfg["auto_weights"] = {"phi": phi, "lam": lam, "mu": mu, "nu": nu}
        print(f"[weights] gradient-balanced: phi={phi:.3g} lam={lam:.3g} "
              f"mu={mu:.3g} nu={nu:.3g}  (lam_scale={a.lam_scale})", flush=True)
        json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)

    gn_bal = None
    if a.gradnorm:
        # GradNorm starts UNIFORM and discovers the balance from measured
        # gradient norms. Seeding it with the gradient-balanced values breaks it:
        # those span four orders of magnitude, and renormalising them to sum to
        # the number of terms drives phi and lam to ~0, switching off the
        # invariance loss entirely.
        gn_bal = GradNorm(model.online.head.weight, None,
                          alpha=a.gradnorm_alpha, lr=a.gradnorm_lr, device=DEVICE)
        w = gn_bal.weights()
        phi, lam, mu, nu = w["inv"], w["cor_sys"], w["var"], w["cov"]
        cfg["gradnorm"] = {"alpha": a.gradnorm_alpha, "lr": a.gradnorm_lr,
                           "every": a.gradnorm_every, "init": w}
        print(f"[gradnorm] on, alpha={a.gradnorm_alpha} every={a.gradnorm_every} "
              f"init phi={phi:.3g} lam={lam:.3g} mu={mu:.3g} nu={nu:.3g}", flush=True)
        json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=2)

    mu_dual = mu
    mu_min, mu_max = a.mu_min, a.mu_max
    best, best_step, streak, step = float("inf"), -1, 0, 0
    while step < a.steps:
        for flux, obs in dl:
            flux, obs = flux.to(DEVICE), obs.to(DEVICE)
            mf, mc = block_mask(flux, obs, a.mask_ratio)
            zo, zp, zt, zs = model(flux, obs, mf, mc)

            if gn_bal is not None and step % a.gradnorm_every == 0:
                terms = {"inv": l_inv(zp, zt), "cor_sys": l_cor_sys(zo, zs),
                         "var": l_var(zo, a.gamma), "cov": l_cov(zo)}
                w, lg = gn_bal.step(terms)
                phi, lam, nu = w["inv"], w["cor_sys"], w["cov"]
                # GradNorm down-weights fast-falling terms, and L_var falls fast
                # exactly while the latent shrinks. The dual-ascent multiplier is
                # kept as a hard floor so the constraint cannot be traded away.
                mu = max(w["var"], mu_dual) if a.dual_mu else w["var"]

            loss, parts = total_loss(zo, zp, zt, zs, phi, lam, mu, nu, a.gamma)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], a.grad_clip)
            opt.step()
            tau = model.update_ema(step)

            # Dual ascent: collapse is a CONSTRAINT (std >= gamma), not a term to
            # trade off. mu rises while the constraint is violated and relaxes
            # once it is met, so it never has to be guessed or swept.
            if a.dual_mu:
                with torch.no_grad():
                    std_med = float(zo.std(dim=0).median())
                mu_dual = float(np.clip(mu_dual + a.dual_eta * (a.gamma - std_med),
                                        mu_min, mu_max))
                mu = max(mu, mu_dual)
            rec = {"step": step, **parts, "tau": tau, "grad_norm": float(gn),
                   "mu": mu, "lam": lam, "nu": nu, "phi": phi,
                   "mu_dual": mu_dual}
            if step % a.eval_every == 0 or step == a.steps - 1:
                v = validate(model, vflux, vobs, cfg, DEVICE, thresholds, streak)
                streak = v["streak"]
                rec.update({f"val_{k}": val for k, val in v.items() if k != "why"})
                if not v["rejected"] and v["val_inv"] < best:
                    best, best_step = v["val_inv"], step
                    torch.save({"online": model.online.state_dict(),
                                "target": model.target.state_dict(),
                                "predictor": model.predictor.state_dict(),
                                "cfg": cfg, "step": step, "val": v},
                               os.path.join(a.out, "part2_best.pt"))
                    rec["saved_best"] = True
                print(f"[part2] {step:5d} inv={parts['inv']:.4f} cor={parts['cor_sys']:.4f} "
                      f"var={parts['var']:.4f} cov={parts['cov']:.4f} | "
                      f"val_inv={v['val_inv']:.4f} erank={v['effective_rank']:.1f} "
                      f"std={v['std_median']:.3f} | w: phi={phi:.2f} lam={lam:.2f} "
                      f"mu={mu:.1f} nu={nu:.2f}"
                      + ("  REJECT " + "; ".join(v["why"][:2]) if v["rejected"] else "  ok"),
                      flush=True)
            with open(mpath, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            step += 1
            if step >= a.steps:
                break

    summary = {"best_val_inv": best if best_step >= 0 else None,
               "best_step": best_step,
               "status": "ok" if best_step >= 0 else "failed_all_collapsed"}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=2)
    print(f"[part2] {summary['status']} best val_inv {best:.4f} @ {best_step}"
          if best_step >= 0 else "[part2] NO VALID CHECKPOINT", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default="artifacts/vicreg_jepa/s15_big.npz")
    p.add_argument("--part1", default="artifacts/vicreg_jepa/part1_s15_big/part1/part1_best.pt")
    p.add_argument("--out", default="artifacts/vicreg_jepa/part2_s15")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--d-state", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--mask-ratio", type=float, default=0.4)
    p.add_argument("--phi", type=float, default=1.0)
    p.add_argument("--lam", type=float, default=25.0)
    p.add_argument("--mu", type=float, default=1.0)
    p.add_argument("--nu", type=float, default=0.01)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--val-n", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--auto-weights", action="store_true",
                   help="set weights from measured gradient norms at init")
    p.add_argument("--lam-scale", type=float, default=1.0,
                   help="multiplier on the balanced lambda; the one real trade-off")
    p.add_argument("--dual-mu", action="store_true",
                   help="treat std >= gamma as a constraint; mu self-tunes")
    p.add_argument("--dual-eta", type=float, default=0.5)
    p.add_argument("--mu-min", type=float, default=0.0)
    p.add_argument("--mu-max", type=float, default=500.0)
    p.add_argument("--gradnorm", action="store_true",
                   help="GradNorm: rebalance weights continuously by gradient norm")
    p.add_argument("--gradnorm-alpha", type=float, default=1.5)
    p.add_argument("--gradnorm-lr", type=float, default=2.5e-2)
    p.add_argument("--gradnorm-every", type=int, default=20)
    main(p.parse_args())
