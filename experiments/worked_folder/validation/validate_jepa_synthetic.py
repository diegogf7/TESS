"""
Empirical check that LatentJEPA (predict-in-latent-space) works and -- the key
worry -- does NOT collapse, using the same synthetic data as validate_synthetic.

Prints, for random-init vs trained:
  - latent std across the batch  (THE collapse tell: ->0 means collapsed)
  - SECTOR / CLASS balanced-accuracy probes on the frozen latent

Run:  python -m src.bot_folder.validate_jepa_synthetic
"""

import numpy as np
import torch

from src.worked_folder.validation.validate_synthetic import make_dataset, probe
from src.worked_folder.physics.latent_jepa import LatentJEPA, jepa_latent_loss

# ----------------------------- config -----------------------------
SEED = 0
N_SAMPLES = 2000
GRID_LENGTH = 256
N_CLASSES = 4
N_SECTORS = 6

D_MODEL = 64
N_LAYERS = 3
N_TOKENS = 16
TOKEN_DIM = 16
MASK_RATIO = 0.5          # standard JEPA default (matches the cluster run)
MOMENTUM = 0.996          # standard EMA momentum (matches the cluster run)
VAR_WEIGHT = 0.05         # gentle anti-collapse / spread safety net

EPOCHS = 40
BATCH_SIZE = 128
LR = 1e-3

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")


def build():
    return LatentJEPA(
        grid_length=GRID_LENGTH, n_tokens=N_TOKENS, token_dim=TOKEN_DIM,
        d_model=D_MODEL, n_layers=N_LAYERS, dropout=0.1,
        mask_ratio=MASK_RATIO, momentum=MOMENTUM,
    ).to(DEVICE)


def pretrain(model, flux, obs, epochs):
    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = flux.shape[0]
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for b in range(0, n, BATCH_SIZE):
            idx = perm[b:b + BATCH_SIZE]
            opt.zero_grad()
            pred, tgt, seg = model(flux[idx], obs[idx])
            loss = jepa_latent_loss(pred, tgt, seg, var_weight=VAR_WEIGHT)
            loss.backward()
            opt.step()
            model.update_target()
            total += loss.item() * len(idx)
        sched.step()
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"  epoch {ep+1:2d}/{epochs}  latent-jepa loss = {total/n:.5f}")


@torch.no_grad()
def get_latents(model, flux, obs):
    model.eval()
    out = []
    for b in range(0, flux.shape[0], BATCH_SIZE):
        z = model.encode(flux[b:b + BATCH_SIZE], obs[b:b + BATCH_SIZE])
        out.append(z.reshape(z.shape[0], -1).cpu().numpy())
    return np.concatenate(out)


def collapse_std(X):
    """mean per-dimension std across the batch -- ~0 means collapsed."""
    return float(X.std(axis=0).mean())


def main():
    print("building synthetic dataset...")
    flux, obs, y_class, y_sector = make_dataset(N_SAMPLES, GRID_LENGTH, N_CLASSES, N_SECTORS, SEED)
    y_class, y_sector = y_class.numpy(), y_sector.numpy()
    print(f"  {N_SAMPLES} curves, {N_CLASSES} classes, {N_SECTORS} sectors")

    print("\n[baseline] random-init JEPA encoder (no pretraining):")
    Xr = get_latents(build(), flux, obs)
    print(f"  latent std (collapse check) = {collapse_std(Xr):.4f}")
    sec_r = probe(Xr, y_sector, "random-init latent -> SECTOR")
    cls_r = probe(Xr, y_class, "random-init latent -> CLASS")

    print("\n[method] LatentJEPA self-supervised pretraining (predict-in-latent-space):")
    model = build()
    pretrain(model, flux, obs, EPOCHS)
    Xt = get_latents(model, flux, obs)
    std_t = collapse_std(Xt)
    print(f"\nlatent std (collapse check) = {std_t:.4f}   (~0 = collapsed, want >> 0)")
    sec = probe(Xt, y_sector, "JEPA latent -> SECTOR  (target)")
    cls = probe(Xt, y_class, "JEPA latent -> CLASS")

    print("\n===================== EVIDENCE SUMMARY =====================")
    print(f"latent std (collapse) : random {collapse_std(Xr):.4f}  ->  trained {std_t:.4f}")
    print(f"SECTOR balanced acc   : random {sec_r:.3f}  ->  trained {sec:.3f}   (chance {1/N_SECTORS:.3f})")
    print(f"CLASS  balanced acc   : random {cls_r:.3f}  ->  trained {cls:.3f}   (chance {1/N_CLASSES:.3f})")
    verdict = "NO COLLAPSE, latent is informative" if std_t > 0.05 and sec > 2.0 / N_SECTORS else "CHECK: possible collapse / weak latent"
    print(f"verdict: {verdict}")
    print("============================================================")


if __name__ == "__main__":
    main()
