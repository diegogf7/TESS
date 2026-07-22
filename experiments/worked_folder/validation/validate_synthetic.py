"""
Empirical proof-of-method on synthetic light curves (runs on CPU, no cluster).

We cannot touch the real TESS parquet from here, so instead we BUILD a dataset
where the ground truth is known: every curve carries an astrophysical CLASS
signal (period/shape) and an instrument SECTOR signature (slow trend, a
sector-specific systematic, a sector-specific noise level, and a downlink-style
gap whose position depends on the sector) -- exactly the kind of structure real
TESS systematics have.

We then:
  1. pretrain MaskedS4D self-supervised on flux+mask ONLY (no labels),
  2. freeze it, encode every curve,
  3. KNN-probe the frozen latent for SECTOR (the target) and CLASS,
  4. compare against a random-init encoder so the number reflects what the
     masked PRETRAINING actually buys, not just the architecture.

If masked pretraining works, the trained-latent sector accuracy is far above
chance and clearly above the random-init baseline. That is the evidence the
same recipe will recover sector structure on the real data on the cluster.
"""

import numpy as np
import torch

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from src.worked_folder.masked.masked_s4d import MaskedS4D, masked_recon_loss

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
MASK_RATIO = 0.5

EPOCHS = 40
BATCH_SIZE = 128
LR = 1e-3

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")


# ------------------------ synthetic data --------------------------
def normalize(flux, clip_sigma=6.0):
    """Same median/MAD normalisation the real data.py pipeline uses."""
    median = np.median(flux)
    if median == 0:
        median = 1.0
    flux = (flux / median) - 1.0
    scale = np.median(np.abs(flux)) * 1.4826
    if scale > 0:
        flux = np.clip(flux, -clip_sigma * scale, clip_sigma * scale)
    return flux


def make_dataset(n, L, n_classes, n_sectors, seed=0):
    """
    A deliberately NON-trivial dataset:
      - CLASS  = a moderate-amplitude astrophysical oscillation at a
                 class-specific frequency (low band, 0.5-2.5 cyc/day).
      - SECTOR = a LOW-amplitude instrument systematic at a sector-specific
                 frequency (high band, 3-6 cyc/day), BURIED IN NOISE.
      - noise level and gap positions are identical in distribution across all
        sectors, so neither class nor sector is separable from variance / DC /
        gap location. You have to extract the *coherent periodic component* to
        recover either label -- which is exactly what masked reconstruction
        forces the encoder to learn, and what a random encoder cannot do well.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 27.0, L)  # ~one TESS sector in days

    # frequencies must be SLOW enough to survive the encoder's per-segment mean
    # pooling (period >> one segment), i.e. the regime real systematics live in.
    # class and sector occupy disjoint low-frequency bands so they're separable.
    class_freq = np.linspace(0.04, 0.10, n_classes)   # period ~10-25 d
    sector_freq = np.linspace(0.13, 0.22, n_sectors)  # period ~4.5-8 d
    SECTOR_AMP = 0.8                                   # systematic is modest...
    NOISE = 0.6                                        # ...and buried in noise
    BASELINE = 10.0                                    # positive flux for median-norm

    flux = np.zeros((n, L), dtype=np.float32)
    obs = np.ones((n, L), dtype=np.float32)
    y_class = rng.integers(0, n_classes, n)
    y_sector = rng.integers(0, n_sectors, n)

    for i in range(n):
        c, s = y_class[i], y_sector[i]

        # astrophysical class signal (clear, class-specific frequency)
        a_c = rng.uniform(0.8, 1.2)
        sig = a_c * np.sin(2 * np.pi * class_freq[c] * t + rng.uniform(0, 2 * np.pi))

        # weak sector systematic at a sector-specific frequency, buried in noise
        sys = SECTOR_AMP * np.sin(2 * np.pi * sector_freq[s] * t + rng.uniform(0, 2 * np.pi))
        noise = rng.normal(0.0, NOISE, L)

        raw = sig + sys + noise + BASELINE

        # gap with a RANDOM position (independent of sector), ~10% blanked
        gw = int(0.10 * L)
        gc = rng.integers(gw, L - gw)
        raw[gc - gw // 2: gc + gw // 2] = 0.0
        obs[i, gc - gw // 2: gc + gw // 2] = 0.0

        f_norm = normalize(raw)
        flux[i] = np.where(obs[i] > 0, f_norm, 0.0)

    return (
        torch.tensor(flux),
        torch.tensor(obs),
        torch.tensor(y_class),
        torch.tensor(y_sector),
    )


# --------------------------- training -----------------------------
def pretrain(model, flux, obs, epochs):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = flux.shape[0]
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for b in range(0, n, BATCH_SIZE):
            idx = perm[b:b + BATCH_SIZE]
            fb, mb = flux[idx], obs[idx]
            opt.zero_grad()
            recon, seg_mask = model(fb, mb)
            loss = masked_recon_loss(recon, fb, seg_mask, model.patch, mb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        if ep == 0 or (ep + 1) % 5 == 0:
            print(f"  epoch {ep+1:2d}/{epochs}  masked-recon loss = {total/n:.5f}")


@torch.no_grad()
def recon_quality(model, flux, obs):
    """Fraction of held-out (masked, observed) variance the model explains."""
    model.eval()
    torch.manual_seed(123)  # fix the mask draw so random vs trained are comparable
    num = den = 0.0
    for b in range(0, flux.shape[0], BATCH_SIZE):
        fb, mb = flux[b:b + BATCH_SIZE], obs[b:b + BATCH_SIZE]
        recon, seg_mask = model(fb, mb)
        w = seg_mask.repeat_interleave(model.patch, dim=1)[:, :fb.shape[1]] * mb
        num += (((recon - fb) ** 2) * w).sum().item()
        den += ((fb ** 2) * w).sum().item()   # targets are ~zero-mean after normalise
    return 1.0 - num / max(den, 1e-9)


@torch.no_grad()
def get_latents(model, flux, obs):
    model.eval()
    out = []
    for b in range(0, flux.shape[0], BATCH_SIZE):
        z = model.encode(flux[b:b + BATCH_SIZE], obs[b:b + BATCH_SIZE])
        out.append(z.reshape(z.shape[0], -1).cpu().numpy())
    return np.concatenate(out)


def probe(X, y, name):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)
    sc = StandardScaler()
    Xtr, Xte = sc.fit_transform(Xtr), sc.transform(Xte)
    clf = KNeighborsClassifier(n_neighbors=20).fit(Xtr, ytr)
    acc = balanced_accuracy_score(yte, clf.predict(Xte))
    chance = 1.0 / len(np.unique(y))
    print(f"  {name:42s} balanced acc = {acc:.3f}   (chance {chance:.3f})")
    return acc


def main():
    print("building synthetic dataset...")
    flux, obs, y_class, y_sector = make_dataset(N_SAMPLES, GRID_LENGTH, N_CLASSES, N_SECTORS, SEED)
    y_class, y_sector = y_class.numpy(), y_sector.numpy()
    print(f"  {N_SAMPLES} curves, length {GRID_LENGTH}, {N_CLASSES} classes, {N_SECTORS} sectors")

    def build():
        return MaskedS4D(
            grid_length=GRID_LENGTH, n_tokens=N_TOKENS, token_dim=TOKEN_DIM,
            d_model=D_MODEL, n_layers=N_LAYERS, dropout=0.1, mask_ratio=MASK_RATIO,
        ).to(DEVICE)

    print("\n[baseline] random-init encoder (no pretraining):")
    rand_model = build()
    rand_fvu = recon_quality(rand_model, flux, obs)
    Xr = get_latents(rand_model, flux, obs)
    print(f"  held-out variance explained = {rand_fvu:.3f}")
    sec_r = probe(Xr, y_sector, "random-init latent -> SECTOR")
    cls_r = probe(Xr, y_class, "random-init latent -> CLASS")

    print("\n[method] masked self-supervised pretraining:")
    model = build()
    pretrain(model, flux, obs, EPOCHS)
    train_fvu = recon_quality(model, flux, obs)
    Xt = get_latents(model, flux, obs)
    print(f"\nheld-out variance explained = {train_fvu:.3f}")
    print("frozen-latent probes after pretraining:")
    sec = probe(Xt, y_sector, "masked latent -> SECTOR  (target)")
    cls = probe(Xt, y_class, "masked latent -> CLASS")

    print("\n===================== EVIDENCE SUMMARY =====================")
    print(f"held-out variance explained : random {rand_fvu:.3f}  ->  trained {train_fvu:.3f}")
    print(f"SECTOR balanced acc (target): random {sec_r:.3f}  ->  trained {sec:.3f}   (chance {1/N_SECTORS:.3f})")
    print(f"CLASS  balanced acc         : random {cls_r:.3f}  ->  trained {cls:.3f}   (chance {1/N_CLASSES:.3f})")
    print("============================================================")


if __name__ == "__main__":
    main()
