"""
LOCAL CPU proof (no cluster) that the per-segment MEAN pooling in S4Model
destroys high-frequency (sub-segment) CLASS signal, and that two IN-FRAMEWORK
fixes recover it -- each fix recovering a different facet:

  (1) mean_std readout    -> recovers class signal carried by sub-segment
                             AMPLITUDE / variance (std per segment != 0 even when
                             the mean averages the oscillation to ~0).
  (2) more tokens / smaller segments -> recovers class signal carried by
                             sub-segment FREQUENCY / shape (the segment now
                             resolves the period instead of averaging over it).

Controlled synthetic (the honest part): 4 classes = 2 AMPLITUDES x 2 FREQUENCIES,
BOTH sub-segment. So:
  - the amplitude axis {classes 0,1} vs {2,3} is recoverable ONLY by std,
  - the frequency axis {0,2} vs {1,3} is recoverable ONLY by finer patches.
A model that gets only one fix should recover ~one axis (~0.5 of a 4-way task);
both fixes should recover both axes. SECTOR is a LOW-frequency systematic that
survives mean pooling -- exactly the real-data regime (sector high, class low).

This is a MECHANISM demo, not a real-data number. The synthetic is deliberately
constructed to isolate the pooling effect; the cluster run on real TESS is the
truth. But it proves the fix targets the right failure mode before spending GPU.

Run:
  /Users/diegogonzalez/.tess_local_venv/bin/python -m src.worked_folder.validate_readout
"""

import numpy as np
import torch

from src.worked_folder.latent_jepa import LatentJEPA, jepa_latent_loss
from src.worked_folder.validate_synthetic import normalize, probe

# ----------------------------- config -----------------------------
SEED = 0
N_SAMPLES = 1600
GRID_LENGTH = 256
N_CLASSES = 4
N_SECTORS = 6

D_MODEL = 64
N_LAYERS = 3
TOKEN_DIM = 16
MASK_RATIO = 0.5
MOMENTUM = 0.996
VAR_WEIGHT = 0.05

EPOCHS = 30
BATCH_SIZE = 128
LR = 1e-3

# segment length in points for the baseline (n_tokens=16): 256/16 = 16 pts.
# t spans 27 days -> dt ~= 0.105 d/pt -> a 16-pt segment ~= 1.69 days.
# class frequencies below are chosen so their period << 1.69 d (sub-segment).
CONFIGS = [
    # (n_tokens, readout, label)
    (16, "mean",     "baseline   mean @16tok (seg~1.7d)"),
    (16, "mean_std", "FIX1       mean+std @16tok"),
    (64, "mean",     "FIX2       mean @64tok (seg~0.42d)"),
    (64, "mean_std", "FIX1+2     mean+std @64tok"),
]

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device("cpu")


def make_dataset_hf(n, L, n_classes, n_sectors, seed=0):
    """4 classes = 2 amplitudes x 2 frequencies, BOTH sub-segment (high-freq)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 27.0, L)

    amps = [0.35, 1.30]            # quiet vs strong pulsator (separable by std)
    freqs = [1.5, 3.0]            # cyc/day, period 0.67d & 0.33d -- both sub-segment
    class_amp = [amps[0], amps[0], amps[1], amps[1]]   # axis: {0,1} low | {2,3} high
    class_freq = [freqs[0], freqs[1], freqs[0], freqs[1]]  # axis: {0,2} | {1,3}

    sector_freq = np.linspace(0.10, 0.20, n_sectors)   # LOW-freq, survives mean pool
    SECTOR_AMP = 0.5
    NOISE = 0.25
    BASELINE = 10.0

    flux = np.zeros((n, L), dtype=np.float32)
    obs = np.ones((n, L), dtype=np.float32)
    y_class = rng.integers(0, n_classes, n)
    y_sector = rng.integers(0, n_sectors, n)

    for i in range(n):
        c, s = y_class[i], y_sector[i]
        sig = class_amp[c] * np.sin(2 * np.pi * class_freq[c] * t + rng.uniform(0, 2 * np.pi))
        sys = SECTOR_AMP * np.sin(2 * np.pi * sector_freq[s] * t + rng.uniform(0, 2 * np.pi))
        noise = rng.normal(0.0, NOISE, L)
        raw = sig + sys + noise + BASELINE

        gw = int(0.10 * L)
        gc = rng.integers(gw, L - gw)
        raw[gc - gw // 2: gc + gw // 2] = 0.0
        obs[i, gc - gw // 2: gc + gw // 2] = 0.0

        f_norm = normalize(raw)
        flux[i] = np.where(obs[i] > 0, f_norm, 0.0)

    return (torch.tensor(flux), torch.tensor(obs),
            torch.tensor(y_class), torch.tensor(y_sector))


def build(n_tokens, readout):
    return LatentJEPA(
        grid_length=GRID_LENGTH, n_tokens=n_tokens, token_dim=TOKEN_DIM,
        d_model=D_MODEL, n_layers=N_LAYERS, dropout=0.1,
        mask_ratio=MASK_RATIO, momentum=MOMENTUM, readout=readout,
    ).to(DEVICE)


def pretrain(model, flux, obs, epochs):
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = flux.shape[0]
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for b in range(0, n, BATCH_SIZE):
            idx = perm[b:b + BATCH_SIZE]
            opt.zero_grad()
            pred, tgt, seg = model(flux[idx], obs[idx])
            jepa_latent_loss(pred, tgt, seg, var_weight=VAR_WEIGHT).backward()
            opt.step()
            model.update_target()
        sched.step()


@torch.no_grad()
def get_latents(model, flux, obs):
    model.eval()
    out = []
    for b in range(0, flux.shape[0], BATCH_SIZE):
        z = model.encode(flux[b:b + BATCH_SIZE], obs[b:b + BATCH_SIZE])
        out.append(z.reshape(z.shape[0], -1).cpu().numpy())
    return np.concatenate(out)


def main():
    print("building HIGH-FREQUENCY-class synthetic dataset...")
    flux, obs, y_class, y_sector = make_dataset_hf(N_SAMPLES, GRID_LENGTH, N_CLASSES, N_SECTORS, SEED)
    y_class, y_sector = y_class.numpy(), y_sector.numpy()
    print(f"  {N_SAMPLES} curves, length {GRID_LENGTH}, {N_CLASSES} classes "
          f"(2 amp x 2 freq, both sub-segment), {N_SECTORS} sectors")

    # fast shape self-check across all configs before any training
    for nt, ro, _ in CONFIGS:
        m = build(nt, ro)
        p, t, s = m(flux[:4], obs[:4])
        assert p.shape == t.shape, (nt, ro, p.shape, t.shape)
    print("  shape self-check OK for all configs\n")

    rows = []
    for n_tokens, readout, label in CONFIGS:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        model = build(n_tokens, readout)

        Xr = get_latents(model, flux, obs)
        cls_r = probe(Xr, y_class, f"[{label}] random CLASS")

        pretrain(model, flux, obs, EPOCHS)
        Xt = get_latents(model, flux, obs)
        std_t = float(Xt.std(axis=0).mean())
        cls = probe(Xt, y_class, f"[{label}] trained CLASS")
        sec = probe(Xt, y_sector, f"[{label}] trained SECTOR")
        rows.append((label, cls_r, cls, sec, std_t))
        print()

    print("\n===================== READOUT / PATCH SWEEP =====================")
    print(f"{'config':36s} {'CLASS(rnd)':>10s} {'CLASS':>8s} {'SECTOR':>8s} {'std':>7s}")
    for label, cls_r, cls, sec, std_t in rows:
        print(f"{label:36s} {cls_r:10.3f} {cls:8.3f} {sec:8.3f} {std_t:7.3f}")
    print(f"\nchance: CLASS {1/N_CLASSES:.3f}   SECTOR {1/N_SECTORS:.3f}")
    print("expectation: baseline CLASS ~chance; FIX1 recovers the amplitude axis;")
    print("FIX2 recovers the frequency axis; FIX1+2 recovers both. SECTOR stays high")
    print("throughout (low-freq, survives mean pooling).")
    print("=================================================================")


if __name__ == "__main__":
    main()
