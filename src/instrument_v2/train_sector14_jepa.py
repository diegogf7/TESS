# All this code is from Claude
"""Sector-14 chip-pair instrument JEPA (v2). See src/instrument_v2/README.md.

Reuses the existing gap-blind machinery UNCHANGED (S4D context encoder, EMA
target encoder, latent predictor, masked smooth-L1 loss, raw-token spread
penalty from src/loss_function/gapblind_fix.py) -- the only new ingredients
are the chip-balanced pair sampling and the two grid arms. NO infilling:
gaps are normalized zeros and the observed mask goes to encoder and loss.

Context = one raw curve; EMA target = a DIFFERENT star, same camera x CCD.

Run:  ARM=shared SEED=0 python -m src.instrument_v2.train_sector14_jepa
Env:  ARM (shared|legacy), SEED, S14_DATA, SPLIT_DIR, ART_DIR, CKPT_DIR,
      EPOCHS, BATCH, LR, VARW, MAX_BATCHES (smoke), NUM_WORKERS
"""

import csv
import os
import random
import subprocess

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.instrument_v2.sector14_dataset import (
    Sector14ChipPairDataset,
    ensure_splits,
    ensure_time_range,
)
from src.loss_function.gapblind_fix import build_gapblind_jepa, gapblind_loss

ARM = os.environ.get("ARM", "shared")
assert ARM in ("shared", "legacy"), f"bad ARM {ARM!r}"
SEED = int(os.environ.get("SEED", "0"))
SECTOR = 14

S14_DATA = os.environ.get("S14_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get("SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
ART_DIR = os.environ.get("ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))
CKPT_DIR = os.environ.get("CKPT_DIR", "/orcd/scratch/orcd/006/diegogon/checkpoints")

EPOCHS = int(os.environ.get("EPOCHS", "100"))
BATCH = int(os.environ.get("BATCH", "256"))
LR = float(os.environ.get("LR", "1e-3"))
VARW = float(os.environ.get("VARW", "0.5"))
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "0"))     # >0 = smoke mode
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def effective_rank(Z):
    """exp(entropy of normalized singular-value energies) of centered latents."""
    Zc = Z - Z.mean(axis=0)
    s = np.linalg.svd(Zc, compute_uv=False)
    p = (s ** 2) / max((s ** 2).sum(), 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def main():
    ckpt_base = os.path.join(CKPT_DIR, f"s14jepa_{ARM}_s{SEED}")
    os.makedirs(ART_DIR, exist_ok=True)
    print(f"git commit: {git_commit()}")
    print(f"config: arm={ARM} seed={SEED} epochs={EPOCHS} batch={BATCH} lr={LR} "
          f"varw={VARW} device={DEVICE} -> {ckpt_base}.pth")
    print(f"data: {S14_DATA}")

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == SECTOR].drop_duplicates("TIC").reset_index(drop=True)

    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, ART_DIR)
    print(f"splits: {len(train_tics)} train / {len(val_tics)} val / {len(test_tics)} test "
          "TICs (mutually disjoint, asserted)")
    t_range = ensure_time_range(ART_DIR, df, train_tics)
    print(f"shared-grid time range (train stars only): {t_range}")

    train_ds = Sector14ChipPairDataset(df, train_tics, ARM, t_range)
    val_ds = Sector14ChipPairDataset(df, val_tics, ARM, t_range)
    assert not (set(train_ds.tics) & test_tics), "test TIC leaked into train dataset"
    assert not (set(val_ds.tics) & test_tics), "test TIC leaked into val dataset"
    print(f"pair datasets: {len(train_ds.tics)} train stars "
          f"({len(train_ds.chip_list)} chips), {len(val_ds.tics)} val stars "
          f"({len(val_ds.chip_list)} chips)")

    train_loader = DataLoader(train_ds, batch_size=BATCH, num_workers=NUM_WORKERS,
                              worker_init_fn=seed_worker,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_ds, batch_size=BATCH, num_workers=NUM_WORKERS,
                            worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(SEED))

    model = build_gapblind_jepa().to(DEVICE)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    metrics_path = os.path.join(ART_DIR, f"metrics_{ARM}_s{SEED}.csv")
    metrics_fields = ["epoch", "train_loss", "val_loss", "latent_std",
                      "effective_rank", "pred_std", "grad_norm"]
    with open(metrics_path, "w", newline="") as fh:
        csv.writer(fh).writerow(metrics_fields)

    for epoch in range(EPOCHS):
        model.train()
        total_loss, total_gnorm, n_batches = 0.0, 0.0, 0
        for batch_idx, (ctx_f, ctx_m, tgt_f, tgt_m) in enumerate(train_loader):
            if MAX_BATCHES and batch_idx >= MAX_BATCHES:
                break
            ctx_f, ctx_m = ctx_f.to(DEVICE), ctx_m.to(DEVICE)
            tgt_f, tgt_m = tgt_f.to(DEVICE), tgt_m.to(DEVICE)

            optimizer.zero_grad()
            prediction, target, context_tokens = model(ctx_f, ctx_m, tgt_f, tgt_m)
            loss = gapblind_loss(prediction, target, context_tokens,
                                 target_mask=tgt_m, var_weight=VARW)
            loss.backward()
            total_gnorm += float(torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1e9))
            optimizer.step()
            model.update_target()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        model.eval()
        val_loss, val_batches = 0.0, 0
        latents, preds = [], []
        with torch.no_grad():
            for batch_idx, (ctx_f, ctx_m, tgt_f, tgt_m) in enumerate(val_loader):
                if MAX_BATCHES and batch_idx >= MAX_BATCHES:
                    break
                ctx_f, ctx_m = ctx_f.to(DEVICE), ctx_m.to(DEVICE)
                tgt_f, tgt_m = tgt_f.to(DEVICE), tgt_m.to(DEVICE)
                prediction, target, context_tokens = model(ctx_f, ctx_m, tgt_f, tgt_m)
                val_loss += gapblind_loss(prediction, target, context_tokens,
                                          target_mask=tgt_m, var_weight=VARW).item()
                val_batches += 1
                z = model.encode(tgt_f, tgt_m)
                latents.append(z.reshape(z.shape[0], -1).cpu().numpy())
                preds.append(prediction.reshape(prediction.shape[0], -1).cpu().numpy())

        Z = np.concatenate(latents)
        P = np.concatenate(preds)
        row = {"epoch": epoch + 1,
               "train_loss": total_loss / max(n_batches, 1),
               "val_loss": val_loss / max(val_batches, 1),
               "latent_std": float(Z.std(axis=0).mean()),
               "effective_rank": effective_rank(Z),
               "pred_std": float(P.std(axis=0).mean()),
               "grad_norm": total_gnorm / max(n_batches, 1)}
        with open(metrics_path, "a", newline="") as fh:
            csv.writer(fh).writerow([row[k] for k in metrics_fields])
        print(f"epoch {epoch + 1}/{EPOCHS}  train {row['train_loss']:.5f}  "
              f"val {row['val_loss']:.5f}  latent_std {row['latent_std']:.4f}  "
              f"erank {row['effective_rank']:.1f}  pred_std {row['pred_std']:.4f}  "
              f"gnorm {row['grad_norm']:.3f}", flush=True)

        torch.save(model.state_dict(), f"{ckpt_base}.pth")          # final/latest
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{ckpt_base}_ep{epoch + 1:03d}.pth")

    print(f"DONE: final checkpoint {ckpt_base}.pth, metrics {metrics_path}")


if __name__ == "__main__":
    main()
