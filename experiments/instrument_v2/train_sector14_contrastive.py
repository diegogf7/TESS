# All this code is from Claude
"""SupCon / hybrid pretraining for the Sector-14 instrument ablation.

Same backbone, data, grid (shared), pairs, masks, and recipe as
train_sector14_jepa.py -- the only change is the objective:
  supcon : supervised contrastive loss over chip labels (both stars of a
           chip pair encoded by the ONLINE context encoder = two views)
  hybrid : gapblind_loss (JEPA prediction + spread penalty) + w * supcon

EMA target updated after every optimizer step in both objectives; dropout is
zero (build_gapblind_jepa). Checkpoints every 10 epochs into a run-specific
ablation directory -- historical s14jepa_* checkpoints are never touched.

Run:  OBJECTIVE=hybrid CONTRASTIVE_WEIGHT=0.5 SEED=0 \
        python -m src.instrument_v2.train_sector14_contrastive
Env:  OBJECTIVE, SEED, CONTRASTIVE_WEIGHT, TEMPERATURE, EPOCHS, BATCH, LR,
      VARW, MAX_BATCHES, S14_DATA, SPLIT_DIR, ART_DIR, ABL_CKPT_DIR, ABL_DIR
"""

import csv
import os
import random
import subprocess

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.instrument_v2.contrastive_loss import supcon_loss
from src.instrument_v2.sector14_dataset import (
    Sector14ChipPairDataset,
    ensure_splits,
    ensure_time_range,
)
from src.instrument_v2.train_sector14_jepa import effective_rank, seed_worker
from src.loss_function.gapblind_fix import build_gapblind_jepa, gapblind_loss

OBJECTIVE = os.environ.get("OBJECTIVE", "supcon")
assert OBJECTIVE in ("supcon", "hybrid"), f"bad OBJECTIVE {OBJECTIVE!r}"
SEED = int(os.environ.get("SEED", "0"))
WEIGHT = float(os.environ.get("CONTRASTIVE_WEIGHT") or "1.0")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.1"))
SECTOR = 14
ARM = "shared"                                   # ablation is shared-grid only

S14_DATA = os.environ.get("S14_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get("SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
ART_DIR = os.environ.get("ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))
ABL_DIR = os.environ.get("ABL_DIR", os.path.join("artifacts", "instrument_v2", "ablation", os.environ.get("RUN_ID", "dev")))
ABL_CKPT_DIR = os.environ.get("ABL_CKPT_DIR",
                              os.path.join("/orcd/scratch/orcd/006/diegogon/checkpoints",
                                           "ablation", os.environ.get("RUN_ID", "dev")))

EPOCHS = int(os.environ.get("EPOCHS", "100"))
BATCH = int(os.environ.get("BATCH", "256"))
LR = float(os.environ.get("LR", "1e-3"))
VARW = float(os.environ.get("VARW", "0.5"))
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "0"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def run_tag(objective=OBJECTIVE, weight=WEIGHT, seed=SEED):
    if objective == "hybrid":
        return f"s14hybrid_w{weight:g}_s{seed}"
    return f"s14supcon_s{seed}"


def total_loss(jepa_term, con_term, objective, weight):
    """Loss composition. hybrid with weight 0 is EXACTLY the JEPA loss."""
    if objective == "supcon":
        return con_term
    return jepa_term if weight == 0 else jepa_term + weight * con_term


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def run_epoch(model, loader, optimizer=None):
    """One pass; returns dict of mean losses (+ latents on eval passes)."""
    training = optimizer is not None
    model.train() if training else model.eval()
    sums = {"jepa": 0.0, "con": 0.0, "total": 0.0}
    n = 0
    latents, preds = [], []
    ctx_manager = torch.enable_grad() if training else torch.no_grad()
    with ctx_manager:
        for batch_idx, (ctx_f, ctx_m, tgt_f, tgt_m, chip) in enumerate(loader):
            if MAX_BATCHES and batch_idx >= MAX_BATCHES:
                break
            ctx_f, ctx_m = ctx_f.to(DEVICE), ctx_m.to(DEVICE)
            tgt_f, tgt_m = tgt_f.to(DEVICE), tgt_m.to(DEVICE)
            chip = chip.to(DEVICE)

            prediction, target, ctx_tokens = model(ctx_f, ctx_m, tgt_f, tgt_m)
            # second view of the SAME chip: the other star through the ONLINE encoder
            tgt_tokens = model.context_encoder(tgt_f.unsqueeze(-1), tgt_m)
            embeddings = torch.cat([ctx_tokens, tgt_tokens], dim=0)
            labels = torch.cat([chip, chip], dim=0)

            jepa_term = gapblind_loss(prediction, target, ctx_tokens,
                                      target_mask=tgt_m, var_weight=VARW)
            con_term = supcon_loss(embeddings, labels, temperature=TEMPERATURE)
            loss = total_loss(jepa_term, con_term, OBJECTIVE, WEIGHT)
            assert torch.isfinite(loss), "non-finite total loss"

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.update_target()            # EMA after EVERY step, both objectives
            else:
                z = model.encode(tgt_f, tgt_m)
                latents.append(z.reshape(z.shape[0], -1).cpu().numpy())
                preds.append(prediction.reshape(prediction.shape[0], -1).cpu().numpy())

            sums["jepa"] += jepa_term.item()
            sums["con"] += con_term.item()
            sums["total"] += loss.item()
            n += 1
    out = {k: v / max(n, 1) for k, v in sums.items()}
    if latents:
        Z = np.concatenate(latents)
        out["latent_std"] = float(Z.std(axis=0).mean())
        out["effective_rank"] = effective_rank(Z)
        out["pred_std"] = float(np.concatenate(preds).std(axis=0).mean())
    return out


def main():
    os.makedirs(ABL_DIR, exist_ok=True)
    os.makedirs(ABL_CKPT_DIR, exist_ok=True)
    tag = run_tag()
    ckpt_base = os.path.join(ABL_CKPT_DIR, tag)
    print(f"git commit: {git_commit()}")
    print(f"config: objective={OBJECTIVE} weight={WEIGHT} temp={TEMPERATURE} seed={SEED} "
          f"epochs={EPOCHS} batch={BATCH} lr={LR} varw={VARW} device={DEVICE} -> {ckpt_base}_epNNN.pth")
    print(f"data: {S14_DATA}")

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == SECTOR].drop_duplicates("TIC").reset_index(drop=True)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, ART_DIR)
    t_range = ensure_time_range(ART_DIR, df, train_tics)
    print(f"splits: {len(train_tics)} train / {len(val_tics)} val / {len(test_tics)} test (disjoint)")

    train_ds = Sector14ChipPairDataset(df, train_tics, ARM, t_range, return_chip=True)
    val_ds = Sector14ChipPairDataset(df, val_tics, ARM, t_range, return_chip=True)
    assert not (set(train_ds.tics) | set(val_ds.tics)) & test_tics, "test TIC leaked into pretraining"

    train_loader = DataLoader(train_ds, batch_size=BATCH, num_workers=NUM_WORKERS,
                              worker_init_fn=seed_worker,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_ds, batch_size=BATCH, num_workers=NUM_WORKERS,
                            worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(SEED))

    model = build_gapblind_jepa().to(DEVICE)     # dropout 0 by construction
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    metrics_path = os.path.join(ABL_DIR, f"metrics_{tag}.csv")
    fields = ["epoch", "train_jepa", "train_con", "train_total",
              "val_jepa", "val_con", "val_total",
              "latent_std", "effective_rank", "pred_std"]
    with open(metrics_path, "w", newline="") as fh:
        csv.writer(fh).writerow(fields)

    for epoch in range(EPOCHS):
        tr = run_epoch(model, train_loader, optimizer)
        scheduler.step()
        va = run_epoch(model, val_loader)
        row = {"epoch": epoch + 1,
               "train_jepa": tr["jepa"], "train_con": tr["con"], "train_total": tr["total"],
               "val_jepa": va["jepa"], "val_con": va["con"], "val_total": va["total"],
               "latent_std": va.get("latent_std", 0.0),
               "effective_rank": va.get("effective_rank", 0.0),
               "pred_std": va.get("pred_std", 0.0)}
        with open(metrics_path, "a", newline="") as fh:
            csv.writer(fh).writerow([row[k] for k in fields])
        print(f"epoch {epoch + 1}/{EPOCHS}  jepa {tr['jepa']:.5f}/{va['jepa']:.5f}  "
              f"con {tr['con']:.5f}/{va['con']:.5f}  total {tr['total']:.5f}/{va['total']:.5f}  "
              f"std {row['latent_std']:.4f}  erank {row['effective_rank']:.1f}", flush=True)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{ckpt_base}_ep{epoch + 1:03d}.pth")

    print(f"DONE: checkpoints {ckpt_base}_ep010..{EPOCHS:03d}.pth, metrics {metrics_path}")


if __name__ == "__main__":
    main()
