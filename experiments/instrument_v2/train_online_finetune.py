# All this code is from Claude
"""Matched fine-tuning from the ONLINE (context) encoder. camccd only.

Identical to train_sector14_matched_finetune.py in every respect (data, grid,
masks, class weighting, head construction+seeding, optimizer, epochs,
validation protocol) EXCEPT the encoder initialization comes from the
`context_encoder` selected in online_pretrain_selection.json. The historical
script is untouched; abl1 checkpoints are never overwritten.

Run:  INIT_ARM=supcon SEED=0 BACKBONE_LR=3e-4 \
        python -m src.instrument_v2.train_online_finetune
Env:  INIT_ARM (jepa|supcon|hybrid), SEED, BACKBONE_LR, ONLINE_MANIFEST,
      NEW_RUN, AUDIT_CKPT_DIR, EPOCHS, BATCH, MAX_BATCHES + data env
"""

import json
import os
import random
import subprocess

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import balanced_accuracy_score

from src.instrument_v2.ablation_config import ONLINE_FT_ARMS
from src.instrument_v2.encoder_source import extract_encoder
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_sector14_jepa import seed_worker
from src.instrument_v2.train_sector14_matched_finetune import (
    FineTuneClassifier, build_supervised_frames, make_head,
)
from src.loss_function.gapblind_fix import build_gapblind_jepa

INIT_ARM = os.environ.get("INIT_ARM", "supcon")
assert INIT_ARM in ONLINE_FT_ARMS, f"bad INIT_ARM {INIT_ARM!r}"
TARGET = "camccd"                                # primary target only
SEED = int(os.environ.get("SEED", "0"))
BACKBONE_LR = float(os.environ.get("BACKBONE_LR", "1e-4"))
HEAD_LR = 1e-3
SECTOR = 14

S14_DATA = os.environ.get("S14_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get("SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
ART_DIR = os.environ.get("ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))
NEW_RUN = os.environ.get("NEW_RUN", os.path.join("artifacts", "instrument_v2", "ablation", "abl1_encoder_audit"))
AUDIT_CKPT_DIR = os.environ.get("AUDIT_CKPT_DIR",
                                os.path.join("/orcd/scratch/orcd/006/diegogon/checkpoints",
                                             "ablation", "abl1_encoder_audit"))
ONLINE_MANIFEST = os.environ.get("ONLINE_MANIFEST",
                                 os.path.join(NEW_RUN, "online_pretrain_selection.json"))
EPOCHS = int(os.environ.get("EPOCHS", "100"))
BATCH = int(os.environ.get("BATCH", "128"))
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "0"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_online_encoder(arm, seed, manifest_path=ONLINE_MANIFEST):
    """The selected ONLINE context encoder (never the EMA target copy)."""
    with open(manifest_path) as fh:
        entry = json.load(fh)["arms"][arm][str(seed)]
    model = build_gapblind_jepa()
    model.load_state_dict(torch.load(entry["checkpoint"], map_location="cpu"))
    return extract_encoder(model, "online")


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(os.path.join(NEW_RUN, "finetune_runs"), exist_ok=True)
    os.makedirs(AUDIT_CKPT_DIR, exist_ok=True)
    tag = f"ftonline_{INIT_ARM}_{TARGET}_s{SEED}_lr{BACKBONE_LR:g}"
    ckpt_path = os.path.join(AUDIT_CKPT_DIR, f"{tag}.pth")
    print(f"git commit: {git_commit()}")
    print(f"config: arm={INIT_ARM} (ONLINE encoder) target={TARGET} seed={SEED} "
          f"backbone_lr={BACKBONE_LR} head_lr={HEAD_LR} epochs={EPOCHS} -> {ckpt_path}")

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == SECTOR].drop_duplicates("TIC").reset_index(drop=True)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, ART_DIR)
    t_range = ensure_time_range(ART_DIR, df, train_tics)
    X, M, y, is_train = build_supervised_frames(df, train_tics, val_tics, test_tics,
                                                t_range, TARGET)
    n_classes = 16
    counts = np.bincount(y[is_train], minlength=n_classes)
    class_weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / n_classes,
                                 dtype=torch.float32).to(DEVICE)
    print(f"{int(is_train.sum())} train / {int((~is_train).sum())} val")

    def loader(mask, shuffle):
        ds = TensorDataset(torch.tensor(X[mask]), torch.tensor(M[mask]),
                           torch.tensor(y[mask], dtype=torch.long))
        return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=NUM_WORKERS,
                          worker_init_fn=seed_worker,
                          generator=torch.Generator().manual_seed(SEED))
    train_loader = loader(is_train, True)
    val_loader = loader(~is_train, False)

    encoder = make_online_encoder(INIT_ARM, SEED)
    for p in encoder.parameters():
        p.requires_grad = True
    head = make_head(n_classes, SEED, TARGET)    # identical head init as abl1
    model = FineTuneClassifier(encoder, head).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": BACKBONE_LR},
        {"params": model.head.parameters(), "lr": HEAD_LR}])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val = -1.0
    for epoch in range(EPOCHS):
        model.train()
        total = 0.0
        for batch_idx, (f, m, t) in enumerate(train_loader):
            if MAX_BATCHES and batch_idx >= MAX_BATCHES:
                break
            f, m, t = f.to(DEVICE), m.to(DEVICE), t.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(f, m), t)
            loss.backward()
            optimizer.step()
            total += loss.item()
        scheduler.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch_idx, (f, m, t) in enumerate(val_loader):
                if MAX_BATCHES and batch_idx >= MAX_BATCHES:
                    break
                logits = model(f.to(DEVICE), m.to(DEVICE))
                preds.append(logits.argmax(dim=1).cpu().numpy())
                trues.append(t.numpy())
        val_bacc = balanced_accuracy_score(np.concatenate(trues), np.concatenate(preds))
        marker = ""
        if val_bacc > best_val:
            best_val = val_bacc
            torch.save(model.state_dict(), ckpt_path)
            marker = "  <- saved best"
        print(f"epoch {epoch + 1}/{EPOCHS}  loss {total / max(len(train_loader), 1):.4f}  "
              f"val bacc {val_bacc:.4f}{marker}", flush=True)

    result = {"tag": tag, "arm": INIT_ARM, "target": TARGET, "seed": SEED,
              "backbone_lr": BACKBONE_LR, "best_val_bacc": best_val,
              "checkpoint": ckpt_path, "encoder_source": "online",
              "git_commit": git_commit()}
    with open(os.path.join(NEW_RUN, "finetune_runs", f"{tag}.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"BEST val bacc {best_val:.4f} -> {ckpt_path}")


if __name__ == "__main__":
    main()
