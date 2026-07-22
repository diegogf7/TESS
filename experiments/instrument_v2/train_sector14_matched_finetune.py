# All this code is from Claude
"""Matched supervised fine-tuning for the instrument ablation.

Every arm (random / jepa / supcon / hybrid) uses the IDENTICAL S4D encoder
architecture and the identical linear head; the ONLY difference is encoder
initialization. The RNG is reset immediately before head construction with a
seed derived from (seed, target) only, so heads are bit-identical across
arms. Data: raw shared-grid Sector 14, masks, no infilling, class-weighted
cross-entropy, head lr 1e-3, backbone lr from env, 100 epochs AdamW + cosine.
Best-VALIDATION checkpoint saved. Test TICs are never loaded here.

Run:  INIT_ARM=jepa TARGET=camccd SEED=0 BACKBONE_LR=3e-4 \
        python -m src.instrument_v2.train_sector14_matched_finetune
Env:  INIT_ARM, TARGET, SEED, BACKBONE_LR, PRETRAIN_MANIFEST, EPOCHS, BATCH,
      MAX_BATCHES, S14_DATA, SPLIT_DIR, ART_DIR, ABL_DIR, ABL_CKPT_DIR
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

from src.instrument_v2.ablation_config import ARMS, TARGETS
from src.instrument_v2.diagnose_chip_common_signal import chip_index
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range, grid_frame
from src.instrument_v2.train_sector14_jepa import seed_worker
from src.loss_function.gapblind_fix import build_gapblind_jepa

INIT_ARM = os.environ.get("INIT_ARM", "random")
TARGET = os.environ.get("TARGET", "camera")
assert INIT_ARM in ARMS and TARGET in TARGETS
SEED = int(os.environ.get("SEED", "0"))
BACKBONE_LR = float(os.environ.get("BACKBONE_LR", "1e-4"))
HEAD_LR = 1e-3
SECTOR = 14

S14_DATA = os.environ.get("S14_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get("SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
ART_DIR = os.environ.get("ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))
ABL_DIR = os.environ.get("ABL_DIR", os.path.join("artifacts", "instrument_v2", "ablation", os.environ.get("RUN_ID", "dev")))
ABL_CKPT_DIR = os.environ.get("ABL_CKPT_DIR",
                              os.path.join("/orcd/scratch/orcd/006/diegogon/checkpoints",
                                           "ablation", os.environ.get("RUN_ID", "dev")))
PRETRAIN_MANIFEST = os.environ.get("PRETRAIN_MANIFEST",
                                   os.path.join(ABL_DIR, "pretrain_selection.json"))
EPOCHS = int(os.environ.get("EPOCHS", "100"))
BATCH = int(os.environ.get("BATCH", "128"))
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "0"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_supervised_frames(df, train_tics, val_tics, test_tics, t_range, target):
    """Grid ONLY train+val rows and build labels. Test TICs never gridded."""
    tic = df["TIC"].astype(str)
    keep = tic.isin(train_tics) | tic.isin(val_tics)
    sub = df[keep].reset_index(drop=True)
    if set(sub["TIC"].astype(str)) & set(test_tics):
        raise RuntimeError("test TIC present in fine-tuning data -- refusing")
    X, M = grid_frame(sub, "shared", t_range)
    chips = np.array([chip_index(c, d) for c, d in zip(sub["camera"], sub["ccd"])])
    y = chips if target == "camccd" else chips // 4
    is_train = sub["TIC"].astype(str).isin(train_tics).to_numpy()
    return X, M, y, is_train


def make_encoder(arm, seed, manifest_path=PRETRAIN_MANIFEST):
    """Identical architecture for every arm; init differs only by checkpoint."""
    if arm == "random":
        torch.manual_seed(seed)
        return build_gapblind_jepa().target_encoder
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    entry = manifest["arms"][arm][str(seed)]
    model = build_gapblind_jepa()
    model.load_state_dict(torch.load(entry["checkpoint"], map_location="cpu"))
    return model.target_encoder                  # the selected EMA encoder


def head_seed(seed, target):
    return 90000 + seed * 10 + TARGETS.index(target)


def make_head(n_classes, seed, target, n_tokens=16, token_dim=16):
    """RNG reset RIGHT BEFORE construction: identical across arms per (seed, target)."""
    torch.manual_seed(head_seed(seed, target))
    return nn.Linear(n_tokens * token_dim, n_classes)


class FineTuneClassifier(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, flux, mask):
        z = self.encoder(flux.unsqueeze(-1), mask)   # (B, 16, 16)
        return self.head(z.reshape(z.shape[0], -1))


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
    os.makedirs(ABL_DIR, exist_ok=True)
    os.makedirs(os.path.join(ABL_DIR, "finetune_runs"), exist_ok=True)
    os.makedirs(ABL_CKPT_DIR, exist_ok=True)
    tag = f"ft_{INIT_ARM}_{TARGET}_s{SEED}_lr{BACKBONE_LR:g}"
    ckpt_path = os.path.join(ABL_CKPT_DIR, f"{tag}.pth")
    print(f"git commit: {git_commit()}")
    print(f"config: arm={INIT_ARM} target={TARGET} seed={SEED} backbone_lr={BACKBONE_LR} "
          f"head_lr={HEAD_LR} epochs={EPOCHS} -> {ckpt_path}")

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == SECTOR].drop_duplicates("TIC").reset_index(drop=True)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, ART_DIR)
    t_range = ensure_time_range(ART_DIR, df, train_tics)
    X, M, y, is_train = build_supervised_frames(df, train_tics, val_tics, test_tics,
                                                t_range, TARGET)
    n_classes = 16 if TARGET == "camccd" else 4
    counts = np.bincount(y[is_train], minlength=n_classes)
    class_weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / n_classes,
                                 dtype=torch.float32).to(DEVICE)
    print(f"{int(is_train.sum())} train / {int((~is_train).sum())} val, "
          f"{n_classes} classes, counts min {counts.min()} max {counts.max()}")

    def loader(part_mask, shuffle):
        ds = TensorDataset(torch.tensor(X[part_mask]), torch.tensor(M[part_mask]),
                           torch.tensor(y[part_mask], dtype=torch.long))
        return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=NUM_WORKERS,
                          worker_init_fn=seed_worker,
                          generator=torch.Generator().manual_seed(SEED))
    train_loader = loader(is_train, True)
    val_loader = loader(~is_train, False)

    encoder = make_encoder(INIT_ARM, SEED)
    for p in encoder.parameters():
        p.requires_grad = True
    head = make_head(n_classes, SEED, TARGET)
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
              "checkpoint": ckpt_path, "git_commit": git_commit()}
    with open(os.path.join(ABL_DIR, "finetune_runs", f"{tag}.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"BEST val bacc {best_val:.4f} -> {ckpt_path}")


if __name__ == "__main__":
    main()
