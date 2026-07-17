# All this code is from Claude
"""LP-FT matched fine-tuning for the instance-subspace experiment (camccd).

Phase 1 (linear probe): backbone + instrument projector FROZEN, head only,
HEAD_EPOCHS. Phase 2 (fine-tune): everything unfrozen, FT_EPOCHS, backbone lr
from env. Identical architecture, head seeding, data, splits, and validation
protocol across arms:

  pretrained     -- selected instance-subspace checkpoint (encoder+projector)
  scratch_proj   -- random S4D + random instrument projector (same arch)
  scratch_direct -- random S4D + direct linear head (no projector; the
                    existing abl1-style architecture)

Test TICs are never loaded here.

Run:  INIT_ARM=pretrained ISJ_ARM=instance_cov GROUP_SIZE=8 SEED=0 \
        BACKBONE_LR=3e-5 python -m src.instrument_v2.train_instance_subspace_finetune
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.metrics import balanced_accuracy_score

from src.instrument_v2.instance_subspace_jepa import build_instance_subspace
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_sector14_jepa import seed_worker
from src.instrument_v2.train_sector14_matched_finetune import (
    build_supervised_frames, make_head,
)

# ISJ_INIT (not INIT_ARM): train_sector14_matched_finetune validates INIT_ARM
# at import time and we import its helpers -- namespaced env avoids collision.
INIT_ARM = os.environ.get("ISJ_INIT", "pretrained")
assert INIT_ARM in ("pretrained", "scratch_proj", "scratch_direct")
ISJ_ARM = os.environ.get("ISJ_ARM", "instance_mean")
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "8"))
SEED = int(os.environ.get("SEED", "0"))
BACKBONE_LR = float(os.environ.get("BACKBONE_LR", "3e-5"))
HEAD_LR = 1e-3
HEAD_EPOCHS = int(os.environ.get("HEAD_EPOCHS", "20"))
FT_EPOCHS = int(os.environ.get("FT_EPOCHS", "80"))
BATCH = int(os.environ.get("BATCH", "128"))
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "0"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
TARGET = "camccd"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

S14_DATA = os.environ.get("S14_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get("SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
BASE_ART_DIR = os.environ.get("BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))
ART_DIR = os.environ.get("ISJ_ART_DIR", os.path.join("artifacts", "instrument_v2", "instance_subspace"))
CKPT_DIR = os.environ.get("ISJ_CKPT_DIR", "/orcd/scratch/orcd/006/diegogon/checkpoints/instance_subspace")


class InstrumentClassifier(nn.Module):
    """encoder -> (optional projector) -> linear head. Identical S4D and
    projector modules across pretrained and scratch arms."""

    def __init__(self, backbone, head, use_projector):
        super().__init__()
        self.backbone = backbone                 # InstanceSubspaceJEPA (holds both)
        self.head = head
        self.use_projector = use_projector

    def features(self, flux, mask):
        tokens = self.backbone.context_encoder(flux.unsqueeze(-1), mask).flatten(1)
        if self.use_projector:
            tokens = self.backbone.instrument_projector(tokens)
        return tokens

    def forward(self, flux, mask):
        return self.head(self.features(flux, mask))

    def backbone_parameters(self):
        params = list(self.backbone.context_encoder.parameters())
        if self.use_projector:
            params += list(self.backbone.instrument_projector.parameters())
        return params


def build_arms_model(init_arm, isj_arm, seed):
    """Identical architecture for every arm; only initialization differs."""
    if init_arm == "pretrained":
        sel_path = os.path.join(ART_DIR, f"selection_isj_{isj_arm}_k{GROUP_SIZE}_s{seed}.json")
        with open(sel_path) as fh:
            ckpt = json.load(fh)["checkpoint"]
        model = build_instance_subspace(isj_arm)
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        return model, True
    torch.manual_seed(seed)
    model = build_instance_subspace(isj_arm)     # random init, same arch
    return model, init_arm == "scratch_proj"


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(ART_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    tag = f"ftisj_{INIT_ARM}_{ISJ_ARM}_k{GROUP_SIZE}_s{SEED}_lr{BACKBONE_LR:g}"
    ckpt_path = os.path.join(CKPT_DIR, f"{tag}.pth")
    print(f"config: {tag} head_epochs={HEAD_EPOCHS} ft_epochs={FT_EPOCHS} -> {ckpt_path}")

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    time_range = ensure_time_range(BASE_ART_DIR, df, train_tics)
    X, M, y, is_train = build_supervised_frames(df, train_tics, val_tics, test_tics,
                                                time_range, TARGET)
    counts = np.bincount(y[is_train], minlength=16)
    class_weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / 16,
                                 dtype=torch.float32).to(DEVICE)

    def loader(mask, shuffle):
        ds = TensorDataset(torch.tensor(X[mask]), torch.tensor(M[mask]),
                           torch.tensor(y[mask], dtype=torch.long))
        return DataLoader(ds, batch_size=BATCH, shuffle=shuffle, num_workers=NUM_WORKERS,
                          worker_init_fn=seed_worker,
                          generator=torch.Generator().manual_seed(SEED))
    train_loader, val_loader = loader(is_train, True), loader(~is_train, False)

    backbone, use_projector = build_arms_model(INIT_ARM, ISJ_ARM, SEED)
    head = make_head(16, SEED, TARGET)           # identical head init across arms
    model = InstrumentClassifier(backbone, head, use_projector).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    def evaluate():
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch_idx, (f, m, t) in enumerate(val_loader):
                if MAX_BATCHES and batch_idx >= MAX_BATCHES:
                    break
                logits = model(f.to(DEVICE), m.to(DEVICE))
                preds.append(logits.argmax(dim=1).cpu().numpy())
                trues.append(t.numpy())
        return balanced_accuracy_score(np.concatenate(trues), np.concatenate(preds))

    def train_phase(n_epochs, optimizer, scheduler, phase, best_val):
        for epoch in range(1, n_epochs + 1):
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
            val_bacc = evaluate()
            marker = ""
            if val_bacc > best_val:
                best_val = val_bacc
                torch.save(model.state_dict(), ckpt_path)
                marker = "  <- saved best"
            print(f"{tag} {phase} {epoch:03d}: loss={total / max(1, len(train_loader)):.5f} "
                  f"val_bacc16={val_bacc:.4f}{marker}", flush=True)
        return best_val

    # phase 1: LP -- backbone (and projector) frozen, head only
    for p in model.backbone.parameters():
        p.requires_grad = False
    opt1 = torch.optim.AdamW(model.head.parameters(), lr=HEAD_LR)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=max(HEAD_EPOCHS, 1))
    best_val = train_phase(HEAD_EPOCHS, opt1, sch1, "LP", -1.0)

    # phase 2: FT -- unfreeze the arm's trainable pieces
    for p in model.backbone_parameters():
        p.requires_grad = True
    opt2 = torch.optim.AdamW([
        {"params": model.backbone_parameters(), "lr": BACKBONE_LR},
        {"params": model.head.parameters(), "lr": HEAD_LR}])
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=max(FT_EPOCHS, 1))
    best_val = train_phase(FT_EPOCHS, opt2, sch2, "FT", best_val)

    result = {"tag": tag, "init_arm": INIT_ARM, "isj_arm": ISJ_ARM,
              "group_size": GROUP_SIZE, "seed": SEED, "backbone_lr": BACKBONE_LR,
              "use_projector": use_projector, "best_val_bacc16": best_val,
              "checkpoint": ckpt_path}
    with open(os.path.join(ART_DIR, f"result_{tag}.json"), "w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
