# All this code is from Claude
"""Matched LP-FT for the area-common-mode comparison (validation-only).

Four initialization arms, identical everything else (head init, batches,
splits, architecture, class-weighted loss, schedule):

  scratch    fresh S4D (seeded)
  groupjepa  online encoder of the selected K=8 group-JEPA
  chip_cm    online encoder of the selected chip common-mode JEPA
  area_cm    online encoder of the selected area common-mode JEPA

LP-FT recipe: LP_EPOCHS head-only epochs (encoder frozen), then FT_EPOCHS
full fine-tuning epochs with a separate backbone LR. Best checkpoint and LR
are selected on validation bacc16 only; test TICs are never loaded.

Run:  INIT_ARM=area_cm SEED=0 BACKBONE_LR=3e-5 \
          python -m src.instrument_v2.train_area_commonmode_finetune
Env:  INIT_ARM, SEED, BACKBONE_LR, HEAD_LR, LP_EPOCHS, FT_EPOCHS, BATCH,
      INIT_SELECTION (chip_cm/area_cm selection JSON; overrides default),
      GROUP_SELECTION, S14_DATA, SPLIT_DIR, BASE_ART_DIR, ACM_ART_DIR,
      ACM_CKPT_DIR, MAX_BATCHES (smoke), NUM_WORKERS
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, TensorDataset

from src.instrument_v2.area_commonmode_jepa import build_area_commonmode_jepa
from src.instrument_v2.diagnose_chip_common_signal import chip_index
from src.instrument_v2.group_level_jepa import build_groupmean_jepa
from src.instrument_v2.sector14_dataset import (
    ensure_splits,
    ensure_time_range,
    grid_frame,
)
from src.instrument_v2.train_sector14_jepa import git_commit, seed_worker

ARMS = ("scratch", "groupjepa", "chip_cm", "area_cm", "v1_area", "v2_area",
        "fixed_teacher")
INIT_ARM = os.environ.get("INIT_ARM", "area_cm")
if INIT_ARM not in ARMS:
    raise ValueError(f"INIT_ARM must be one of {ARMS}")
SEED = int(os.environ.get("SEED", "0"))
BACKBONE_LR = float(os.environ.get("BACKBONE_LR", "3e-5"))
HEAD_LR = float(os.environ.get("HEAD_LR", "1e-3"))
LP_EPOCHS = int(os.environ.get("LP_EPOCHS", "20"))
FT_EPOCHS = int(os.environ.get("FT_EPOCHS", "80"))
BATCH = int(os.environ.get("BATCH", "128"))
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "0"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

S14_DATA = os.environ.get(
    "S14_DATA",
    "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get(
    "SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
BASE_ART_DIR = os.environ.get(
    "BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))
ART_DIR = os.environ.get(
    "ACM_ART_DIR", os.path.join("artifacts", "instrument_v2", "area_commonmode_v1"))
CKPT_DIR = os.environ.get(
    "ACM_CKPT_DIR", "/orcd/scratch/orcd/006/diegogon/checkpoints/area_commonmode")
GROUP_SELECTION = os.environ.get(
    "GROUP_SELECTION",
    os.path.join("artifacts", "instrument_v2", "group_level",
                 f"selection_s14groupmean_k8_s{SEED}.json"))
INIT_SELECTION = os.environ.get("INIT_SELECTION", "")


def build_frames(df, train_tics, val_tics, test_tics, time_range):
    """Gridded train+val curves with chip labels (same recipe as the group-
    level matched finetune; inlined because that module validates ITS env
    vars at import time and rejects our arm names)."""
    tic = df["TIC"].astype(str)
    keep = tic.isin(train_tics) | tic.isin(val_tics)
    frame = df[keep].reset_index(drop=True)
    if set(frame["TIC"].astype(str)) & set(test_tics):
        raise RuntimeError("test TIC entered fine-tuning")
    flux, mask = grid_frame(frame, "shared", time_range)
    labels = np.asarray(
        [chip_index(camera, ccd)
         for camera, ccd in zip(frame["camera"], frame["ccd"])],
        dtype=np.int64)
    is_train = frame["TIC"].astype(str).isin(train_tics).to_numpy()
    return flux, mask, labels, is_train


def make_encoder():
    """Encoder per arm. scratch is seeded; SSL arms load the ONLINE encoder."""
    if INIT_ARM == "scratch":
        torch.manual_seed(SEED)
        return build_area_commonmode_jepa().context_encoder
    if INIT_ARM == "groupjepa":
        with open(GROUP_SELECTION) as handle:
            selection = json.load(handle)
        model = build_groupmean_jepa()
        model.load_state_dict(torch.load(selection["checkpoint"],
                                         map_location="cpu"))
        return model.context_encoder
    if not INIT_SELECTION:
        raise RuntimeError(f"INIT_SELECTION required for arm {INIT_ARM}")
    with open(INIT_SELECTION) as handle:
        selection = json.load(handle)
    if selection.get("skipped"):
        raise RuntimeError(f"selection {INIT_SELECTION} was skipped")
    if INIT_ARM == "fixed_teacher":
        # Stage-B student saved its plain S4Model encoder state separately.
        encoder = build_area_commonmode_jepa().context_encoder
        encoder.load_state_dict(torch.load(selection["encoder_checkpoint"],
                                           map_location="cpu"))
        return encoder
    model = build_area_commonmode_jepa()
    model.load_state_dict(torch.load(selection["checkpoint"],
                                     map_location="cpu"))
    return model.context_encoder


def make_head():
    # Identical head init across arms: same formula as the existing matched
    # camccd benchmarks, keyed only on seed.
    torch.manual_seed(90000 + SEED * 10 + 1)
    return nn.Linear(16 * 16, 16)


class Classifier(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, flux, mask):
        return self.head(self.encoder(flux.unsqueeze(-1), mask).flatten(1))


def evaluate(model, loader):
    model.eval()
    predicted, actual = [], []
    with torch.no_grad():
        for flux, mask, target in loader:
            logits = model(flux.to(DEVICE), mask.to(DEVICE))
            predicted.append(logits.argmax(dim=1).cpu().numpy())
            actual.append(target.numpy())
    return float(balanced_accuracy_score(np.concatenate(actual),
                                         np.concatenate(predicted)))


def train_phase(model, loader, val_loader, criterion, optimizer, scheduler,
                epochs, phase, tag, checkpoint_path, best):
    for epoch in range(1, epochs + 1):
        model.train()
        total, batches = 0.0, 0
        for batch_idx, (flux, mask, target) in enumerate(loader):
            if MAX_BATCHES and batch_idx >= MAX_BATCHES:
                break
            optimizer.zero_grad()
            loss = criterion(model(flux.to(DEVICE), mask.to(DEVICE)),
                             target.to(DEVICE))
            loss.backward()
            optimizer.step()
            total += loss.item()
            batches += 1
        if scheduler is not None:
            scheduler.step()
        val_bacc = evaluate(model, val_loader)
        marker = ""
        if val_bacc > best["bacc"]:
            best.update(bacc=val_bacc, phase=phase, epoch=epoch)
            torch.save(model.state_dict(), checkpoint_path)
            marker = " <- best"
        print(f"{tag} {phase} epoch {epoch:03d}: "
              f"loss={total / max(1, batches):.5f} "
              f"val_bacc16={val_bacc:.4f}{marker}", flush=True)
    return best


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(ART_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    tag = f"lpft_{INIT_ARM}_s{SEED}_lr{BACKBONE_LR:g}"
    checkpoint_path = os.path.join(CKPT_DIR, f"{tag}.pth")
    print(f"git commit: {git_commit()}", flush=True)
    print(f"config: {tag} lp={LP_EPOCHS} ft={FT_EPOCHS} head_lr={HEAD_LR} "
          f"device={DEVICE}", flush=True)

    frame = pd.read_parquet(S14_DATA)
    frame = frame[frame["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    time_range = ensure_time_range(BASE_ART_DIR, frame, train_tics)
    flux, mask, labels, is_train = build_frames(
        frame, train_tics, val_tics, test_tics, time_range)

    counts = np.bincount(labels[is_train], minlength=16)
    weights = torch.tensor(counts.sum() / np.maximum(counts, 1) / 16,
                           dtype=torch.float32, device=DEVICE)

    def make_loader(select, shuffle):
        dataset = TensorDataset(torch.from_numpy(flux[select]),
                                torch.from_numpy(mask[select]),
                                torch.from_numpy(labels[select]))
        return DataLoader(dataset, batch_size=BATCH, shuffle=shuffle,
                          num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
                          generator=torch.Generator().manual_seed(SEED))

    train_loader = make_loader(is_train, True)
    val_loader = make_loader(~is_train, False)
    model = Classifier(make_encoder(), make_head()).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights)
    best = {"bacc": -1.0, "phase": None, "epoch": None}

    # ---- phase 1: linear probe (encoder frozen, head only) ----
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    head_optimizer = torch.optim.AdamW(model.head.parameters(), lr=HEAD_LR)
    best = train_phase(model, train_loader, val_loader, criterion,
                       head_optimizer, None, LP_EPOCHS, "lp", tag,
                       checkpoint_path, best)

    # ---- phase 2: full fine-tuning ----
    for parameter in model.encoder.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": BACKBONE_LR},
        {"params": model.head.parameters(), "lr": HEAD_LR},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=FT_EPOCHS)
    best = train_phase(model, train_loader, val_loader, criterion, optimizer,
                       scheduler, FT_EPOCHS, "ft", tag, checkpoint_path, best)

    result = {"tag": tag, "arm": INIT_ARM, "seed": SEED,
              "backbone_lr": BACKBONE_LR, "head_lr": HEAD_LR,
              "lp_epochs": LP_EPOCHS, "ft_epochs": FT_EPOCHS,
              "best_val_bacc16": best["bacc"], "best_phase": best["phase"],
              "best_epoch": best["epoch"], "init_selection": INIT_SELECTION or
              (GROUP_SELECTION if INIT_ARM == "groupjepa" else None),
              "checkpoint": checkpoint_path, "git_commit": git_commit()}
    with open(os.path.join(ART_DIR, f"result_{tag}.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
