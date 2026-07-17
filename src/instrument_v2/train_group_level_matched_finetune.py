"""Matched fine-tuning for random versus group-level-JEPA initialization.

This mirrors ``train_sector14_matched_finetune`` but accepts a group-JEPA
checkpoint and can initialize from either the online or EMA encoder.  Test TICs
are never loaded; the best checkpoint is chosen by validation bacc16 only.
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

from src.instrument_v2.diagnose_chip_common_signal import chip_index
from src.instrument_v2.group_level_jepa import build_groupmean_jepa
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range, grid_frame
from src.instrument_v2.train_sector14_jepa import seed_worker


INIT_ARM = os.environ.get("INIT_ARM", "group")
if INIT_ARM not in {"random", "group"}:
    raise ValueError("INIT_ARM must be random or group")
SEED = int(os.environ.get("SEED", "0"))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "8"))
ENCODER_VIEW = os.environ.get("ENCODER_VIEW", "online")
if ENCODER_VIEW not in {"online", "ema"}:
    raise ValueError("ENCODER_VIEW must be online or ema")
BACKBONE_LR = float(os.environ.get("BACKBONE_LR", "1e-4"))
HEAD_LR = float(os.environ.get("HEAD_LR", "1e-3"))
EPOCHS = int(os.environ.get("EPOCHS", "100"))
BATCH = int(os.environ.get("BATCH", "128"))
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "0"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

S14_DATA = os.environ.get(
    "S14_DATA",
    "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet",
)
SPLIT_DIR = os.environ.get(
    "SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic")
)
BASE_ART_DIR = os.environ.get(
    "BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa")
)
ART_DIR = os.environ.get(
    "GROUP_ART_DIR", os.path.join("artifacts", "instrument_v2", "group_level")
)
CKPT_DIR = os.environ.get(
    "GROUP_CKPT_DIR",
    "/orcd/scratch/orcd/006/diegogon/checkpoints/group_level",
)
SELECTION_PATH = os.environ.get(
    "GROUP_SELECTION",
    os.path.join(ART_DIR, f"selection_s14groupmean_k{GROUP_SIZE}_s{SEED}.json"),
)


def build_frames(df, train_tics, val_tics, test_tics, time_range):
    tic = df["TIC"].astype(str)
    keep = tic.isin(train_tics) | tic.isin(val_tics)
    frame = df[keep].reset_index(drop=True)
    if set(frame["TIC"].astype(str)) & set(test_tics):
        raise RuntimeError("test TIC entered group fine-tuning")
    flux, mask = grid_frame(frame, "shared", time_range)
    labels = np.asarray(
        [chip_index(camera, ccd) for camera, ccd in zip(frame["camera"], frame["ccd"])],
        dtype=np.int64,
    )
    is_train = frame["TIC"].astype(str).isin(train_tics).to_numpy()
    return flux, mask, labels, is_train


def make_encoder():
    if INIT_ARM == "random":
        torch.manual_seed(SEED)
        return build_groupmean_jepa().context_encoder
    with open(SELECTION_PATH) as handle:
        selection = json.load(handle)
    model = build_groupmean_jepa()
    model.load_state_dict(torch.load(selection["checkpoint"], map_location="cpu"))
    return model.context_encoder if ENCODER_VIEW == "online" else model.target_encoder


def make_head():
    # Same seed formula as the existing camccd matched benchmark.
    torch.manual_seed(90000 + SEED * 10 + 1)
    return nn.Linear(16 * 16, 16)


class Classifier(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, flux, mask):
        tokens = self.encoder(flux.unsqueeze(-1), mask)
        return self.head(tokens.flatten(1))


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(ART_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    tag = (
        f"ft_{INIT_ARM}_k{GROUP_SIZE}_{ENCODER_VIEW}_camccd_s{SEED}"
        f"_lr{BACKBONE_LR:g}"
    )
    checkpoint_path = os.path.join(CKPT_DIR, f"{tag}.pth")

    frame = pd.read_parquet(S14_DATA)
    frame = frame[frame["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    time_range = ensure_time_range(BASE_ART_DIR, frame, train_tics)
    flux, mask, labels, is_train = build_frames(
        frame, train_tics, val_tics, test_tics, time_range
    )
    counts = np.bincount(labels[is_train], minlength=16)
    weights = torch.tensor(
        counts.sum() / np.maximum(counts, 1) / 16,
        dtype=torch.float32,
        device=DEVICE,
    )

    def make_loader(select, shuffle):
        dataset = TensorDataset(
            torch.from_numpy(flux[select]),
            torch.from_numpy(mask[select]),
            torch.from_numpy(labels[select]),
        )
        return DataLoader(
            dataset,
            batch_size=BATCH,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(SEED),
        )

    train_loader = make_loader(is_train, True)
    val_loader = make_loader(~is_train, False)
    model = Classifier(make_encoder(), make_head()).to(DEVICE)
    for parameter in model.encoder.parameters():
        parameter.requires_grad = True
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": BACKBONE_LR},
            {"params": model.head.parameters(), "lr": HEAD_LR},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss, batches = 0.0, 0
        for batch_idx, (batch_flux, batch_mask, target) in enumerate(train_loader):
            if MAX_BATCHES and batch_idx >= MAX_BATCHES:
                break
            batch_flux = batch_flux.to(DEVICE)
            batch_mask = batch_mask.to(DEVICE)
            target = target.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_flux, batch_mask), target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batches += 1
        scheduler.step()

        model.eval()
        predicted, actual = [], []
        with torch.no_grad():
            for batch_flux, batch_mask, target in val_loader:
                logits = model(batch_flux.to(DEVICE), batch_mask.to(DEVICE))
                predicted.append(logits.argmax(dim=1).cpu().numpy())
                actual.append(target.numpy())
        val_bacc = float(
            balanced_accuracy_score(np.concatenate(actual), np.concatenate(predicted))
        )
        marker = ""
        if val_bacc > best:
            best = val_bacc
            torch.save(model.state_dict(), checkpoint_path)
            marker = " <- best"
        print(
            f"{tag} epoch {epoch:03d}: loss={train_loss / max(1, batches):.5f} "
            f"val_bacc16={val_bacc:.4f}{marker}",
            flush=True,
        )

    result = {
        "tag": tag,
        "arm": INIT_ARM,
        "group_size": GROUP_SIZE,
        "encoder_view": ENCODER_VIEW,
        "seed": SEED,
        "backbone_lr": BACKBONE_LR,
        "best_val_bacc16": best,
        "checkpoint": checkpoint_path,
    }
    with open(os.path.join(ART_DIR, f"result_{tag}.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
