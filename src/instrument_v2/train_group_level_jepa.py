"""Low-cost screening trainer for the isolated group-level-latent ablation.

The data, S4D, predictor, JEPA loss, optimizer, and split stay matched to the
Sector-14 pair-JEPA.  Only the pretraining unit changes from one star to the
mean latent of a disjoint same-chip star set.
"""

from __future__ import annotations

import csv
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from src.instrument_v2.group_level_dataset import Sector14ChipGroupDataset
from src.instrument_v2.group_level_jepa import build_groupmean_jepa
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_sector14_jepa import effective_rank, seed_worker
from src.loss_function.gapblind_fix import gapblind_loss


SEED = int(os.environ.get("SEED", "0"))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "8"))
EPOCHS = int(os.environ.get("EPOCHS", "20"))
BATCH = int(os.environ.get("BATCH", str(max(8, 256 // GROUP_SIZE))))
LR = float(os.environ.get("LR", "1e-3"))
VARW = float(os.environ.get("VARW", "0.5"))
PROBE_EVERY = int(os.environ.get("PROBE_EVERY", "2"))
PROBE_VIEW = os.environ.get("PROBE_VIEW", "online")
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


def individual_latents(model, dataset, view):
    pieces = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset.X), 256):
            flux = torch.from_numpy(dataset.X[start : start + 256]).to(DEVICE)
            mask = torch.from_numpy(dataset.M[start : start + 256]).to(DEVICE)
            tokens = model.encode(flux, mask, view=view)
            pieces.append(tokens.flatten(1).cpu().numpy())
    return np.concatenate(pieces)


def fast_probe(train_z, train_y, val_z, val_y):
    n_components = min(64, train_z.shape[0] - 1, train_z.shape[1])
    classifier = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components, random_state=0),
        LogisticRegression(
            max_iter=3000, C=1.0, class_weight="balanced", random_state=0
        ),
    )
    classifier.fit(train_z, train_y)
    return float(balanced_accuracy_score(val_y, classifier.predict(val_z)))


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, latents, predictions, batches = 0.0, [], [], 0
    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for batch_idx, (ctx_f, ctx_m, tgt_f, tgt_m, _) in enumerate(loader):
            if MAX_BATCHES and batch_idx >= MAX_BATCHES:
                break
            ctx_f, ctx_m = ctx_f.to(DEVICE), ctx_m.to(DEVICE)
            tgt_f, tgt_m = tgt_f.to(DEVICE), tgt_m.to(DEVICE)
            prediction, target, context_group, _ = model(ctx_f, ctx_m, tgt_f, tgt_m)
            # Fractional group coverage retains the existing token-weighted loss.
            target_group_mask = tgt_m.float().mean(dim=1)
            loss = gapblind_loss(
                prediction,
                target,
                context_group,
                target_mask=target_group_mask,
                var_weight=VARW,
            )
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.update_target()
            else:
                latents.append(target.flatten(1).cpu().numpy())
                predictions.append(prediction.flatten(1).cpu().numpy())
            total += loss.item()
            batches += 1
    result = {"loss": total / max(1, batches)}
    if latents:
        latent = np.concatenate(latents)
        result.update(
            latent_std=float(latent.std(axis=0).mean()),
            effective_rank=effective_rank(latent),
            pred_std=float(np.concatenate(predictions).std(axis=0).mean()),
        )
    return result


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(ART_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    tag = f"s14groupmean_k{GROUP_SIZE}_s{SEED}"
    best_path = os.path.join(CKPT_DIR, f"{tag}_best.pth")

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    time_range = ensure_time_range(BASE_ART_DIR, df, train_tics)
    train_ds = Sector14ChipGroupDataset(
        df,
        train_tics,
        "shared",
        time_range,
        group_size=GROUP_SIZE,
        return_chip=True,
    )
    val_ds = Sector14ChipGroupDataset(
        df,
        val_tics,
        "shared",
        time_range,
        group_size=GROUP_SIZE,
        return_chip=True,
    )
    assert not (set(train_ds.tics) | set(val_ds.tics)) & test_tics
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH,
        num_workers=NUM_WORKERS,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(SEED),
    )

    model = build_groupmean_jepa().to(DEVICE)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=LR
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    fields = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_probe_bacc16",
        "latent_std",
        "effective_rank",
        "pred_std",
    ]
    metrics_path = os.path.join(ART_DIR, f"metrics_{tag}.csv")
    with open(metrics_path, "w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()

    train_y = train_ds.chips
    val_y = val_ds.chips
    best_probe, best_epoch = -1.0, None
    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, optimizer)
        scheduler.step()
        val_metrics = run_epoch(model, val_loader)
        probe = np.nan
        if epoch == 1 or epoch % PROBE_EVERY == 0 or epoch == EPOCHS:
            train_z = individual_latents(model, train_ds, PROBE_VIEW)
            val_z = individual_latents(model, val_ds, PROBE_VIEW)
            probe = fast_probe(train_z, train_y, val_z, val_y)
            if probe > best_probe:
                best_probe, best_epoch = probe, epoch
                torch.save(model.state_dict(), best_path)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_probe_bacc16": probe,
            "latent_std": val_metrics["latent_std"],
            "effective_rank": val_metrics["effective_rank"],
            "pred_std": val_metrics["pred_std"],
        }
        with open(metrics_path, "a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        print(
            f"{tag} epoch {epoch:03d}: train={row['train_loss']:.5f} "
            f"val={row['val_loss']:.5f} probe={probe:.4f} "
            f"rank={row['effective_rank']:.1f}",
            flush=True,
        )

    selection = {
        "tag": tag,
        "group_size": GROUP_SIZE,
        "seed": SEED,
        "probe_view": PROBE_VIEW,
        "best_val_probe_bacc16": best_probe,
        "best_epoch": best_epoch,
        "checkpoint": best_path,
    }
    with open(os.path.join(ART_DIR, f"selection_{tag}.json"), "w") as handle:
        json.dump(selection, handle, indent=2)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
