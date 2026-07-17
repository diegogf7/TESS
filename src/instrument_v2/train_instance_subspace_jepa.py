# All this code is from Claude
"""Validation-gated screening trainer for the Instance-to-Subspace JEPA.

Data, splits, grid, chip-uniform disjoint-set sampling and compute budget are
matched to the group-level trainer; only the objective changes by ARM. The
frozen probe (and everything downstream) reads encode_instrument() -- the
individual-star projector output -- because that is the representation the
losses now constrain directly.

Also records, per evaluated epoch: frozen val bacc16, effective rank,
per-dimension std, disjoint-set target cosine, and the three loss components.
The matched random baseline (same architecture, untrained) is probed once at
epoch 0 and stored in the selection json for the promotion rule.

Run:  ARM=instance_cov GROUP_SIZE=8 SEED=0 EPOCHS=20 \
        python -m src.instrument_v2.train_instance_subspace_jepa
Env:  ARM, GROUP_SIZE, SEED, EPOCHS, PROBE_EVERY, MAX_BATCHES, ISJ_ART_DIR,
      ISJ_CKPT_DIR + shared data env. Test TICs are never loaded.
"""

from __future__ import annotations

import csv
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.instrument_v2.group_level_dataset import Sector14ChipGroupDataset
from src.instrument_v2.instance_subspace_jepa import ARMS, build_instance_subspace
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_group_level_jepa import fast_probe
from src.instrument_v2.train_sector14_jepa import effective_rank, seed_worker

# ISJ_ARM (not ARM): the historical sector14 trainer validates ARM at import,
# and we import helpers from it -- namespaced env avoids the collision.
ARM = os.environ.get("ISJ_ARM", "instance_mean")
assert ARM in ARMS, f"bad ISJ_ARM {ARM!r}"
SEED = int(os.environ.get("SEED", "0"))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "8"))
EPOCHS = int(os.environ.get("EPOCHS", "20"))
BATCH = int(os.environ.get("BATCH", str(max(8, 256 // GROUP_SIZE))))
LR = float(os.environ.get("LR", "1e-3"))
PROBE_EVERY = int(os.environ.get("PROBE_EVERY", "2"))
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "0"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

S14_DATA = os.environ.get("S14_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get("SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
BASE_ART_DIR = os.environ.get("BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))
ART_DIR = os.environ.get("ISJ_ART_DIR", os.path.join("artifacts", "instrument_v2", "instance_subspace"))
CKPT_DIR = os.environ.get("ISJ_CKPT_DIR", "/orcd/scratch/orcd/006/diegogon/checkpoints/instance_subspace")


def instrument_latents(model, dataset):
    """Individual-star instrument representations via encode_instrument()."""
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(dataset.X), 256):
            flux = torch.from_numpy(dataset.X[start:start + 256]).to(DEVICE)
            mask = torch.from_numpy(dataset.M[start:start + 256]).to(DEVICE)
            pieces.append(model.encode_instrument(flux, mask, source="online").cpu().numpy())
    return np.concatenate(pieces)


def probe_and_stats(model, train_ds, val_ds):
    train_z = instrument_latents(model, train_ds)
    val_z = instrument_latents(model, val_ds)
    bacc = fast_probe(train_z, train_ds.chips, val_z, val_ds.chips)
    return {"val_probe_bacc16": bacc,
            "effective_rank": effective_rank(val_z),
            "per_dim_std": float(val_z.std(axis=0).mean())}


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    sums = {"loss": 0.0, "pred_loss": 0.0, "var_loss": 0.0, "cov_loss": 0.0}
    cosines, batches = [], 0
    grad_ctx = torch.enable_grad() if training else torch.no_grad()
    with grad_ctx:
        for batch_idx, (ctx_f, ctx_m, tgt_f, tgt_m, _) in enumerate(loader):
            if MAX_BATCHES and batch_idx >= MAX_BATCHES:
                break
            ctx_f, ctx_m = ctx_f.to(DEVICE), ctx_m.to(DEVICE)
            tgt_f, tgt_m = tgt_f.to(DEVICE), tgt_m.to(DEVICE)
            out = model(ctx_f, ctx_m, tgt_f, tgt_m)
            assert torch.isfinite(out["loss"]), "non-finite loss"
            if training:
                optimizer.zero_grad()
                out["loss"].backward()
                optimizer.step()
                model.update_target()
            else:
                # disjoint-set target agreement: context vs target set codes
                cosines.append(model.target_cosine(ctx_f, ctx_m, tgt_f, tgt_m))
            for k in sums:
                sums[k] += float(out[k])
            batches += 1
    result = {k: v / max(1, batches) for k, v in sums.items()}
    if cosines:
        result["target_cosine"] = float(np.mean(cosines))
    return result


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(ART_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    tag = f"isj_{ARM}_k{GROUP_SIZE}_s{SEED}"
    best_path = os.path.join(CKPT_DIR, f"{tag}_best.pth")

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    time_range = ensure_time_range(BASE_ART_DIR, df, train_tics)
    train_ds = Sector14ChipGroupDataset(df, train_tics, "shared", time_range,
                                        group_size=GROUP_SIZE, return_chip=True)
    val_ds = Sector14ChipGroupDataset(df, val_tics, "shared", time_range,
                                      group_size=GROUP_SIZE, return_chip=True)
    assert not (set(train_ds.tics) | set(val_ds.tics)) & test_tics, \
        "test TIC leaked into pretraining"

    train_loader = DataLoader(train_ds, batch_size=BATCH, num_workers=NUM_WORKERS,
                              worker_init_fn=seed_worker,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_ds, batch_size=BATCH, num_workers=NUM_WORKERS,
                            worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(SEED))

    model = build_instance_subspace(ARM).to(DEVICE)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # matched random baseline: SAME architecture, untrained, probed once
    random_stats = probe_and_stats(model, train_ds, val_ds)
    print(f"{tag} epoch 000 (RANDOM baseline): probe={random_stats['val_probe_bacc16']:.4f} "
          f"rank={random_stats['effective_rank']:.1f}", flush=True)

    fields = ["epoch", "train_loss", "train_pred", "train_var", "train_cov",
              "val_loss", "val_pred", "target_cosine", "val_probe_bacc16",
              "effective_rank", "per_dim_std"]
    metrics_path = os.path.join(ART_DIR, f"metrics_{tag}.csv")
    with open(metrics_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fields).writeheader()

    best_probe, best_epoch, best_stats = -1.0, None, None
    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, optimizer)
        scheduler.step()
        va = run_epoch(model, val_loader)
        stats = {"val_probe_bacc16": np.nan, "effective_rank": np.nan,
                 "per_dim_std": np.nan}
        if epoch == 1 or epoch % PROBE_EVERY == 0 or epoch == EPOCHS:
            stats = probe_and_stats(model, train_ds, val_ds)
            if stats["val_probe_bacc16"] > best_probe:
                best_probe = stats["val_probe_bacc16"]
                best_epoch, best_stats = epoch, stats
                torch.save(model.state_dict(), best_path)
        row = {"epoch": epoch, "train_loss": tr["loss"], "train_pred": tr["pred_loss"],
               "train_var": tr["var_loss"], "train_cov": tr["cov_loss"],
               "val_loss": va["loss"], "val_pred": va["pred_loss"],
               "target_cosine": va.get("target_cosine", np.nan), **stats}
        with open(metrics_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fields).writerow(row)
        print(f"{tag} epoch {epoch:03d}: loss={row['train_loss']:.5f} "
              f"pred={row['train_pred']:.5f} var={row['train_var']:.4f} "
              f"cov={row['train_cov']:.4f} tgt_cos={row['target_cosine']:.4f} "
              f"probe={row['val_probe_bacc16']:.4f} rank={row['effective_rank']:.1f}",
              flush=True)

    selection = {"tag": tag, "arm": ARM, "group_size": GROUP_SIZE, "seed": SEED,
                 "best_val_probe_bacc16": best_probe, "best_epoch": best_epoch,
                 "best_effective_rank": (best_stats or {}).get("effective_rank"),
                 "random_probe_bacc16": random_stats["val_probe_bacc16"],
                 "random_effective_rank": random_stats["effective_rank"],
                 "checkpoint": best_path}
    with open(os.path.join(ART_DIR, f"selection_{tag}.json"), "w") as fh:
        json.dump(selection, fh, indent=2)
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
