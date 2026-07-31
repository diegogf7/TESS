from __future__ import annotations
"""Train the shared-S4D systematics autoencoder on leave-one-out group medians.

masked Smooth-L1 per curve (averaged over valid cadences), then averaged across
all 32 curves x groups. Shared encoder+decoder applied to every curve; a single
loss backprops through both. No teacher/EMA/PCA/MAD/JEPA/latent-matching/variance.
    python -m src.shared_s4d.train
"""

import csv
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.instrument_v2.area_commonmode_dataset import Sector14GroupStatDataset, ensure_area_column
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_sector14_jepa import git_commit, seed_worker
from src.shared_s4d.dataset import AreaGroupLOODataset
from src.shared_s4d.model import build_model, preprocessing_config, GRID, LATENT_DIM

SEED = int(os.environ.get("SEED", "0"))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "32"))
N_STARS = int(os.environ.get("N_STARS", "1000"))
TARGET_MIN_VALID = int(os.environ.get("TARGET_MIN_VALID", "4"))
EPOCHS = int(os.environ.get("EPOCHS", "30"))
LR = float(os.environ.get("LR", "1e-3"))
GROUPS_PER_BATCH = int(os.environ.get("GROUPS_PER_BATCH", "8"))    # groups/batch (x32 curves)
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
REQUIRE_FULL = os.environ.get("REQUIRE_FULL", "1").lower() not in ("0", "false", "no")
COLLAPSE_STD = float(os.environ.get("COLLAPSE_STD", "1e-3"))
USE_AMP = os.environ.get("USE_AMP", "0") == "1"                    # S4D complex kernels -> default off
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "0"))             # >0 = smoke
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

S14_DATA = os.environ.get("S14_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2.parquet")
SPLIT_DIR = os.environ.get("SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "dense_v2_split"))
BASE_ART_DIR = os.environ.get("BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa_dense_v2"))
ART_DIR = os.environ.get("ART_DIR", os.path.join("artifacts", "shared_s4d", "systematics_v1"))
CKPT_DIR = os.environ.get("CKPT_DIR", "/orcd/scratch/orcd/006/diegogon/checkpoints/shared_s4d_systematics_v1")


def masked_smooth_l1(pred, target, valid):
    """Per-curve Smooth-L1 averaged over VALID cadences, then averaged over curves."""
    per = F.smooth_l1_loss(pred, target, reduction="none") * valid
    denom = valid.sum(dim=1).clamp(min=1.0)
    per_curve = per.sum(dim=1) / denom
    has = valid.sum(dim=1) > 0
    return per_curve[has].mean() if bool(has.any()) else per_curve.sum() * 0.0


def run_epoch(model, loader, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    tot, nb, cad_sum = 0.0, 0, 0.0
    lat_sum = lat_sqsum = None
    lat_n = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for bi, batch in enumerate(loader):
            if MAX_BATCHES and bi >= MAX_BATCHES:
                break
            Xg, Mg, Tg, Vg = [t.to(DEVICE) for t in batch]       # (B, 32, L)
            n = Xg.shape[0] * GROUP_SIZE
            x = Xg.reshape(n, GRID); m = Mg.reshape(n, GRID)
            t = Tg.reshape(n, GRID); v = Vg.reshape(n, GRID)
            with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP and DEVICE.type == "cuda"):
                s_hat, latent = model(x, m)                      # shared enc+dec on all 32*B curves
                loss = masked_smooth_l1(s_hat, t, v)
            if training:
                optimizer.zero_grad()
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
            tot += float(loss.detach()); nb += 1
            cad_sum += float(v.sum(dim=1).mean())
            l = latent.detach().float()
            lat_sum = l.sum(0) if lat_sum is None else lat_sum + l.sum(0)
            lat_sqsum = (l * l).sum(0) if lat_sqsum is None else lat_sqsum + (l * l).sum(0)
            lat_n += l.shape[0]
    mean = lat_sum / max(1, lat_n)
    latent_std = float(((lat_sqsum / max(1, lat_n)) - mean * mean).clamp(min=0).sqrt().mean()) if lat_n else 0.0
    return tot / max(1, nb), cad_sum / max(1, nb), latent_std


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs(ART_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)
    tag = f"shared_s4d_g{GROUP_SIZE}_z{LATENT_DIM}_s{SEED}"
    ckpt_base = os.path.join(CKPT_DIR, tag)
    print("================ config ================", flush=True)
    print(f"  git {git_commit()}  tag {tag}  device {DEVICE}", flush=True)
    print(f"  N_STARS {N_STARS}  GROUP {GROUP_SIZE}  target_min_valid {TARGET_MIN_VALID}", flush=True)
    print(f"  EPOCHS {EPOCHS}  LR {LR}  groups/batch {GROUPS_PER_BATCH}  require_full {REQUIRE_FULL}  amp {USE_AMP}", flush=True)
    print("========================================", flush=True)

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    df = ensure_area_column(df)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, train_tics)
    base_tr = Sector14GroupStatDataset(df, train_tics, t_range, "area", GROUP_SIZE, min_valid=16)
    base_va = Sector14GroupStatDataset(df, val_tics, t_range, "area", GROUP_SIZE, min_valid=16)
    assert not (set(base_tr.tics) | set(base_va.tics)) & test_tics, "test TIC leaked"

    train_ds = AreaGroupLOODataset(base_tr.X, base_tr.M, base_tr.areas, base_tr.tics,
                                   n_stars=N_STARS, group_size=GROUP_SIZE,
                                   target_min_valid=TARGET_MIN_VALID, seed=SEED,
                                   require_full=REQUIRE_FULL, resample=True)
    val_ds = AreaGroupLOODataset(base_va.X, base_va.M, base_va.areas, base_va.tics,
                                 n_stars=N_STARS, group_size=GROUP_SIZE,
                                 target_min_valid=TARGET_MIN_VALID, seed=SEED,
                                 require_full=False, resample=False)
    print(f"train: {len(train_ds.eligible)} areas -> {len(train_ds)} groups/epoch | "
          f"val: {len(val_ds.eligible)} areas -> {len(val_ds)} groups", flush=True)

    model = build_model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    with open(os.path.join(ART_DIR, "preprocessing.json"), "w") as fh:
        json.dump(preprocessing_config(), fh, indent=2)
    fields = ["epoch", "train_loss", "val_loss", "val_valid_cadences", "latent_std"]
    metrics_path = os.path.join(ART_DIR, f"metrics_{tag}.csv")
    with open(metrics_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fields).writeheader()

    best = {"val_loss": float("inf"), "epoch": None}
    collapsed = False
    for epoch in range(1, EPOCHS + 1):
        train_ds.set_epoch(epoch)
        train_ds.assert_contracts()
        gen = torch.Generator().manual_seed(SEED + epoch)
        tl = DataLoader(train_ds, batch_size=GROUPS_PER_BATCH, shuffle=True, num_workers=NUM_WORKERS,
                        worker_init_fn=seed_worker, generator=gen, drop_last=True)
        vl = DataLoader(val_ds, batch_size=GROUPS_PER_BATCH, num_workers=NUM_WORKERS,
                        worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(SEED), drop_last=False)

        train_loss, _, _ = run_epoch(model, tl, optimizer, scaler)
        scheduler.step()
        val_loss, val_cad, latent_std = run_epoch(model, vl)

        with open(metrics_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fields).writerow(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                 "val_valid_cadences": val_cad, "latent_std": latent_std})
        marker = ""
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "latent_std": latent_std}
            torch.save({"model": model.state_dict(), "config": preprocessing_config(),
                        "epoch": epoch, "val_loss": val_loss}, f"{ckpt_base}_best.pth")
            torch.save(model.encoder.state_dict(), f"{ckpt_base}_best_encoder.pth")
            torch.save(model.decoder.state_dict(), f"{ckpt_base}_best_decoder.pth")
            marker = " <- best"
        print(f"[epoch {epoch:02d}] train={train_loss:.5f} val={val_loss:.5f} "
              f"valid_cad={val_cad:.0f} latent_std={latent_std:.4f}{marker}", flush=True)

        if latent_std < COLLAPSE_STD:
            print(f"!! LATENT COLLAPSE: std {latent_std:.2e} < {COLLAPSE_STD:.1e} -- stopping", flush=True)
            collapsed = True
            break

    selection = {"tag": tag, "seed": SEED, "group_size": GROUP_SIZE, "latent_dim": LATENT_DIM,
                 "n_stars": N_STARS, "target_min_valid": TARGET_MIN_VALID, "epochs": EPOCHS,
                 "require_full": REQUIRE_FULL, "collapsed": collapsed, "best": best,
                 "checkpoint": f"{ckpt_base}_best.pth",
                 "encoder_checkpoint": f"{ckpt_base}_best_encoder.pth",
                 "decoder_checkpoint": f"{ckpt_base}_best_decoder.pth",
                 "preprocessing": os.path.join(ART_DIR, "preprocessing.json"),
                 "git_commit": git_commit()}
    with open(os.path.join(ART_DIR, f"selection_{tag}.json"), "w") as fh:
        json.dump(selection, fh, indent=2, default=float)
    print(json.dumps(selection, indent=2, default=float), flush=True)
    print(f"BEST epoch {best['epoch']} val_loss {best['val_loss']:.5f}"
          f"{' (COLLAPSED)' if collapsed else ''}", flush=True)


if __name__ == "__main__":
    main()
