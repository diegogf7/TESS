from __future__ import annotations
"""Self-supervised instrument correction: shared S4D -> 32 latent -> MLP -> 1024
correction c; residual r=x-c; minimize squared pairwise RESIDUAL correlation +
lambda_size * normalized correction energy. ONE backward + ONE step per 32-group.
    LAMBDA_SIZE=0.01 python -m src.shared_s4d.train_correction
"""

import csv
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.instrument_v2.area_commonmode_dataset import Sector14GroupStatDataset, ensure_area_column
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_sector14_jepa import git_commit, seed_worker, effective_rank
from src.shared_s4d.ae_dataset import AreaGroupAEDataset
from src.shared_s4d.model import build_model, preprocessing_config, GRID, LATENT_DIM
from src.shared_s4d.correction_losses import (
    masked_pairwise_residual_correlation, normalized_correction_energy, mean_abs_pairwise_corr)

SEED = int(os.environ.get("SEED", "0"))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "32"))
N_STARS = int(os.environ.get("N_STARS", "1000"))
EPOCHS = int(os.environ.get("EPOCHS", "30"))
LR = float(os.environ.get("LR", "1e-3"))
LAMBDA_SIZE = float(os.environ.get("LAMBDA_SIZE", "0.01"))       # test 0.001 / 0.01 / 0.1
MIN_OVERLAP = int(os.environ.get("MIN_OVERLAP", "64"))          # min shared observed cadences per pair
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
REQUIRE_FULL = os.environ.get("REQUIRE_FULL", "1").lower() not in ("0", "false", "no")
COLLAPSE_STD = float(os.environ.get("COLLAPSE_STD", "1e-3"))
USE_AMP = os.environ.get("USE_AMP", "0") == "1"                  # S4D complex kernels -> default off
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "0"))          # >0 = smoke
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

S14_DATA = os.environ.get("S14_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2.parquet")
SPLIT_DIR = os.environ.get("SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "dense_v2_split"))
BASE_ART_DIR = os.environ.get("BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa_dense_v2"))
ART_DIR = os.environ.get("ART_DIR", os.path.join("artifacts", "shared_s4d", "correction_v1"))
CKPT_DIR = os.environ.get("CKPT_DIR", "/orcd/scratch/orcd/006/diegogon/checkpoints/shared_s4d_correction_v1")


def run_train_epoch(model, loader, optimizer, scaler):
    model.train()
    tot = shl = szl = 0.0; nb = 0
    for bi, (Xg, Mg) in enumerate(loader):
        if MAX_BATCHES and bi >= MAX_BATCHES:
            break
        Xg = Xg.squeeze(0).to(DEVICE); Mg = Mg.squeeze(0).to(DEVICE)   # ONE group (32, L)
        optimizer.zero_grad()
        with torch.autocast(device_type=DEVICE.type, enabled=USE_AMP and DEVICE.type == "cuda"):
            corrections, latents = model(Xg, Mg)                      # c_i (32, L)
            residuals = Xg - corrections                              # r_i = x_i - c_i
            shared_loss = masked_pairwise_residual_correlation(residuals, Mg, MIN_OVERLAP)
            size_loss = normalized_correction_energy(corrections, Xg, Mg)
            group_loss = shared_loss + LAMBDA_SIZE * size_loss        # combine, THEN backward
        if scaler.is_enabled():
            scaler.scale(group_loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            group_loss.backward(); optimizer.step()                   # one backward + one step / group
        tot += float(group_loss.detach()); shl += float(shared_loss.detach()); szl += float(size_loss.detach()); nb += 1
    return tot / max(1, nb), shl / max(1, nb), szl / max(1, nb)


def val_metrics(model, loader):
    model.eval()
    befs, afts, ratios, shareds, sizes, lat_chunks = [], [], [], [], [], []
    with torch.no_grad():
        for bi, (Xg, Mg) in enumerate(loader):
            if MAX_BATCHES and bi >= MAX_BATCHES:
                break
            Xg = Xg.squeeze(0).to(DEVICE); Mg = Mg.squeeze(0).to(DEVICE)
            corrections, lat = model(Xg, Mg)
            residuals = Xg - corrections
            befs.append(mean_abs_pairwise_corr(Xg, Mg, MIN_OVERLAP))
            afts.append(mean_abs_pairwise_corr(residuals, Mg, MIN_OVERLAP))
            shareds.append(float(masked_pairwise_residual_correlation(residuals, Mg, MIN_OVERLAP)))
            sizes.append(float(normalized_correction_energy(corrections, Xg, Mg)))
            m = (Mg > 0).float()
            crms = torch.sqrt((m * corrections ** 2).sum(1) / m.sum(1).clamp(min=1.0))
            xrms = torch.sqrt((m * Xg ** 2).sum(1) / m.sum(1).clamp(min=1.0))
            ratios.append(float((crms / xrms.clamp(min=1e-8)).mean()))
            lat_chunks.append(lat.cpu().numpy())
    bef = float(np.nanmean(befs)); aft = float(np.nanmean(afts))
    Z = np.concatenate(lat_chunks, 0); shared = float(np.mean(shareds)); size = float(np.mean(sizes))
    return {"corr_before": bef, "corr_after": aft,
            "pct_reduction": float(100.0 * (bef - aft) / bef) if bef > 1e-8 else float("nan"),
            "corr_rms_over_input_rms": float(np.mean(ratios)),
            "shared_loss": shared, "size_loss": size, "total_loss": shared + LAMBDA_SIZE * size,
            "latent_std": float(Z.std(0).mean()), "effective_rank": float(effective_rank(Z))}


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs(ART_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)
    tag = f"shared_s4d_corr_g{GROUP_SIZE}_z{LATENT_DIM}_lam{LAMBDA_SIZE}_s{SEED}"
    ckpt_base = os.path.join(CKPT_DIR, tag)
    print(f"git {git_commit()}  tag {tag}  device {DEVICE}  lambda_size {LAMBDA_SIZE}  "
          f"min_overlap {MIN_OVERLAP}  require_full {REQUIRE_FULL}  amp {USE_AMP}", flush=True)

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    df = ensure_area_column(df)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, train_tics)
    base_tr = Sector14GroupStatDataset(df, train_tics, t_range, "area", GROUP_SIZE, min_valid=16)
    base_va = Sector14GroupStatDataset(df, val_tics, t_range, "area", GROUP_SIZE, min_valid=16)
    assert not (set(base_tr.tics) | set(base_va.tics)) & test_tics, "test TIC leaked"

    train_ds = AreaGroupAEDataset(base_tr.X, base_tr.M, base_tr.areas, base_tr.tics,
                                  n_stars=N_STARS, group_size=GROUP_SIZE, seed=SEED,
                                  require_full=REQUIRE_FULL, resample=True)
    val_ds = AreaGroupAEDataset(base_va.X, base_va.M, base_va.areas, base_va.tics,
                                n_stars=N_STARS, group_size=GROUP_SIZE, seed=SEED,
                                require_full=False, resample=False)         # fixed deterministic val groups
    print(f"train: {len(train_ds.eligible)} areas -> {len(train_ds)} groups/epoch | "
          f"val: {len(val_ds.eligible)} areas -> {len(val_ds)} groups", flush=True)

    model = build_model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    with open(os.path.join(ART_DIR, "preprocessing.json"), "w") as fh:
        json.dump(preprocessing_config(), fh, indent=2)
    fields = ["epoch", "train_total", "train_shared", "train_size", "val_total", "val_shared",
              "val_size", "corr_before", "corr_after", "pct_reduction", "corr_rms_over_input_rms",
              "latent_std", "effective_rank"]
    metrics_path = os.path.join(ART_DIR, f"metrics_{tag}.csv")
    with open(metrics_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fields).writeheader()

    best = {"val_total": float("inf"), "epoch": None}
    collapsed = False
    for epoch in range(1, EPOCHS + 1):
        train_ds.set_epoch(epoch)
        train_ds.assert_contracts()
        tl = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=NUM_WORKERS,
                        worker_init_fn=seed_worker, generator=torch.Generator().manual_seed(SEED + epoch))
        vl = DataLoader(val_ds, batch_size=1, num_workers=NUM_WORKERS, worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(SEED))

        tr_tot, tr_sh, tr_sz = run_train_epoch(model, tl, optimizer, scaler)
        scheduler.step()
        vm = val_metrics(model, vl)

        with open(metrics_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fields).writerow(
                {"epoch": epoch, "train_total": tr_tot, "train_shared": tr_sh, "train_size": tr_sz,
                 "val_total": vm["total_loss"], "val_shared": vm["shared_loss"], "val_size": vm["size_loss"],
                 "corr_before": vm["corr_before"], "corr_after": vm["corr_after"],
                 "pct_reduction": vm["pct_reduction"], "corr_rms_over_input_rms": vm["corr_rms_over_input_rms"],
                 "latent_std": vm["latent_std"], "effective_rank": vm["effective_rank"]})
        marker = ""
        if vm["total_loss"] < best["val_total"]:
            best = {"val_total": vm["total_loss"], "epoch": epoch, "corr_before": vm["corr_before"],
                    "corr_after": vm["corr_after"], "pct_reduction": vm["pct_reduction"],
                    "corr_rms_over_input_rms": vm["corr_rms_over_input_rms"]}
            torch.save({"model": model.state_dict(), "config": preprocessing_config(),
                        "epoch": epoch, "lambda_size": LAMBDA_SIZE, "val_total": vm["total_loss"]},
                       f"{ckpt_base}_best.pth")
            torch.save(model.encoder.state_dict(), f"{ckpt_base}_best_encoder.pth")
            torch.save(model.decoder.state_dict(), f"{ckpt_base}_best_decoder.pth")
            marker = " <- best"
        print(f"[epoch {epoch:02d}] total={vm['total_loss']:.4f} shared={vm['shared_loss']:.4f} "
              f"size={vm['size_loss']:.4f} corr {vm['corr_before']:.3f}->{vm['corr_after']:.3f} "
              f"(-{vm['pct_reduction']:.0f}%) c/x_rms={vm['corr_rms_over_input_rms']:.3f} "
              f"lstd={vm['latent_std']:.3f} erank={vm['effective_rank']:.1f}{marker}", flush=True)

        if vm["latent_std"] < COLLAPSE_STD:
            print(f"!! LATENT COLLAPSE: std {vm['latent_std']:.2e} < {COLLAPSE_STD:.1e} -- stopping", flush=True)
            collapsed = True
            break

    selection = {"tag": tag, "seed": SEED, "group_size": GROUP_SIZE, "latent_dim": LATENT_DIM,
                 "n_stars": N_STARS, "epochs": EPOCHS, "lambda_size": LAMBDA_SIZE, "min_overlap": MIN_OVERLAP,
                 "require_full": REQUIRE_FULL, "collapsed": collapsed, "best": best,
                 "checkpoint": f"{ckpt_base}_best.pth", "encoder_checkpoint": f"{ckpt_base}_best_encoder.pth",
                 "decoder_checkpoint": f"{ckpt_base}_best_decoder.pth",
                 "preprocessing": os.path.join(ART_DIR, "preprocessing.json"), "git_commit": git_commit()}
    with open(os.path.join(ART_DIR, f"selection_{tag}.json"), "w") as fh:
        json.dump(selection, fh, indent=2, default=float)
    print(json.dumps(selection, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
