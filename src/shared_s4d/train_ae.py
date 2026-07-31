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
from src.instrument_v2.train_sector14_jepa import git_commit, seed_worker, effective_rank
from src.shared_s4d.ae_dataset import AreaGroupAEDataset
from src.shared_s4d.model import build_model, preprocessing_config, GRID, LATENT_DIM

SEED = int(os.environ.get("SEED", "0"))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "32"))
N_STARS = int(os.environ.get("N_STARS", "1000"))
EPOCHS = int(os.environ.get("EPOCHS", "30"))
LR = float(os.environ.get("LR", "1e-3"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
REQUIRE_FULL = os.environ.get("REQUIRE_FULL", "1").lower() not in ("0", "false", "no")
COLLAPSE_STD = float(os.environ.get("COLLAPSE_STD", "1e-3"))
USE_AMP = os.environ.get("USE_AMP", "0") == "1"                    # S4D complex kernels -> default off
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "0"))            # >0 = smoke
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

S14_DATA = os.environ.get("S14_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2.parquet")
SPLIT_DIR = os.environ.get("SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "dense_v2_split"))
BASE_ART_DIR = os.environ.get("BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa_dense_v2"))
ART_DIR = os.environ.get("ART_DIR", os.path.join("artifacts", "shared_s4d", "autoencoder_v1"))
CKPT_DIR = os.environ.get("CKPT_DIR", "/orcd/scratch/orcd/006/diegogon/checkpoints/shared_s4d_autoencoder_v1")



def masked_smooth_l1(prediction, target, valid, reduction = "mean"):
    per = F.smooth_l1_loss(prediction, target, reduction = "none") * valid
    per_curve = per.sum(dim = 1) / valid.sum(dim = 1).clamp(min = 1.0)

    if reduction == "per_curve":
        return per_curve
    
    has = valid.sum(dim = 1) > 0

    return per_curve[has].mean() if bool(has.any()) else per_curve.sum() * 0.0

def masked_pearson(prediction, target, valid):

    n = valid.sum(dim = 1).clamp(min = 1.0)
    pm = (prediction * valid).sum(1) / n; tm = (target * valid).sum(1) / n
    pc = (prediction - pm[:, None]) * valid; tc = (target - tm[:, None]) * valid

    ps = torch.sqrt((pc * pc).sum(1)); ts = torch.sqrt((tc * tc).sum(1))

    r = (pc * tc).sum(1) / (ps * ts).clamp(min = 1e-8)
    r[(ps < 1e-6) | (ts < 1e-6)] = float("nan")

    return r

def run_train_epoch(model, loader, optimizer, scaler):
    model.train()
    total, nb = 0.0, 0

    for bi, (Xg, Mg) in enumerate(loader):
        if MAX_BATCHES and bi >= MAX_BATCHES:

            break

        Xg = Xg.squeeze(0).to(DEVICE); Mg = Mg.squeeze(0).to(DEVICE)
        optimizer.zero_grad()

        with torch.autocast(device_type = DEVICE.type, enabled = USE_AMP and DEVICE.type == "cuda"):
            reconstruction, latents = model(Xg, Mg)
            losses = masked_smooth_l1(reconstruction, Xg, Mg, reduction = "per_curve")

            group_loss = losses.mean()

        if scaler.is_enabled():
            scaler.scale(group_loss).backward(); scaler.step(optimizer); scaler.update()
        else:
            group_loss.backward(); optimizer.step()

        total += float(group_loss.detach()); nb += 1

    return total / max(1, nb)


def val_metrics(model, loader):

    model.eval()
    losses, rmses, correlations, cadences, lat_chunks = [], [], [], [], []

    with torch.no_grad():
        for bi, (Xg, Mg) in enumerate(loader):

            if MAX_BATCHES and bi >= MAX_BATCHES:
                break

            Xg = Xg.squeeze(0).to(DEVICE); Mg = Mg.squeeze(0).to(DEVICE)

            reconstruction, latitude = model(Xg, Mg)
            v = (Mg > 0).float()

            losses.append(masked_smooth_l1(reconstruction, Xg, Mg, reduction="per_curve").cpu().numpy())
            num = ((reconstruction - Xg) ** 2 * v).sum(1) / v.sum(1).clamp(min=1.0)
            rmses.append(num.sqrt().cpu().numpy())
            correlations.append(masked_pearson(reconstruction, Xg, v).cpu().numpy())
            cadences.append(v.sum(1).cpu().numpy())
            lat_chunks.append(latitude.cpu().numpy())
    L = np.concatenate(losses); R = np.concatenate(rmses)
    C = np.concatenate(correlations); Cad = np.concatenate(cadences); Z = np.concatenate(lat_chunks, 0)
    return {"smooth_l1": float(np.mean(L)), "rmse": float(np.median(R)),
            "corr": float(np.nanmedian(C)), "latent_std": float(Z.std(0).mean()),
            "effective_rank": float(effective_rank(Z)), "valid_cadences": float(np.mean(Cad))}

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs(ART_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)
    tag = f"shared_s4d_ae_g{GROUP_SIZE}_z{LATENT_DIM}_s{SEED}"
    ckpt_base = os.path.join(CKPT_DIR, tag)
    print(f"git {git_commit()}  tag {tag}  device {DEVICE}  require_full {REQUIRE_FULL}  amp {USE_AMP}", flush=True)

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
                                require_full=False, resample=False)
    print(f"train: {len(train_ds.eligible)} areas -> {len(train_ds)} groups/epoch | "
          f"val: {len(val_ds.eligible)} areas -> {len(val_ds)} groups", flush=True)

    model = build_model().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    with open(os.path.join(ART_DIR, "preprocessing.json"), "w") as fh:
        json.dump(preprocessing_config(), fh, indent=2)
    fields = ["epoch", "train_loss", "val_loss", "val_rmse", "val_corr",
              "latent_std", "effective_rank", "valid_cadences"]
    metrics_path = os.path.join(ART_DIR, f"metrics_{tag}.csv")
    with open(metrics_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fields).writeheader()

    best = {"val_loss": float("inf"), "epoch": None}
    collapsed = False
    for epoch in range(1, EPOCHS + 1):
        train_ds.set_epoch(epoch)
        train_ds.assert_contracts()                                   # 31 groups/area, 32 unique, disjoint
        train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=NUM_WORKERS,
                                  worker_init_fn=seed_worker,
                                  generator=torch.Generator().manual_seed(SEED + epoch))
        val_loader = DataLoader(val_ds, batch_size=1, num_workers=NUM_WORKERS,
                                worker_init_fn=seed_worker,
                                generator=torch.Generator().manual_seed(SEED))

        train_loss = run_train_epoch(model, train_loader, optimizer, scaler)
        scheduler.step()
        vm = val_metrics(model, val_loader)

        with open(metrics_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fields).writerow(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": vm["smooth_l1"],
                 "val_rmse": vm["rmse"], "val_corr": vm["corr"], "latent_std": vm["latent_std"],
                 "effective_rank": vm["effective_rank"], "valid_cadences": vm["valid_cadences"]})
        marker = ""
        if vm["smooth_l1"] < best["val_loss"]:
            best = {"val_loss": vm["smooth_l1"], "epoch": epoch, "val_corr": vm["corr"],
                    "val_rmse": vm["rmse"], "effective_rank": vm["effective_rank"]}
            torch.save({"model": model.state_dict(), "config": preprocessing_config(),
                        "epoch": epoch, "val_loss": vm["smooth_l1"]}, f"{ckpt_base}_best.pth")
            torch.save(model.encoder.state_dict(), f"{ckpt_base}_best_encoder.pth")
            torch.save(model.decoder.state_dict(), f"{ckpt_base}_best_decoder.pth")
            marker = " <- best"
        print(f"[epoch {epoch:02d}] train={train_loss:.5f} val={vm['smooth_l1']:.5f} "
              f"rmse={vm['rmse']:.4f} corr={vm['corr']:.4f} latent_std={vm['latent_std']:.4f} "
              f"erank={vm['effective_rank']:.1f} valid_cad={vm['valid_cadences']:.0f}{marker}", flush=True)

        if vm["latent_std"] < COLLAPSE_STD:
            print(f"!! LATENT COLLAPSE: std {vm['latent_std']:.2e} < {COLLAPSE_STD:.1e} -- stopping", flush=True)
            collapsed = True
            break

    selection = {"tag": tag, "seed": SEED, "group_size": GROUP_SIZE, "latent_dim": LATENT_DIM,
                 "n_stars": N_STARS, "epochs": EPOCHS, "require_full": REQUIRE_FULL,
                 "collapsed": collapsed, "best": best, "checkpoint": f"{ckpt_base}_best.pth",
                 "encoder_checkpoint": f"{ckpt_base}_best_encoder.pth",
                 "decoder_checkpoint": f"{ckpt_base}_best_decoder.pth",
                 "preprocessing": os.path.join(ART_DIR, "preprocessing.json"), "git_commit": git_commit()}
    with open(os.path.join(ART_DIR, f"selection_{tag}.json"), "w") as fh:
        json.dump(selection, fh, indent=2, default=float)
    print(json.dumps(selection, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
