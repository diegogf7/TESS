# All this code is from Claude
"""Stage B trainer: distill single stars toward the frozen regional teacher.

Seed-0 pilot, validation-only. Every epoch prints the losses, frozen
camera / CCD / camCCD probes, effective rank, latent std, and a teacher
state-hash verification (crashes if the frozen teacher ever changes).

Run:  python -m src.instrument_v2.train_fixed_teacher_instrument_jepa
Env:  SEED, EPOCHS, BATCH, LR, VARW, K, TEACHER_SELECTION, GROUP_SELECTION,
      S14_DATA, SPLIT_DIR, BASE_ART_DIR, FRT_ART_DIR, FRT_CKPT_DIR,
      MAX_BATCHES (smoke), NUM_WORKERS
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

from src.instrument_v2.area_commonmode_dataset import (
    Sector14GroupStatDataset,
    ensure_area_column,
)
from src.instrument_v2.fixed_teacher_instrument_jepa import (
    build_fixed_teacher_jepa,
    fixed_teacher_loss,
    load_frozen_teacher,
    load_student_warmstart,
)
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_group_level_jepa import fast_probe, individual_latents
from src.instrument_v2.train_sector14_jepa import (
    effective_rank,
    git_commit,
    seed_worker,
)

SEED = int(os.environ.get("SEED", "0"))
K = int(os.environ.get("K", "8"))
EPOCHS = int(os.environ.get("EPOCHS", "30"))
PREDICTOR = os.environ.get("PREDICTOR", "mlp")
SELECT_VIEW = os.environ.get("SELECT_VIEW", "online")   # online | predicted
MIN_EPOCHS = int(os.environ.get("MIN_EPOCHS", "1"))
PATIENCE = int(os.environ.get("PATIENCE", "0"))          # 0 = no early stop
BATCH = int(os.environ.get("BATCH", "64"))
LR = float(os.environ.get("LR", "1e-3"))
VARW = float(os.environ.get("VARW", "0.5"))
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
    "FRT_ART_DIR",
    os.path.join("artifacts", "instrument_v2", "fixed_regional_teacher_v1"))
CKPT_DIR = os.environ.get(
    "FRT_CKPT_DIR",
    "/orcd/scratch/orcd/006/diegogon/checkpoints/fixed_regional_teacher_v1")
TEACHER_SELECTION = os.environ.get(
    "TEACHER_SELECTION",
    os.path.join(ART_DIR, f"selection_regteacher_k{K}_s{SEED}.json"))
GROUP_SELECTION = os.environ.get(
    "GROUP_SELECTION",
    os.path.join("artifacts", "instrument_v2", "group_level",
                 f"selection_s14groupmean_k8_s{SEED}.json"))


def stack_stats(median, log_mad):
    return torch.stack([median, log_mad], dim=-1)


def selection_metric(select_view, encoder_bacc, predicted_bacc):
    """Checkpoint-selection and early-stopping metric (camCCD only).
    'online' = the S4D encoder output BEFORE the predictor -- the physics
    frozen-probe benchmark representation. 'predicted' = predictor output."""
    return predicted_bacc if select_view == "predicted" else encoder_bacc


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, batches = 0.0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(loader):
            if MAX_BATCHES and batch_idx >= MAX_BATCHES:
                break
            (ctx_f, ctx_m, median, log_mad, valid, _, _) = \
                [tensor.to(DEVICE) for tensor in batch]
            prediction, target, tokens = model(
                ctx_f, ctx_m, stack_stats(median, log_mad), valid)
            loss = fixed_teacher_loss(prediction, target, tokens, valid,
                                      var_weight=VARW)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # NO EMA update: the teacher is frozen by construction.
            total += float(loss.detach())
            batches += 1
    return total / max(1, batches)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(ART_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    tag = (f"frtstudent_tx_k{K}_s{SEED}" if PREDICTOR == "transformer"
           else f"frtstudent_k{K}_s{SEED}")
    ckpt_base = os.path.join(CKPT_DIR, tag)

    print(f"git commit: {git_commit()}", flush=True)
    print(f"config: {tag} epochs={EPOCHS} batch={BATCH} lr={LR} varw={VARW} "
          f"device={DEVICE}", flush=True)

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    df = ensure_area_column(df)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, train_tics)

    train_ds = Sector14GroupStatDataset(df, train_tics, t_range, "area", K)
    val_ds = Sector14GroupStatDataset(df, val_tics, t_range, "area", K)
    assert not (set(train_ds.tics) | set(val_ds.tics)) & test_tics, \
        "test TIC leaked into student training"

    train_loader = DataLoader(train_ds, batch_size=BATCH,
                              num_workers=NUM_WORKERS,
                              worker_init_fn=seed_worker,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_ds, batch_size=BATCH,
                            num_workers=NUM_WORKERS,
                            worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(SEED))

    model = build_fixed_teacher_jepa().to(DEVICE)
    teacher_info = load_frozen_teacher(model, TEACHER_SELECTION)
    student_info = load_student_warmstart(model, GROUP_SELECTION)
    teacher_hash = model.teacher_hash()
    print(f"frozen teacher: {teacher_info['checkpoint']} "
          f"(hash {teacher_hash[:16]}...)", flush=True)
    print(f"student warm-start: {student_info['checkpoint']}", flush=True)

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=EPOCHS)

    camera = train_ds.chips // 4 + 1
    val_camera = val_ds.chips // 4 + 1
    ccd = train_ds.chips % 4 + 1
    val_ccd = val_ds.chips % 4 + 1

    fields = ["epoch", "train_loss", "val_loss", "val_camccd_bacc",
              "val_pred_camccd_bacc", "val_camera_bacc", "val_ccd_bacc",
              "effective_rank", "pred_effective_rank",
              "latent_std", "teacher_hash_ok"]
    metrics_path = os.path.join(ART_DIR, f"metrics_{tag}.csv")
    with open(metrics_path, "w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()

    best = {"camccd_bacc": -1.0, "select_bacc": -1.0, "epoch": None}
    epochs_since_best = 0
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer)
        scheduler.step()
        val_loss = run_epoch(model, val_loader)

        hash_ok = model.teacher_hash() == teacher_hash
        if not hash_ok:
            raise RuntimeError("FROZEN TEACHER CHANGED -- protocol violation")

        train_z = individual_latents(model, train_ds, "online")
        val_z = individual_latents(model, val_ds, "online")
        train_p = individual_latents(model, train_ds, "predicted")
        val_p = individual_latents(model, val_ds, "predicted")
        camccd_bacc = fast_probe(train_z, train_ds.chips, val_z, val_ds.chips)
        pred_camccd = fast_probe(train_p, train_ds.chips, val_p, val_ds.chips)
        camera_bacc = fast_probe(train_z, camera, val_z, val_camera)
        ccd_bacc = fast_probe(train_z, ccd, val_z, val_ccd)
        erank = effective_rank(val_z)
        pred_erank = effective_rank(val_p)
        latent_std = float(val_z.std(axis=0).mean())

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "val_camccd_bacc": camccd_bacc,
               "val_pred_camccd_bacc": pred_camccd,
               "val_camera_bacc": camera_bacc,
               "val_ccd_bacc": ccd_bacc, "effective_rank": erank,
               "pred_effective_rank": pred_erank,
               "latent_std": latent_std, "teacher_hash_ok": hash_ok}
        with open(metrics_path, "a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        print(f"student epoch {epoch:03d}: train={train_loss:.5f} "
              f"val={val_loss:.5f} camccd={camccd_bacc:.4f} "
              f"pred_camccd={pred_camccd:.4f} camera={camera_bacc:.4f} "
              f"ccd={ccd_bacc:.4f} erank={erank:.1f} "
              f"pred_erank={pred_erank:.1f} std={latent_std:.4f} "
              f"teacher_hash_ok={hash_ok}", flush=True)

        torch.save(model.state_dict(), f"{ckpt_base}_latest.pth")
        if epoch % 5 == 0:
            torch.save(model.state_dict(), f"{ckpt_base}_ep{epoch:03d}.pth")
        # Checkpoint selection metric: camCCD only, on SELECT_VIEW latents.
        select_bacc = selection_metric(SELECT_VIEW, camccd_bacc, pred_camccd)
        if select_bacc > best["select_bacc"]:
            best = {"camccd_bacc": camccd_bacc, "pred_camccd_bacc": pred_camccd,
                    "select_bacc": select_bacc, "select_view": SELECT_VIEW,
                    "epoch": epoch,
                    "camera_bacc": camera_bacc, "ccd_bacc": ccd_bacc,
                    "effective_rank": erank, "pred_effective_rank": pred_erank,
                    "latent_std": latent_std}
            torch.save(model.state_dict(), f"{ckpt_base}_best.pth")
            torch.save(model.student.state_dict(),
                       f"{ckpt_base}_best_student_encoder.pth")
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        if PATIENCE and epoch >= MIN_EPOCHS and epochs_since_best >= PATIENCE:
            print(f"early stop at epoch {epoch} "
                  f"({epochs_since_best} epochs without improvement)", flush=True)
            break

    selection = {"tag": tag, "seed": SEED, "k": K, "epochs": EPOCHS,
                 "predictor_type": PREDICTOR, "select_view": SELECT_VIEW,
                 "best": best, "checkpoint": f"{ckpt_base}_best.pth",
                 "encoder_checkpoint": f"{ckpt_base}_best_student_encoder.pth",
                 "teacher_selection": TEACHER_SELECTION,
                 "teacher_hash": teacher_hash,
                 "teacher_hash_verified_every_epoch": True,
                 "student_warmstart": GROUP_SELECTION,
                 "git_commit": git_commit()}
    with open(os.path.join(ART_DIR, f"selection_{tag}.json"), "w") as handle:
        json.dump(selection, handle, indent=2)
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
