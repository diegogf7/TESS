from __future__ import annotations
"""Retrain the Sector-14 instrument Stage-B student (S4D encoder + MLP predictor)
with DYNAMIC per-area group sampling: 1,000 context stars per eligible area per
epoch, each with a fresh random 32-star same-area group (context excluded),
resampled every epoch from deterministic seeds.

UNCHANGED and reused: architecture, preprocessing, splits, K=8 area CBVs, the
frozen regional S4D teacher (loaded from the existing Stage-A selection, never
retrained), the instrument-JEPA loss, and hyperparameters. Only the training-set
SAMPLING and the checkpoint-selection score change.

Selection: the epoch with the best (lowest) predicted-latent validation loss --
i.e. how well the predictor matches the frozen teacher on a fixed deterministic
validation set. The full model (student + frozen teacher + predictor) is saved so
the decoder stage can load it as a FixedTeacherInstrumentJEPA.

    python -m src.instrument_v2.train_instrument_dynamic_groups
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
from src.instrument_v2.dynamic_group_dataset import DynamicAreaGroupDataset
from src.instrument_v2.fixed_teacher_instrument_jepa import (
    FixedTeacherInstrumentJEPA, fixed_teacher_loss, load_frozen_teacher,
)
from src.instrument_v2.regional_cbv import build_or_load_area_bases
from src.instrument_v2.regional_group_teacher import state_hash
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_sector14_jepa import git_commit, seed_worker

SEED = int(os.environ.get("SEED", "0"))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "32"))
CBV_RANK = int(os.environ.get("CBV_RANK", "8"))
MIN_VALID_STARS = int(os.environ.get("MIN_VALID_STARS", "16"))
assert 1 <= MIN_VALID_STARS <= GROUP_SIZE, (MIN_VALID_STARS, GROUP_SIZE)
N_CONTEXT = int(os.environ.get("N_CONTEXT", "1000"))         # context stars per area per epoch
REQUIRE_FULL = os.environ.get("REQUIRE_FULL", "1").lower() not in ("0", "false", "no")  # 0 = keep areas < N_CONTEXT (use min(N_CONTEXT, available))
EPOCHS = int(os.environ.get("EPOCHS", "15"))
LR = float(os.environ.get("LR", "1e-3"))
VARW = float(os.environ.get("VARW", "0.5"))
BATCH = int(os.environ.get("BATCH", "64"))
RIDGE_LAMBDA = float(os.environ.get("RIDGE_LAMBDA", "1e-2"))
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "0"))         # >0 = smoke
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

S14_DATA = os.environ.get(
    "S14_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2.parquet")
SPLIT_DIR = os.environ.get("SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "dense_v2_split"))
BASE_ART_DIR = os.environ.get("BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa_dense_v2"))
# existing (frozen) pipeline supplies the CBV bases + Stage-A teacher selection
SRC_ART_DIR = os.environ.get("SRC_ART_DIR", os.path.join("artifacts", "instrument_v2", "custom_group32_cbv8_mlp_qclean_v1"))
TEACHER_SELECTION = os.environ.get(
    "TEACHER_SELECTION",
    os.path.join(SRC_ART_DIR, f"selection_regteacher_cbv_g{GROUP_SIZE}_r{CBV_RANK}_mv{MIN_VALID_STARS}_s{SEED}.json"))
# new outputs (never overwrite the existing pipeline)
ART_DIR = os.environ.get("GROUP_ART_DIR", os.path.join("artifacts", "instrument_v2", "custom_group32_cbv8_mlp_dynamic1000_v1"))
CKPT_DIR = os.environ.get("CKPT_DIR", "/orcd/scratch/orcd/006/diegogon/checkpoints/custom_group32_cbv8_mlp_dynamic1000_v1")


def stack_stats(median, log_mad):
    return torch.stack([median, log_mad], dim=-1)


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, batches = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for bi, batch in enumerate(loader):
            if MAX_BATCHES and bi >= MAX_BATCHES:
                break
            ctx_f, ctx_m, median, log_mad, valid, _, _ = [t.to(DEVICE) for t in batch]
            prediction, target, tokens = model(ctx_f, ctx_m, stack_stats(median, log_mad), valid)
            loss = fixed_teacher_loss(prediction, target, tokens, valid, var_weight=VARW)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()                                 # NO EMA: teacher frozen
            total += float(loss.detach())
            batches += 1
    return total / max(1, batches)


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs(ART_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)
    tag = f"group_cbv_mlp_dyn{N_CONTEXT}_g{GROUP_SIZE}_r{CBV_RANK}_mv{MIN_VALID_STARS}_s{SEED}"
    ckpt_base = os.path.join(CKPT_DIR, tag)
    print("================ resolved configuration ================", flush=True)
    print(f"  git commit    : {git_commit()}", flush=True)
    print(f"  tag           : {tag}", flush=True)
    print(f"  N_CONTEXT/GROUP/MINVALID: {N_CONTEXT}/{GROUP_SIZE}/{MIN_VALID_STARS}", flush=True)
    print(f"  EPOCHS/LR/VARW/BATCH    : {EPOCHS}/{LR}/{VARW}/{BATCH}", flush=True)
    print(f"  S14_DATA      : {S14_DATA}", flush=True)
    print(f"  TEACHER_SEL   : {TEACHER_SELECTION}", flush=True)
    print(f"  ART_DIR/CKPT  : {ART_DIR} | {CKPT_DIR}", flush=True)
    print("========================================================", flush=True)

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    df = ensure_area_column(df)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, train_tics)

    # gridded arrays (reuse the exact preprocessing) for train + val
    base_train = Sector14GroupStatDataset(df, train_tics, t_range, "area", GROUP_SIZE, min_valid=MIN_VALID_STARS)
    base_val = Sector14GroupStatDataset(df, val_tics, t_range, "area", GROUP_SIZE, min_valid=MIN_VALID_STARS)
    train_set, val_set = set(base_train.tics), set(base_val.tics)
    assert not (train_set | val_set) & test_tics, "test TIC leaked into training/validation"
    assert not train_set & val_set, "train/val TIC overlap"

    # frozen K=8 CBV bases (train TICs only; identical to the existing pipeline by determinism)
    bases = build_or_load_area_bases(base_train.X, base_train.M, base_train.areas,
                                     sorted(base_train.tics), CBV_RANK, ART_DIR, GROUP_SIZE, MIN_VALID_STARS)

    train_ds = DynamicAreaGroupDataset(base_train.X, base_train.M, base_train.areas, base_train.tics,
                                       bases, GROUP_SIZE, MIN_VALID_STARS, RIDGE_LAMBDA,
                                       n_context=N_CONTEXT, seed=SEED, resample=True,
                                       require_full=REQUIRE_FULL)
    val_ds = DynamicAreaGroupDataset(base_val.X, base_val.M, base_val.areas, base_val.tics,
                                     bases, GROUP_SIZE, MIN_VALID_STARS, RIDGE_LAMBDA,
                                     n_context=None, seed=SEED, resample=False)   # fixed deterministic val
    floor = N_CONTEXT if REQUIRE_FULL else GROUP_SIZE + 1
    print(f"REQUIRE_FULL={REQUIRE_FULL} -> area eligibility floor: >= {floor} stars", flush=True)
    print(f"train eligible areas: {len(train_ds.eligible)} "
          f"-> {len(train_ds)} context examples/epoch "
          f"({'exactly' if REQUIRE_FULL else 'up to'} {N_CONTEXT}/area)", flush=True)
    print(f"train EXCLUDED areas (< {floor} stars): {len(train_ds.excluded)} "
          f"{train_ds.excluded}", flush=True)
    print(f"val eligible areas: {len(val_ds.eligible)} -> {len(val_ds)} fixed examples", flush=True)
    with open(os.path.join(ART_DIR, f"area_eligibility_{tag}.json"), "w") as fh:
        json.dump({"n_context": N_CONTEXT, "train_eligible": train_ds.eligible,
                   "train_excluded_area_to_nstars": train_ds.excluded,
                   "val_eligible": val_ds.eligible}, fh, indent=2)

    model = FixedTeacherInstrumentJEPA(n_tokens=16, token_dim=16, d_model=256, n_layers=4,
                                       readout="mean", predictor_type="mlp").to(DEVICE)
    load_frozen_teacher(model, TEACHER_SELECTION)                # existing Stage-A teacher, frozen
    teacher_hash = model.teacher_hash()
    print(f"frozen teacher hash {teacher_hash[:16]}...", flush=True)

    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    fields = ["epoch", "train_loss", "val_loss", "teacher_hash_ok", "n_train_examples"]
    metrics_path = os.path.join(ART_DIR, f"metrics_{tag}.csv")
    with open(metrics_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fields).writeheader()

    best = {"val_loss": float("inf"), "epoch": None}
    for epoch in range(1, EPOCHS + 1):
        train_ds.set_epoch(epoch)                                # resample this epoch's groups
        train_ds.assert_contracts()                             # item-9 guarantees, every epoch
        gen = torch.Generator().manual_seed(SEED + epoch)
        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS,
                                  worker_init_fn=seed_worker, generator=gen, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH, num_workers=NUM_WORKERS,
                                worker_init_fn=seed_worker,
                                generator=torch.Generator().manual_seed(SEED))

        train_loss = run_epoch(model, train_loader, optimizer)
        scheduler.step()
        val_loss = run_epoch(model, val_loader)                 # predicted-latent validation score
        hash_ok = model.teacher_hash() == teacher_hash
        if not hash_ok:
            raise RuntimeError("FROZEN TEACHER CHANGED -- protocol violation")

        with open(metrics_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fields).writerow(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                 "teacher_hash_ok": hash_ok, "n_train_examples": len(train_ds)})
        marker = ""
        if val_loss < best["val_loss"]:
            best = {"val_loss": val_loss, "epoch": epoch, "train_loss": train_loss}
            torch.save(model.state_dict(), f"{ckpt_base}_best.pth")            # full model for decode
            torch.save(model.student.state_dict(), f"{ckpt_base}_best_student_encoder.pth")
            torch.save(model.predictor.state_dict(), f"{ckpt_base}_best_predictor.pth")
            marker = " <- best"
        torch.save(model.state_dict(), f"{ckpt_base}_latest.pth")
        print(f"epoch {epoch:02d}: train={train_loss:.5f} val={val_loss:.5f} "
              f"teacher_ok={hash_ok} n={len(train_ds)}{marker}", flush=True)

    selection = {"tag": tag, "seed": SEED, "n_context": N_CONTEXT, "require_full": REQUIRE_FULL,
                 "n_train_areas": len(train_ds.eligible), "group_size": GROUP_SIZE,
                 "cbv_rank": CBV_RANK, "min_valid": MIN_VALID_STARS, "ridge_lambda": RIDGE_LAMBDA,
                 "epochs": EPOCHS, "lr": LR, "var_weight": VARW, "batch": BATCH,
                 "select_metric": "min predicted-latent validation loss", "best": best,
                 "checkpoint": f"{ckpt_base}_best.pth",
                 "encoder_checkpoint": f"{ckpt_base}_best_student_encoder.pth",
                 "predictor_checkpoint": f"{ckpt_base}_best_predictor.pth",
                 "teacher_selection": TEACHER_SELECTION, "teacher_hash": teacher_hash,
                 "teacher_hash_verified_every_epoch": True,
                 "train_eligible_areas": len(train_ds.eligible),
                 "train_excluded_areas": len(train_ds.excluded),
                 "git_commit": git_commit()}
    with open(os.path.join(ART_DIR, f"selection_{tag}.json"), "w") as fh:
        json.dump(selection, fh, indent=2, default=float)
    print(json.dumps(selection, indent=2, default=float), flush=True)
    print(f"BEST epoch {best['epoch']} val_loss {best['val_loss']:.5f} -> {ckpt_base}_best.pth", flush=True)


if __name__ == "__main__":
    main()
