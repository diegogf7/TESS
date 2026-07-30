from __future__ import annotations
"""Part A (scaled): regional group teacher with 32-star groups, exactly 1,000
training stars per area, and exactly 1,000 Group-A/Group-B pairs per area per
epoch (resampled each epoch by deterministic [seed, epoch, area, pair] seeds).

Architecture and loss are the existing regional teacher: Group A -> online S4D
-> predictor; Group B -> EMA S4D target; masked Smooth-L1 + spread (gapblind
loss); EMA update each step. Trains exactly 20 epochs, NO early stopping, with
fixed deterministic validation pairs so every epoch is comparable. Selection =
best validation area balanced accuracy. Saves latest, best full checkpoint, best
EMA encoder, per-epoch metrics, and a Part-B-compatible selection JSON tagged
regteacher_g32_n1000_s0.

    python -m src.instrument_v2.train_regional_teacher_scaled
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
from src.instrument_v2.dynamic_pair_dataset import DynamicAreaPairDataset
from src.instrument_v2.regional_group_teacher import build_regional_teacher, state_hash
from src.instrument_v2.regional_cbv import build_or_load_area_bases
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_group_level_jepa import fast_probe
from src.instrument_v2.train_sector14_jepa import effective_rank, git_commit, seed_worker
from src.loss_function.gapblind_fix import gapblind_loss

SEED = int(os.environ.get("SEED", "0"))
GROUP_SIZE = int(os.environ.get("GROUP_SIZE", "32"))
CBV_RANK = int(os.environ.get("CBV_RANK", "8"))
MIN_VALID_STARS = int(os.environ.get("MIN_VALID_STARS", "16"))
N_STARS = int(os.environ.get("N_STARS", "1000"))            # exactly this many stars per area
N_PAIRS = int(os.environ.get("N_PAIRS", "1000"))            # pairs per area per epoch
REQUIRE_FULL = os.environ.get("REQUIRE_FULL", "1").lower() not in ("0", "false", "no")  # 0 = use all available stars/area (no 1000 floor)
N_PAIRS_VAL = int(os.environ.get("N_PAIRS_VAL", "200"))     # fixed deterministic validation pairs/area
EPOCHS = int(os.environ.get("EPOCHS", "20"))               # exactly 20, no early stopping
LR = float(os.environ.get("LR", "1e-3"))
VARW = float(os.environ.get("VARW", "0.5"))
BATCH = int(os.environ.get("BATCH", "64"))
RIDGE_LAMBDA = float(os.environ.get("RIDGE_LAMBDA", "1e-2"))
N_PROBE_DRAWS = int(os.environ.get("N_PROBE_DRAWS", "6"))
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "0"))       # >0 = smoke
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

S14_DATA = os.environ.get("S14_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2.parquet")
SPLIT_DIR = os.environ.get("SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "dense_v2_split"))
BASE_ART_DIR = os.environ.get("BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa_dense_v2"))
ART_DIR = os.environ.get("GROUP_ART_DIR", os.path.join("artifacts", "instrument_v2", "regteacher_g32_n1000_v1"))
CKPT_DIR = os.environ.get("CKPT_DIR", "/orcd/scratch/orcd/006/diegogon/checkpoints/regteacher_g32_n1000_v1")


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, batches = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for bi, batch in enumerate(loader):
            if MAX_BATCHES and bi >= MAX_BATCHES:
                break
            stats_a, valid_a, stats_b, valid_b, _, _ = [t.to(DEVICE) for t in batch]
            prediction, target, context_tokens = model(stats_a, valid_a, stats_b, valid_b)
            loss = gapblind_loss(prediction, target, context_tokens, target_mask=valid_b, var_weight=VARW)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.update_target()                        # EMA update
            total += float(loss.detach())
            batches += 1
    return total / max(1, batches)


def area_features(model, pairs, view="online", seed=0):
    """Encode N_PROBE_DRAWS groups drawn from each eligible area's pool."""
    feats, areas = [], []
    rng = np.random.default_rng(seed)
    model.eval()
    with torch.no_grad():
        for a in pairs.eligible:
            pool = pairs.pool[a]
            for _ in range(N_PROBE_DRAWS):
                rows = rng.choice(pool, size=pairs.group_size, replace=False)
                stats, valid = pairs.group_input(rows, a)
                tok = model.encode(stats.unsqueeze(0).to(DEVICE), valid.unsqueeze(0).to(DEVICE), view=view)
                feats.append(tok.flatten(1).cpu().numpy()[0]); areas.append(int(a))
    return np.asarray(feats), np.asarray(areas)


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    os.makedirs(ART_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)
    tag = f"regteacher_g{GROUP_SIZE}_n{N_STARS}_s{SEED}"
    ckpt_base = os.path.join(CKPT_DIR, tag)
    print("================ resolved configuration ================", flush=True)
    print(f"  git commit  : {git_commit()}", flush=True)
    print(f"  tag         : {tag}", flush=True)
    print(f"  GROUP/RANK/MINVALID : {GROUP_SIZE}/{CBV_RANK}/{MIN_VALID_STARS}", flush=True)
    print(f"  N_STARS/N_PAIRS/EPOCHS: {N_STARS}/{N_PAIRS}/{EPOCHS} (no early stop)", flush=True)
    print(f"  REQUIRE_FULL: {REQUIRE_FULL}  ({'exactly N_STARS/area, hard-fail if short' if REQUIRE_FULL else 'use ALL available stars/area, drop areas < 64'})", flush=True)
    print(f"  S14_DATA    : {S14_DATA}", flush=True)
    print(f"  ART/CKPT    : {ART_DIR} | {CKPT_DIR}", flush=True)
    print("========================================================", flush=True)

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    df = ensure_area_column(df)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, train_tics)

    base_train = Sector14GroupStatDataset(df, train_tics, t_range, "area", GROUP_SIZE, min_valid=MIN_VALID_STARS)
    base_val = Sector14GroupStatDataset(df, val_tics, t_range, "area", GROUP_SIZE, min_valid=MIN_VALID_STARS)
    assert not (set(base_train.tics) | set(base_val.tics)) & test_tics, "test TIC leaked"
    assert not set(base_train.tics) & set(base_val.tics), "train/val overlap"

    bases = build_or_load_area_bases(base_train.X, base_train.M, base_train.areas,
                                     sorted(base_train.tics), CBV_RANK, ART_DIR, GROUP_SIZE, MIN_VALID_STARS)

    # print the per-area train count report up front (also fails inside the dataset if any < N_STARS)
    counts = {}
    for a in base_train.areas:
        counts[int(a)] = counts.get(int(a), 0) + 1
    short = {a: n for a, n in counts.items() if a in bases and n < N_STARS}
    if short:
        print(f"AREA-COUNT REPORT: {len(short)} areas below {N_STARS} training stars:", flush=True)
        for a, n in sorted(short.items()):
            print(f"    area {a}: {n}", flush=True)

    train_pairs = DynamicAreaPairDataset(base_train.X, base_train.M, base_train.areas, base_train.tics,
                                         bases, GROUP_SIZE, MIN_VALID_STARS, RIDGE_LAMBDA,
                                         n_stars=N_STARS, n_pairs=N_PAIRS, seed=SEED,
                                         resample=True, require_full=REQUIRE_FULL)   # REQUIRE_FULL=0 -> use all available stars/area
    val_pairs = DynamicAreaPairDataset(base_val.X, base_val.M, base_val.areas, base_val.tics,
                                       bases, GROUP_SIZE, MIN_VALID_STARS, RIDGE_LAMBDA,
                                       n_stars=N_STARS, n_pairs=N_PAIRS_VAL, seed=SEED,
                                       resample=False, require_full=False)   # fixed deterministic val
    print(f"train areas: {len(train_pairs.eligible)} x {N_PAIRS} pairs = {len(train_pairs)} pairs/epoch", flush=True)
    print(f"val areas:   {len(val_pairs.eligible)} x {N_PAIRS_VAL} pairs = {len(val_pairs)} fixed pairs", flush=True)

    model = build_regional_teacher().to(DEVICE)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    ema_hash0 = state_hash(model.ema_encoder)

    fields = ["epoch", "train_loss", "val_loss", "val_area_bacc", "effective_rank", "ema_moving"]
    metrics_path = os.path.join(ART_DIR, f"metrics_{tag}.csv")
    with open(metrics_path, "w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fields).writeheader()

    best = {"area_bacc": -1.0, "epoch": None}
    for epoch in range(1, EPOCHS + 1):
        train_pairs.set_epoch(epoch)
        train_pairs.assert_contracts()                       # item-9 pair guarantees, every epoch
        gen = torch.Generator().manual_seed(SEED + epoch)
        train_loader = DataLoader(train_pairs, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS,
                                  worker_init_fn=seed_worker, generator=gen, drop_last=True)
        val_loader = DataLoader(val_pairs, batch_size=BATCH, num_workers=NUM_WORKERS,
                                worker_init_fn=seed_worker,
                                generator=torch.Generator().manual_seed(SEED), drop_last=True)

        train_loss = run_epoch(model, train_loader, optimizer)
        scheduler.step()
        val_loss = run_epoch(model, val_loader)
        tr_f, tr_a = area_features(model, train_pairs, view="online", seed=SEED)
        va_f, va_a = area_features(model, val_pairs, view="online", seed=SEED + 1)
        area_bacc = fast_probe(tr_f, tr_a, va_f, va_a)
        erank = effective_rank(va_f)
        ema_moving = state_hash(model.ema_encoder) != ema_hash0

        with open(metrics_path, "a", newline="") as fh:
            csv.DictWriter(fh, fieldnames=fields).writerow(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                 "val_area_bacc": area_bacc, "effective_rank": erank, "ema_moving": ema_moving})
        torch.save(model.state_dict(), f"{ckpt_base}_latest.pth")
        marker = ""
        if area_bacc > best["area_bacc"]:
            best = {"area_bacc": area_bacc, "epoch": epoch, "val_loss": val_loss,
                    "train_loss": train_loss, "effective_rank": erank}
            torch.save(model.state_dict(), f"{ckpt_base}_best.pth")              # full model (Part-B loader reads ema_encoder.*)
            torch.save(model.ema_encoder.state_dict(), f"{ckpt_base}_best_ema_encoder.pth")
            marker = " <- best"
        print(f"[A] epoch {epoch:02d}: train={train_loss:.5f} val={val_loss:.5f} "
              f"area={area_bacc:.4f} erank={erank:.1f} ema_moving={ema_moving}{marker}", flush=True)

    selection = {"tag": tag, "seed": SEED, "group_size": GROUP_SIZE, "cbv_rank": CBV_RANK,
                 "min_valid": MIN_VALID_STARS, "require_full": REQUIRE_FULL, "n_stars_per_area": N_STARS,
                 "n_pairs_per_area_per_epoch": N_PAIRS, "epochs": EPOCHS,
                 "lr": LR, "var_weight": VARW, "batch": BATCH, "ridge_lambda": RIDGE_LAMBDA,
                 "best": best, "checkpoint": f"{ckpt_base}_best.pth",
                 "ema_encoder_checkpoint": f"{ckpt_base}_best_ema_encoder.pth",
                 "n_train_areas": len(train_pairs.eligible), "n_val_areas": len(val_pairs.eligible),
                 "git_commit": git_commit()}
    with open(os.path.join(ART_DIR, f"selection_{tag}.json"), "w") as fh:
        json.dump(selection, fh, indent=2, default=float)
    print(json.dumps(selection, indent=2, default=float), flush=True)
    print(f"BEST epoch {best['epoch']} area_bacc {best['area_bacc']:.4f} -> {ckpt_base}_best.pth", flush=True)


if __name__ == "__main__":
    main()
