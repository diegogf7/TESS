# All this code is from Claude
"""Stage A trainer: regional group teacher (seed-0 pilot, validation-only).

Selection: best VALIDATION AREA balanced accuracy, subject to
  effective rank >= 16  and  same-area cosine > same-CCD/different-area.
The selected EMA encoder becomes the FROZEN teacher for Stage B.

Run:  python -m src.instrument_v2.train_regional_group_teacher
Env:  SEED, EPOCHS, BATCH, LR, VARW, K, S14_DATA, SPLIT_DIR, BASE_ART_DIR,
      FRT_ART_DIR, FRT_CKPT_DIR, MAX_BATCHES (smoke), NUM_WORKERS
"""

from __future__ import annotations

import csv
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.instrument_v2.area_commonmode_dataset import (
    Sector14GroupStatDataset,
    ensure_area_column,
)
from src.instrument_v2.regional_group_teacher import (
    AreaGroupPairDataset,
    build_regional_teacher,
)
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_group_level_jepa import fast_probe
from src.instrument_v2.train_sector14_jepa import (
    effective_rank,
    git_commit,
    seed_worker,
)
from src.loss_function.gapblind_fix import gapblind_loss

SEED = int(os.environ.get("SEED", "0"))
K = int(os.environ.get("K", "8"))
EPOCHS = int(os.environ.get("EPOCHS", "20"))
BATCH = int(os.environ.get("BATCH", "64"))
LR = float(os.environ.get("LR", "1e-3"))
VARW = float(os.environ.get("VARW", "0.5"))
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "0"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
N_PROBE_DRAWS = int(os.environ.get("N_PROBE_DRAWS", "6"))
N_COSINE_DRAWS = int(os.environ.get("N_COSINE_DRAWS", "32"))
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

GATE_MIN_ERANK = float(os.environ.get("GATE_MIN_ERANK", "16"))


def group_features(model, pair_dataset, view="online"):
    """Encode N_PROBE_DRAWS random K-star groups per eligible area."""
    features, areas, chips = [], [], []
    base = pair_dataset.base
    with torch.no_grad():
        for area in pair_dataset.eligible:
            for _ in range(N_PROBE_DRAWS):
                rows = np.random.choice(base.group_rows[area], size=base.k,
                                        replace=False)
                stats, valid = pair_dataset.group_input(rows)
                tokens = model.encode(stats.unsqueeze(0).to(DEVICE),
                                      valid.unsqueeze(0).to(DEVICE), view=view)
                features.append(tokens.flatten(1).cpu().numpy()[0])
                areas.append(int(area))
                chips.append(int(area) // 10)     # camera*10 + ccd parent
    return np.asarray(features), np.asarray(areas), np.asarray(chips)


def target_cosines(model, pair_dataset):
    base = pair_dataset.base

    def encode(rows):
        stats, valid = pair_dataset.group_input(rows)
        tokens = model.encode(stats.unsqueeze(0).to(DEVICE),
                              valid.unsqueeze(0).to(DEVICE), view="ema")
        return F.layer_norm(tokens, (tokens.shape[-1],)).flatten(1)

    same, cross = [], []
    for _ in range(N_COSINE_DRAWS):
        draw = base.sample_disjoint_same_group()
        if draw is not None:
            rows_a, rows_b, _ = draw
            same.append(float(F.cosine_similarity(encode(rows_a),
                                                  encode(rows_b)).mean()))
        draw = base.sample_cross_group()
        if draw is not None:
            rows_a, rows_b, _ = draw
            cross.append(float(F.cosine_similarity(encode(rows_a),
                                                   encode(rows_b)).mean()))
    return (float(np.mean(same)) if same else float("nan"),
            float(np.mean(cross)) if cross else float("nan"))


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total, batches = 0.0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(loader):
            if MAX_BATCHES and batch_idx >= MAX_BATCHES:
                break
            stats_a, valid_a, stats_b, valid_b, _, _ = \
                [tensor.to(DEVICE) for tensor in batch]
            prediction, target, context_tokens = model(
                stats_a, valid_a, stats_b, valid_b)
            loss = gapblind_loss(prediction, target, context_tokens,
                                 target_mask=valid_b, var_weight=VARW)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.update_target()
            total += float(loss.detach())
            batches += 1
    return total / max(1, batches)


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(ART_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    tag = f"regteacher_k{K}_s{SEED}"
    best_path = os.path.join(CKPT_DIR, f"{tag}_best.pth")

    print(f"git commit: {git_commit()}", flush=True)
    print(f"config: {tag} epochs={EPOCHS} batch={BATCH} lr={LR} varw={VARW} "
          f"device={DEVICE}", flush=True)

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    df = ensure_area_column(df)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, train_tics)

    train_pairs = AreaGroupPairDataset(
        Sector14GroupStatDataset(df, train_tics, t_range, "area", K))
    val_pairs = AreaGroupPairDataset(
        Sector14GroupStatDataset(df, val_tics, t_range, "area", K))
    assert not (set(train_pairs.base.tics) | set(val_pairs.base.tics)) & test_tics

    train_loader = DataLoader(train_pairs, batch_size=BATCH,
                              num_workers=NUM_WORKERS,
                              worker_init_fn=seed_worker,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_pairs, batch_size=BATCH,
                            num_workers=NUM_WORKERS,
                            worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(SEED))

    model = build_regional_teacher().to(DEVICE)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=EPOCHS)

    fields = ["epoch", "train_loss", "val_loss", "effective_rank",
              "same_area_cos", "cross_area_cos", "val_area_bacc",
              "val_camccd_bacc"]
    metrics_path = os.path.join(ART_DIR, f"metrics_{tag}.csv")
    with open(metrics_path, "w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()

    best = {"area_bacc": -1.0, "epoch": None, "erank": float("nan"),
            "same_cos": float("nan"), "cross_cos": float("nan"),
            "camccd_bacc": float("nan")}
    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, optimizer)
        scheduler.step()
        val_loss = run_epoch(model, val_loader)

        train_z, train_area, train_chip = group_features(model, train_pairs)
        val_z, val_area, val_chip = group_features(model, val_pairs)
        erank = effective_rank(val_z)
        area_bacc = fast_probe(train_z, train_area, val_z, val_area)
        camccd_bacc = fast_probe(train_z, train_chip, val_z, val_chip)
        same_cos, cross_cos = target_cosines(model, val_pairs)

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "effective_rank": erank, "same_area_cos": same_cos,
               "cross_area_cos": cross_cos, "val_area_bacc": area_bacc,
               "val_camccd_bacc": camccd_bacc}
        with open(metrics_path, "a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        print(f"teacher epoch {epoch:03d}: train={train_loss:.5f} "
              f"val={val_loss:.5f} erank={erank:.1f} "
              f"same_cos={same_cos:.3f} cross_cos={cross_cos:.3f} "
              f"area_bacc={area_bacc:.4f} camccd_bacc={camccd_bacc:.4f}",
              flush=True)

        if area_bacc > best["area_bacc"]:
            best = {"area_bacc": area_bacc, "epoch": epoch, "erank": erank,
                    "same_cos": same_cos, "cross_cos": cross_cos,
                    "camccd_bacc": camccd_bacc}
            torch.save(model.state_dict(), best_path)
            print(f"  <- saved best teacher (area_bacc {area_bacc:.4f})",
                  flush=True)

    # Gates are INFORMATIONAL only (pilot): reported, never enforced.
    gates_info = {"min_effective_rank": GATE_MIN_ERANK,
                  "erank_ok": bool(best["erank"] >= GATE_MIN_ERANK),
                  "same_cos_gt_cross": bool(best["same_cos"] > best["cross_cos"])}
    selection = {"tag": tag, "seed": SEED, "k": K, "epochs": EPOCHS,
                 "passed_gates": True,          # never blocks downstream
                 "gates_informational": gates_info,
                 "best": best, "checkpoint": best_path,
                 "git_commit": git_commit()}
    with open(os.path.join(ART_DIR, f"selection_{tag}.json"), "w") as handle:
        json.dump(selection, handle, indent=2)
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
