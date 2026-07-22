# All this code is from Claude
"""Trainer for the area/chip common-mode JEPA (validation-only protocol).

One invocation = one (grouping, target, K, seed) cell, used both for the
seed-0 screen (EPOCHS=20) and the 3-seed confirmation (EPOCHS=50). Test
TICs are never loaded. Writes an area/chip star-count diagnostic BEFORE
training and exits cleanly (rc 0, skipped=True selection) when the
requested K is not supported by the star counts.

Run:  GROUPING=area TARGET=median_mad K=8 SEED=0 \
          python -m src.instrument_v2.train_area_commonmode_jepa
Env:  GROUPING (chip|area), TARGET (median|median_mad), K, SEED, EPOCHS,
      BATCH, LR, VARW, MAD_WEIGHT, WARMSTART (1|0), WARMSTART_SELECTION,
      S14_DATA, AREA_SOURCE, SPLIT_DIR, BASE_ART_DIR, ACM_ART_DIR,
      ACM_CKPT_DIR, PROBE_EVERY, MAX_BATCHES (smoke), NUM_WORKERS
"""

from __future__ import annotations

import csv
import json
import math
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
    valid_k_values,
)
from src.instrument_v2.area_commonmode_jepa import (
    build_area_commonmode_jepa,
    commonmode_loss,
    load_group_jepa_warmstart,
)
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_group_level_jepa import fast_probe, individual_latents
from src.instrument_v2.train_sector14_jepa import (
    effective_rank,
    git_commit,
    seed_worker,
)

GROUPING = os.environ.get("GROUPING", "area")
TARGET = os.environ.get("TARGET", "median_mad")
assert GROUPING in ("chip", "area"), f"bad GROUPING {GROUPING!r}"
assert TARGET in ("median", "median_mad"), f"bad TARGET {TARGET!r}"
K = int(os.environ.get("K", "8"))
SEED = int(os.environ.get("SEED", "0"))
EPOCHS = int(os.environ.get("EPOCHS", "20"))
BATCH = int(os.environ.get("BATCH", "64"))
LR = float(os.environ.get("LR", "1e-3"))
VARW = float(os.environ.get("VARW", "0.5"))
MAD_WEIGHT = float(os.environ.get("MAD_WEIGHT", "0.25"))
COV_WEIGHT = float(os.environ.get("COV_WEIGHT", "0.0"))
WARMSTART = os.environ.get("WARMSTART", "1") == "1"
PROBE_EVERY = int(os.environ.get("PROBE_EVERY", "2"))
MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "0"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "4"))
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
    "ACM_ART_DIR", os.path.join("artifacts", "instrument_v2", "area_commonmode_v1"))
CKPT_DIR = os.environ.get(
    "ACM_CKPT_DIR", "/orcd/scratch/orcd/006/diegogon/checkpoints/area_commonmode")
WARMSTART_SELECTION = os.environ.get(
    "WARMSTART_SELECTION",
    os.path.join("artifacts", "instrument_v2", "group_level",
                 f"selection_s14groupmean_k8_s{SEED}.json"))

# promotion gates (screen selection refuses collapsed models)
GATE_MIN_ERANK = float(os.environ.get("GATE_MIN_ERANK", "16"))
GATE_MIN_SAME_COS = float(os.environ.get("GATE_MIN_SAME_COS", "0.90"))
GATE_MIN_PROBE = float(os.environ.get("GATE_MIN_PROBE", "0.0"))  # v2: 0.44


def target_cosine_stats(model, dataset, n_draws=N_COSINE_DRAWS):
    """Mean cosine between EMA-teacher latents of (a) two disjoint same-group
    median targets and (b) two sibling-group median targets."""
    def encode_target(rows):
        median, _, valid = dataset.group_target_tensors(rows)
        tokens = model._teach(median.unsqueeze(0).to(DEVICE),
                              valid.unsqueeze(0).to(DEVICE))
        return tokens.flatten(1)

    same, cross = [], []
    for _ in range(n_draws):
        draw = dataset.sample_disjoint_same_group()
        if draw is not None:
            rows_a, rows_b, _ = draw
            same.append(float(F.cosine_similarity(
                encode_target(rows_a), encode_target(rows_b)).mean()))
        draw = dataset.sample_cross_group()
        if draw is not None:
            rows_a, rows_b, _ = draw
            cross.append(float(F.cosine_similarity(
                encode_target(rows_a), encode_target(rows_b)).mean()))
    return (float(np.mean(same)) if same else float("nan"),
            float(np.mean(cross)) if cross else float("nan"))


def run_epoch(model, loader, optimizer=None):
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "median_loss": 0.0, "mad_loss": 0.0,
              "var_loss": 0.0, "cov_loss": 0.0, "valid_stars": 0.0}
    batches = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_idx, batch in enumerate(loader):
            if MAX_BATCHES and batch_idx >= MAX_BATCHES:
                break
            (ctx_f, ctx_m, median, log_mad, valid, n_observed, _) = \
                [tensor.to(DEVICE) for tensor in batch]
            outputs = model(ctx_f, ctx_m, median, log_mad, valid, target=TARGET)
            pred_median, target_median, pred_mad, target_mad, tokens = outputs
            loss, parts = commonmode_loss(
                pred_median, target_median, pred_mad, target_mad, tokens,
                valid, mad_weight=MAD_WEIGHT, var_weight=VARW,
                cov_weight=COV_WEIGHT)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                model.update_target()
            totals["loss"] += float(loss.detach())
            totals["median_loss"] += parts["median_loss"]
            totals["mad_loss"] += parts["mad_loss"]
            totals["var_loss"] += parts["var_loss"]
            totals["cov_loss"] += parts["cov_loss"]
            totals["valid_stars"] += float(
                (n_observed * valid).sum() / valid.sum().clamp(min=1))
            batches += 1
    return {key: value / max(1, batches) for key, value in totals.items()}


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    os.makedirs(ART_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    tag = f"acm_{GROUPING}_{TARGET}_k{K}_cw{COV_WEIGHT:g}_s{SEED}"
    best_path = os.path.join(CKPT_DIR, f"{tag}_best.pth")
    selection_path = os.path.join(ART_DIR, f"selection_{tag}.json")

    print(f"git commit: {git_commit()}", flush=True)
    print(f"config: {tag} epochs={EPOCHS} batch={BATCH} lr={LR} varw={VARW} "
          f"mad_w={MAD_WEIGHT} warmstart={WARMSTART} device={DEVICE}", flush=True)

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    df = ensure_area_column(df)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, train_tics)

    train_ds = Sector14GroupStatDataset(df, train_tics, t_range, GROUPING, K)
    val_ds = Sector14GroupStatDataset(df, val_tics, t_range, GROUPING, K)
    assert not (set(train_ds.tics) | set(val_ds.tics)) & test_tics, \
        "test TIC leaked into common-mode training"

    # -------- star-count diagnostic BEFORE training + automatic K skip
    counts_path = os.path.join(ART_DIR, f"group_counts_{GROUPING}_train.csv")
    rows = train_ds.group_count_table()
    with open(counts_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"group-count diagnostic -> {counts_path} "
          f"({sum(r['usable'] for r in rows)}/{len(rows)} groups usable at K={K})",
          flush=True)
    supported = valid_k_values(train_ds)
    if K not in supported:
        selection = {"tag": tag, "skipped": True,
                     "reason": f"K={K} unsupported for {GROUPING} "
                               f"(supported: {supported})"}
        with open(selection_path, "w") as handle:
            json.dump(selection, handle, indent=2)
        print(json.dumps(selection, indent=2), flush=True)
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH, num_workers=NUM_WORKERS,
                              worker_init_fn=seed_worker,
                              generator=torch.Generator().manual_seed(SEED))
    val_loader = DataLoader(val_ds, batch_size=BATCH, num_workers=NUM_WORKERS,
                            worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(SEED))

    model = build_area_commonmode_jepa().to(DEVICE)
    warmstart_info = None
    if WARMSTART:
        warmstart_info = load_group_jepa_warmstart(model, WARMSTART_SELECTION)
        print(f"warm-start: online+EMA encoders from "
              f"{warmstart_info['checkpoint']}", flush=True)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    fields = ["epoch", "train_loss", "train_median_loss", "train_mad_loss",
              "train_var_loss", "train_cov_loss",
              "val_loss", "val_median_loss", "val_mad_loss",
              "val_var_loss", "val_cov_loss",
              "val_probe_bacc16", "latent_std", "effective_rank",
              "same_group_cos", "cross_group_cos", "mean_valid_stars"]
    metrics_path = os.path.join(ART_DIR, f"metrics_{tag}.csv")
    with open(metrics_path, "w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fields).writeheader()

    best = {"probe": -1.0, "epoch": None, "erank": float("nan"),
            "same_cos": float("nan"), "cross_cos": float("nan")}
    for epoch in range(1, EPOCHS + 1):
        train_metrics = run_epoch(model, train_loader, optimizer)
        scheduler.step()
        val_metrics = run_epoch(model, val_loader)

        train_z = individual_latents(model, train_ds, "online")
        val_z = individual_latents(model, val_ds, "online")
        latent_std = float(val_z.std(axis=0).mean())
        erank = effective_rank(val_z)
        same_cos, cross_cos = target_cosine_stats(model, val_ds)

        probe = float("nan")
        if epoch == 1 or epoch % PROBE_EVERY == 0 or epoch == EPOCHS:
            probe = fast_probe(train_z, train_ds.chips, val_z, val_ds.chips)
            if probe > best["probe"]:
                best = {"probe": probe, "epoch": epoch, "erank": erank,
                        "same_cos": same_cos, "cross_cos": cross_cos}
                torch.save(model.state_dict(), best_path)

        row = {"epoch": epoch, "train_loss": train_metrics["loss"],
               "train_median_loss": train_metrics["median_loss"],
               "train_mad_loss": train_metrics["mad_loss"],
               "train_var_loss": train_metrics["var_loss"],
               "train_cov_loss": train_metrics["cov_loss"],
               "val_loss": val_metrics["loss"],
               "val_median_loss": val_metrics["median_loss"],
               "val_mad_loss": val_metrics["mad_loss"],
               "val_var_loss": val_metrics["var_loss"],
               "val_cov_loss": val_metrics["cov_loss"],
               "val_probe_bacc16": probe, "latent_std": latent_std,
               "effective_rank": erank, "same_group_cos": same_cos,
               "cross_group_cos": cross_cos,
               "mean_valid_stars": val_metrics["valid_stars"]}
        with open(metrics_path, "a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writerow(row)
        print(f"{tag} epoch {epoch:03d}: train={row['train_loss']:.5f} "
              f"val={row['val_loss']:.5f} probe={probe:.4f} "
              f"rank={erank:.1f} same_cos={same_cos:.3f} "
              f"cross_cos={cross_cos:.3f}", flush=True)

    gates = {"effective_rank_ok": bool(best["erank"] >= GATE_MIN_ERANK),
             "same_cos_ok": bool(best["same_cos"] >= GATE_MIN_SAME_COS),
             "same_gt_cross": bool(best["same_cos"] > best["cross_cos"]),
             "probe_ok": bool(best["probe"] > GATE_MIN_PROBE)}
    selection = {"tag": tag, "grouping": GROUPING, "target": TARGET, "k": K,
                 "cov_weight": COV_WEIGHT,
                 "seed": SEED, "epochs": EPOCHS, "skipped": False,
                 "warmstart": WARMSTART_SELECTION if WARMSTART else None,
                 "best_val_probe_bacc16": best["probe"],
                 "best_epoch": best["epoch"],
                 "best_effective_rank": best["erank"],
                 "best_same_group_cos": best["same_cos"],
                 "best_cross_group_cos": best["cross_cos"],
                 "gates": gates, "passes_gates": all(gates.values()),
                 "checkpoint": best_path, "git_commit": git_commit()}
    with open(selection_path, "w") as handle:
        json.dump(selection, handle, indent=2)
    print(json.dumps(selection, indent=2), flush=True)


if __name__ == "__main__":
    main()
