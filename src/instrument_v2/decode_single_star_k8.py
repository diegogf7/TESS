from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.instrument_v2.area_commonmode_dataset import(
    Sector14GroupStatDataset,
    ensure_area_column,
    group_statistics


)

from src.instrument_v2.fixed_teacher_instrument_jepa import FixedTeacherInstrumentJEPA
from src.instrument_v2.regional_group_teacher import state_hash
from src.instrument_v2.regional_cbv import build_or_load_area_bases, ridge_reconstruct
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_sector14_jepa import git_commit


#these variables are all from CLAUDE bc I kept messing it up

SEED = int(os.environ.get("SEED", "0"))
K = int(os.environ.get("K", "8"))
RIDGE_LAMBDA = float(os.environ.get("RIDGE_LAMBDA", "1e-2"))
GROUP_MIN_VALID = int(os.environ.get("GROUP_MIN_VALID", "4"))
EPOCHS = int(os.environ.get("EPOCHS", "20"))
BATCH = int(os.environ.get("BATCH", "64"))
LR = float(os.environ.get("LR", "1e-3"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

S14_DATA = os.environ.get(
    "S14_DATA",
    "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get(
    "SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
BASE_ART_DIR = os.environ.get(
    "BASE_ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))
GROUP_ART_DIR = os.environ.get(
    "GROUP_ART_DIR",
    os.path.join("artifacts", "instrument_v2", "custom_group_cbv_k8_mlp_v1"))
CKPT_DIR = os.environ.get(
    "CKPT_DIR",
    "/orcd/scratch/orcd/006/diegogon/checkpoints/custom_group_cbv_k8_mlp_v1")
STAGE_B_CKPT = os.environ.get(
    "STAGE_B_CKPT", os.path.join(CKPT_DIR, f"group_cbv_mlp_k{K}_s{SEED}_best.pth"))
OUT_DIR = os.environ.get(
    "OUT_DIR",
    os.path.join("artifacts", "instrument_v2", "custom_group_cbv_k8_mlp_v1",
                 "single_star_decode"))


def build_decoder():
    final_model = nn.Sequential(
        nn.Flatten(), nn.LayerNorm(256), nn.Linear(256, 512), nn.GELU(),
        nn.Linear(512, 1024))
    
    return final_model


def masked_smooth_l1(prediction, target, mask):

    per_loss = F.smooth_l1_loss(prediction, target, reduction = "None")

    final = (per_loss * mask).sum() / mask.sum().clamp(min = 1.0)

    return final

def masked_metrics(pred, target, mask):

    observed = mask > 0

    prediction, tar = pred[observed], target[observed]

    if prediction.size < 2:
        return np.nan, np.nan, np.nan
    

    correlation = float(np.corrcoef(prediction, tar)[0,1]) if prediction.std() > 0 and tar.std() > 0 else np.nan

    root_mean = float(np.sqrt(np.mean((prediction - tar) ** 2)))
    ss_total = float(np.sum((tar - tar.mean()) ** 2))

    r_squared = float(1.0 - np.sum((tar - prediction) ** 2) / ss_total) if ss_total >0 else np.nan

    return correlation, root_mean, r_squared


def teacher_latent(model, reconstruction, log_mad, valid):

    stats = torch.tensor(np.stack([reconstruction, log_mad], -1), dtype = torch.float32)[None].to(DEVICE)
    v = torch.tensor(valid, dtype = torch.float32)[None].to(DEVICE)

    tokens = model.teacher(stats, v)

    return F.layer_norm(tokens, (tokens.shape[-1],))


def deterministic_area_row(ds):

    order = np.argsort(np.asarray(ds.tics, dtype = str))

    rank = np.empty(len(ds.tics), dtype = np.int64)

    rank[order] = np.arange(len(ds.tics))
    out = {}

    for i, a in enumerate(ds.areas):

        out.setdefault(int(a), []).append(i)
    
    return {a: list(np.asarray(rows)[np.argsort(rank[rows])]) for a, rows in out.items()}

def build_decoder_trainset(model, train_ds, bases):

    area_rows = deterministic_area_row(train_ds)

    latitude = []
    target = []
    mask = []

    with torch.no_grad():

        for area, B in bases.items():
            rows = area_rows.get(area, [])
            for g in range(len(rows) // K):

                group = rows[g * K: (g+1) * K]
                median, log_mad, valid, _ = group_statistics(train_ds.X[group], train_ds.M[group], train_ds.min_valid)

                reconstruction = ridge_reconstruct(median, valid, B, RIDGE_LAMBDA)

                latitude.append(teacher_latent(model, reconstruction, log_mad, valid).unsqueeze(0).cpu().numpy())
                target.append(reconstruction)
                mask.append(valid)
    
    return (np.stacak(latitude).astype(np.float32), np.stack(target).astype(np.float32), np.stack(mask).astype(np.float32))


def reference_target(ds, area_rows, i, bases):

    


