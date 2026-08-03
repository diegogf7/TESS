from __future__ import annotations
"""Validate the self-supervised instrument-correction model.
  full : val metrics (before/after pairwise |corr|, % reduction, correction/input
         & cleaned/input RMS, loss components, latent std + effective rank) +
         the 4 reject flags + plots (original / correction / cleaned  and
         before/after group-correlation heatmaps).
    EVAL_MODE=full  python -m src.shared_s4d.eval_correction
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.instrument_v2.area_commonmode_dataset import Sector14GroupStatDataset, ensure_area_column
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_sector14_jepa import seed_worker, effective_rank
from src.shared_s4d.ae_dataset import AreaGroupAEDataset
from src.shared_s4d.model import build_model, GRID, LATENT_DIM
from src.shared_s4d.correction_losses import (
    _pairwise_corr, masked_pairwise_residual_correlation, normalized_correction_energy, mean_abs_pairwise_corr)
from src.shared_s4d.train_correction import (
    SEED, GROUP_SIZE, N_STARS, LAMBDA_SIZE, MIN_OVERLAP, COLLAPSE_STD, DEVICE,
    S14_DATA, SPLIT_DIR, BASE_ART_DIR, ART_DIR, CKPT_DIR)

CKPT = os.environ.get("CKPT", os.path.join(
    CKPT_DIR, f"shared_s4d_corr_g{GROUP_SIZE}_z{LATENT_DIM}_lam{LAMBDA_SIZE}_s{SEED}_best.pth"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(ART_DIR, "eval"))

def load_model(ckpt_path):
    model = build_model().to(DEVICE)
    ck = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model

def evaluate(model, val_ds):
    dl = DataLoader(val_ds, batch_size=1, num_workers=2, worker_init_fn=seed_worker,
                    generator=torch.Generator().manual_seed(SEED))
    bef, aft, cx, rx, sh, sz, lat = [], [], [], [], [], [], []
    with torch.no_grad():
        for Xg, Mg in dl:
            Xg = Xg.squeeze(0).to(DEVICE); Mg = Mg.squeeze(0).to(DEVICE)
            c, z = model(Xg, Mg); r = Xg - c; m = (Mg > 0).float()
            bef.append(mean_abs_pairwise_corr(Xg, Mg, MIN_OVERLAP))
            aft.append(mean_abs_pairwise_corr(r, Mg, MIN_OVERLAP))
            xr = torch.sqrt((m * Xg ** 2).sum(1) / m.sum(1).clamp(min=1.0)).clamp(min=1e-8)
            cx.append(float((torch.sqrt((m * c ** 2).sum(1) / m.sum(1).clamp(min=1.0)) / xr).mean()))
            rx.append(float((torch.sqrt((m * r ** 2).sum(1) / m.sum(1).clamp(min=1.0)) / xr).mean()))
            sh.append(float(masked_pairwise_residual_correlation(r, Mg, MIN_OVERLAP)))
            sz.append(float(normalized_correction_energy(c, Xg, Mg)))
            lat.append(z.cpu().numpy())
    Z = np.concatenate(lat, 0)
    b = float(np.nanmean(bef)); a = float(np.nanmean(aft))
    return {"corr_before": b, "corr_after": a,
            "pct_reduction": float(100.0 * (b - a) / b) if b > 1e-8 else float("nan"),
            "correction_rms_over_input_rms": float(np.mean(cx)),
            "cleaned_rms_over_input_rms": float(np.mean(rx)),
            "shared_loss": float(np.mean(sh)), "size_loss": float(np.mean(sz)),
            "latent_std": float(Z.std(0).mean()), "effective_rank": float(effective_rank(Z)),
            "n_val_groups": len(val_ds)}

def reject_flags(m):
    f = {"correction_nearly_identical_to_input": bool(m["correction_rms_over_input_rms"] > 0.9),
         "cleaned_nearly_zero": bool(m["cleaned_rms_over_input_rms"] < 0.1),
         "correction_zero_but_corr_unchanged":
             bool(m["correction_rms_over_input_rms"] < 0.05 and (m["pct_reduction"] < 2.0)),
         "latent_collapse": bool(m["latent_std"] < COLLAPSE_STD)}
    f["any"] = any(f.values())
    return f

@torch.no_grad()
def corr_matrix(curves, masks, min_overlap):
    """(N, N) |pairwise Pearson| matrix (NaN where a pair lacks min overlap)."""
    N = curves.shape[0]
    corr, valid = _pairwise_corr(curves, masks, min_overlap)
    ii, jj = torch.triu_indices(N, N, offset=1)          # same order as _pairwise_corr
    C = np.full((N, N), np.nan); np.fill_diagonal(C, 1.0)
    cc = corr.abs().cpu().numpy(); vv = valid.cpu().numpy()
    ii = ii.numpy(); jj = jj.numpy()
    for k in range(len(cc)):
        v = cc[k] if vv[k] else np.nan
        C[ii[k], jj[k]] = v; C[jj[k], ii[k]] = v
    return C

def plot_examples(model, val_ds, n, out_path):
    Xg, Mg = val_ds[0]
    with torch.no_grad():
        c, _ = model(Xg.to(DEVICE), Mg.to(DEVICE))
    x = Xg.numpy(); m = Mg.numpy(); cc = c.cpu().numpy(); r = x - cc
    g = np.arange(GRID); n = min(n, x.shape[0])
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for i in range(n):
        obs = m[i] > 0; ax = axes[i]
        ax.plot(g[obs], x[i][obs], ".", ms=2, color="0.6", label="original")
        ax.plot(g, cc[i], color="tab:red", lw=0.9, label="correction c")
        ax.plot(g[obs], r[i][obs], ".", ms=2, color="tab:blue", label="cleaned x-c")
        if i == 0:
            ax.legend(fontsize=7, ncol=3, loc="upper right")
        ax.set_ylabel(f"curve {i}")
    fig.suptitle("instrument correction: original / correction / cleaned")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)

def plot_corr_matrices(model, val_ds, out_path):
    Xg, Mg = val_ds[0]
    Xg = Xg.to(DEVICE); Mg = Mg.to(DEVICE)
    with torch.no_grad():
        c, _ = model(Xg, Mg)
    r = Xg - c
    B = corr_matrix(Xg, Mg, MIN_OVERLAP); A = corr_matrix(r, Mg, MIN_OVERLAP)
    off = ~np.eye(len(B), dtype=bool)
    fig, ax = plt.subplots(1, 2, figsize=(11, 5))
    for axi, Mtx, ttl in ((ax[0], B, "before"), (ax[1], A, "after")):
        im = axi.imshow(Mtx, vmin=0, vmax=1, cmap="viridis")
        axi.set_title(f"|pairwise corr| {ttl}   mean={np.nanmean(Mtx[off]):.3f}", fontsize=9)
        fig.colorbar(im, ax=axi, fraction=0.046)
    fig.suptitle("group correlation before vs after correction")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    df = ensure_area_column(df)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, train_tics)
    base_va = Sector14GroupStatDataset(df, val_tics, t_range, "area", GROUP_SIZE, min_valid=16)
    val_ds = AreaGroupAEDataset(base_va.X, base_va.M, base_va.areas, base_va.tics,
                                n_stars=N_STARS, group_size=GROUP_SIZE, seed=SEED,
                                require_full=False, resample=False)
    model = load_model(CKPT)
    metrics = evaluate(model, val_ds)
    flags = reject_flags(metrics)
    plot_examples(model, val_ds, 6, os.path.join(OUT_DIR, "correction_examples.png"))
    plot_corr_matrices(model, val_ds, os.path.join(OUT_DIR, "correlation_before_after.png"))
    report = {"ckpt": CKPT, "lambda_size": LAMBDA_SIZE, "metrics": metrics, "reject_flags": flags}
    with open(os.path.join(OUT_DIR, "eval_correction.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    if flags["any"]:
        print("!! REJECT: " + ", ".join(k for k, v in flags.items() if v and k != "any"), flush=True)
    print(f"wrote correction_examples.png + correlation_before_after.png + eval_correction.json to {OUT_DIR}", flush=True)

if __name__ == "__main__":
    main()
