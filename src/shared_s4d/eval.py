from __future__ import annotations
"""Validate the shared-S4D systematics model.
  full      : plots (input / LOO target / predicted systematics / cleaned) +
              per-curve SmoothL1, RMSE, Pearson vs LOO target on the val split.
  synthetic : shared straight-line systematic -> a freshly trained model must
              recover the line from a single curve (architecture sanity, no data).

    EVAL_MODE=full      python -m src.shared_s4d.eval      # needs a trained ckpt + data
    EVAL_MODE=synthetic python -m src.shared_s4d.eval      # self-contained
"""

import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.instrument_v2.area_commonmode_dataset import Sector14GroupStatDataset, ensure_area_column
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.train_sector14_jepa import seed_worker
from src.shared_s4d.dataset import AreaGroupLOODataset
from src.shared_s4d.model import build_model, GRID, LATENT_DIM
from src.shared_s4d.train import (SEED, GROUP_SIZE, N_STARS, TARGET_MIN_VALID, DEVICE,
                                  S14_DATA, SPLIT_DIR, BASE_ART_DIR, ART_DIR, CKPT_DIR,
                                  masked_smooth_l1)

EVAL_MODE = os.environ.get("EVAL_MODE", "full")
CKPT = os.environ.get("CKPT", os.path.join(CKPT_DIR, f"shared_s4d_g{GROUP_SIZE}_z{LATENT_DIM}_s{SEED}_best.pth"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(ART_DIR, "eval"))


def load_model(ckpt_path):
    model = build_model().to(DEVICE)
    ck = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _per_curve_metrics(pred, target, valid):
    """SmoothL1 / RMSE / Pearson / R^2 per curve over VALID cadences. arrays (N, L)."""
    rows = []
    for i in range(pred.shape[0]):
        v = valid[i] > 0
        if v.sum() < 2:
            continue
        p, t = pred[i][v], target[i][v]
        sl1 = float(F.smooth_l1_loss(torch.tensor(p), torch.tensor(t)))
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        corr = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-8 and t.std() > 1e-8 else np.nan
        ss_tot = float(np.sum((t - t.mean()) ** 2))
        r2 = float(1.0 - np.sum((t - p) ** 2) / ss_tot) if ss_tot > 1e-12 else np.nan
        rows.append((sl1, rmse, corr, r2))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 4)



def evaluate(model, val_ds):
    dl = DataLoader(val_ds, batch_size=8, num_workers=2, worker_init_fn=seed_worker,
                    generator=torch.Generator().manual_seed(SEED))
    chunks = []
    with torch.no_grad():
        for Xg, Mg, Tg, Vg in dl:
            b = Xg.shape[0] * GROUP_SIZE
            x = Xg.reshape(b, GRID).to(DEVICE); m = Mg.reshape(b, GRID).to(DEVICE)
            s_hat, _ = model(x, m)
            chunks.append(_per_curve_metrics(s_hat.cpu().numpy(),
                                             Tg.reshape(b, GRID).numpy(),
                                             Vg.reshape(b, GRID).numpy()))
    M = np.concatenate([c for c in chunks if len(c)], axis=0)

    def q(col):
        c = M[:, col]; c = c[np.isfinite(c)]
        return {"median": float(np.median(c)), "q1": float(np.percentile(c, 25)),
                "q3": float(np.percentile(c, 75)), "n": int(len(c))}
    return {"smooth_l1": q(0), "rmse": q(1), "pearson": q(2), "r2": q(3)}


def plot_examples(model, val_ds, n, out_path):
    Xg, Mg, Tg, Vg = val_ds[0]                            # first group (32 curves)
    with torch.no_grad():
        s_hat, _ = model(Xg.to(DEVICE), Mg.to(DEVICE))
    x = Xg.numpy(); m = Mg.numpy(); T = Tg.numpy(); V = Vg.numpy(); s = s_hat.cpu().numpy()
    g = np.arange(GRID)
    n = min(n, x.shape[0])
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharex=True)
    axes = np.atleast_1d(axes)
    for i in range(n):
        ax = axes[i]; obs = m[i] > 0; val = V[i] > 0
        ax.plot(g[obs], x[i][obs], ".", ms=2, color="0.6", label="input x")
        ax.plot(g[val], T[i][val], color="tab:green", lw=1.0, label="LOO target")
        ax.plot(g, s[i], color="tab:red", lw=1.0, label="pred systematics")
        ax.plot(g[obs], (x[i] - s[i])[obs], ".", ms=2, color="tab:blue", label="cleaned x - s")
        if i == 0:
            ax.legend(fontsize=7, ncol=4, loc="upper right")
        ax.set_ylabel(f"curve {i}")
    fig.suptitle("shared-S4D systematics: input / LOO target / prediction / cleaned")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def synthetic_sanity(n_groups=16, steps=400, batch_groups=4, noise=0.05, seed=0):
    """Shared straight line per group + per-star noise; a fresh model trained on
    the LOO medians must recover the line from ONE curve (corr > 0.9)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(-1, 1, GRID).astype(np.float32)
    Xs, Ts, lines = [], [], []
    for _ in range(n_groups):
        line = (rng.uniform(-2, 2) * t + rng.uniform(-1, 1)).astype(np.float32)
        X = line[None, :] + noise * rng.standard_normal((GROUP_SIZE, GRID)).astype(np.float32)
        T = np.stack([np.median(np.delete(X, i, axis=0), axis=0) for i in range(GROUP_SIZE)])
        Xs.append(X); Ts.append(T.astype(np.float32)); lines.append(line)
    Xall = torch.tensor(np.stack(Xs)); Tall = torch.tensor(np.stack(Ts))
    model = build_model().to(DEVICE); opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    loss = torch.tensor(0.0)
    for _ in range(steps):
        idx = rng.integers(0, n_groups, size=batch_groups)
        x = Xall[idx].reshape(-1, GRID).to(DEVICE); m = torch.ones_like(x)
        tt = Tall[idx].reshape(-1, GRID).to(DEVICE); vv = torch.ones_like(tt)
        s, _ = model(x, m)
        loss = masked_smooth_l1(s, tt, vv)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        one = Xall[0, 0:1].to(DEVICE)
        pred = model(one, torch.ones_like(one))[0].cpu().numpy()[0]
    corr = float(np.corrcoef(pred, lines[0])[0, 1])
    print(f"synthetic sanity: corr(pred, line)={corr:.4f}  final_loss={float(loss):.5f}", flush=True)
    assert corr > 0.9, f"straight-line systematic not recovered (corr={corr:.3f})"
    return corr


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if EVAL_MODE == "synthetic":
        synthetic_sanity()
        return

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    df = ensure_area_column(df)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, train_tics)
    base_va = Sector14GroupStatDataset(df, val_tics, t_range, "area", GROUP_SIZE, min_valid=16)
    val_ds = AreaGroupLOODataset(base_va.X, base_va.M, base_va.areas, base_va.tics,
                                 n_stars=N_STARS, group_size=GROUP_SIZE,
                                 target_min_valid=TARGET_MIN_VALID, seed=SEED,
                                 require_full=False, resample=False)

    model = load_model(CKPT)
    metrics = evaluate(model, val_ds)
    plot_examples(model, val_ds, n=6, out_path=os.path.join(OUT_DIR, "examples.png"))
    corr = synthetic_sanity()
    report = {"ckpt": CKPT, "metrics_vs_loo_target": metrics, "synthetic_line_corr": corr,
              "n_val_groups": len(val_ds)}
    with open(os.path.join(OUT_DIR, "eval.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote examples.png + eval.json to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
