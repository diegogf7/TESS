from __future__ import annotations
"""Validate the shared-S4D per-curve autoencoder.
  full : per-curve SmoothL1/RMSE/Pearson/R^2 (recon vs original) on val + 4-panel
         plots (original / reconstruction / residual / 32-D latent).
  test : the 7 structural tests (self-contained, synthetic).

    EVAL_MODE=full  python -m src.shared_s4d.eval_ae     # needs ckpt + data
    EVAL_MODE=test  python -m src.shared_s4d.eval_ae     # self-contained
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
from src.shared_s4d.ae_dataset import AreaGroupAEDataset
from src.shared_s4d.model import build_model, GRID, LATENT_DIM
from src.shared_s4d.train_ae import (SEED, GROUP_SIZE, N_STARS, DEVICE, S14_DATA, SPLIT_DIR,
                                     BASE_ART_DIR, ART_DIR, CKPT_DIR, masked_smooth_l1)

EVAL_MODE = os.environ.get("EVAL_MODE", "full")
CKPT = os.environ.get("CKPT", os.path.join(CKPT_DIR, f"shared_s4d_ae_g{GROUP_SIZE}_z{LATENT_DIM}_s{SEED}_best.pth"))
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
        ss = float(np.sum((t - t.mean()) ** 2))
        r2 = float(1.0 - np.sum((t - p) ** 2) / ss) if ss > 1e-12 else np.nan
        rows.append((sl1, rmse, corr, r2))
    return np.asarray(rows, dtype=np.float64).reshape(-1, 4)


def evaluate(model, val_ds):
    dl = DataLoader(val_ds, batch_size=1, num_workers=2, worker_init_fn=seed_worker,
                    generator=torch.Generator().manual_seed(SEED))
    chunks = []
    with torch.no_grad():
        for Xg, Mg in dl:
            Xg = Xg.squeeze(0).to(DEVICE); Mg = Mg.squeeze(0).to(DEVICE)
            recon, _ = model(Xg, Mg)
            chunks.append(_per_curve_metrics(recon.cpu().numpy(), Xg.cpu().numpy(), Mg.cpu().numpy()))
    M = np.concatenate([c for c in chunks if len(c)], axis=0)

    def q(col):
        c = M[:, col]; c = c[np.isfinite(c)]
        return {"median": float(np.median(c)), "q1": float(np.percentile(c, 25)),
                "q3": float(np.percentile(c, 75)), "n": int(len(c))}
    return {"smooth_l1": q(0), "rmse": q(1), "pearson": q(2), "r2": q(3)}


def plot_examples(model, val_ds, n, out_path):
    Xg, Mg = val_ds[0]
    with torch.no_grad():
        recon, lat = model(Xg.to(DEVICE), Mg.to(DEVICE))
    x = Xg.numpy(); m = Mg.numpy(); r = recon.cpu().numpy(); z = lat.cpu().numpy()
    g = np.arange(GRID); n = min(n, x.shape[0])
    fig, axes = plt.subplots(n, 2, figsize=(15, 2.4 * n),
                             gridspec_kw={"width_ratios": [3, 1]})
    axes = np.atleast_2d(axes)
    for i in range(n):
        obs = m[i] > 0
        axL = axes[i, 0]
        axL.plot(g[obs], x[i][obs], ".", ms=2, color="0.6", label="original")
        axL.plot(g, r[i], color="tab:red", lw=0.8, label="reconstruction")
        axL.plot(g[obs], (x[i] - r[i])[obs], ".", ms=1.5, color="tab:blue", label="residual")
        if i == 0:
            axL.legend(fontsize=7, ncol=3, loc="upper right")
        axL.set_ylabel(f"curve {i}")
        axes[i, 1].bar(np.arange(LATENT_DIM), z[i], color="tab:green")
        if i == 0:
            axes[i, 1].set_title("32-D latent", fontsize=8)
    fig.suptitle("shared-S4D autoencoder: original / reconstruction / residual  +  latent")
    fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def run_tests():
    """The 7 structural tests, self-contained on synthetic data."""
    import src.shared_s4d.train_ae as T
    GS = GROUP_SIZE
    rng = np.random.default_rng(0)
    areas, tics = [], []
    for a in (111, 232):
        areas += [a] * 1000; tics += [f"T{a}_{i}" for i in range(1000)]
    N = len(areas)
    X = rng.standard_normal((N, GRID)).astype(np.float32); Mfull = np.ones((N, GRID), np.float32)
    ds = AreaGroupAEDataset(X, Mfull, np.array(areas), np.array(tics),
                            n_stars=1000, group_size=GS, seed=0, require_full=True, resample=True)

    # 1) exactly 31 groups from 1000 curves per area
    per = {}
    for rows, a in ds.items:
        per[a] = per.get(a, 0) + 1
    assert per == {111: 31, 232: 31}, per

    # 2) every group = 32 curves from ONE area
    for rows, a in ds.items:
        assert len(rows) == GS and set(ds.areas[rows].tolist()) == {a}, "group spans areas / wrong size"

    # 3) groups change deterministically between epochs
    ds.set_epoch(1); e1 = [(r.tobytes(), a) for r, a in ds.items]
    ds.set_epoch(2); e2 = [(r.tobytes(), a) for r, a in ds.items]
    ds.set_epoch(1); e1b = [(r.tobytes(), a) for r, a in ds.items]
    assert e1 != e2 and e1 == e1b, "epoch determinism/rotation broken"

    torch.manual_seed(0); model = build_model()
    Xg, Mg = ds[0]
    with torch.no_grad():
        recon, _ = model(Xg, Mg)

    # 4) reconstruction matched to the CORRECT original (matched target fits better than shuffled)
    perm = torch.randperm(GS)
    assert float(masked_smooth_l1(recon, Xg, Mg)) < float(masked_smooth_l1(recon, Xg[perm], Mg[perm])), \
        "reconstruction not aligned with its own original"

    # 5) missing cadences do not affect the loss
    Mg2 = Mg.clone(); Mg2[:, :100] = 0.0
    l0 = float(masked_smooth_l1(recon, Xg, Mg2))
    rc = recon.clone(); rc[:, :100] += 1e3            # corrupt ONLY masked cadences
    assert abs(l0 - float(masked_smooth_l1(rc, Xg, Mg2))) < 1e-4, "masked cadences leaked into loss"

    # 6) exactly one optimizer step per complete group
    steps = {"n": 0}
    class CountAdamW(torch.optim.AdamW):
        def step(self, *a, **k):
            steps["n"] += 1
            return super().step(*a, **k)
    torch.manual_seed(0); m2 = build_model()
    opt = CountAdamW(m2.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    old = T.MAX_BATCHES; T.MAX_BATCHES = 3
    T.run_train_epoch(m2, DataLoader(ds, batch_size=1, shuffle=False), opt, scaler)
    T.MAX_BATCHES = old
    assert steps["n"] == 3, f"expected 3 steps (one/group), got {steps['n']}"

    # 7) encoder + decoder are single SHARED modules applied to all curves
    m3 = build_model()
    r3, _ = m3(Xg, Mg)
    masked_smooth_l1(r3, Xg, Mg).backward()
    assert m3.encoder.encoder.weight.grad is not None, "shared encoder got no gradient"
    assert m3.decoder[0].weight.grad is not None, "shared decoder got no gradient"
    n_enc = sum(p.numel() for p in m3.encoder.parameters())
    with torch.no_grad():
        assert m3(Xg[:8], Mg[:8])[0].shape[0] == 8 and m3(Xg, Mg)[0].shape[0] == 32
    assert sum(p.numel() for p in m3.encoder.parameters()) == n_enc, "param count changed with group size"

    print("ALL 7 TESTS PASSED: 31 groups, 32/area, epoch-determinism, recon-alignment,\n"
          "  masked-cadence invariance, one-step-per-group, shared enc/dec params", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    if EVAL_MODE == "test":
        run_tests()
        return

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
    plot_examples(model, val_ds, n=6, out_path=os.path.join(OUT_DIR, "ae_examples.png"))
    run_tests()
    with open(os.path.join(OUT_DIR, "eval_ae.json"), "w") as fh:
        json.dump({"ckpt": CKPT, "metrics_recon_vs_original": metrics, "n_val_groups": len(val_ds)}, fh, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"wrote ae_examples.png + eval_ae.json to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
