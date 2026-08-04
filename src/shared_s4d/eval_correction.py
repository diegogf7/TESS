from __future__ import annotations
"""Per-curve validation for the self-supervised correction model.
Overall before/after correlation PLUS per-curve top-K fixed-covariance metrics,
the high_shared_signal_but_flat_correction flag, and two plot sets
(random curves + the strongest-systematics curves).
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
from src.shared_s4d.model import build_model, experiment_tag, GRID, LATENT_DIM, N_TOKENS, TOKEN_DIM
from src.shared_s4d.correction_losses import (_select_topk_peers, mean_abs_pairwise_corr,
                                              pairwise_window_cov_loss)
from src.shared_s4d.train_correction import (SEED, GROUP_SIZE, N_STARS, LAMBDA_SIZE, MIN_OVERLAP,
                                             TOPK_PEERS, LOSS_MODE, GROUPING_MODE, COLLAPSE_STD, DEVICE,
                                             S14_DATA, SPLIT_DIR, BASE_ART_DIR, ART_DIR, CKPT_DIR)
from src.instrument_v2.train_sector14_jepa import effective_rank

CKPT = os.environ.get("CKPT", os.path.join(CKPT_DIR, experiment_tag(   # auto-locate via shared tag
    LOSS_MODE, GROUPING_MODE, GROUP_SIZE, N_TOKENS, TOKEN_DIM, LAMBDA_SIZE, SEED) + "_best.pth"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(ART_DIR, "eval"))
FLAT_THRESH = 0.05


def load_model(ckpt_path):
    model = build_model().to(DEVICE)
    ck = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ck["model"] if "model" in ck else ck)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


@torch.no_grad()
def _group_percurve(Xg, Mg, corrections):
    """Per-curve metrics for one group. Returns dict of (N,) arrays."""
    m = (Mg > 0).to(Xg.dtype)
    xmean = (Xg * m).sum(1) / m.sum(1).clamp(min=1.0)
    xc = (Xg - xmean[:, None]) * m
    xvar = (xc * xc).sum(1) / m.sum(1).clamp(min=1.0)               # (N,)
    resid = Xg - corrections
    peers = _select_topk_peers(Xg, Mg, TOPK_PEERS, MIN_OVERLAP)
    N = Xg.shape[0]
    before = np.full(N, np.nan); after = np.full(N, np.nan); cpm = np.full(N, np.nan)

    def fixed_cov(a, i, j):                                         # Cov over shared / sqrt(xvar_i xvar_j)
        sh = m[i] * m[j]; n = sh.sum().clamp(min=1.0)
        am = (a[i] * sh).sum() / n; bm = (a[j] * sh).sum() / n
        cov = (((a[i] - am) * sh) * ((a[j] - bm) * sh)).sum() / n
        return float((cov / (torch.sqrt(xvar[i] * xvar[j]) + 1e-6)) ** 2)

    for i in range(N):
        p = peers[i]
        if p.numel() == 0:
            continue
        pl = p.tolist()
        before[i] = float(np.mean([fixed_cov(Xg, i, j) for j in pl]))
        after[i] = float(np.mean([fixed_cov(resid, i, j) for j in pl]))
        pmask = (m[pl].sum(0) > 0).to(Xg.dtype)                     # cadences some peer observed
        peer_mean = (Xg[pl] * m[pl]).sum(0) / m[pl].sum(0).clamp(min=1.0)
        v = pmask > 0
        c = corrections[i]
        if int(v.sum()) >= 2 and c[v].std() > 1e-8 and peer_mean[v].std() > 1e-8:
            cpm[i] = float(torch.corrcoef(torch.stack([c[v], peer_mean[v]]))[0, 1])

    crms = torch.sqrt((m * corrections ** 2).sum(1) / m.sum(1).clamp(min=1.0))
    xrms = torch.sqrt((m * Xg ** 2).sum(1) / m.sum(1).clamp(min=1.0)).clamp(min=1e-8)
    clnrms = torch.sqrt((m * resid ** 2).sum(1) / m.sum(1).clamp(min=1.0))
    return {"before": before, "after": after, "corr_c_peermean": cpm,
            "c_over_x": (crms / xrms).cpu().numpy(), "cleaned_over_x": (clnrms / xrms).cpu().numpy()}


def _q(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if a.size == 0:
        return {"median": np.nan, "q1": np.nan, "q3": np.nan, "p10": np.nan, "p90": np.nan, "n": 0}
    return {"median": float(np.median(a)), "q1": float(np.percentile(a, 25)),
            "q3": float(np.percentile(a, 75)), "p10": float(np.percentile(a, 10)),
            "p90": float(np.percentile(a, 90)), "n": int(a.size)}


def evaluate(model, val_ds):
    dl = DataLoader(val_ds, batch_size=1, num_workers=2, worker_init_fn=seed_worker,
                    generator=torch.Generator().manual_seed(SEED))
    cols = {k: [] for k in ("before", "after", "corr_c_peermean", "c_over_x", "cleaned_over_x")}
    befs, afts, caps, pwbs, pwas = [], [], [], [], []
    corr_chunks = []                                               # subsample of corrections for effective rank
    n_nan = 0
    strongest = []                                                 # top-6 curves by before-cov, for plotting
    with torch.no_grad():
        for Xg, Mg in dl:
            Xg = Xg.squeeze(0).to(DEVICE); Mg = Mg.squeeze(0).to(DEVICE)
            corr, _ = model(Xg, Mg)
            n_nan += int((~torch.isfinite(corr)).sum())            # NaN/Inf in corrections
            pwbs.append(float(pairwise_window_cov_loss(Xg, Xg, Mg)))          # PRIMARY: pairwise-window loss
            pwas.append(float(pairwise_window_cov_loss(Xg - corr, Xg, Mg)))
            befs.append(mean_abs_pairwise_corr(Xg, Mg, MIN_OVERLAP))
            afts.append(mean_abs_pairwise_corr(Xg - corr, Mg, MIN_OVERLAP))
            m = (Mg > 0).float()                                   # soft-cap ratio = mean(c^2)/var(x)
            xmean = (Xg * m).sum(1) / m.sum(1).clamp(min=1.0)
            xvar = (((Xg - xmean[:, None]) * m) ** 2).sum(1) / m.sum(1).clamp(min=1.0) + 1e-6
            energy_ratio = (m * corr ** 2).sum(1) / m.sum(1).clamp(min=1.0) / xvar
            caps.append(float((energy_ratio > 0.5).float().mean()))
            pc = _group_percurve(Xg, Mg, corr)
            for k in cols:
                cols[k].append(pc[k])
            if len(corr_chunks) < 400:                             # cap ~ up to 400 groups for the SVD
                corr_chunks.append(corr.cpu().numpy())
            bc = pc["before"]
            for i in range(Xg.shape[0]):                            # track strongest-systematics curves
                if np.isfinite(bc[i]):
                    strongest.append((bc[i], Xg[i].cpu().numpy(), corr[i].cpu().numpy(), Mg[i].cpu().numpy()))
            strongest = sorted(strongest, key=lambda t: -t[0])[:6]

    A = {k: np.concatenate(v) for k, v in cols.items()}
    corr_mat = np.concatenate(corr_chunks, 0) if corr_chunks else np.zeros((1, GRID))
    corr_erank = float(effective_rank(corr_mat))                  # effective rank of the CORRECTION shapes
    pct = 100.0 * (A["before"] - A["after"]) / np.where(A["before"] > 1e-8, A["before"], np.nan)

    # failure flag: top-quartile ORIGINAL shared cov AND flat correction
    hi = A["before"] >= np.nanpercentile(A["before"], 75)
    flat = A["c_over_x"] < FLAT_THRESH
    n_flag = int(np.sum(hi & flat))
    pwb = float(np.mean(pwbs)); pwa = float(np.mean(pwas))
    metrics = {
        "pw_before": pwb, "pw_after": pwa,
        "pw_reduction": float(100.0 * (pwb - pwa) / pwb) if pwb > 1e-12 else float("nan"),
        "overall_corr_before": float(np.nanmean(befs)), "overall_corr_after": float(np.nanmean(afts)),
        "topk_cov_before": _q(A["before"]), "topk_cov_after": _q(A["after"]),
        "pct_reduction_per_curve": _q(pct), "c_over_x": _q(A["c_over_x"]),
        "cleaned_over_x": _q(A["cleaned_over_x"]), "corr_correction_peermean": _q(A["corr_c_peermean"]),
        "frac_flat_correction": float(np.mean(A["c_over_x"] < FLAT_THRESH)),
        "frac_over_cap": float(np.mean(caps)), "correction_effective_rank": corr_erank,
        "n_nan_corrections": int(n_nan), "n_curves": int(A["before"].size),
    }
    flags = {
        "high_shared_signal_but_flat_correction": {"n": n_flag,
            "frac": float(n_flag / max(1, int(hi.sum())))},
        "correction_nearly_identical_to_input": bool(_q(A["c_over_x"])["median"] > 0.9),
        "cleaned_nearly_zero": bool(_q(A["cleaned_over_x"])["median"] < 0.1),
        "nan_in_corrections": bool(n_nan > 0),
    }
    flags["any_degeneracy"] = bool(n_flag > 0 or flags["correction_nearly_identical_to_input"]
                                   or flags["cleaned_nearly_zero"] or flags["nan_in_corrections"])
    return metrics, flags, strongest


def plot_curves(items, out_path, title):
    n = len(items)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharex=True)
    axes = np.atleast_1d(axes)
    g = np.arange(GRID)
    for k, (bc, x, c, mk) in enumerate(items):
        obs = mk > 0; ax = axes[k]
        ax.plot(g[obs], x[obs], ".", ms=2, color="0.6", label="original")
        ax.plot(g, c, color="tab:red", lw=0.9, label="correction c")
        ax.plot(g[obs], (x - c)[obs], ".", ms=2, color="tab:blue", label="cleaned x-c")
        if k == 0:
            ax.legend(fontsize=7, ncol=3, loc="upper right")
        ax.set_ylabel(f"before_cov={bc:.2f}")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(out_path, dpi=130); plt.close(fig)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == 14].drop_duplicates("TIC").reset_index(drop=True)
    df = ensure_area_column(df)
    tr, va, te = ensure_splits(SPLIT_DIR, BASE_ART_DIR)
    t_range = ensure_time_range(BASE_ART_DIR, df, tr)
    # base loader only supplies X/M/areas/tics; fix its group_size at 32 so min_valid=16
    # stays valid for GROUP_SIZE<16 (mirrors the train_correction fix).
    base_va = Sector14GroupStatDataset(df, va, t_range, "area", 32, min_valid=16)

    def coords(cols):                                              # STAR_X/STAR_Y or ra/dec aligned to val stars
        if not set(cols) <= set(df.columns):
            raise RuntimeError(f"GROUPING_MODE={GROUPING_MODE} needs {cols}, absent from parquet "
                               "-- regenerate the dataset (see train_correction.detxy_for).")
        sub = df.set_index(df["TIC"].astype(str))
        return sub.loc[[str(t) for t in base_va.tics], cols].to_numpy(dtype=float)
    radec = coords(["ra", "dec"]) if GROUPING_MODE == "nearest" else None
    detxy = coords(["STAR_X", "STAR_Y"]) if GROUPING_MODE == "detector_nearest" else None
    val_ds = AreaGroupAEDataset(base_va.X, base_va.M, base_va.areas, base_va.tics,
                                n_stars=N_STARS, group_size=GROUP_SIZE, seed=SEED,
                                require_full=False, resample=False,
                                grouping_mode=GROUPING_MODE, radec=radec, detxy=detxy)
    model = load_model(CKPT)
    metrics, flags, strongest = evaluate(model, val_ds)

    # plot 1: random curves (first group); plot 2: strongest-systematics curves
    Xg, Mg = val_ds[0]
    with torch.no_grad():
        c0, _ = model(Xg.to(DEVICE), Mg.to(DEVICE))
    rand_items = [(np.nan, Xg[i].numpy(), c0[i].cpu().numpy(), Mg[i].numpy()) for i in range(6)]
    plot_curves(rand_items, os.path.join(OUT_DIR, "correction_random.png"), "random validation curves")
    plot_curves(strongest, os.path.join(OUT_DIR, "correction_strongest.png"),
                "strongest-systematics curves (do they still get flat corrections?)")

    report = {"ckpt": CKPT, "loss_mode": LOSS_MODE, "lambda_size": LAMBDA_SIZE,
              "topk_peers": TOPK_PEERS, "metrics": metrics, "reject_flags": flags}
    with open(os.path.join(OUT_DIR, "eval_correction.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    if flags["any_degeneracy"]:
        print("!! DEGENERACY FLAG(S) TRIPPED", flush=True)
    print(f"wrote correction_random.png + correction_strongest.png + eval_correction.json to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
