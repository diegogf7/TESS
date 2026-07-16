# All this code is from Claude
"""Frozen-encoder probe evaluation for the Sector-14 chip-pair JEPA.

Protocol (leak-proof by construction):
  - encoders are FROZEN (final checkpoints; no best-val selection);
  - a balanced logistic-regression probe is fit on TRAIN latents;
  - probe hyperparameters (C x PCA dim) are selected on VALIDATION only;
  - the selected probe is evaluated exactly ONCE on TEST TICs.

Rows reported: shared-grid JEPA and legacy-grid JEPA (seeds 0/1/2), a random
untrained encoder per arm, and the Step-1 shared-grid PCA reference read from
the diagnostic's results.csv. Per-seed metrics plus mean/std/bootstrap-CI
across seeds.

Run:  python -m src.instrument_v2.eval_sector14_jepa
Env:  S14_DATA, SPLIT_DIR, ART_DIR, CKPT_DIR, SEEDS (default 0,1,2), ARMS
"""

import json
import os
import subprocess

import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.instrument_v2.diagnose_chip_common_signal import chip_index, pair_cosine_auc
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range, grid_frame
from src.instrument_v2.train_sector14_jepa import effective_rank
from src.loss_function.gapblind_fix import build_gapblind_jepa

SECTOR = 14
S14_DATA = os.environ.get("S14_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get("SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
ART_DIR = os.environ.get("ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))
CKPT_DIR = os.environ.get("CKPT_DIR", "/orcd/scratch/orcd/006/diegogon/checkpoints")
SEEDS = tuple(int(s) for s in os.environ.get("SEEDS", "0,1,2").split(","))
ARMS = tuple(os.environ.get("ARMS", "shared,legacy").split(","))
BATCH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

C_GRID = (0.01, 0.1, 1.0, 10.0)
PCA_GRID = (None, 32, 64)
N_BOOTSTRAP = 1000


def encode_all(model, X, M):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(X), BATCH):
            f = torch.tensor(X[start:start + BATCH]).to(DEVICE)
            m = torch.tensor(M[start:start + BATCH]).to(DEVICE)
            z = model.encode(f, m)
            pieces.append(z.reshape(z.shape[0], -1).cpu().numpy())
    return np.concatenate(pieces)


def probe_with_val_selection(Ztr, ytr, Zval, yval, Zte, yte):
    """Fit on train, pick (C, pca) on validation bacc16, evaluate ONCE on test."""
    best = None
    for pca_dim in PCA_GRID:
        for C in C_GRID:
            steps = [StandardScaler()]
            if pca_dim is not None and pca_dim < Ztr.shape[1]:
                steps.append(PCA(n_components=pca_dim, random_state=0))
            steps.append(LogisticRegression(max_iter=3000, C=C, class_weight="balanced"))
            clf = make_pipeline(*steps)
            clf.fit(Ztr, ytr)
            val_bacc = balanced_accuracy_score(yval, clf.predict(Zval))
            if best is None or val_bacc > best[0]:
                best = (val_bacc, C, pca_dim, clf)
    val_bacc, C, pca_dim, clf = best
    pred = clf.predict(Zte)                      # the single test evaluation
    return {
        "probe_C": C, "probe_pca": pca_dim or 0, "val_bacc16": val_bacc,
        "bacc_16way": balanced_accuracy_score(yte, pred),
        "bacc_camera": balanced_accuracy_score(yte // 4, pred // 4),
        "bacc_ccd": balanced_accuracy_score(yte % 4, pred % 4),
        "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
    }, pred


def bootstrap_ci_bacc(yte, pred, n_boot=N_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(yte))
    scores = [balanced_accuracy_score(yte[i], pred[i])
              for i in (rng.choice(idx_all, len(idx_all), replace=True)
                        for _ in range(n_boot))]
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def step1_pca_reference():
    path = os.path.join(SPLIT_DIR, "results.csv")
    if not os.path.exists(path):
        return None
    r = pd.read_csv(path)
    r = r[(r["representation"] == "shared_sector_1024") & (r["condition"] == "real")]
    if len(r) == 0:
        return None
    best = r.loc[r["bacc_16way"].idxmax()]
    return {"model": "step1_pca_shared", "K": int(best["K"]),
            "bacc_16way": float(best["bacc_16way"]),
            "bacc_camera": float(best["bacc_camera"]),
            "bacc_ccd": float(best["bacc_ccd"])}


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    os.makedirs(ART_DIR, exist_ok=True)
    print(f"git commit: {git_commit()}")
    print(f"config: arms={ARMS} seeds={SEEDS} ckpts={CKPT_DIR} -> {ART_DIR}")
    print(f"data: {S14_DATA}")

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == SECTOR].drop_duplicates("TIC").reset_index(drop=True)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, ART_DIR)
    t_range = ensure_time_range(ART_DIR, df, train_tics)

    tic = df["TIC"].astype(str)
    part = np.where(tic.isin(train_tics), "train",
                    np.where(tic.isin(val_tics), "val",
                             np.where(tic.isin(test_tics), "test", "none")))
    chips = np.array([chip_index(c, d) for c, d in zip(df["camera"], df["ccd"])])
    print(f"stars: {int((part=='train').sum())} train / {int((part=='val').sum())} val "
          f"/ {int((part=='test').sum())} test")

    grids = {}
    for arm in ARMS:
        grids[arm] = grid_frame(df, arm, t_range)
        print(f"gridded arm {arm}")

    rows = []
    for arm in ARMS:
        X, M = grids[arm]
        for kind, seed in [("jepa", s) for s in SEEDS] + [("random", 12345)]:
            if kind == "jepa":
                path = os.path.join(CKPT_DIR, f"s14jepa_{arm}_s{seed}.pth")
                if not os.path.exists(path):
                    print(f"MISSING checkpoint {path} -- skipping")
                    continue
                model = build_gapblind_jepa().to(DEVICE)
                model.load_state_dict(torch.load(path, map_location=DEVICE))
            else:
                torch.manual_seed(seed)          # fixed random init, reproducible
                model = build_gapblind_jepa().to(DEVICE)
            Z = encode_all(model, X, M)
            Ztr, ytr = Z[part == "train"], chips[part == "train"]
            Zval, yval = Z[part == "val"], chips[part == "val"]
            Zte, yte = Z[part == "test"], chips[part == "test"]
            met, pred = probe_with_val_selection(Ztr, ytr, Zval, yval, Zte, yte)
            lo, hi = bootstrap_ci_bacc(yte, pred)
            row = {"model": f"{kind}_{arm}", "arm": arm, "seed": seed,
                   "bacc_16way_lo": lo, "bacc_16way_hi": hi,
                   "pair_cosine_auc": pair_cosine_auc(Zte, yte),
                   "latent_std": float(Zte.std(axis=0).mean()),
                   "effective_rank": effective_rank(Zte), **met}
            rows.append(row)
            print(f"{row['model']:16s} seed {seed:5d}  "
                  f"bacc16 {row['bacc_16way']:.4f} [{lo:.4f},{hi:.4f}]  "
                  f"cam {row['bacc_camera']:.4f}  ccd {row['bacc_ccd']:.4f}  "
                  f"f1 {row['macro_f1']:.4f}  aucPair {row['pair_cosine_auc']:.4f}  "
                  f"std {row['latent_std']:.3f}  erank {row['effective_rank']:.1f}  "
                  f"(C={row['probe_C']}, pca={row['probe_pca']}, val {row['val_bacc16']:.4f})",
                  flush=True)

    # ---------------- aggregate across seeds (JEPA rows only) ----------------
    agg = []
    rng = np.random.default_rng(0)
    for arm in ARMS:
        vals = {m: [r[m] for r in rows if r["model"] == f"jepa_{arm}"]
                for m in ("bacc_16way", "bacc_camera", "bacc_ccd", "macro_f1",
                          "pair_cosine_auc", "latent_std", "effective_rank")}
        if not vals["bacc_16way"]:
            continue
        entry = {"model": f"jepa_{arm}", "n_seeds": len(vals["bacc_16way"])}
        for m, v in vals.items():
            v = np.asarray(v, dtype=float)
            boots = [np.mean(rng.choice(v, len(v), replace=True))
                     for _ in range(N_BOOTSTRAP)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            entry[m] = {"mean": float(v.mean()), "std": float(v.std(ddof=0)),
                        "ci95": [float(lo), float(hi)], "per_seed": v.tolist()}
        agg.append(entry)
        b = entry["bacc_16way"]
        print(f"AGG {entry['model']:14s} bacc16 {b['mean']:.4f} +/- {b['std']:.4f} "
              f"ci[{b['ci95'][0]:.4f},{b['ci95'][1]:.4f}] over {entry['n_seeds']} seeds")

    ref = step1_pca_reference()
    if ref:
        print(f"REFERENCE step-1 PCA (shared, K={ref['K']}): "
              f"bacc16 {ref['bacc_16way']:.4f}  cam {ref['bacc_camera']:.4f}  "
              f"ccd {ref['bacc_ccd']:.4f}")

    pd.DataFrame(rows).to_csv(os.path.join(ART_DIR, "eval_results.csv"), index=False)
    with open(os.path.join(ART_DIR, "eval_summary.json"), "w") as fh:
        json.dump({"git_commit": git_commit(), "rows": rows, "aggregate": agg,
                   "step1_pca_reference": ref}, fh, indent=2, default=float)
    print(f"\nwrote eval_results.csv and eval_summary.json to {ART_DIR}")


if __name__ == "__main__":
    main()
