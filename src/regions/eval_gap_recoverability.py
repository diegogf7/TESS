#All this code is from Claude

import os
import subprocess

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.data.data import normalize, resample_to_grid
from src.loss_function.gap_infill import infill_gaps

DATA_PATH = os.environ.get("GAP_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_probe20k_area.parquet")
N_STARS = int(os.environ.get("GAP_N_STARS", "2000"))
N_BOOTSTRAP = int(os.environ.get("GAP_N_BOOTSTRAP", "500"))
SEED = int(os.environ.get("GAP_SEED", "0"))
GRID_LENGTH = 1024
BATCH = 256

rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def rolling_mean(x, window):
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


def cadence_features(flux):
    """Local texture features for one (L,) flux array. Gaps infilled by
    independent resampling lose local autocorrelation; these features are
    designed to expose exactly that."""
    padded = np.pad(flux, 1, mode="edge")
    d1 = np.abs(padded[1:-1] - padded[:-2])                    # |x_t - x_{t-1}|
    d1_fwd = np.abs(padded[2:] - padded[1:-1])                 # |x_{t+1} - x_t|
    d2 = np.abs(padded[2:] - 2 * padded[1:-1] + padded[:-2])   # second difference
    padded2 = np.pad(flux, 2, mode="edge")
    lag2 = np.abs(padded2[2:-2] - padded2[:-4])                # |x_t - x_{t-2}|
    roll_var = rolling_mean(flux ** 2, 5) - rolling_mean(flux, 5) ** 2
    roll_d1 = rolling_mean(d1, 5)
    return np.stack([d1, d1_fwd, d2, lag2, roll_var, roll_d1], axis=1)


print(f"git commit: {git_commit()}")
print(f"data: {DATA_PATH}")
print(f"config: N_STARS={N_STARS} N_BOOTSTRAP={N_BOOTSTRAP} SEED={SEED}")

df = pd.read_parquet(DATA_PATH, columns=["GAIADR3", "time", "flux"])
stars = df["GAIADR3"].to_numpy()
unique_stars = np.unique(stars)
chosen = rng.choice(unique_stars, size=min(N_STARS, len(unique_stars)), replace=False)
df = df[df["GAIADR3"].isin(set(chosen.tolist()))].reset_index(drop=True)
print(f"{len(df)} curves from {df['GAIADR3'].nunique()} stars")

# --- run the exact production pipeline: normalize -> grid -> infill ---
grid_all, mask_all = [], []
for _, row in df.iterrows():
    flux = normalize(np.array(row["flux"]))
    grid_flux, observed = resample_to_grid(np.array(row["time"]), flux, GRID_LENGTH)
    grid_all.append(grid_flux.astype(np.float32))
    mask_all.append(observed.astype(np.float32))
grid_all = np.stack(grid_all)   # (n_curves, L) zero-filled gaps
mask_all = np.stack(mask_all)   # 1 = observed, 0 = gap

infilled = np.empty_like(grid_all)
for start in range(0, len(grid_all), BATCH):
    flux_t = torch.tensor(grid_all[start:start + BATCH])
    mask_t = torch.tensor(mask_all[start:start + BATCH])
    filled, _ = infill_gaps(flux_t, mask_t)
    infilled[start:start + BATCH] = filled.numpy()

# --- per-cadence design matrices ---
n_curves, L = infilled.shape
X_infilled = np.concatenate([cadence_features(f) for f in infilled])
X_zerofill = np.concatenate([cadence_features(f) for f in grid_all])
y = (1.0 - mask_all).reshape(-1)                       # 1 = was a gap
groups = np.repeat(df["GAIADR3"].to_numpy(), L)        # star id per cadence

y_shuffled = mask_all.copy()
for i in range(n_curves):                              # shuffle labels within each curve
    rng.shuffle(y_shuffled[i])
y_shuffled = (1.0 - y_shuffled).reshape(-1)

prevalence = y.mean()
print(f"{len(y)} cadences, gap prevalence {prevalence:.4f}")

train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2,
                                             random_state=SEED).split(X_infilled, y, groups))
test_stars = groups[test_idx]
unique_test_stars = np.unique(test_stars)
star_rows = {s: np.flatnonzero(test_stars == s) for s in unique_test_stars}
print(f"split: {len(train_idx)} train / {len(test_idx)} test cadences, "
      f"{len(unique_test_stars)} test stars (star-disjoint)")


def bootstrap_ci(y_true, y_score, y_pred, metric_rng):
    aucs, baccs = [], []
    for _ in range(N_BOOTSTRAP):
        pick = metric_rng.choice(unique_test_stars, size=len(unique_test_stars), replace=True)
        rows = np.concatenate([star_rows[s] for s in pick])
        if len(np.unique(y_true[rows])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[rows], y_score[rows]))
        baccs.append(balanced_accuracy_score(y_true[rows], y_pred[rows]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    blo, bhi = np.percentile(baccs, [2.5, 97.5])
    return (lo, hi), (blo, bhi)


def run_variant(name, X, y_var):
    scaler = StandardScaler().fit(X[train_idx])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(scaler.transform(X[train_idx]), y_var[train_idx])

    Xt = scaler.transform(X[test_idx])
    score = clf.predict_proba(Xt)[:, 1]
    pred = clf.predict(Xt)
    y_true = y_var[test_idx]

    auc = roc_auc_score(y_true, score)
    bacc = balanced_accuracy_score(y_true, pred)
    (auc_lo, auc_hi), (b_lo, b_hi) = bootstrap_ci(y_true, score, pred,
                                                  np.random.default_rng(SEED + 1))
    print(f"{name:18s} AUC {auc:.4f} [95% CI {auc_lo:.4f}, {auc_hi:.4f}]  "
          f"bal-acc {bacc:.4f} [{b_lo:.4f}, {b_hi:.4f}]")
    return auc


auc_ceiling = run_variant("zerofill_ceiling", X_zerofill, y)
auc_floor = run_variant("shuffled_floor", X_infilled, y_shuffled)
auc_real = run_variant("infilled (REAL)", X_infilled, y)

print()
if auc_ceiling < 0.9:
    print("WARNING: ceiling control is weak -- features may not detect gaps at all; "
          "the real-data verdict below is not trustworthy.")
if abs(auc_floor - 0.5) > 0.05:
    print("WARNING: shuffled floor deviates from 0.5 -- leakage in the audit itself.")
if auc_real <= 0.55:
    print(f"VERDICT: gaps are NOT meaningfully recoverable after infilling "
          f"(AUC {auc_real:.3f}). Gap-blindness is supported.")
else:
    print(f"VERDICT: gaps ARE partially recoverable after infilling "
          f"(AUC {auc_real:.3f} > 0.55). Do NOT claim gap-blindness; "
          f"fix infilling (e.g. autocorrelation-preserving fill) before "
          f"interpreting sector/chip accuracies as flux-borne.")
