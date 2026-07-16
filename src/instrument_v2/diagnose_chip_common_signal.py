# All this code is from Claude
"""Diagnostic: do Sector-14 light curves share a cadence-aligned common-mode
signal per camera x CCD chip?

Method: fit a per-chip PCA/SVD "common mode" basis on TRAIN stars only, then
classify each TEST star as the chip whose basis reconstructs it best
(masked least-squares, error on genuinely observed cadences only). If exact
cadence alignment beats the legacy per-curve grid, the chip signal lives in
shared temporal structure -- the motivation for a cadence-aligned JEPA.

Three representations of the same curves:
  legacy_local_1024   per-curve linspace(time[0], time[-1], 1024)
                      (mirrors src/data/data.py resample_to_grid, torch-free)
  shared_sector_1024  one global Sector-14 time range -> 1024 bins for all stars
  exact_cadence       cadence_num - first_cadence -> tensor index
                      (mirrors src/data/cadence_dataset.py, torch-free)

Controls: mask-only LR, within-curve shuffled flux, summary-stats-only LR.

Requires a parquet WITH cadence_num (src/tglc/extract_raw_parquet_cadence.py).
Fails loudly if none is found -- never falls back to relative time.

Run:  python -m src.instrument_v2.diagnose_chip_common_signal
Env:  DIAG_DATA (explicit parquet), DIAG_SECTOR, ART_DIR, N_BOOTSTRAP, SEED
Tests: python -m src.tests.test_chip_common_signal
"""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd

from scipy.interpolate import interp1d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = "/orcd/scratch/orcd/006/diegogon/tglc_primary"
CANDIDATE_PARQUETS = (
    os.path.join(DATA_DIR, "tglc_raw_cadence_all.parquet"),
    os.path.join(DATA_DIR, "tglc_raw_cadence_s14.parquet"),
)
REQUIRED_COLUMNS = ("TIC", "sector", "camera", "ccd", "time", "flux", "cadence_num")
QUALITY_COLUMNS = ("TESS_flags", "TGLC_flags")

SECTOR = int(os.environ.get("DIAG_SECTOR", "14"))
SEED = int(os.environ.get("SEED", "42"))
K_LIST = tuple(int(k) for k in os.environ.get("K_LIST", "1,4,8,16").split(","))
GRID_1024 = 1024
N_BOOTSTRAP = int(os.environ.get("N_BOOTSTRAP", "1000"))
MAX_PAIR_STARS = 600  # cap for the O(n^2) star-pair cosine AUC
ART_DIR = os.environ.get("ART_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))

N_CHIPS = 16


# ---------------------------------------------------------------- data
def find_dataset():
    """First candidate parquet whose schema has every required column.

    Reads only parquet footers (cheap). Fails loudly if nothing qualifies --
    per spec we NEVER fall back to relative time when cadence_num is missing.
    """
    import pyarrow.parquet as pq

    explicit = os.environ.get("DIAG_DATA")
    candidates = (explicit,) if explicit else CANDIDATE_PARQUETS
    report = []
    for path in candidates:
        if not os.path.exists(path):
            report.append(f"  {path}: does not exist")
            continue
        names = set(pq.ParquetFile(path).schema_arrow.names)
        missing = [c for c in REQUIRED_COLUMNS if c not in names]
        if missing:
            report.append(f"  {path}: missing columns {missing}")
            continue
        return path
    raise RuntimeError(
        "No parquet with cadence_num found. Candidates checked:\n"
        + "\n".join(report)
        + "\nGenerate one with: python -m src.tglc.extract_raw_parquet_cadence"
        + "\n(or point DIAG_DATA at a parquet that has cadence_num)."
    )


def load_sector(path, sector):
    df = pd.read_parquet(path, filters=[("sector", "==", sector)])
    if len(df) == 0:
        raise RuntimeError(f"no rows for sector {sector} in {path}")
    bad = df["cadence_num"].isna()
    if bad.any():
        raise RuntimeError(f"{int(bad.sum())}/{len(df)} sector-{sector} rows have "
                           f"null cadence_num in {path} -- refusing to continue")
    n_dupes = int(df.duplicated("TIC").sum())
    if n_dupes:
        print(f"WARNING: dropping {n_dupes} duplicate-TIC rows (keeping first)")
        df = df.drop_duplicates("TIC")
    return df.reset_index(drop=True)


def chip_index(camera, ccd):
    """camera x ccd -> stable 0..15 label: (camera-1)*4 + (ccd-1)."""
    camera, ccd = int(camera), int(ccd)
    if not (1 <= camera <= 4 and 1 <= ccd <= 4):
        raise ValueError(f"camera/ccd out of range: {camera}/{ccd}")
    return (camera - 1) * 4 + (ccd - 1)


def chip_name(idx):
    return f"cam{idx // 4 + 1}-ccd{idx % 4 + 1}"


def make_split(tics, seed=SEED, test_frac=0.2):
    """Deterministic TIC-disjoint split: sort unique TICs, permute with `seed`."""
    unique = np.sort(np.unique(np.asarray(tics, dtype=str)))
    perm = np.random.default_rng(seed).permutation(len(unique))
    n_test = max(1, int(round(test_frac * len(unique))))
    test = set(unique[perm[:n_test]].tolist())
    train = set(unique[perm[n_test:]].tolist())
    assert not (train & test)
    return train, test


def normalize_median_mad(flux):
    """Robust z-score from OBSERVED flux only (curves store observed points)."""
    flux = np.asarray(flux, dtype=np.float64)
    med = float(np.median(flux))
    mad = float(np.median(np.abs(flux - med)))
    scale = 1.4826 * mad
    if scale <= 0:
        scale = 1.0
    return (flux - med) / scale, med, mad


# ---------------------------------------------------- representations
def legacy_local_grid(time, flux, grid_length=GRID_1024):
    """Mirror of src/data/data.py resample_to_grid on one NORMALIZED curve."""
    time = np.asarray(time, dtype=np.float64)
    time_grid = np.linspace(time[0], time[len(time) - 1], grid_length)
    grid_flux = interp1d(time, flux)(time_grid)

    cadence = np.median(np.diff(time))
    idx = np.searchsorted(time, time_grid)
    idx = np.clip(idx, 1, len(time) - 1)
    distance = np.minimum(time_grid - time[idx - 1], time[idx] - time_grid)
    observed = (distance <= 3 * cadence).astype(np.float32)
    return np.where(observed > 0, grid_flux, 0.0).astype(np.float32), observed


def build_legacy_local(times, fluxes, grid_length=GRID_1024):
    X = np.zeros((len(fluxes), grid_length), dtype=np.float32)
    M = np.zeros_like(X)
    for i, (t, f) in enumerate(zip(times, fluxes)):
        X[i], M[i] = legacy_local_grid(t, f, grid_length)
    return X, M


def build_shared_sector(times, fluxes, grid_length=GRID_1024):
    """One global sector time range -> grid_length bins, identical for all stars.
    Multiple observations in a bin are averaged; a bin is observed if any land."""
    t0 = min(float(np.min(t)) for t in times)
    t1 = max(float(np.max(t)) for t in times)
    span = max(t1 - t0, 1e-9)
    X = np.zeros((len(fluxes), grid_length), dtype=np.float32)
    M = np.zeros_like(X)
    for i, (t, f) in enumerate(zip(times, fluxes)):
        b = np.clip(((np.asarray(t) - t0) / span * grid_length).astype(np.int64),
                    0, grid_length - 1)
        total = np.zeros(grid_length)
        count = np.zeros(grid_length)
        np.add.at(total, b, np.asarray(f, dtype=np.float64))
        np.add.at(count, b, 1.0)
        hit = count > 0
        X[i, hit] = (total[hit] / count[hit]).astype(np.float32)
        M[i, hit] = 1.0
    return X, M


def exact_cadence_grid_params(cadences, pad_to_multiple=16):
    """Mirror of src/data/cadence_dataset.sector_cadence_grid (torch-free)."""
    first = min(int(np.min(c)) for c in cadences)
    last = max(int(np.max(c)) for c in cadences)
    length = last - first + 1
    if pad_to_multiple:
        remainder = length % pad_to_multiple
        if remainder:
            length += pad_to_multiple - remainder
    return first, length


def build_exact_cadence(cadences, fluxes, pad_to_multiple=16):
    """index = cadence_num - first_cadence: same cadence -> same index, always."""
    first, length = exact_cadence_grid_params(cadences, pad_to_multiple)
    X = np.zeros((len(fluxes), length), dtype=np.float32)
    M = np.zeros_like(X)
    for i, (c, f) in enumerate(zip(cadences, fluxes)):
        idx = np.asarray(c, dtype=np.int64) - first
        if idx.min() < 0 or idx.max() >= length:
            raise ValueError("cadence_num outside the sector grid")
        X[i, idx] = np.asarray(f, dtype=np.float32)
        M[i, idx] = 1.0
    return X, M, first


# ------------------------------------------------ common-mode PCA model
def fit_chip_bases(X_train, M_train, chips_train, k_max, min_stars=4):
    """Per-chip masked mean + SVD basis from TRAIN rows only.

    Returns ({chip: (mean, components, n_train)}, skipped_chips). Unobserved
    entries contribute neither to the mean nor to the SVD (zero after center).
    """
    bases, skipped = {}, []
    for chip in range(N_CHIPS):
        rows = np.flatnonzero(chips_train == chip)
        if len(rows) < min_stars:
            skipped.append((chip, len(rows)))
            continue
        Xc, Mc = X_train[rows], M_train[rows]
        count = Mc.sum(axis=0)
        mean = np.where(count > 0, (Xc * Mc).sum(axis=0) / np.maximum(count, 1), 0.0)
        centered = (Xc - mean) * Mc
        k = min(k_max, len(rows) - 1)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        bases[chip] = (mean.astype(np.float64), vt[:k].astype(np.float64), len(rows))
    return bases, skipped


def recon_error(x, m, mean, components, k):
    """Masked least-squares reconstruction MSE over OBSERVED cadences only.

    Unobserved entries of x never enter: the design matrix and target are both
    restricted to observed columns before solving.
    """
    obs = m > 0
    k = min(k, components.shape[0])
    if obs.sum() <= k + 1:
        return np.nan
    A = components[:k][:, obs].T                # (n_obs, k)
    b = (x - mean)[obs].astype(np.float64)
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = b - A @ coef
    return float(np.mean(resid ** 2))


def predict_chips(X_test, M_test, bases, k):
    """Errors (n_test, 16) under every chip basis; prediction = argmin."""
    n = len(X_test)
    errors = np.full((n, N_CHIPS), np.nan)
    for chip, (mean, comps, _) in bases.items():
        for i in range(n):
            errors[i, chip] = recon_error(X_test[i], M_test[i], mean, comps, k)
    valid = ~np.all(np.isnan(errors), axis=1)
    pred = np.full(n, -1, dtype=np.int64)
    pred[valid] = np.nanargmin(errors[valid], axis=1)
    return pred, errors, int((~valid).sum())


# ----------------------------------------------------------- metrics
def compute_metrics(y_true, y_pred, errors=None):
    ok = y_pred >= 0
    yt, yp = y_true[ok], y_pred[ok]
    out = {
        "n_evaluated": int(ok.sum()),
        "n_skipped": int((~ok).sum()),
        "bacc_16way": balanced_accuracy_score(yt, yp),
        "bacc_camera": balanced_accuracy_score(yt // 4, yp // 4),
        "bacc_ccd": balanced_accuracy_score(yt % 4, yp % 4),
        "macro_f1": f1_score(yt, yp, average="macro", zero_division=0),
    }
    if errors is not None:
        e = errors[ok]
        pair_ok = ~np.isnan(e)
        labels = (np.arange(N_CHIPS)[None, :] == yt[:, None])
        out["auc_star_chip"] = roc_auc_score(labels[pair_ok].ravel(),
                                             (-e[pair_ok]).ravel())
    return out


def bootstrap_ci(y_true, y_pred, n_boot=N_BOOTSTRAP, seed=SEED):
    """95% CI for the 16-way balanced accuracy over test-star resamples."""
    ok = y_pred >= 0
    yt, yp = y_true[ok], y_pred[ok]
    rng = np.random.default_rng(seed)
    idx_all = np.arange(len(yt))
    scores = []
    for _ in range(n_boot):
        idx = rng.choice(idx_all, len(idx_all), replace=True)
        scores.append(balanced_accuracy_score(yt[idx], yp[idx]))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)


def pair_cosine_auc(X_test, y_test, seed=SEED, max_stars=MAX_PAIR_STARS):
    """Same-chip vs different-chip AUC of star-pair cosine similarity
    (zero-filled vectors; alignment-sensitive by construction)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(X_test))
    if len(idx) > max_stars:
        idx = rng.choice(idx, max_stars, replace=False)
    V = X_test[idx].astype(np.float64)
    norms = np.linalg.norm(V, axis=1)
    norms[norms == 0] = 1.0
    V = V / norms[:, None]
    S = V @ V.T
    iu = np.triu_indices(len(idx), k=1)
    same = (y_test[idx][:, None] == y_test[idx][None, :])[iu]
    return roc_auc_score(same, S[iu])


def shuffle_within_curves(X, M, seed=SEED):
    """Control: permute each curve's OBSERVED values over its observed positions.
    Preserves mask and marginal flux distribution; destroys temporal alignment."""
    rng = np.random.default_rng(seed)
    Xs = X.copy()
    for i in range(len(X)):
        obs = np.flatnonzero(M[i] > 0)
        Xs[i, obs] = X[i, obs][rng.permutation(len(obs))]
    return Xs


def logreg_bacc(F_train, y_train, F_test, y_test):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, class_weight="balanced"))
    clf.fit(F_train, y_train)
    pred = clf.predict(F_test)
    return balanced_accuracy_score(y_test, pred), pred


# ------------------------------------------------------------- plots
def save_plots(results, confusions, art_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reps = sorted({r["representation"] for r in results if r["condition"] == "real"})
    fig, axes = plt.subplots(1, len(confusions), figsize=(6 * len(confusions), 5.5))
    axes = np.atleast_1d(axes)
    for ax, (title, cm) in zip(axes, confusions.items()):
        row_sums = cm.sum(axis=1, keepdims=True)
        ax.imshow(cm / np.maximum(row_sums, 1), cmap="viridis", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("predicted chip")
        ax.set_ylabel("true chip")
        ticks = range(N_CHIPS)
        ax.set_xticks(ticks, [chip_name(i)[3:] for i in ticks], rotation=90, fontsize=6)
        ax.set_yticks(ticks, [chip_name(i)[3:] for i in ticks], fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(art_dir, "confusion_matrices.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.8 / len(reps)
    for j, rep in enumerate(reps):
        rows = [r for r in results
                if r["representation"] == rep and r["condition"] == "real" and r["K"] > 0]
        rows.sort(key=lambda r: r["K"])
        xs = np.arange(len(rows)) + j * width
        ax.bar(xs, [r["bacc_16way"] for r in rows], width=width, label=rep)
        for x, r in zip(xs, rows):
            ax.plot([x, x], [r["bacc_16way_lo"], r["bacc_16way_hi"]], color="black", lw=1)
    ks = sorted({r["K"] for r in results if r["condition"] == "real" and r["K"] > 0})
    ax.set_xticks(np.arange(len(ks)) + 0.4 - width / 2, [f"K={k}" for k in ks])
    ax.axhline(1 / N_CHIPS, color="gray", ls=":", label="chance (1/16)")
    for r in results:
        if r["condition"] == "control" and r["representation"] == "summary_stats":
            ax.axhline(r["bacc_16way"], color="red", ls="--", lw=1,
                       label="summary-stats control")
    ax.set_ylabel("16-way chip balanced accuracy")
    ax.set_title(f"Chip common-mode classification, sector {SECTOR} "
                 "(bars: real data; error bars: bootstrap 95% CI)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(art_dir, "alignment_comparison.png"), dpi=150)
    plt.close(fig)


# --------------------------------------------------------------- main
def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    os.makedirs(ART_DIR, exist_ok=True)
    print(f"git commit: {git_commit()}")
    print(f"command: {' '.join(sys.argv) or 'python -m src.instrument_v2.diagnose_chip_common_signal'}")
    print(f"config: sector={SECTOR} seed={SEED} K={K_LIST} bootstrap={N_BOOTSTRAP} -> {ART_DIR}")

    path = find_dataset()
    print(f"data: {path}")
    df = load_sector(path, SECTOR)
    quality_present = [c for c in QUALITY_COLUMNS if c in df.columns and df[c].notna().all()]
    chips = np.array([chip_index(c, d) for c, d in zip(df["camera"], df["ccd"])])
    counts = np.bincount(chips, minlength=N_CHIPS)
    print(f"{len(df)} stars, quality columns present: {quality_present or 'none'}")
    print("class counts: " + " ".join(f"{chip_name(i)}={counts[i]}" for i in range(N_CHIPS)))

    train_tics, test_tics = make_split(df["TIC"])
    with open(os.path.join(ART_DIR, "split_train_tics.txt"), "w") as fh:
        fh.write("\n".join(sorted(train_tics)))
    with open(os.path.join(ART_DIR, "split_test_tics.txt"), "w") as fh:
        fh.write("\n".join(sorted(test_tics)))
    is_train = df["TIC"].astype(str).isin(train_tics).to_numpy()
    print(f"split: {int(is_train.sum())} train / {int((~is_train).sum())} test stars (TIC-disjoint, seed {SEED})")

    normed, meds, mads, npts = [], [], [], []
    for f in df["flux"]:
        nf, med, mad = normalize_median_mad(f)
        normed.append(nf)
        meds.append(med)
        mads.append(mad)
        npts.append(len(nf))
    times = [np.asarray(t, dtype=np.float64) for t in df["time"]]
    cads = [np.asarray(c, dtype=np.int64) for c in df["cadence_num"]]

    print("building representations...")
    reps = {}
    reps["legacy_local_1024"] = build_legacy_local(times, normed)
    reps["shared_sector_1024"] = build_shared_sector(times, normed)
    Xe, Me, first_cadence = build_exact_cadence(cads, normed)
    reps["exact_cadence"] = (Xe, Me)
    print(f"exact_cadence grid: first_cadence={first_cadence}, length={Xe.shape[1]}")

    results, confusions = [], {}
    y = chips
    for rep_name, (X, M) in reps.items():
        Xtr, Mtr, ytr = X[is_train], M[is_train], y[is_train]
        Xte, Mte, yte = X[~is_train], M[~is_train], y[~is_train]
        for condition in ("real", "shuffled"):
            if condition == "shuffled":
                Xtr_c = shuffle_within_curves(Xtr, Mtr, seed=SEED + 1)
                Xte_c = shuffle_within_curves(Xte, Mte, seed=SEED + 2)
            else:
                Xtr_c, Xte_c = Xtr, Xte
            bases, skipped_chips = fit_chip_bases(Xtr_c, Mtr, ytr, k_max=max(K_LIST))
            best = None
            for k in K_LIST:
                pred, errors, n_unpredicted = predict_chips(Xte_c, Mte, bases, k)
                met = compute_metrics(yte, pred, errors)
                lo, hi = bootstrap_ci(yte, pred)
                row = {"representation": rep_name, "condition": condition, "K": k,
                       "bacc_16way_lo": lo, "bacc_16way_hi": hi,
                       "skipped_chips": len(skipped_chips), **met}
                if condition == "real":
                    row["pair_cosine_auc"] = pair_cosine_auc(Xte_c, yte)
                results.append(row)
                print(f"{rep_name:20s} {condition:8s} K={k:2d}  "
                      f"bacc16 {met['bacc_16way']:.4f} [{lo:.4f},{hi:.4f}]  "
                      f"cam {met['bacc_camera']:.4f}  ccd {met['bacc_ccd']:.4f}  "
                      f"f1 {met['macro_f1']:.4f}")
                if condition == "real" and (best is None or met["bacc_16way"] > best[0]):
                    ok = pred >= 0
                    best = (met["bacc_16way"], k,
                            confusion_matrix(yte[ok], pred[ok], labels=range(N_CHIPS)))
            if condition == "real" and best is not None:
                confusions[f"{rep_name} (K={best[1]}, bacc {best[0]:.3f})"] = best[2]

        mask_bacc, _ = logreg_bacc(Mtr, ytr, Mte, yte)
        results.append({"representation": rep_name, "condition": "control", "K": 0,
                        "control": "mask_only", "bacc_16way": mask_bacc,
                        "n_evaluated": len(yte)})
        print(f"{rep_name:20s} control  mask-only LR bacc16 {mask_bacc:.4f}")

    stats = np.column_stack([meds, mads,
                             1.0 - reps["exact_cadence"][1].mean(axis=1), npts])
    stats_bacc, _ = logreg_bacc(stats[is_train], y[is_train], stats[~is_train], y[~is_train])
    results.append({"representation": "summary_stats", "condition": "control", "K": 0,
                    "control": "summary_stats", "bacc_16way": stats_bacc,
                    "n_evaluated": int((~is_train).sum())})
    print(f"{'summary_stats':20s} control  median/MAD/missing/length LR bacc16 {stats_bacc:.4f}")

    pd.DataFrame(results).to_csv(os.path.join(ART_DIR, "results.csv"), index=False)
    summary = {
        "git_commit": git_commit(), "data": path, "sector": SECTOR, "seed": SEED,
        "n_stars": len(df), "n_train": int(is_train.sum()), "n_test": int((~is_train).sum()),
        "class_counts": {chip_name(i): int(counts[i]) for i in range(N_CHIPS)},
        "quality_columns_present": quality_present,
        "exact_cadence_grid": {"first_cadence": int(first_cadence), "length": int(Xe.shape[1])},
        "K_list": list(K_LIST), "n_bootstrap": N_BOOTSTRAP,
        "results": results,
    }
    with open(os.path.join(ART_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    save_plots(results, confusions, ART_DIR)
    print(f"\nwrote results.csv, summary.json, split TIC lists, and plots to {ART_DIR}")


if __name__ == "__main__":
    main()
