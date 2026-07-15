#this is all claude code

import os
import subprocess
import warnings

import numpy as np
import torch

from torch.utils.data import DataLoader
from src.data.data import SectorDataset
from src.loss_function.gap_infill import infill_gaps
from src.worked_folder.instrument.instrument_jepa import build_instrument_jepa

from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.simplefilter("error", ConvergenceWarning)  # fail loudly, caught per cell

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_PATH = os.environ.get("JEPA_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_probe20k_area.parquet")
CKPT_DIR = os.environ.get("CKPT_DIR", "/orcd/scratch/orcd/006/diegogon/checkpoints")
ENCODERS = os.environ.get("ENCODERS", "sector_s0,sector_s1,sector_s2,chip_s0,chip_s1,chip_s2,random,rawflux").split(",")
PCA_DIMS = os.environ.get("PCA_DIMS", "none,16,32,64,128,256").split(",")
N_INFILLS = int(os.environ.get("N_INFILLS", "5"))
N_REPEATS = int(os.environ.get("N_REPEATS", "5"))
N_BOOTSTRAP = int(os.environ.get("N_BOOTSTRAP", "300"))
SEED = int(os.environ.get("SEED", "0"))
BATCH_SIZE = 256
MIN_PER_CLASS = int(os.environ.get("MIN_PER_CLASS", "10"))
MIN_PER_CLASS_AREA = int(os.environ.get("MIN_PER_CLASS_AREA", "5"))


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


print(f"git commit: {git_commit()}")
print(f"data: {DATA_PATH}")
print(f"checkpoints: {CKPT_DIR}")
print(f"config: encoders={ENCODERS} pca={PCA_DIMS} n_infills={N_INFILLS} "
      f"n_repeats={N_REPEATS} n_bootstrap={N_BOOTSTRAP} seed={SEED} "
      f"min_per_class={MIN_PER_CLASS} (area {MIN_PER_CLASS_AREA})")

# ---------------- data: grid once, reuse for every encoder ----------------
dataset = SectorDataset(DATA_PATH, grid_length=1024)
loader = DataLoader(dataset, BATCH_SIZE, shuffle=False, num_workers=2)

flux_chunks, mask_chunks = [], []
for flux, mask, _ in loader:
    flux_chunks.append(flux)
    mask_chunks.append(mask)
all_flux = torch.cat(flux_chunks)
all_mask = torch.cat(mask_chunks)

sectors = dataset.df["sector"].to_numpy()
camera = dataset.df["camera"].to_numpy()
ccd = dataset.df["ccd"].to_numpy()
cam_ccd = camera * 10 + ccd
if "area" in dataset.df.columns:
    area = dataset.df["area"].to_numpy()
else:
    area = None
    print("WARNING: no 'area' column in dataset -- skipping AREA label")

LABELS = [("CAMERA", camera, MIN_PER_CLASS),
          ("CAMCCD_16way", cam_ccd, MIN_PER_CLASS)]
if area is not None:
    LABELS.append(("AREA", area, MIN_PER_CLASS_AREA))
LABELS.append(("CCD_secondary", ccd, MIN_PER_CLASS))

print(f"{len(dataset)} curves, {len(np.unique(sectors))} sectors, "
      f"{dataset.df['GAIADR3'].nunique() if 'GAIADR3' in dataset.df.columns else 'n/a'} stars")
for name, y, _ in LABELS:
    print(f"  {name}: {len(np.unique(y))} classes, chance {1.0 / len(np.unique(y)):.5f}")


def encode_with_model(model):
    """Average N_INFILLS deterministic infills. Reseeding before each encoder
    guarantees every encoder sees the exact same fabricated values."""
    torch.manual_seed(SEED)
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(all_flux), BATCH_SIZE):
            flux = all_flux[start:start + BATCH_SIZE].to(DEVICE)
            mask = all_mask[start:start + BATCH_SIZE].to(DEVICE)
            z_sum = 0.0
            for _ in range(N_INFILLS):
                filled, _ = infill_gaps(flux, mask)
                z_sum = z_sum + model.encode(filled, None)
            z = z_sum / N_INFILLS
            pieces.append(z.reshape(z.shape[0], -1).cpu().numpy())
    return np.concatenate(pieces)


def encode_rawflux():
    torch.manual_seed(SEED)
    pieces = []
    with torch.no_grad():
        for start in range(0, len(all_flux), BATCH_SIZE):
            flux = all_flux[start:start + BATCH_SIZE].to(DEVICE)
            mask = all_mask[start:start + BATCH_SIZE].to(DEVICE)
            f_sum = 0.0
            for _ in range(N_INFILLS):
                filled, _ = infill_gaps(flux, mask)
                f_sum = f_sum + filled
            pieces.append((f_sum / N_INFILLS).cpu().numpy())
    return np.concatenate(pieces)


features = {}
for name in ENCODERS:
    if name == "rawflux":
        features[name] = encode_rawflux()
        print(f"encoded rawflux baseline: {features[name].shape}")
    elif name == "random":
        torch.manual_seed(12345)  # fixed random init, reproducible
        model = build_instrument_jepa().to(DEVICE)
        features[name] = encode_with_model(model)
        print(f"encoded random-init encoder: {features[name].shape}")
    else:
        path = os.path.join(CKPT_DIR, f"instrument_gapblind_{name}.pth")
        model = build_instrument_jepa().to(DEVICE)
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        features[name] = encode_with_model(model)
        print(f"encoded {name} ({path}): {features[name].shape}")


# ---------------- probes ----------------
def make_probe(pca_dim):
    steps = [StandardScaler()]
    if pca_dim is not None:
        steps.append(PCA(n_components=pca_dim, random_state=0))
    steps.append(LogisticRegression(max_iter=5000, class_weight="balanced"))
    return make_pipeline(*steps)


def fit_score(X_tr, y_tr, X_te, y_te, pca_dim):
    probe = make_probe(pca_dim)
    probe.fit(X_tr, y_tr)
    return balanced_accuracy_score(y_te, probe.predict(X_te)), probe.predict(X_te)


def within_sector(X, y, pca_dim, min_per_class):
    sector_means, skipped = [], 0
    for sec in np.unique(sectors):
        rows = np.flatnonzero(sectors == sec)
        y_sec, X_sec = y[rows], X[rows]
        classes, counts = np.unique(y_sec, return_counts=True)
        if len(classes) < 2 or counts.min() < min_per_class:
            skipped += 1
            continue
        scores = []
        for r in range(N_REPEATS):
            tr, te = train_test_split(np.arange(len(y_sec)), test_size=0.2,
                                      stratify=y_sec, random_state=r)
            s, _ = fit_score(X_sec[tr], y_sec[tr], X_sec[te], y_sec[te], pca_dim)
            scores.append(s)
        sector_means.append(np.mean(scores))
    if not sector_means:
        return None, (None, None), skipped
    sector_means = np.asarray(sector_means)
    boot_rng = np.random.default_rng(SEED)
    boots = [np.mean(boot_rng.choice(sector_means, len(sector_means), replace=True))
             for _ in range(1000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(sector_means.mean()), (lo, hi), skipped


def leave_sector_out(X, y, pca_dim):
    y_pred = np.empty(len(y), dtype=y.dtype)
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups=sectors):
        _, pred = fit_score(X[tr], y[tr], X[te], y[te], pca_dim)
        y_pred[te] = pred
    point = balanced_accuracy_score(y, y_pred)
    boot_rng = np.random.default_rng(SEED)
    idx_all = np.arange(len(y))  # probe set is one row per star -> row bootstrap = star bootstrap
    boots = []
    for _ in range(N_BOOTSTRAP):
        idx = boot_rng.choice(idx_all, len(idx_all), replace=True)
        boots.append(balanced_accuracy_score(y[idx], y_pred[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(point), (lo, hi)


results = {}
for enc in ENCODERS:
    X = features[enc]
    for pca_str in PCA_DIMS:
        pca_dim = None if pca_str == "none" else int(pca_str)
        if pca_dim is not None and pca_dim >= X.shape[1]:
            continue  # PCA dim must be < feature dim
        for label, y, min_pc in LABELS:
            try:
                ws_mean, (ws_lo, ws_hi), skipped = within_sector(X, y, pca_dim, min_pc)
                if len(np.unique(sectors)) >= 2:
                    lso_mean, (lso_lo, lso_hi) = leave_sector_out(X, y, pca_dim)
                else:  # single-sector data: LSO undefined
                    lso_mean, lso_lo, lso_hi = float("nan"), float("nan"), float("nan")
            except ConvergenceWarning as w:
                print(f"FAILED {enc} pca={pca_str} {label}: ConvergenceWarning ({w})")
                results[(enc, label, pca_str, "within")] = np.nan
                results[(enc, label, pca_str, "lso")] = np.nan
                continue
            results[(enc, label, pca_str, "within")] = ws_mean
            results[(enc, label, pca_str, "lso")] = lso_mean
            ws_txt = ("skipped-all" if ws_mean is None else
                      f"{ws_mean:.4f} [{ws_lo:.4f},{ws_hi:.4f}] ({skipped} sect skipped)")
            print(f"{enc:10s} pca={pca_str:5s} {label:14s} "
                  f"within {ws_txt}  |  LSO {lso_mean:.4f} [{lso_lo:.4f},{lso_hi:.4f}]")

# ---------------- paired chip - sector differences ----------------
seeds = [s for s in range(3)
         if f"sector_s{s}" in ENCODERS and f"chip_s{s}" in ENCODERS]
if seeds:
    print("\npaired chip-minus-sector differences (same seed, same infills, same splits):")
    for pca_str in PCA_DIMS:
        for label, _, _ in LABELS:
            for proto in ("within", "lso"):
                diffs = []
                for s in seeds:
                    a = results.get((f"chip_s{s}", label, pca_str, proto))
                    b = results.get((f"sector_s{s}", label, pca_str, proto))
                    if a is None or b is None or np.isnan(a) or np.isnan(b):
                        continue
                    diffs.append(a - b)
                if diffs:
                    per_seed = " ".join(f"{d:+.4f}" for d in diffs)
                    print(f"  pca={pca_str:5s} {label:14s} {proto:6s} "
                          f"mean {np.mean(diffs):+.4f}  per-seed [{per_seed}]")
print("\nDONE")
