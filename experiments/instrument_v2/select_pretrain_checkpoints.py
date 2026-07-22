# All this code is from Claude
"""Validation-only pretraining checkpoint selection for the ablation.

For every arm (random / jepa / supcon / hybrid-per-weight) and seed, scores
epochs 10..100 with frozen-probe validation balanced accuracy (16-way) and
selects checkpoint epoch + probe (C, PCA). For hybrid, one GLOBAL contrastive
weight is chosen by mean validation bacc16 across the three seeds.

The test split is NEVER loaded, gridded, encoded, or scored here: latents are
built exclusively from train+val rows (asserted).

Run:  python -m src.instrument_v2.select_pretrain_checkpoints
Env:  RUN_ID / ABL_DIR / ABL_CKPT_DIR, CKPT_DIR (jepa epochs), S14_DATA,
      SPLIT_DIR, ART_DIR
Out:  $ABL_DIR/pretrain_selection.json
"""

import glob
import json
import os
import subprocess

import numpy as np
import pandas as pd
import torch

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.instrument_v2.ablation_config import HYBRID_WEIGHTS, SEEDS
from src.instrument_v2.diagnose_chip_common_signal import chip_index
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range, grid_frame
from src.loss_function.gapblind_fix import build_gapblind_jepa

SECTOR = 14
S14_DATA = os.environ.get("S14_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get("SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
ART_DIR = os.environ.get("ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))
ABL_DIR = os.environ.get("ABL_DIR", os.path.join("artifacts", "instrument_v2", "ablation", os.environ.get("RUN_ID", "dev")))
ABL_CKPT_DIR = os.environ.get("ABL_CKPT_DIR",
                              os.path.join("/orcd/scratch/orcd/006/diegogon/checkpoints",
                                           "ablation", os.environ.get("RUN_ID", "dev")))
CKPT_DIR = os.environ.get("CKPT_DIR", "/orcd/scratch/orcd/006/diegogon/checkpoints")
BATCH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
C_GRID = (0.1, 1.0, 10.0)
PCA_GRID = (None, 64)


def load_trainval(df, train_tics, val_tics, test_tics, t_range):
    """Grid ONLY train+val rows. Raises if a test TIC slips in."""
    tic = df["TIC"].astype(str)
    keep = tic.isin(train_tics) | tic.isin(val_tics)
    sub = df[keep].reset_index(drop=True)
    if set(sub["TIC"].astype(str)) & set(test_tics):
        raise RuntimeError("test TIC present in selection data -- refusing")
    X, M = grid_frame(sub, "shared", t_range)
    chips = np.array([chip_index(c, d) for c, d in zip(sub["camera"], sub["ccd"])])
    is_train = sub["TIC"].astype(str).isin(train_tics).to_numpy()
    return X, M, chips, is_train


def encode(model, X, M):
    model.eval()
    pieces = []
    with torch.no_grad():
        for start in range(0, len(X), BATCH):
            f = torch.tensor(X[start:start + BATCH]).to(DEVICE)
            m = torch.tensor(M[start:start + BATCH]).to(DEVICE)
            z = model.encode(f, m)
            pieces.append(z.reshape(z.shape[0], -1).cpu().numpy())
    return np.concatenate(pieces)


def fit_probe(Ztr, ytr, C, pca_dim):
    steps = [StandardScaler()]
    if pca_dim is not None and pca_dim < Ztr.shape[1]:
        steps.append(PCA(n_components=pca_dim, random_state=0))
    steps.append(LogisticRegression(max_iter=3000, C=C, class_weight="balanced"))
    clf = make_pipeline(*steps)
    clf.fit(Ztr, ytr)
    return clf


def best_probe(Ztr, ytr, Zval, yval):
    """(val_bacc, C, pca_dim) maximizing validation 16-way bacc."""
    best = None
    for pca_dim in PCA_GRID:
        for C in C_GRID:
            clf = fit_probe(Ztr, ytr, C, pca_dim)
            bacc = balanced_accuracy_score(yval, clf.predict(Zval))
            if best is None or bacc > best[0]:
                best = (bacc, C, pca_dim)
    return best


def fast_score(Ztr, ytr, Zval, yval):
    """Single cheap probe (C=1, PCA-64) used only to RANK epochs; the full
    hyperparameter grid runs once on the winning epoch. Validation-only."""
    clf = fit_probe(Ztr, ytr, 1.0, 64)
    return balanced_accuracy_score(yval, clf.predict(Zval))


def arm_checkpoints(arm, seed):
    """{epoch: path} of every saved epoch for an arm/seed (None for random)."""
    if arm == "random":
        return {0: None}
    if arm == "jepa":
        pattern = os.path.join(CKPT_DIR, f"s14jepa_shared_s{seed}_ep*.pth")
    elif arm == "supcon":
        pattern = os.path.join(ABL_CKPT_DIR, f"s14supcon_s{seed}_ep*.pth")
    else:                                       # hybrid_w{w}
        weight = arm.split("_w")[1]
        pattern = os.path.join(ABL_CKPT_DIR, f"s14hybrid_w{float(weight):g}_s{seed}_ep*.pth")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise RuntimeError(f"no checkpoints found for arm={arm} seed={seed}: {pattern}")
    return {int(p.rsplit("_ep", 1)[1][:3]): p for p in paths}


def build_model(arm, seed, path):
    if arm == "random":
        torch.manual_seed(seed)                 # per-seed scratch init
        return build_gapblind_jepa().to(DEVICE)
    model = build_gapblind_jepa().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    os.makedirs(ABL_DIR, exist_ok=True)
    print(f"git commit: {git_commit()}")
    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == SECTOR].drop_duplicates("TIC").reset_index(drop=True)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, ART_DIR)
    t_range = ensure_time_range(ART_DIR, df, train_tics)
    X, M, chips, is_train = load_trainval(df, train_tics, val_tics, test_tics, t_range)
    ytr, yval = chips[is_train], chips[~is_train]
    print(f"selection data: {int(is_train.sum())} train / {int((~is_train).sum())} val "
          "(test split untouched)")

    arms = ["random", "jepa", "supcon"] + [f"hybrid_w{w}" for w in HYBRID_WEIGHTS]
    selection = {}
    for arm in arms:
        selection[arm] = {}
        for seed in SEEDS:
            # pass 1: rank epochs with the fast probe, keep the winner's latents
            winner = None                        # (fast_bacc, epoch, path, Z)
            for epoch, path in sorted(arm_checkpoints(arm, seed).items()):
                model = build_model(arm, seed, path)
                Z = encode(model, X, M)
                fb = fast_score(Z[is_train], ytr, Z[~is_train], yval)
                print(f"{arm:12s} s{seed} ep{epoch:3d}  fast val bacc16 {fb:.4f}", flush=True)
                if winner is None or fb > winner[0]:
                    winner = (fb, epoch, path, Z)
            # pass 2: full probe grid on the winning epoch only
            _, epoch, path, Z = winner
            bacc, C, pca_dim = best_probe(Z[is_train], ytr, Z[~is_train], yval)
            cam = balanced_accuracy_score(
                yval // 4, fit_probe(Z[is_train], ytr // 4, C, pca_dim).predict(Z[~is_train]))
            ccd = balanced_accuracy_score(
                yval % 4, fit_probe(Z[is_train], ytr % 4, C, pca_dim).predict(Z[~is_train]))
            selection[arm][str(seed)] = {
                "checkpoint": path, "epoch": epoch, "probe_C": C,
                "probe_pca": pca_dim or 0, "val_bacc16": bacc,
                "val_bacc_camera": cam, "val_bacc_ccd": ccd}
            print(f"SELECTED {arm} s{seed}: ep{epoch} val {bacc:.4f} "
                  f"(C={C}, pca={pca_dim or 0})")

    # one global hybrid weight by mean val bacc16 across seeds
    weight_means = {w: float(np.mean([selection[f"hybrid_w{w}"][str(s)]["val_bacc16"]
                                      for s in SEEDS])) for w in HYBRID_WEIGHTS}
    hybrid_weight = max(weight_means, key=weight_means.get)
    selection["hybrid"] = selection[f"hybrid_w{hybrid_weight}"]
    print(f"hybrid weight means {weight_means} -> chosen w={hybrid_weight}")

    manifest = {"git_commit": git_commit(), "hybrid_weight": hybrid_weight,
                "hybrid_weight_val_means": weight_means, "arms": selection}
    out = os.path.join(ABL_DIR, "pretrain_selection.json")
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2, default=float)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
