# All this code is from Claude
"""Fixed-checkpoint online-vs-EMA audit.

For every arm/seed, take the EXACT checkpoint abl1 selected (validation-only,
EMA-scored) and compare the two encoders INSIDE that checkpoint: online
(context) features vs EMA (target) features. Probes are fit on train with
C/PCA selected separately per source on validation, then evaluated on the
existing test split. This isolates the encoder-source variable completely.

Random arms are evaluated once per seed: online and EMA are verified
identical at init, so one row (source="identical") covers both.

Nothing is retrained; no checkpoint is written or modified.

Run:  python -m src.instrument_v2.eval_encoder_source_fixed
Env:  OLD_RUN (abl1 dir), NEW_RUN, S14_DATA, SPLIT_DIR, ART_DIR
Out:  $NEW_RUN/fixed_results.{csv,json}, $NEW_RUN/fixed_preds.npz
"""

import json
import os
import subprocess

import numpy as np
import pandas as pd
import torch

from sklearn.metrics import balanced_accuracy_score

from src.instrument_v2.ablation_config import ARMS, SEEDS
from src.instrument_v2.diagnose_chip_common_signal import chip_index
from src.instrument_v2.encoder_source import (
    assert_same_architecture, encode_features, encoders_identical, param_distance,
)
from src.instrument_v2.eval_final_ablation import metrics_16way
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range, grid_frame
from src.instrument_v2.select_pretrain_checkpoints import best_probe, build_model, fit_probe
from src.instrument_v2.train_sector14_jepa import effective_rank

SECTOR = 14
S14_DATA = os.environ.get("S14_DATA", "/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14.parquet")
SPLIT_DIR = os.environ.get("SPLIT_DIR", os.path.join("artifacts", "instrument_v2", "chip_signal_diagnostic"))
ART_DIR = os.environ.get("ART_DIR", os.path.join("artifacts", "instrument_v2", "sector14_jepa"))
OLD_RUN = os.environ.get("OLD_RUN", os.path.join("artifacts", "instrument_v2", "ablation", "abl1"))
NEW_RUN = os.environ.get("NEW_RUN", os.path.join("artifacts", "instrument_v2", "ablation", "abl1_encoder_audit"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main():
    os.makedirs(NEW_RUN, exist_ok=True)
    print(f"git commit: {git_commit()}")
    with open(os.path.join(OLD_RUN, "pretrain_selection.json")) as fh:
        old_sel = json.load(fh)

    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == SECTOR].drop_duplicates("TIC").reset_index(drop=True)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, ART_DIR)
    t_range = ensure_time_range(ART_DIR, df, train_tics)
    tic = df["TIC"].astype(str)
    part = np.where(tic.isin(train_tics), "train",
                    np.where(tic.isin(val_tics), "val", "test"))
    chips = np.array([chip_index(c, d) for c, d in zip(df["camera"], df["ccd"])])
    X, M = grid_frame(df, "shared", t_range)
    tr, va, te = part == "train", part == "val", part == "test"
    print(f"{int(tr.sum())} train / {int(va.sum())} val / {int(te.sum())} test")

    rows, preds = [], {}
    for arm in ARMS:
        for seed in SEEDS:
            entry = old_sel["arms"][arm][str(seed)]
            model = build_model(arm, seed, entry["checkpoint"])
            assert_same_architecture(model)
            identical = encoders_identical(model)
            dist = param_distance(model)
            if arm == "random":
                assert identical, "random init online/EMA differ -- should be impossible"
                sources = ("identical",)
            else:
                assert not identical, f"{arm} s{seed}: online == EMA after training?!"
                sources = ("online", "ema")
            for source in sources:
                actual = "online" if source == "identical" else source
                Z = encode_features(model, actual, X, M, DEVICE)
                bacc_val, C, pca_dim = best_probe(Z[tr], chips[tr], Z[va], chips[va])
                clf = fit_probe(Z[tr], chips[tr], C, pca_dim)
                pred = clf.predict(Z[te])
                preds[f"{arm}_s{seed}_{source}"] = pred
                met = metrics_16way(chips[te], pred)
                rows.append({"arm": arm, "seed": seed, "source": source,
                             "epoch": entry["epoch"], "checkpoint": entry["checkpoint"],
                             "probe_C": C, "probe_pca": pca_dim or 0,
                             "val_bacc16": bacc_val, "param_distance": dist,
                             "latent_std": float(Z[te].std(axis=0).mean()),
                             "effective_rank": effective_rank(Z[te]), **met})
                print(f"{arm:8s} s{seed} {source:9s} ep{entry['epoch']:3d}  "
                      f"test bacc16 {met['bacc_16way']:.4f}  val {bacc_val:.4f}  "
                      f"dist {dist:.4f}  erank {rows[-1]['effective_rank']:.1f}", flush=True)

    # per-arm paired summary at the fixed checkpoint
    for arm in ("jepa", "supcon", "hybrid"):
        d = [next(r["bacc_16way"] for r in rows if r["arm"] == arm and r["seed"] == s
                  and r["source"] == "online")
             - next(r["bacc_16way"] for r in rows if r["arm"] == arm and r["seed"] == s
                    and r["source"] == "ema") for s in SEEDS]
        print(f"FIXED-CKPT online-minus-ema {arm:8s}: per-seed {['%+.4f' % x for x in d]} "
              f"mean {np.mean(d):+.4f}")

    pd.DataFrame(rows).to_csv(os.path.join(NEW_RUN, "fixed_results.csv"), index=False)
    with open(os.path.join(NEW_RUN, "fixed_results.json"), "w") as fh:
        json.dump({"git_commit": git_commit(), "rows": rows}, fh, indent=2, default=float)
    np.savez(os.path.join(NEW_RUN, "fixed_preds.npz"),
             y_test=chips[te], **{k: v for k, v in preds.items()})
    print(f"wrote fixed_results.csv/json + fixed_preds.npz to {NEW_RUN}")


if __name__ == "__main__":
    main()
