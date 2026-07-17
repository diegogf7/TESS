# All this code is from Claude
"""Independent validation-only checkpoint selection for the ONLINE encoder.

Same protocol as select_pretrain_checkpoints.py (fast-probe epoch ranking,
full C/PCA grid on the winner, hybrid weight chosen by mean validation bacc16
across seeds) but features come from `context_encoder` via encode_features()
-- never model.encode(), which would silently return the EMA copy.

Test TICs are never loaded, encoded, or scored here.

Run:  python -m src.instrument_v2.select_online_checkpoints
Env:  NEW_RUN, plus select_pretrain_checkpoints' env (ABL_CKPT_DIR must point
      at abl1's checkpoint dir; CKPT_DIR at the main dir with the jepa epochs)
Out:  $NEW_RUN/online_pretrain_selection.json
"""

import json
import os

import numpy as np
import pandas as pd

from sklearn.metrics import balanced_accuracy_score

from src.instrument_v2.ablation_config import HYBRID_WEIGHTS, SEEDS
from src.instrument_v2.encoder_source import encode_features
from src.instrument_v2.sector14_dataset import ensure_splits, ensure_time_range
from src.instrument_v2.select_pretrain_checkpoints import (
    ART_DIR, DEVICE, S14_DATA, SECTOR, SPLIT_DIR,
    arm_checkpoints, best_probe, build_model, fast_score, fit_probe, git_commit,
    load_trainval,
)

NEW_RUN = os.environ.get("NEW_RUN", os.path.join("artifacts", "instrument_v2", "ablation", "abl1_encoder_audit"))


def main():
    os.makedirs(NEW_RUN, exist_ok=True)
    print(f"git commit: {git_commit()}")
    df = pd.read_parquet(S14_DATA)
    df = df[df["sector"] == SECTOR].drop_duplicates("TIC").reset_index(drop=True)
    train_tics, val_tics, test_tics = ensure_splits(SPLIT_DIR, ART_DIR)
    t_range = ensure_time_range(ART_DIR, df, train_tics)
    X, M, chips, is_train = load_trainval(df, train_tics, val_tics, test_tics, t_range)
    ytr, yval = chips[is_train], chips[~is_train]
    print(f"online selection data: {int(is_train.sum())} train / {int((~is_train).sum())} val "
          "(test untouched)")

    arms = ["jepa", "supcon"] + [f"hybrid_w{w}" for w in HYBRID_WEIGHTS]
    selection = {}
    for arm in arms:
        selection[arm] = {}
        for seed in SEEDS:
            winner = None
            for epoch, path in sorted(arm_checkpoints(arm, seed).items()):
                model = build_model(arm, seed, path)
                Z = encode_features(model, "online", X, M, DEVICE)
                fb = fast_score(Z[is_train], ytr, Z[~is_train], yval)
                print(f"{arm:12s} s{seed} ep{epoch:3d}  ONLINE fast val bacc16 {fb:.4f}",
                      flush=True)
                if winner is None or fb > winner[0]:
                    winner = (fb, epoch, path, Z)
            _, epoch, path, Z = winner
            bacc, C, pca_dim = best_probe(Z[is_train], ytr, Z[~is_train], yval)
            selection[arm][str(seed)] = {
                "checkpoint": path, "epoch": epoch, "probe_C": C,
                "probe_pca": pca_dim or 0, "val_bacc16": bacc, "source": "online"}
            print(f"SELECTED online {arm} s{seed}: ep{epoch} val {bacc:.4f} "
                  f"(C={C}, pca={pca_dim or 0})")

    weight_means = {w: float(np.mean([selection[f"hybrid_w{w}"][str(s)]["val_bacc16"]
                                      for s in SEEDS])) for w in HYBRID_WEIGHTS}
    hybrid_weight = max(weight_means, key=weight_means.get)
    selection["hybrid"] = selection[f"hybrid_w{hybrid_weight}"]
    print(f"online hybrid weight means {weight_means} -> chosen w={hybrid_weight}")

    manifest = {"git_commit": git_commit(), "hybrid_weight": hybrid_weight,
                "hybrid_weight_val_means": weight_means, "arms": selection}
    out = os.path.join(NEW_RUN, "online_pretrain_selection.json")
    with open(out, "w") as fh:
        json.dump(manifest, fh, indent=2, default=float)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
