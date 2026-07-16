# All this code is from Claude
"""Backbone-LR selection for the matched fine-tuning arms (validation only).

Reads the 72 per-run result jsons written by train_sector14_matched_finetune,
FAILS if any expected (arm, target, seed, lr) run is missing, and for every
(arm, target) picks ONE backbone learning rate by mean validation balanced
accuracy across seeds 0/1/2. Test results are never read (they don't exist
yet by construction).

Run:  python -m src.instrument_v2.select_finetune_models
Out:  $ABL_DIR/finetune_selection.json
"""

import json
import os

import numpy as np

from src.instrument_v2.ablation_config import ARMS, BACKBONE_LRS, SEEDS, TARGETS

ABL_DIR = os.environ.get("ABL_DIR", os.path.join("artifacts", "instrument_v2", "ablation", os.environ.get("RUN_ID", "dev")))
RUNS_DIR = os.path.join(ABL_DIR, "finetune_runs")


def load_run(arm, target, seed, lr):
    tag = f"ft_{arm}_{target}_s{seed}_lr{float(lr):g}"
    path = os.path.join(RUNS_DIR, f"{tag}.json")
    if not os.path.exists(path):
        raise RuntimeError(f"missing fine-tune run result: {path}")
    with open(path) as fh:
        return json.load(fh)


def main():
    missing = []
    for arm in ARMS:
        for target in TARGETS:
            for seed in SEEDS:
                for lr in BACKBONE_LRS:
                    tag = f"ft_{arm}_{target}_s{seed}_lr{float(lr):g}"
                    if not os.path.exists(os.path.join(RUNS_DIR, f"{tag}.json")):
                        missing.append(tag)
    if missing:
        raise RuntimeError(f"{len(missing)} fine-tune runs missing, e.g. {missing[:5]}")

    selection = {}
    for arm in ARMS:
        selection[arm] = {}
        for target in TARGETS:
            lr_means = {}
            for lr in BACKBONE_LRS:
                vals = [load_run(arm, target, s, lr)["best_val_bacc"] for s in SEEDS]
                lr_means[lr] = float(np.mean(vals))
            best_lr = max(lr_means, key=lr_means.get)
            runs = {str(s): load_run(arm, target, s, best_lr) for s in SEEDS}
            selection[arm][target] = {
                "backbone_lr": best_lr, "lr_val_means": lr_means,
                "checkpoints": {s: r["checkpoint"] for s, r in runs.items()},
                "val_bacc_per_seed": {s: r["best_val_bacc"] for s, r in runs.items()},
            }
            print(f"{arm:8s} {target:7s} lr means {lr_means} -> {best_lr}")

    out = os.path.join(ABL_DIR, "finetune_selection.json")
    with open(out, "w") as fh:
        json.dump(selection, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
