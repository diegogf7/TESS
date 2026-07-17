# All this code is from Claude
"""Validation-only backbone-LR selection for the online fine-tunes (27 runs).

One LR per arm (jepa/supcon/hybrid, camccd) by mean validation balanced
accuracy over seeds 0/1/2. Fails loudly if any run is missing. No test access.

Run:  python -m src.instrument_v2.select_online_finetune
Out:  $NEW_RUN/online_finetune_selection.json
"""

import json
import os

import numpy as np

from src.instrument_v2.ablation_config import BACKBONE_LRS, ONLINE_FT_ARMS, SEEDS

NEW_RUN = os.environ.get("NEW_RUN", os.path.join("artifacts", "instrument_v2", "ablation", "abl1_encoder_audit"))
RUNS_DIR = os.path.join(NEW_RUN, "finetune_runs")


def load_run(arm, seed, lr):
    tag = f"ftonline_{arm}_camccd_s{seed}_lr{float(lr):g}"
    path = os.path.join(RUNS_DIR, f"{tag}.json")
    if not os.path.exists(path):
        raise RuntimeError(f"missing online fine-tune result: {path}")
    with open(path) as fh:
        return json.load(fh)


def main():
    missing = [f"ftonline_{a}_camccd_s{s}_lr{float(lr):g}"
               for a in ONLINE_FT_ARMS for s in SEEDS for lr in BACKBONE_LRS
               if not os.path.exists(os.path.join(
                   RUNS_DIR, f"ftonline_{a}_camccd_s{s}_lr{float(lr):g}.json"))]
    if missing:
        raise RuntimeError(f"{len(missing)} runs missing, e.g. {missing[:5]}")

    selection = {}
    for arm in ONLINE_FT_ARMS:
        lr_means = {lr: float(np.mean([load_run(arm, s, lr)["best_val_bacc"]
                                       for s in SEEDS])) for lr in BACKBONE_LRS}
        best_lr = max(lr_means, key=lr_means.get)
        runs = {str(s): load_run(arm, s, best_lr) for s in SEEDS}
        selection[arm] = {"backbone_lr": best_lr, "lr_val_means": lr_means,
                          "checkpoints": {s: r["checkpoint"] for s, r in runs.items()},
                          "val_bacc_per_seed": {s: r["best_val_bacc"] for s, r in runs.items()}}
        print(f"{arm:8s} lr means {lr_means} -> {best_lr}")

    out = os.path.join(NEW_RUN, "online_finetune_selection.json")
    with open(out, "w") as fh:
        json.dump(selection, fh, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
