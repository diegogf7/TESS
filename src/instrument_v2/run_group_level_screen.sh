#!/usr/bin/env bash
set -euo pipefail

# Run inside one allocated GPU job.  Stage 1 is cheap validation-only screening;
# only the winning K is promoted.  TEST is touched once, after all choices.
export GROUP_ART_DIR="${GROUP_ART_DIR:-artifacts/instrument_v2/group_level}"
export GROUP_CKPT_DIR="${GROUP_CKPT_DIR:-/orcd/scratch/orcd/006/diegogon/checkpoints/group_level}"
export PROBE_VIEW="${PROBE_VIEW:-online}"
mkdir -p "$GROUP_ART_DIR" "$GROUP_CKPT_DIR"

for group_size in 2 4 8 16; do
  GROUP_SIZE="$group_size" SEED=0 EPOCHS="${SCREEN_EPOCHS:-20}" PROBE_EVERY=2 \
    python -m src.instrument_v2.train_group_level_jepa
done

best_k="$({ python - <<'PY'
import glob, json
import os
rows = []
root = os.environ["GROUP_ART_DIR"]
for path in glob.glob(os.path.join(root, "selection_s14groupmean_k*_s0.json")):
    with open(path) as handle:
        row = json.load(handle)
    rows.append((row["best_val_probe_bacc16"], row["group_size"]))
if not rows:
    raise SystemExit("no screening selections found")
print(max(rows)[1])
PY
} )"
echo "Promoting validation winner GROUP_SIZE=$best_k"

for seed in 0 1 2; do
  GROUP_SIZE="$best_k" SEED="$seed" EPOCHS="${CONFIRM_EPOCHS:-50}" PROBE_EVERY=5 \
    python -m src.instrument_v2.train_group_level_jepa
done

for seed in 0 1 2; do
  for arm in random group; do
    for backbone_lr in 1e-4 3e-4 1e-3; do
      INIT_ARM="$arm" GROUP_SIZE="$best_k" ENCODER_VIEW="$PROBE_VIEW" SEED="$seed" \
        BACKBONE_LR="$backbone_lr" EPOCHS="${FINETUNE_EPOCHS:-100}" \
        python -m src.instrument_v2.train_group_level_matched_finetune
    done
  done
done

# Do not spend the reused test split unless group initialization first beats
# the matched scratch rerun on validation, averaged over the three seeds.
GROUP_SIZE="$best_k" ENCODER_VIEW="$PROBE_VIEW" python - <<'PY'
import glob
import json
import os

root = os.environ["GROUP_ART_DIR"]
k = int(os.environ["GROUP_SIZE"])
view = os.environ["ENCODER_VIEW"]
means = {}
for arm in ("random", "group"):
    seed_best = []
    for seed in (0, 1, 2):
        pattern = os.path.join(
            root, f"result_ft_{arm}_k{k}_{view}_camccd_s{seed}_lr*.json"
        )
        rows = []
        for path in glob.glob(pattern):
            with open(path) as handle:
                rows.append(json.load(handle))
        if not rows:
            raise SystemExit(f"missing validation results: {pattern}")
        seed_best.append(max(row["best_val_bacc16"] for row in rows))
    means[arm] = sum(seed_best) / len(seed_best)
print(f"validation gate: group={means['group']:.4f}, random={means['random']:.4f}")
if means["group"] <= means["random"]:
    raise SystemExit(
        "GROUP-LEVEL LATENT DID NOT BEAT SCRATCH ON VALIDATION; test remains untouched"
    )
PY

# Single gated test access after K, epochs, view, LR, and checkpoints are fixed.
GROUP_SIZE="$best_k" ENCODER_VIEW="$PROBE_VIEW" \
  python -m src.instrument_v2.eval_group_level_test
