#!/usr/bin/env bash
# All this code is from Claude
# Validation-gated Instance-to-Subspace JEPA screen. Run inside ONE GPU job.
# Stage 1: 1 seed, K in {8,16}, 20 epochs, all arms -> promotion rule.
# Stage 2: 3 seeds, 50 epochs, frozen validation first, then LP-FT sweep.
# The reused test split is NEVER touched; the final gate only reports whether
# a test evaluation would be justified.
set -euo pipefail

export ISJ_ART_DIR="${ISJ_ART_DIR:-artifacts/instrument_v2/instance_subspace}"
export ISJ_CKPT_DIR="${ISJ_CKPT_DIR:-/orcd/scratch/orcd/006/diegogon/checkpoints/instance_subspace}"
mkdir -p "$ISJ_ART_DIR" "$ISJ_CKPT_DIR"

ARMS_LIST="mean_to_mean instance_mean instance_mean_var instance_cov"

# ---------------- stage 1: cheap screening ----------------
for K in 8 16; do
  for ARM in $ARMS_LIST; do
    ISJ_ARM="$ARM" GROUP_SIZE="$K" SEED=0 EPOCHS="${SCREEN_EPOCHS:-20}" PROBE_EVERY=2 \
      python -m src.instrument_v2.train_instance_subspace_jepa
  done
done

# ---------------- promotion rule (validation only) ----------------
PROMOTED="$(python - <<'PY'
import glob, json, os
root = os.environ["ISJ_ART_DIR"]
rows = {}
for path in glob.glob(os.path.join(root, "selection_isj_*_s0.json")):
    with open(path) as fh:
        r = json.load(fh)
    rows[(r["arm"], r["group_size"])] = r
best = None
for (arm, k), r in rows.items():
    if arm == "mean_to_mean":
        continue
    margin = r["best_val_probe_bacc16"] - r["random_probe_bacc16"]
    m2m = rows.get(("mean_to_mean", k))
    beats_m2m = m2m is None or r["best_val_probe_bacc16"] > m2m["best_val_probe_bacc16"]
    ok = margin >= 0.02 and (r["best_effective_rank"] or 0) > 32 and beats_m2m
    print(f"# {arm} K={k}: probe={r['best_val_probe_bacc16']:.4f} "
          f"random={r['random_probe_bacc16']:.4f} margin={margin:+.4f} "
          f"rank={r['best_effective_rank']:.1f} beats_mean_to_mean={beats_m2m} "
          f"promotable={ok}")
    if ok and (best is None or r["best_val_probe_bacc16"] > best[0]):
        best = (r["best_val_probe_bacc16"], arm, k)
print(f"{best[1]} {best[2]}" if best else "NONE")
PY
)"
echo "$PROMOTED"
LAST_LINE="$(echo "$PROMOTED" | tail -1)"
if [ "$LAST_LINE" = "NONE" ]; then
  echo "NO ARM PROMOTED: no instance arm beat matched random by >=0.02 with rank>32"
  echo "and beat mean_to_mean. Stopping before stage 2; test remains untouched."
  exit 0
fi
BEST_ARM="$(echo "$LAST_LINE" | cut -d' ' -f1)"
BEST_K="$(echo "$LAST_LINE" | cut -d' ' -f2)"
echo "PROMOTED: ARM=$BEST_ARM K=$BEST_K"

# ---------------- stage 2: three-seed confirmation ----------------
for SEED in 0 1 2; do
  ISJ_ARM="$BEST_ARM" GROUP_SIZE="$BEST_K" SEED="$SEED" EPOCHS="${CONFIRM_EPOCHS:-50}" \
    PROBE_EVERY=5 python -m src.instrument_v2.train_instance_subspace_jepa
done

# frozen validation comparison across seeds before any fine-tuning
python - <<PY
import json, os
root = os.environ["ISJ_ART_DIR"]
probes, randoms = [], []
for seed in (0, 1, 2):
    with open(os.path.join(root, f"selection_isj_${BEST_ARM}_k${BEST_K}_s{seed}.json")) as fh:
        r = json.load(fh)
    probes.append(r["best_val_probe_bacc16"]); randoms.append(r["random_probe_bacc16"])
mean_p = sum(probes) / 3; mean_r = sum(randoms) / 3
print(f"stage-2 frozen val: pretrained={mean_p:.4f} random={mean_r:.4f} per-seed={probes}")
if mean_p <= mean_r:
    raise SystemExit("FROZEN CONFIRMATION FAILED: pretrained <= random on validation")
PY

# ---------------- LP-FT sweep: 3 arms x 3 seeds x 3 lrs ----------------
for SEED in 0 1 2; do
  for INIT in pretrained scratch_proj scratch_direct; do
    for LR in 1e-5 3e-5 1e-4; do
      ISJ_INIT="$INIT" ISJ_ARM="$BEST_ARM" GROUP_SIZE="$BEST_K" SEED="$SEED" \
        BACKBONE_LR="$LR" python -m src.instrument_v2.train_instance_subspace_finetune
    done
  done
done

# ---------------- test gate (report only; test is NOT evaluated here) ----------------
ISJ_BEST_ARM="$BEST_ARM" ISJ_BEST_K="$BEST_K" python - <<'PY'
import glob, json, os
root = os.environ["ISJ_ART_DIR"]
arm, k = os.environ["ISJ_BEST_ARM"], int(os.environ["ISJ_BEST_K"])
best_lr_val = {}
for init in ("pretrained", "scratch_proj", "scratch_direct"):
    per_lr = {}
    for lr in ("1e-05", "3e-05", "0.0001"):
        vals = []
        for seed in (0, 1, 2):
            paths = glob.glob(os.path.join(
                root, f"result_ftisj_{init}_{arm}_k{k}_s{seed}_lr{float(lr):g}.json"))
            if not paths:
                raise SystemExit(f"missing LP-FT result for {init} s{seed} lr{lr}")
            with open(paths[0]) as fh:
                vals.append(json.load(fh)["best_val_bacc16"])
        per_lr[lr] = vals
    lr = max(per_lr, key=lambda key: sum(per_lr[key]))
    best_lr_val[init] = per_lr[lr]
    print(f"{init}: lr={lr} per-seed={per_lr[lr]} mean={sum(per_lr[lr]) / 3:.4f}")
p, sp, sd = (best_lr_val[i] for i in ("pretrained", "scratch_proj", "scratch_direct"))
mean = lambda v: sum(v) / len(v)
all_seeds_positive = all(pi > max(si, di) for pi, si, di in zip(p, sp, sd))
if mean(p) > mean(sp) and mean(p) > mean(sd) and all_seeds_positive:
    print("TEST GATE PASSED: instance-subspace beats both scratch baselines on mean "
          "validation AND in every seed. A single gated test evaluation is justified "
          "(run separately; this script never touches test).")
else:
    print("TEST GATE REFUSED: pretrained does not beat both scratch baselines "
          "consistently on validation; test remains untouched.")
PY
