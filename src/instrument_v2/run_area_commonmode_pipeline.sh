#!/bin/bash -l
# All this code is from Claude
# Stage runner for the area common-mode pipeline. Invoked by sbatch with
# STAGE (and SLURM_ARRAY_TASK_ID for array stages) in the environment.
# Stages: smoke | screen | screen_select | confirm | finetune | final
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"
export PYTHONUNBUFFERED=1
export ACM_ART_DIR=${ACM_ART_DIR:-artifacts/instrument_v2/area_commonmode_v1}
SCREEN_SELECTION=$ACM_ART_DIR/screen_selection.json

chosen() {  # chosen <area|chip> <grouping|target|k>
  $PY - "$SCREEN_SELECTION" "$1" "$2" <<'EOF'
import json, sys
with open(sys.argv[1]) as fh:
    print(json.load(fh)["chosen"][sys.argv[2]][sys.argv[3]])
EOF
}

echo "=== node $(hostname) | stage ${STAGE:?set STAGE} | task ${SLURM_ARRAY_TASK_ID:-n/a} | commit $(git rev-parse --short HEAD) ==="

case "$STAGE" in
  smoke)
    # 2-batch end-to-end pass of both arms and the LP-FT harness.
    MAX_BATCHES=2 EPOCHS=1 GROUPING=chip TARGET=median K=8 SEED=0 WARMSTART=${SMOKE_WARMSTART:-1} \
      $PY -m src.instrument_v2.train_area_commonmode_jepa
    MAX_BATCHES=2 EPOCHS=1 GROUPING=area TARGET=median_mad K=8 SEED=0 WARMSTART=${SMOKE_WARMSTART:-1} \
      $PY -m src.instrument_v2.train_area_commonmode_jepa
    MAX_BATCHES=2 LP_EPOCHS=1 FT_EPOCHS=1 INIT_ARM=scratch SEED=0 BACKBONE_LR=1e-4 \
      $PY -m src.instrument_v2.train_area_commonmode_finetune
    ;;

  screen)
    # 12 seed-0 cells: grouping{chip,area} x target{median,median_mad} x K{8,16,32}
    IDX=${SLURM_ARRAY_TASK_ID:?screen needs an array id}
    GROUPINGS=(chip area); TARGETS=(median median_mad); KS=(8 16 32)
    export GROUPING=${GROUPINGS[$((IDX / 6))]}
    export TARGET=${TARGETS[$(((IDX % 6) / 3))]}
    export K=${KS[$((IDX % 3))]}
    export SEED=0 EPOCHS=${SCREEN_EPOCHS:-20}
    $PY -m src.instrument_v2.train_area_commonmode_jepa
    ;;

  screen_select)
    STAGE=screen_select $PY -m src.instrument_v2.report_area_commonmode
    ;;

  confirm)
    # 6 cells: chosen config x seeds 0-2 (0-2 = area, 3-5 = chip), 50 epochs
    IDX=${SLURM_ARRAY_TASK_ID:?confirm needs an array id}
    if [ "$IDX" -lt 3 ]; then WHICH=area; else WHICH=chip; fi
    export GROUPING=$(chosen $WHICH grouping)
    export TARGET=$(chosen $WHICH target)
    export K=$(chosen $WHICH k)
    export SEED=$((IDX % 3)) EPOCHS=${CONFIRM_EPOCHS:-50}
    $PY -m src.instrument_v2.train_area_commonmode_jepa
    ;;

  finetune)
    # 36 cells: arm{scratch,groupjepa,chip_cm,area_cm} x seed{0-2} x lr{1e-5,3e-5,1e-4}
    IDX=${SLURM_ARRAY_TASK_ID:?finetune needs an array id}
    ARMS=(scratch groupjepa chip_cm area_cm); LRS=(1e-5 3e-5 1e-4)
    export INIT_ARM=${ARMS[$((IDX / 9))]}
    export SEED=$(((IDX % 9) / 3))
    export BACKBONE_LR=${LRS[$((IDX % 3))]}
    if [ "$INIT_ARM" = "chip_cm" ] || [ "$INIT_ARM" = "area_cm" ]; then
      WHICH=${INIT_ARM%_cm}
      TAG="acm_$(chosen $WHICH grouping)_$(chosen $WHICH target)_k$(chosen $WHICH k)_s${SEED}"
      export INIT_SELECTION=$ACM_ART_DIR/selection_${TAG}.json
    fi
    $PY -m src.instrument_v2.train_area_commonmode_finetune
    ;;

  final)
    STAGE=final $PY -m src.instrument_v2.report_area_commonmode
    ;;

  *)
    echo "unknown STAGE '$STAGE'"; exit 1 ;;
esac
echo "=== STAGE $STAGE DONE ==="
