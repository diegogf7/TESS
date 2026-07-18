#!/bin/bash -l
# All this code is from Claude
# Stage runner for area_commonmode_v2. Every stage exits 0 even when it
# self-skips (diagnostic failed / selection refused), so the afterok chain
# always reaches the final report -- unlike v1, where refusal cancelled it.
# Stages: diag | screen | select | confirm | finetune | final
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"
export PYTHONUNBUFFERED=1
export ACM_ART_DIR=${ACM2_ART_DIR:-artifacts/instrument_v2/area_commonmode_v2}
export ACM2_ART_DIR=$ACM_ART_DIR
export ACM_CKPT_DIR=${ACM2_CKPT_DIR:-/orcd/scratch/orcd/006/diegogon/checkpoints/area_commonmode_v2}
export GROUPING=area TARGET=median_mad K=8 GATE_MIN_PROBE=0.44
V1_ART=artifacts/instrument_v2/area_commonmode_v1
SELECTION=$ACM_ART_DIR/screen_selection.json
DIAG=$ACM_ART_DIR/raw_region_diagnostic.json

json_get() {  # json_get <file> <dotted.path>; prints value or "null"
  $PY - "$1" "$2" <<'EOF'
import json, sys
try:
    node = json.load(open(sys.argv[1]))
    for key in sys.argv[2].split("."):
        node = node[key]
    print(node)
except Exception:
    print("null")
EOF
}

diag_passed()    { [ "$(json_get "$DIAG" passes)" = "True" ]; }
have_selection() { [ "$(json_get "$SELECTION" selected.cov_weight)" != "null" ]; }

echo "=== node $(hostname) | stage ${STAGE:?set STAGE} | task ${SLURM_ARRAY_TASK_ID:-n/a} | commit $(git rev-parse --short HEAD) ==="

case "$STAGE" in
  diag)
    $PY -m src.instrument_v2.diagnose_raw_area_commonmode
    ;;

  screen)
    if ! diag_passed; then
      echo "SKIP screen: raw region diagnostic did not pass"; exit 0
    fi
    IDX=${SLURM_ARRAY_TASK_ID:?screen needs an array id}
    WEIGHTS=(0.0 0.001 0.01 0.05)
    export COV_WEIGHT=${WEIGHTS[$IDX]} SEED=0 EPOCHS=${SCREEN_EPOCHS:-20}
    $PY -m src.instrument_v2.train_area_commonmode_jepa
    ;;

  select)
    STAGE=screen_select $PY -m src.instrument_v2.report_area_commonmode_v2
    ;;

  confirm)
    if ! have_selection; then
      echo "SKIP confirm: no covariance weight was promoted"; exit 0
    fi
    IDX=${SLURM_ARRAY_TASK_ID:?confirm needs an array id}
    export COV_WEIGHT=$(json_get "$SELECTION" selected.cov_weight)
    export SEED=$IDX EPOCHS=${CONFIRM_EPOCHS:-50}
    $PY -m src.instrument_v2.train_area_commonmode_jepa
    ;;

  finetune)
    if ! have_selection; then
      echo "SKIP finetune: no covariance weight was promoted"; exit 0
    fi
    IDX=${SLURM_ARRAY_TASK_ID:?finetune needs an array id}
    ARMS=(scratch groupjepa v1_area v2_area); LRS=(1e-5 3e-5 1e-4)
    export INIT_ARM=${ARMS[$((IDX / 9))]}
    export SEED=$(((IDX % 9) / 3))
    export BACKBONE_LR=${LRS[$((IDX % 3))]}
    W=$(json_get "$SELECTION" selected.cov_weight)
    case "$INIT_ARM" in
      v1_area)
        # v1 never reached confirmation; its only checkpoints are the seed-0
        # 20-epoch screen models -- used for every seed (documented limitation).
        export INIT_SELECTION=$V1_ART/selection_acm_area_median_mad_k8_s0.json ;;
      v2_area)
        export INIT_SELECTION=$ACM_ART_DIR/selection_acm_area_median_mad_k8_cw${W}_s${SEED}.json ;;
    esac
    $PY -m src.instrument_v2.train_area_commonmode_finetune
    ;;

  final)
    STAGE=final $PY -m src.instrument_v2.report_area_commonmode_v2
    ;;

  *)
    echo "unknown STAGE '$STAGE'"; exit 1 ;;
esac
echo "=== STAGE $STAGE DONE ==="
