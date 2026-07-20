#!/bin/bash -l
# All this code is from Claude
# Stage runner for the fixed-regional-teacher pilot (seed 0, val-only).
# Every stage exits 0 on self-skip so the final report is always produced.
# Stages: teacher | student | probes | finetune | final
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"
export PYTHONUNBUFFERED=1
export FRT_ART_DIR=${FRT_ART_DIR:-artifacts/instrument_v2/fixed_regional_teacher_v1}
export FRT_CKPT_DIR=${FRT_CKPT_DIR:-/orcd/scratch/orcd/006/diegogon/checkpoints/fixed_regional_teacher_v1}
export SEED=0 K=8
TEACHER_SEL=$FRT_ART_DIR/selection_regteacher_k8_s0.json
STUDENT_SEL=$FRT_ART_DIR/selection_frtstudent_k8_s0.json

json_get() {
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

teacher_passed() { [ "$(json_get "$TEACHER_SEL" passed_gates)" = "True" ]; }
have_student()   { [ -f "$STUDENT_SEL" ]; }

echo "=== node $(hostname) | stage ${STAGE:?set STAGE} | task ${SLURM_ARRAY_TASK_ID:-n/a} | commit $(git rev-parse --short HEAD) ==="

case "$STAGE" in
  teacher)
    EPOCHS=${TEACHER_EPOCHS:-20} $PY -m src.instrument_v2.train_regional_group_teacher
    ;;

  student)
    if ! teacher_passed; then
      echo "SKIP student: regional teacher did not pass its gates"; exit 0
    fi
    EPOCHS=${STUDENT_EPOCHS:-30} $PY -m src.instrument_v2.train_fixed_teacher_instrument_jepa
    ;;

  probes)
    STAGE=probes $PY -m src.instrument_v2.eval_fixed_teacher_instrument
    ;;

  finetune)
    if ! have_student; then
      echo "SKIP finetune: no student was trained"; exit 0
    fi
    IDX=${SLURM_ARRAY_TASK_ID:?finetune needs an array id}
    ARMS=(scratch groupjepa fixed_teacher); LRS=(1e-5 3e-5 1e-4)
    export INIT_ARM=${ARMS[$((IDX / 3))]}
    export BACKBONE_LR=${LRS[$((IDX % 3))]}
    export ACM_ART_DIR=$FRT_ART_DIR ACM_CKPT_DIR=$FRT_CKPT_DIR
    if [ "$INIT_ARM" = "fixed_teacher" ]; then
      export INIT_SELECTION=$STUDENT_SEL
    fi
    $PY -m src.instrument_v2.train_area_commonmode_finetune
    ;;

  final)
    STAGE=report $PY -m src.instrument_v2.eval_fixed_teacher_instrument
    ;;

  *)
    echo "unknown STAGE '$STAGE'"; exit 1 ;;
esac
echo "=== STAGE $STAGE DONE ==="
