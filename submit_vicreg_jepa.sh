#!/bin/bash
#SBATCH --job-name=vicreg_jepa
#SBATCH --partition=ou_mki_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Full protocol for the VICReg-JEPA experiment.
#
#   STAGE=cache      build the real-curve cache + split manifest
#   STAGE=part1      train the frozen systematics encoder
#   STAGE=pilot      lambda x mu sweep, selection on validation only
#   STAGE=ablations  controls + ablations, 3 seeds each
#   STAGE=final      the single test-set evaluation
#
# Example:
#   sbatch --export=ALL,STAGE=part1 submit_vicreg_jepa.sh
#
set -euo pipefail

source "${CONDA_ROOT:-$SCRATCH/miniforge3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-lightcurve}"

REPO="${REPO:-$PWD}"
cd "$REPO"

STAGE="${STAGE:-part1}"
OUT="${OUT:-artifacts/vicreg_jepa}"
NPZ="${NPZ:-$OUT/curves.npz}"
PARQUET="${PARQUET:-$S14_DATA/*.parquet}"
SEEDS="${SEEDS:-0 1 2}"
STEPS1="${STEPS1:-8000}"
STEPS2="${STEPS2:-30000}"
PILOT_STEPS="${PILOT_STEPS:-3000}"

mkdir -p "$OUT"
echo "stage=$STAGE  out=$OUT  npz=$NPZ  git=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

case "$STAGE" in
  cache)
    python -m src.vicreg_jepa.real_data \
      --parquet-glob "$PARQUET" --out "$NPZ" \
      ${LABEL_CSV:+--label-csv "$LABEL_CSV"}
    python -m src.vicreg_jepa.test_real_data --npz "$NPZ"
    ;;

  part1)
    python -m src.vicreg_jepa.train --part 1 --source real --npz "$NPZ" \
      --steps1 "$STEPS1" --seed 0 --out "$OUT/part1_run"
    ;;

  pilot)
    python -m src.vicreg_jepa.sweep --stage pilot --source real --npz "$NPZ" \
      --part1-ckpt "$OUT/part1_run/part1/part1_best.pt" \
      --steps2 "$PILOT_STEPS" --seed 0 --out "$OUT/pilot"
    ;;

  ablations)
    python -m src.vicreg_jepa.sweep --stage ablations --source real --npz "$NPZ" \
      --part1-ckpt "$OUT/part1_run/part1/part1_best.pt" \
      --steps2 "$STEPS2" --seeds $SEEDS --out "$OUT/ablations"
    ;;

  final)
    # RUNS must be a JSON map {name: [ckpt per seed]}; built by the ablation stage.
    : "${RUNS:?set RUNS to the JSON checkpoint map}"
    python -m src.vicreg_jepa.final_eval --source real --npz "$NPZ" \
      --part1-ckpt "$OUT/part1_run/part1/part1_best.pt" \
      --runs "$RUNS" --out "$OUT/final"
    ;;

  *)
    echo "unknown STAGE=$STAGE" >&2; exit 2 ;;
esac

echo "stage $STAGE done"
