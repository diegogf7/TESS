#!/bin/bash -l
#SBATCH -J anomaly_20k
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /orcd/scratch/orcd/006/diegogon/logs/anomaly_20k_%j.out
#SBATCH -e /orcd/scratch/orcd/006/diegogon/logs/anomaly_20k_%j.err

# 20k chip-balanced anchors, PCA sized to 90% retained variance.
# GPU: 20k anchors means 160k peer encodings plus 80k masked physics passes, which is
# ~40 min on CPU and seconds on a GPU. Everything else is unchanged and frozen.
#
#   RUN_NAME=multichip_5sectors_v1 \
#   PARQUET=/orcd/.../disentangle_5sec/cross_sector_raw.parquet \
#   OLD_SCORES=.../snapshot_20260806_134803/anomaly_flows/anomaly_scores.csv \
#   sbatch disentangle_attempt/submit_anomaly_large.sh
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

RUN_NAME=${RUN_NAME:?set RUN_NAME}
PARQUET=${PARQUET:?set PARQUET}
RUN_DIR="disentangle_attempt/outputs/$RUN_NAME"
OUT_DIR=${OUT_DIR:-"$RUN_DIR/anomaly_analysis_20k_pca90"}
OLD_SCORES=${OLD_SCORES:-}

mkdir -p /orcd/scratch/orcd/006/diegogon/logs

echo "================ 20k anomaly analysis ================"
echo "  node       : $(hostname)"
echo "  run        : $RUN_NAME"
echo "  output     : $OUT_DIR"
echo "  reference  : $([ -f "$RUN_DIR/reference_context.pt" ] && echo present || echo MISSING)"
echo "  last epochs:"; tail -3 "$RUN_DIR/history.csv" 2>/dev/null || echo "    (none)"
echo "======================================================"

ARGS=(--checkpoint "$RUN_DIR/best.pt" --parquet "$PARQUET" --output-dir "$OUT_DIR" --seed 42)
[ -n "$OLD_SCORES" ] && ARGS+=(--old-scores "$OLD_SCORES")

$PY -m disentangle_attempt.anomaly_analysis_large "${ARGS[@]}"

echo "done: $OUT_DIR"
