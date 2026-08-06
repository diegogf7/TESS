#!/bin/bash -l
#SBATCH -J disentangle_analysis
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 01:00:00
#SBATCH -o /orcd/scratch/orcd/006/diegogon/logs/disentangle_analysis_%j.out
#SBATCH -e /orcd/scratch/orcd/006/diegogon/logs/disentangle_analysis_%j.err

# Latent UMAPs + flow-based anomaly scores for a checkpoint, INCLUDING one that is
# still training: best.pt is copied first so a concurrent torch.save cannot be read
# half-written, and the candidate galleries are skipped until reference_context.pt
# exists (it is only written when training finishes).
#
#   RUN_NAME=multichip_5sectors_v1 \
#   PARQUET=/orcd/.../disentangle_5sec/cross_sector_raw.parquet \
#   sbatch disentangle_attempt/submit_analysis.sh
#
# CPU only -- the encoders run over ~1000 curves, which is seconds.
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

RUN_NAME=${RUN_NAME:?set RUN_NAME}
PARQUET=${PARQUET:?set PARQUET}
MAX_SAMPLES=${MAX_SAMPLES:-1000}
RUN_DIR="disentangle_attempt/outputs/$RUN_NAME"
SNAP_DIR=${SNAP_DIR:-"$RUN_DIR/snapshot_$(date +%Y%m%d_%H%M%S)"}

mkdir -p "$SNAP_DIR" /orcd/scratch/orcd/006/diegogon/logs
cp "$RUN_DIR/best.pt" "$SNAP_DIR/best.pt"          # frozen copy, safe to read
[ -f "$RUN_DIR/reference_context.pt" ] && cp "$RUN_DIR/reference_context.pt" "$SNAP_DIR/" || true
[ -f "$RUN_DIR/history.csv" ] && cp "$RUN_DIR/history.csv" "$SNAP_DIR/" || true

echo "================ analysis ================"
echo "  node      : $(hostname)"
echo "  run       : $RUN_NAME"
echo "  snapshot  : $SNAP_DIR"
echo "  epochs so far:"; tail -3 "$RUN_DIR/history.csv" 2>/dev/null || echo "    (no history yet)"
echo "=========================================="

$PY -m disentangle_attempt.plot_latent_umaps \
    --checkpoint "$SNAP_DIR/best.pt" --parquet "$PARQUET" \
    --max-samples "$MAX_SAMPLES" --output-dir "$SNAP_DIR/umaps"

$PY -m disentangle_attempt.fit_anomaly_flows \
    --latents-dir "$SNAP_DIR/umaps" --checkpoint "$SNAP_DIR/best.pt" \
    --parquet "$PARQUET" --output-dir "$SNAP_DIR/anomaly_flows" --seed 42

echo "done: $SNAP_DIR"
