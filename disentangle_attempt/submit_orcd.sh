#!/bin/bash -l
#SBATCH -J disentangle_xsector
#SBATCH -p ou_mki_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 08:00:00
#SBATCH -o /orcd/scratch/orcd/006/diegogon/logs/disentangle_xsector_%j.out
#SBATCH -e /orcd/scratch/orcd/006/diegogon/logs/disentangle_xsector_%j.err

# Cross-sector disentangle attempt: train + branch-use tests + quiet-reference
# selection, then the counterfactual inference plot.
#
#   sbatch disentangle_attempt/submit_orcd.sh
#
# PREREQUISITE -- the data. This experiment needs the SAME TIC in two sectors on a
# chip dense enough for eight detector-nearest peers. Neither existing parquet gives
# that: the dense_v2 set is Sector 14 only, and tglc_raw_cadence_all.parquet samples
# each sector independently, so a TIC almost never repeats. Check first, on the LOGIN
# node (compute nodes have no outbound network):
#
#   python -m disentangle_attempt.dataset \
#     /orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_all.parquet
#
# If it reports too few multi-sector TICs, build the patch (~290 MB, ~10 min):
#
#   DA_DATA_DIR=/orcd/scratch/orcd/006/diegogon/tglc_primary/disentangle_patch \
#   N_STARS=6000 N_WORKERS=16 python -m disentangle_attempt.fetch_data
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

CONFIG=${CONFIG:-disentangle_attempt/config_orcd.yaml}
PARQUET=${PARQUET:-/orcd/scratch/orcd/006/diegogon/tglc_primary/disentangle_patch/cross_sector_raw.parquet}
RUN_NAME=${RUN_NAME:-orcd_$(date +%Y%m%d_%H%M%S)}

mkdir -p /orcd/scratch/orcd/006/diegogon/logs

echo "================ resolved configuration ================"
echo "  node        : $(hostname)"
echo "  git commit  : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "  python      : $PY"
echo "  config      : $CONFIG"
echo "  parquet     : $PARQUET"
echo "  run name    : $RUN_NAME"
echo "========================================================"

# -e not -f: fetch_data writes a parquet DIRECTORY (one file per chip).
[ -e "$PARQUET" ] || { echo "FATAL: $PARQUET missing -- see the header of this script"; exit 1; }

DA_PARQUET="$PARQUET" $PY -m disentangle_attempt.smoke_test
$PY -m disentangle_attempt.train --config "$CONFIG" --parquet "$PARQUET" --run-name "$RUN_NAME"
$PY -m disentangle_attempt.infer \
    --checkpoint "disentangle_attempt/outputs/$RUN_NAME/best.pt" --parquet "$PARQUET"
$PY -m disentangle_attempt.plot_curves \
    --checkpoint "disentangle_attempt/outputs/$RUN_NAME/best.pt" --parquet "$PARQUET" --n-stars 6

echo "done: disentangle_attempt/outputs/$RUN_NAME"
