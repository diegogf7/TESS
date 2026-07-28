#!/bin/bash -l
#SBATCH -J cbv_failcorr
#SBATCH -p mit_normal
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -o cbv_failcorr_%j.out

# Validation-only failure-correlation analysis (brightness vs weak-signal vs
# detector position). No model, no GPU -- pandas/sklearn only. Reuses the
# existing oracle per-example metrics (joined by TIC).
#   sbatch -p mit_normal src/instrument_v2/submit_analyze_cbv_failure_correlations.sh
set -euo pipefail

PY=${PY:-/home/diegogon/orcd/scratch/miniforge3/envs/lightcurve/bin/python}
REPO=${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}
cd "$REPO"

export SEED=${SEED:-0}
export GROUP_SIZE=${GROUP_SIZE:-32}
export CBV_RANK=${CBV_RANK:-8}
export MIN_VALID_STARS=${MIN_VALID_STARS:-16}
DENSE_V2=/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_raw_cadence_s14_dense_v2.parquet
export S14_DATA=${S14_DATA:-$DENSE_V2}
export SPLIT_DIR=${SPLIT_DIR:-artifacts/instrument_v2/dense_v2_split}
export BASE_ART_DIR=${BASE_ART_DIR:-artifacts/instrument_v2/sector14_jepa_dense_v2}
export GROUP_ART_DIR=${GROUP_ART_DIR:-artifacts/instrument_v2/custom_group32_cbv8_mlp_qclean_v1}
export ORACLE_CSV=${ORACLE_CSV:-$GROUP_ART_DIR/oracle_ceiling/per_example_metrics.csv}
export POSITIONS_PATH=${POSITIONS_PATH:-/orcd/scratch/orcd/006/diegogon/tglc_primary/tglc_positions.parquet}
export EXPECTED_N=${EXPECTED_N:-10498}

echo "================ resolved configuration ================"
echo "  node          : $(hostname)"
echo "  git commit    : $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "  ORACLE_CSV    : $ORACLE_CSV"
echo "  POSITIONS_PATH: $POSITIONS_PATH"
echo "  S14_DATA      : $S14_DATA"
echo "  EXPECTED_N    : $EXPECTED_N"
echo "========================================================"

"$PY" -m src.instrument_v2.analyze_cbv_failure_correlations
echo "=== DONE cbv_failcorr ==="
