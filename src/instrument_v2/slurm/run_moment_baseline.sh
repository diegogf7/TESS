#!/bin/bash
#SBATCH --job-name=moment_baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=moment_baseline_%j.out
#SBATCH --error=moment_baseline_%j.err
#
# MOMENT zero-shot vs the physics JEPA on our PhyTS-matched S14 cohort.
#
#   sbatch --partition=ou_mki_gpu src/instrument_v2/slurm/run_moment_baseline.sh
#
# This MUST be a batch job, not a login-node run. The dense_v2 TGLC parquet is
# written as a single row group, so read_parquet pulls the whole file into memory
# and the login-node watchdog kills the session (taking the ssh connection with it).
# Hence --mem=128G.
#
# Weights are expected to be pre-cached by cache_moment_weights.sh, because compute
# nodes may have no outbound network. HF_HUB_OFFLINE makes a missing cache fail
# immediately and legibly instead of hanging on a network call.
set -euo pipefail

source "$SCRATCH/miniforge3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-lightcurve}"
cd "${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export MOMENT_ARM="${MOMENT_ARM:-both}"
export PYTHONUNBUFFERED=1

echo "=== $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import torch;print('cuda', torch.cuda.is_available(), torch.__version__)"
date

python -m src.instrument_v2.eval_moment_phyts_baseline

echo
date
echo "summary: artifacts/instrument_v2/moment_phyts_baseline/summary.json"
