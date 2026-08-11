# Shared environment for the multi-chip scale-up on ORCD.
# Sourced by every script here.  Edit this file, not the individual scripts.

# Where the repository lives on the cluster.
export REPO="${REPO:-/orcd/scratch/orcd/006/diegogon/TESS}"

# Where the ~22 GB of data goes (FITS + parquet chunks + final parquet).
export DATA_DIR="${DATA_DIR:-/orcd/scratch/orcd/006/diegogon/multichip_data}"

# Where the dataset cache, checkpoints, metrics and plots go.
export OUT_DIR="${OUT_DIR:-/orcd/scratch/orcd/006/diegogon/multichip_out}"

export LOG_DIR="${LOG_DIR:-/orcd/scratch/orcd/006/diegogon/logs}"

# Sectors 1-5, all 16 camera/CCD chips each = 80 chips.
export SECTORS="${SECTORS:-1,2,3,4,5}"
export CHIPS="${CHIPS:-all}"
export STARS_PER_CHIP="${STARS_PER_CHIP:-3000}"

# Slurm partitions.  GPU work goes to ou_mki_gpu, CPU work to pg_mki_aryeh.
export GPU_PARTITION="${GPU_PARTITION:-ou_mki_gpu}"
export CPU_PARTITION="${CPU_PARTITION:-pg_mki_aryeh}"

export CONFIG="${CONFIG:-$REPO/disentangle_attempt/config_multichip_s1_s5.yaml}"

# Conda env used by the existing submit_train.sh.  The calling scripts run with
# `set -u`, but conda's own activation scripts reference unset variables, so relax it
# for the duration and restore it afterwards.
_conda_profile="${SCRATCH:-}/miniforge3/etc/profile.d/conda.sh"
if [ -n "${SCRATCH:-}" ] && [ -f "$_conda_profile" ]; then
  case "$-" in *u*) _restore_u=1; set +u ;; *) _restore_u=0 ;; esac
  source "$_conda_profile"
  conda activate "${CONDA_ENV:-lightcurve}"
  [ "$_restore_u" = "1" ] && set -u
  unset _restore_u
else
  echo "note: \$SCRATCH/miniforge3 not found; using the python already on PATH" >&2
fi
unset _conda_profile

mkdir -p "$DATA_DIR" "$OUT_DIR" "$LOG_DIR"
cd "$REPO"
