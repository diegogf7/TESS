#!/bin/bash
# Pre-download the MOMENT weights on a LOGIN NODE, before submitting the GPU job.
#
#   bash src/instrument_v2/slurm/cache_moment_weights.sh
#
# Compute nodes may have no outbound network, so the GPU job runs with
# HF_HUB_OFFLINE=1 and reads from this cache. This step is safe on a login node:
# it only downloads ~1.3 GB of weights into $HOME/.cache/huggingface and never
# touches the large parquet that gets sessions killed.
set -euo pipefail

source "$SCRATCH/miniforge3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-lightcurve}"

MOMENT_MODEL="${MOMENT_MODEL:-AutonLab/MOMENT-1-large}"
echo "caching $MOMENT_MODEL ..."

python - <<PY
import os
from momentfm import MOMENTPipeline

name = os.environ.get("MOMENT_MODEL", "AutonLab/MOMENT-1-large")
model = MOMENTPipeline.from_pretrained(name, model_kwargs={"task_name": "embedding"})
model.init()
seq_len = getattr(getattr(model, "config", object()), "seq_len", None)
n = sum(p.numel() for p in model.parameters())
print(f"cached {name}: {n/1e6:.1f}M parameters, seq_len={seq_len}")
PY

echo
echo "cache: ${HF_HOME:-$HOME/.cache/huggingface}"
echo "next: sbatch --partition=ou_mki_gpu src/instrument_v2/slurm/run_moment_baseline.sh"
