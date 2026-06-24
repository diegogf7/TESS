#!/bin/bash -l
#SBATCH -J jepa_sweep
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -o jepa_sweep_%j.out

# Sweep the readout / patch-size fixes for the latent JEPA, all in one job.
# Goal: push the frozen CLASS probe above the masked-S4D baseline (0.473).
#
# Each config trains (100 ep) then evals (KNN probe SECTOR + CLASS) into its OWN
# checkpoint, driven purely by env vars (train_jepa.py + eval_jepa.py both read
# them via build_latent_jepa, so the architecture can't drift between them).
#
# Configs are ordered most-promising first, so a wall-time cutoff still leaves
# the best ones done. ~25-30 min each on an L40S.

cd /orcd/scratch/orcd/006/diegogon/TESS
conda activate lightcurve || source activate lightcurve

echo "=== node: $(hostname) ==="
nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"
mkdir -p /orcd/scratch/orcd/006/diegogon/checkpoints
CKPT_DIR=/orcd/scratch/orcd/006/diegogon/checkpoints

# "n_tokens:readout:checkpoint_name"
CONFIGS=(
  "16:mean_std:latent_jepa_ms16.pth"   # FIX1 only: per-segment std (recovers amplitude)
  "32:mean_std:latent_jepa_ms32.pth"   # FIX1+2: std + smaller patches (top pick)
  "64:mean_std:latent_jepa_ms64.pth"   # FIX1+2 aggressive: even finer patches
  "16:mean:latent_jepa_baseline.pth"   # control: reproduces the ~0.42 baseline
)

for cfg in "${CONFIGS[@]}"; do
  IFS=":" read -r NTOK READOUT CKPT <<< "$cfg"
  export JEPA_NTOKENS=$NTOK
  export JEPA_READOUT=$READOUT
  export JEPA_CKPT=$CKPT_DIR/$CKPT
  echo ""
  echo "############################################################"
  echo "### CONFIG  n_tokens=$NTOK  readout=$READOUT  -> $CKPT"
  echo "############################################################"
  echo "=== TRAIN ==="
  python -m src.worked_folder.train_jepa || { echo "TRAIN FAILED for $cfg"; continue; }
  echo "=== EVAL (freeze + KNN probe SECTOR/CLASS) ==="
  python -m src.worked_folder.eval_jepa || echo "EVAL FAILED for $cfg"
done

echo ""
echo "=== SWEEP DONE -- grep this log for 'CONFIG' and 'balanced accuracy' ==="
