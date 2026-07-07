#!/bin/bash -l
#SBATCH -J predictor_ab
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -o predictor_ab_%j.out

# A/B: transformer vs per-token MLP predictor, both at the ms16 config.
# 2 runs per arm (no fixed seed -> independent draws; compare spreads, not singles).
# Arms interleaved so a wall-time cutoff still leaves one run of each.

cd /orcd/scratch/orcd/006/diegogon/TESS
conda activate lightcurve || source activate lightcurve

echo "=== node: $(hostname) ==="
nvidia-smi
mkdir -p /orcd/scratch/orcd/006/diegogon/checkpoints
CKPT_DIR=/orcd/scratch/orcd/006/diegogon/checkpoints

export JEPA_NTOKENS=16
export JEPA_READOUT=mean_std

# "predictor:checkpoint_name"
CONFIGS=(
  "transformer:latent_jepa_tr_run1.pth"
  "mlp:latent_jepa_mlp_run1.pth"
  "transformer:latent_jepa_tr_run2.pth"
  "mlp:latent_jepa_mlp_run2.pth"
)

for cfg in "${CONFIGS[@]}"; do
  IFS=":" read -r PRED CKPT <<< "$cfg"
  export JEPA_PREDICTOR=$PRED
  export JEPA_CKPT=$CKPT_DIR/$CKPT
  echo ""
  echo "############################################################"
  echo "### CONFIG  predictor=$PRED  -> $CKPT"
  echo "############################################################"
  echo "=== TRAIN ==="
  python -m src.worked_folder.train_jepa || { echo "TRAIN FAILED for $cfg"; continue; }
  echo "=== EVAL ==="
  python -m src.worked_folder.eval_jepa || echo "EVAL FAILED for $cfg"
done

echo ""
echo "=== A/B DONE -- grep this log for 'CONFIG' and 'balanced accuracy' ==="
