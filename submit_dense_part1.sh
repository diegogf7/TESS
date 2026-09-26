#!/bin/bash
#SBATCH --job-name=dense_part1
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=6:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#
# Part 1 on a DENSE patch with tight groups. At 0.07 deg (about 12 TESS pixels)
# 99% of stars still have 32 neighbours; the scattered-light ramp is close to
# identical over that scale, whereas the 1 deg groups used before span ~170 px
# and average over genuinely different illumination.
set -u
cd "${WORK}/TESS"
PY="${WORK}/venv_rocm/bin/python"
export PYTHONUNBUFFERED=1 TESS_DEVICE=cuda
NPZ=artifacts/vicreg_jepa/dense_s15.npz
RADIUS="${RADIUS:-0.07}"
OUT="${OUT:-artifacts/vicreg_jepa/dense_r07}"

$PY -c "import torch;print('gpu',torch.cuda.get_device_name(0))"
echo "git $(git rev-parse --short HEAD)  radius=$RADIUS"
$PY -m src.vicreg_jepa.train --part 1 --source real --npz "$NPZ" \
  --group-radius "$RADIUS" \
  --steps1 12000 --eval-every 500 --val-batches 12 --seed 0 --out "$OUT"
echo "=== DONE ==="
