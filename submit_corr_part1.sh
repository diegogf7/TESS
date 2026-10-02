#!/bin/bash
#SBATCH --job-name=corr_part1
#SBATCH --partition=mi2101x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
set -u
cd "${WORK}/TESS"
PY="${WORK}/venv_rocm/bin/python"
export PYTHONUNBUFFERED=1 TESS_DEVICE=cuda
$PY -c "import torch;print('gpu',torch.cuda.get_device_name(0))"
echo "git $(git rev-parse --short HEAD)"
$PY -m src.vicreg_jepa.corr_part1 \
  --npz "${NPZ:-artifacts/vicreg_jepa/ccd3_fill.npz}" \
  --out "${OUT:-artifacts/vicreg_jepa/corr_part1}" \
  --steps "${STEPS:-4000}" --dec-steps "${DEC_STEPS:-3000}" \
  --d-model "${DMODEL:-128}" --n-layers "${NLAYERS:-4}" \
  --objective "${OBJECTIVE:-correlation}" \
  --alpha "${ALPHA:-1.0}" --mu1 "${MU1:-1.0}" --nu1 "${NU1:-0.04}" \
  --group-size 32 --group-radius 0.0583
