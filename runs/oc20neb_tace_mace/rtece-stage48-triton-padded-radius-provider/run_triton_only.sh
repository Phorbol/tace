#!/bin/bash
#SBATCH --job-name=rtece-triton-only
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=00:10:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-triton-only-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-triton-only-%j.err

set -euo pipefail
set -x
cd /home/gengjianrui/bin/tace
export PYTHONPATH=/home/gengjianrui/bin/tace:${PYTHONPATH:-}
export PATH=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin:${PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
PY=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python
OUT=runs/oc20neb_tace_mace/rtece-stage48-triton-padded-radius-provider
MODEL=runs/oc20neb_tace_mace/rtece-scalar-678773/rtece_element_density/rtece_scalar_best.pt
DFT=/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz
COMMON=(
  --model "${MODEL}"
  --configs "${DFT}"
  --variant rtece_element_density
  --limit-configs 1024
  --force-mode auto
  --device cuda
  --default-dtype float32
  --measure-passes 1
  --graph-construction-backend torch_radius_nopbc
  --trajectory-replay-steps 20
  --trajectory-skin-margin 0.002
  --trajectory-displacement-std 0.001
  --graph-update-backend torch_radius_nopbc_triton_padded
)

"${PY}" benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py   "${COMMON[@]}"   --trajectory-update-only   --output "${OUT}/radial8h24_direct_active_triton_padded_invalid_update_only.json"

"${PY}" benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py   "${COMMON[@]}"   --output "${OUT}/radial8h24_direct_active_triton_padded_invalid_model_updates.json"
