#!/bin/bash
#SBATCH --job-name=rtece-triton-radius
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=00:25:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-triton-radius-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-triton-radius-%j.err

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
)

run_backend() {
  local backend=$1
  local label=$2
  shift 2
  "${PY}" benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py     "${COMMON[@]}"     --graph-update-backend "${backend}"     "$@"     --trajectory-update-only     --output "${OUT}/radial8h24_direct_active_${label}_invalid_update_only.json"

  "${PY}" benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py     "${COMMON[@]}"     --graph-update-backend "${backend}"     "$@"     --output "${OUT}/radial8h24_direct_active_${label}_invalid_model_updates.json"
}

run_backend torch_radius_nopbc_grouped grouped
run_backend torch_radius_nopbc_grouped_chunked chunk256 --graph-update-chunk-configs 256
run_backend torch_radius_nopbc_ragged ragged
run_backend torch_radius_nopbc_triton_padded triton_padded
