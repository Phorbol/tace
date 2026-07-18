#!/bin/bash
#SBATCH --job-name=rtece-counted-stress
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=00:30:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-counted-stress-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-counted-stress-%j.err

set -uo pipefail
set -x
cd /home/gengjianrui/bin/tace
export PYTHONPATH=/home/gengjianrui/bin/tace:${PYTHONPATH:-}
export PATH=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin:${PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
PY=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python
OUT=runs/oc20neb_tace_mace/rtece-stage50-triton-counted-batch-stress
MODEL=runs/oc20neb_tace_mace/rtece-scalar-678773/rtece_element_density/rtece_scalar_best.pt
DFT=/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz

run_one() {
  local limit=$1
  local backend=$2
  local label=$3
  local output="${OUT}/radial8h24_direct_active_${label}_limit${limit}_invalid_update_only.json"
  local status="${OUT}/radial8h24_direct_active_${label}_limit${limit}_invalid_update_only.status"
  "${PY}" benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py \
    --model "${MODEL}" \
    --configs "${DFT}" \
    --variant rtece_element_density \
    --limit-configs "${limit}" \
    --force-mode auto \
    --device cuda \
    --default-dtype float32 \
    --measure-passes 1 \
    --graph-construction-backend torch_radius_nopbc \
    --trajectory-replay-steps 20 \
    --trajectory-skin-margin 0.002 \
    --trajectory-displacement-std 0.001 \
    --trajectory-update-only \
    --graph-update-backend "${backend}" \
    --output "${output}"
  local rc=$?
  echo "${rc}" > "${status}"
  return 0
}

for limit in 1024 2048 4096 8192; do
  run_one "${limit}" torch_radius_nopbc_triton_padded triton_padded
  run_one "${limit}" torch_radius_nopbc_triton_counted triton_counted
done
