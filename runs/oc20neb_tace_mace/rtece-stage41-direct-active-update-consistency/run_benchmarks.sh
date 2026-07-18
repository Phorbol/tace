#!/bin/bash
#SBATCH --job-name=rtece-active-update
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=00:30:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-active-update-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-active-update-%j.err

set -euo pipefail
set -x
cd /home/gengjianrui/bin/tace
export PYTHONPATH=/home/gengjianrui/bin/tace:${PYTHONPATH:-}
export PATH=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin:${PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
PY=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python
OUT=runs/oc20neb_tace_mace/rtece-stage41-direct-active-update-consistency
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
)

run_invalid_update_backend() {
  local backend=$1
  "${PY}" benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py     "${COMMON[@]}"     --trajectory-replay-steps 20     --trajectory-skin-margin 0.002     --trajectory-displacement-std 0.001     --graph-update-backend "${backend}"     --trajectory-update-only     --output "${OUT}/radial8h24_direct_active_${backend}_invalid_update_only.json"

  "${PY}" benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py     "${COMMON[@]}"     --trajectory-replay-steps 20     --trajectory-skin-margin 0.002     --trajectory-displacement-std 0.001     --graph-update-backend "${backend}"     --output "${OUT}/radial8h24_direct_active_${backend}_invalid_model_updates.json"
}

"${PY}" benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py   "${COMMON[@]}"   --trajectory-replay-steps 2000   --trajectory-skin-margin 0.004   --trajectory-displacement-std 0.001   --output "${OUT}/radial8h24_direct_active_valid_cached_steps2000.json"

run_invalid_update_backend cached_topology
run_invalid_update_backend torch_radius_nopbc
