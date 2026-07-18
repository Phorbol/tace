#!/bin/bash
#SBATCH --job-name=rtece-active-graph
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=00:18:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-active-graph-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-active-graph-%j.err

set -euo pipefail
set -x
cd /home/gengjianrui/bin/tace
export PYTHONPATH=/home/gengjianrui/bin/tace:${PYTHONPATH:-}
export PATH=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin:${PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
PY=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python
OUT=runs/oc20neb_tace_mace/rtece-stage38-active-graph-construction
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
  --measure-passes 3
)

run_backend() {
  local backend=$1
  ${PY} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py     "${COMMON[@]}"     --graph-construction-backend "${backend}"     --output "${OUT}/radial8h24_${backend}_prebuilt.json"

  ${PY} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py     "${COMMON[@]}"     --graph-construction-backend "${backend}"     --include-graph-construction     --batch-graph-construction     --measure-passes 1     --output "${OUT}/radial8h24_${backend}_rebuild_batch.json"

  ${PY} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py     "${COMMON[@]}"     --graph-construction-backend "${backend}"     --trajectory-replay-steps 2000     --trajectory-skin-margin 0.004     --trajectory-displacement-std 0.001     --measure-passes 1     --output "${OUT}/radial8h24_${backend}_trajectory_cached_steps2000.json"
}

run_backend ase_neighborlist
run_backend torch_radius_nopbc
