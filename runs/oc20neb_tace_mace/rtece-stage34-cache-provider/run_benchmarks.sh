#!/bin/bash
#SBATCH --job-name=rtece-cache-provider
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=00:12:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-cache-provider-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-cache-provider-%j.err

set -euo pipefail
set -x
cd /home/gengjianrui/bin/tace
export PYTHONPATH=/home/gengjianrui/bin/tace:${PYTHONPATH:-}
export PATH=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin:${PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
PY=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python
OUT=runs/oc20neb_tace_mace/rtece-stage34-cache-provider
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
  --trajectory-displacement-std 0.001
  --trajectory-skin-margin 0.004
  --trajectory-replay-steps 2000
)

${PY} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py   "${COMMON[@]}"   --output "${OUT}/radial8h24_validity_only_skin004_steps2000.json"   --measure-passes 5   --trajectory-validity-only

${PY} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py   "${COMMON[@]}"   --output "${OUT}/radial8h24_cached_model_skin004_steps2000.json"   --measure-passes 3
