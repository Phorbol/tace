#!/bin/bash
#SBATCH --job-name=rtece-skin-valid
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=00:15:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-skin-valid-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-skin-valid-%j.err

set -euo pipefail
set -x
cd /home/gengjianrui/bin/tace
export PYTHONPATH=/home/gengjianrui/bin/tace:${PYTHONPATH:-}
export PATH=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin:${PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
PY=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python
OUT=runs/oc20neb_tace_mace/rtece-stage33-skin-validity
MODEL=runs/oc20neb_tace_mace/rtece-scalar-678773/rtece_element_density/rtece_scalar_best.pt
DFT=/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz
COMMON=(
  --model "${MODEL}"
  --configs "${DFT}"
  --variant rtece_element_density
  --limit-configs 256
  --force-mode auto
  --device cuda
  --default-dtype float32
  --trajectory-displacement-std 0.001
  --measure-passes 1
)

${PY} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py   "${COMMON[@]}"   --output "${OUT}/radial8h24_skin_cached_steps200.json"   --trajectory-replay-steps 200

${PY} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py   "${COMMON[@]}"   --output "${OUT}/radial8h24_skin008_steps200.json"   --trajectory-replay-steps 200   --trajectory-skin-margin 0.008

${PY} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py   "${COMMON[@]}"   --output "${OUT}/radial8h24_skin004_steps100.json"   --trajectory-replay-steps 100   --trajectory-skin-margin 0.004

${PY} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py   "${COMMON[@]}"   --output "${OUT}/radial8h24_skin002_steps20.json"   --trajectory-replay-steps 20   --trajectory-skin-margin 0.002
