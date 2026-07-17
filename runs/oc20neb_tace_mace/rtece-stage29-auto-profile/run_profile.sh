#!/bin/bash
#SBATCH --job-name=rtece-auto-prof
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=00:45:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-auto-prof-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-auto-prof-%j.err

set -euo pipefail
set -x
cd /home/gengjianrui/bin/tace
export PYTHONPATH=/home/gengjianrui/bin/tace:${PYTHONPATH:-}
export PATH=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin:${PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
PY=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python
OUT=runs/oc20neb_tace_mace/rtece-stage29-auto-profile
MODEL=runs/oc20neb_tace_mace/rtece-scalar-678773/rtece_element_density/rtece_scalar_best.pt
DFT=/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz

${PY} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py \
  --model "${MODEL}" \
  --configs "${DFT}" \
  --output "${OUT}/radial8h24_auto_prebuilt_1024.json" \
  --variant rtece_element_density \
  --limit-configs 1024 \
  --measure-passes 5 \
  --force-mode auto \
  --device cuda \
  --default-dtype float32

${PY} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py \
  --model "${MODEL}" \
  --configs "${DFT}" \
  --output "${OUT}/radial8h24_auto_include_graph_1024.json" \
  --variant rtece_element_density \
  --limit-configs 1024 \
  --measure-passes 3 \
  --force-mode auto \
  --include-graph-construction \
  --device cuda \
  --default-dtype float32

${PY} benchmarks/oc20neb_tace_mace/profile_rtece_scalar.py \
  --model "${MODEL}" \
  --configs "${DFT}" \
  --output "${OUT}/radial8h24_auto_profile_1024.json" \
  --trace-output "${OUT}/radial8h24_auto_profile_1024_trace.json" \
  --limit-configs 1024 \
  --warmup-passes 3 \
  --profile-passes 5 \
  --top-ops 25 \
  --force-mode auto \
  --device cuda \
  --default-dtype float32
