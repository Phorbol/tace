#!/bin/bash
#SBATCH --job-name=rtece-traj-k2000
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=00:10:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-traj-k2000-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-traj-k2000-%j.err

set -euo pipefail
set -x
cd /home/gengjianrui/bin/tace
export PYTHONPATH=/home/gengjianrui/bin/tace:${PYTHONPATH:-}
export PATH=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin:${PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
PY=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python
OUT=runs/oc20neb_tace_mace/rtece-stage32-trajectory-replay
MODEL=runs/oc20neb_tace_mace/rtece-scalar-678773/rtece_element_density/rtece_scalar_best.pt
DFT=/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz
${PY} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py   --model "${MODEL}"   --configs "${DFT}"   --variant rtece_element_density   --limit-configs 1024   --force-mode auto   --device cuda   --default-dtype float32   --trajectory-displacement-std 0.001   --output "${OUT}/radial8h24_traj_rebuild_k2000_steps2001.json"   --measure-passes 1   --trajectory-replay-steps 2001   --trajectory-rebuild-interval 2000
