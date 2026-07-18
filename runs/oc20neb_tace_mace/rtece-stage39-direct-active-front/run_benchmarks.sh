#!/bin/bash
#SBATCH --job-name=rtece-active-front
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=01:00:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-active-front-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-active-front-%j.err

set -euo pipefail
set -x
cd /home/gengjianrui/bin/tace
export PYTHONPATH=/home/gengjianrui/bin/tace:${PYTHONPATH:-}
export PATH=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin:${PATH}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
PY=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python
OUT=runs/oc20neb_tace_mace/rtece-stage39-direct-active-front
DFT=/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz
TEACHER=runs/oc20neb_tace_mace/tece-distill-20260717/teacher_valid.extxyz
MODE=auto

run_one() {
  local name=$1
  local model=$2
  local configs=$3
  local suffix=$4
  ${PY} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py     --model "${model}"     --configs "${configs}"     --output "${OUT}/${name}_${suffix}_direct_active.json"     --variant rtece_element_density     --limit-configs 4096     --measure-passes 5     --force-mode "${MODE}"     --device cuda     --default-dtype float32     --graph-construction-backend torch_radius_nopbc
}

run_pair() {
  local name=$1
  local model=$2
  run_one "${name}" "${model}" "${DFT}" dft
  run_one "${name}" "${model}" "${TEACHER}" teacher
}

run_pair radial4h16 runs/oc20neb_tace_mace/rtece-stage23-radial4-hidden16/rtece_element_density/rtece_scalar_best.pt
run_pair radial5h16_stage24 runs/oc20neb_tace_mace/rtece-stage24-radial5-hidden16/rtece_element_density/rtece_scalar_best.pt
run_pair radial5h16_seed2_long runs/oc20neb_tace_mace/rtece-stage26-radial5-hidden16-seed2-long/rtece_element_density/rtece_scalar_best.pt
run_pair radial8h24 runs/oc20neb_tace_mace/rtece-scalar-678773/rtece_element_density/rtece_scalar_best.pt
run_pair radial8h32 runs/oc20neb_tace_mace/rtece-scalar-678749/rtece_element_density/rtece_scalar_best.pt
