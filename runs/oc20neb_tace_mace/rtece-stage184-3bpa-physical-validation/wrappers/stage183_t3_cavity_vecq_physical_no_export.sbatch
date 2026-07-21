#!/bin/bash
#SBATCH --job-name=rtece-st184-stage183_t3_cavity_vecq
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=02:30:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-st184-stage183_t3_cavity_vecq-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-st184-stage183_t3_cavity_vecq-%j.err

set -eo pipefail
set -x
TACE_ROOT=/home/gengjianrui/bin/tace
TACE_PYTHON=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python
PATH=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin:${PATH}
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
TORCH_CUDA_ARCH_LIST=7.0
mkdir -p /home/gengjianrui/bin/logs
cd ${TACE_ROOT}
${TACE_PYTHON} -V

mkdir -p runs/oc20neb_tace_mace/rtece-stage184-3bpa-physical-validation/diagnostics/stage183_t3_cavity_vecq
PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/dimer_scan_rtece.py --checkpoint runs/oc20neb_tace_mace/rtece-stage183-3bpa-representation-ladder/results/stage183_t3_cavity_vecq/rtece_scalar_best.pt --output-json runs/oc20neb_tace_mace/rtece-stage184-3bpa-physical-validation/diagnostics/stage183_t3_cavity_vecq/stage183_t3_cavity_vecq_dimer.json --output-md runs/oc20neb_tace_mace/rtece-stage184-3bpa-physical-validation/diagnostics/stage183_t3_cavity_vecq/stage183_t3_cavity_vecq_dimer.md --route stage183_t3_cavity_vecq --pairs C-N C-O C-H N-H O-H C-C N-N --num-points 16 --min-scale 0.5 --max-scale 5.0 --device cuda --default-dtype float32 --force-mode autograd --neighborlist-backend ase
PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/rattle_relax_rtece.py --checkpoint runs/oc20neb_tace_mace/rtece-stage183-3bpa-representation-ladder/results/stage183_t3_cavity_vecq/rtece_scalar_best.pt --configs datasets/3BPA/dataset_3BPA/test_300K.xyz --output-json runs/oc20neb_tace_mace/rtece-stage184-3bpa-physical-validation/diagnostics/stage183_t3_cavity_vecq/stage183_t3_cavity_vecq_rattle.json --output-md runs/oc20neb_tace_mace/rtece-stage184-3bpa-physical-validation/diagnostics/stage183_t3_cavity_vecq/stage183_t3_cavity_vecq_rattle.md --route stage183_t3_cavity_vecq --start-config 0 --limit-configs 8 --rattle-std 0.05 --rattle-seed 20260721 --fmax 0.05 --max-steps 10 --device cuda --default-dtype float32 --force-mode autograd --neighborlist-backend matscipy
