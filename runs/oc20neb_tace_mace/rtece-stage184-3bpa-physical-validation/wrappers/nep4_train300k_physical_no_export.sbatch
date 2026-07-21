#!/bin/bash
#SBATCH --job-name=rtece-st184-nep4-phys
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=03:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-st184-nep4-phys-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-st184-nep4-phys-%j.err

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

module load gpumd/4.8-cuda12.4
mkdir -p runs/oc20neb_tace_mace/rtece-stage184-3bpa-physical-validation/diagnostics/nep4_train300k runs/oc20neb_tace_mace/rtece-stage184-3bpa-physical-validation/diagnostics/nep4_train300k/nep_prediction_work
PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/physical_stage145_community.py --engine nep --model-artifact runs/oc20neb_tace_mace/rtece-stage182-3bpa-closure/results/nep4_train300k/nep.txt --configs datasets/3BPA/dataset_3BPA/test_300K.xyz --output-json runs/oc20neb_tace_mace/rtece-stage184-3bpa-physical-validation/diagnostics/nep4_train300k/nep4_train300k_community_physical.json --row-name nep4_train300k --run-dir runs/oc20neb_tace_mace/rtece-stage184-3bpa-physical-validation/diagnostics/nep4_train300k/nep_prediction_work --pairs C-N C-O C-H N-H O-H C-C N-N --dimer-points 16 --min-scale 0.5 --max-scale 5.0 --start-config 0 --limit-configs 8 --rattle-std 0.05 --rattle-seed 20260721 --fmax 0.05 --max-steps 10 --device cuda
