#!/bin/bash
#SBATCH --job-name=rtece-st182-deepmd-bench
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=01:40:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-st182-deepmd-bench-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-st182-deepmd-bench-%j.err

set -eo pipefail
set -x
TACE_ROOT=/home/gengjianrui/bin/tace
TACE_PYTHON=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python
PATH=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin:${PATH}
mkdir -p /home/gengjianrui/bin/logs
cd ${TACE_ROOT}
${TACE_PYTHON} -V

module load deepmd-kit/3.1.2
mkdir -p runs/oc20neb_tace_mace/rtece-stage182-3bpa-closure/diagnostics/deepmd_dpa1_zero_train300k
PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/benchmark_stage145_community.py --engine deepmd --model-artifact runs/oc20neb_tace_mace/rtece-stage182-3bpa-closure/results/deepmd_dpa1_zero_train300k/frozen_model.pth --configs datasets/3BPA/dataset_3BPA/test_300K.xyz --output runs/oc20neb_tace_mace/rtece-stage182-3bpa-closure/diagnostics/deepmd_dpa1_zero_train300k/test_300K_benchmark.json --row-name deepmd_dpa1_zero_train300k_test_300K --energy-key energy --forces-key forces --start-config 0 --limit-configs 512 --measure-passes 5 --device cuda
PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/benchmark_stage145_community.py --engine deepmd --model-artifact runs/oc20neb_tace_mace/rtece-stage182-3bpa-closure/results/deepmd_dpa1_zero_train300k/frozen_model.pth --configs datasets/3BPA/dataset_3BPA/test_600K.xyz --output runs/oc20neb_tace_mace/rtece-stage182-3bpa-closure/diagnostics/deepmd_dpa1_zero_train300k/test_600K_benchmark.json --row-name deepmd_dpa1_zero_train300k_test_600K --energy-key energy --forces-key forces --start-config 0 --limit-configs 512 --measure-passes 5 --device cuda
PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/benchmark_stage145_community.py --engine deepmd --model-artifact runs/oc20neb_tace_mace/rtece-stage182-3bpa-closure/results/deepmd_dpa1_zero_train300k/frozen_model.pth --configs datasets/3BPA/dataset_3BPA/test_1200K.xyz --output runs/oc20neb_tace_mace/rtece-stage182-3bpa-closure/diagnostics/deepmd_dpa1_zero_train300k/test_1200K_benchmark.json --row-name deepmd_dpa1_zero_train300k_test_1200K --energy-key energy --forces-key forces --start-config 0 --limit-configs 512 --measure-passes 5 --device cuda
PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/benchmark_stage145_community.py --engine deepmd --model-artifact runs/oc20neb_tace_mace/rtece-stage182-3bpa-closure/results/deepmd_dpa1_zero_train300k/frozen_model.pth --configs datasets/3BPA/dataset_3BPA/test_dih.xyz --output runs/oc20neb_tace_mace/rtece-stage182-3bpa-closure/diagnostics/deepmd_dpa1_zero_train300k/test_dih_benchmark.json --row-name deepmd_dpa1_zero_train300k_test_dih --energy-key energy --forces-key forces --start-config 0 --limit-configs 512 --measure-passes 5 --device cuda
