#!/bin/bash
#SBATCH --job-name=rtece-st181-rmd17-bench
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=01:00:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-st181-rmd17-bench-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-st181-rmd17-bench-%j.err

set -eo pipefail
set -x
TACE_ROOT=/home/gengjianrui/bin/tace
ENV_DIR=/home/gengjianrui/bin/.venvs/tace-mace-cu126
TACE_PYTHON="${ENV_DIR}/bin/python"
PATH="${ENV_DIR}/bin:${PATH}"
DATASET_ROOT=datasets/3BPA/dataset_3BPA
RMD17_ROOT=datasets/rMD17
RESULTS_ROOT=runs/oc20neb_tace_mace/rtece-stage181-conventional-md/results
DIAGNOSTICS_ROOT=runs/oc20neb_tace_mace/rtece-stage181-conventional-md/diagnostics
mkdir -p /home/gengjianrui/bin/logs "${RESULTS_ROOT}" "${DIAGNOSTICS_ROOT}"
cd "${TACE_ROOT}"
"${TACE_PYTHON}" -V
nvidia-smi -L

mkdir -p runs/oc20neb_tace_mace/rtece-stage181-conventional-md/diagnostics
PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" "${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model runs/oc20neb_tace_mace/rtece-stage181-conventional-md/results/stage181_rmd17_ethanol/rtece_scalar_best.pt --configs "${RMD17_ROOT}/converted_extxyz/rmd17_ethanol_test.extxyz" --output runs/oc20neb_tace_mace/rtece-stage181-conventional-md/diagnostics/stage181_rmd17_ethanol_test_benchmark.json --variant stage181_rmd17_ethanol_test --start-config 0 --limit-configs 512 --measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist
