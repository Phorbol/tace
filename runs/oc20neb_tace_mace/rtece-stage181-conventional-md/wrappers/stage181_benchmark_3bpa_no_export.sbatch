#!/bin/bash
#SBATCH --job-name=rtece-st181-3bpa-bench
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=02:00:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-st181-3bpa-bench-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-st181-3bpa-bench-%j.err

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

mkdir -p runs/oc20neb_tace_mace/rtece-stage181-conventional-md/diagnostics/stage181_3bpa_train300K
PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" "${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model runs/oc20neb_tace_mace/rtece-stage181-conventional-md/results/stage181_3bpa_train300K/rtece_scalar_best.pt --configs "${DATASET_ROOT}/test_300K.xyz" --output runs/oc20neb_tace_mace/rtece-stage181-conventional-md/diagnostics/stage181_3bpa_train300K/test_300K_benchmark.json --variant stage181_3bpa_train300K_test_300K --start-config 0 --limit-configs 512 --measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist
PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" "${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model runs/oc20neb_tace_mace/rtece-stage181-conventional-md/results/stage181_3bpa_train300K/rtece_scalar_best.pt --configs "${DATASET_ROOT}/test_600K.xyz" --output runs/oc20neb_tace_mace/rtece-stage181-conventional-md/diagnostics/stage181_3bpa_train300K/test_600K_benchmark.json --variant stage181_3bpa_train300K_test_600K --start-config 0 --limit-configs 512 --measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist
PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" "${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model runs/oc20neb_tace_mace/rtece-stage181-conventional-md/results/stage181_3bpa_train300K/rtece_scalar_best.pt --configs "${DATASET_ROOT}/test_1200K.xyz" --output runs/oc20neb_tace_mace/rtece-stage181-conventional-md/diagnostics/stage181_3bpa_train300K/test_1200K_benchmark.json --variant stage181_3bpa_train300K_test_1200K --start-config 0 --limit-configs 512 --measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist
PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" "${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model runs/oc20neb_tace_mace/rtece-stage181-conventional-md/results/stage181_3bpa_train300K/rtece_scalar_best.pt --configs "${DATASET_ROOT}/test_dih.xyz" --output runs/oc20neb_tace_mace/rtece-stage181-conventional-md/diagnostics/stage181_3bpa_train300K/test_dih_benchmark.json --variant stage181_3bpa_train300K_test_dih --start-config 0 --limit-configs 512 --measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist
