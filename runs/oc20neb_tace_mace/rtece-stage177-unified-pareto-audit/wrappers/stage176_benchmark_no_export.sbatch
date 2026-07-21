#!/bin/bash
#SBATCH --job-name=rtece177-bench
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=03:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece177-bench-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece177-bench-%j.err

set -eo pipefail
set -x
PYTHONUNBUFFERED=1
TACE_ROOT=${TACE_ROOT:-/home/gengjianrui/bin/tace}
ENV_DIR=${ENV_DIR:-/home/gengjianrui/bin/.venvs/tace-mace-cu126}
TACE_PYTHON=${TACE_PYTHON:-${ENV_DIR}/bin/python}
PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-}
PATH=${ENV_DIR}/bin:${PATH}
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-7.0}
mkdir -p /home/gengjianrui/bin/logs
cd "${TACE_ROOT}"
env PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" PYTHONPATH="${PYTHONPATH}" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" "${TACE_PYTHON}" -V
nvidia-smi -L
OUT_ROOT="runs/oc20neb_tace_mace/rtece-stage177-unified-pareto-audit/stage176_benchmark"
CHECKPOINT="runs/oc20neb_tace_mace/rtece-stage176-production-local-l0-front/train_smoke/rtece_scalar_best.pt"
DFT_VALID_FILE="/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz"
TEACHER_VALID_FILE="runs/oc20neb_tace_mace/tece-distill-20260717/teacher_valid.extxyz"
mkdir -p "${OUT_ROOT}"
env PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" PYTHONPATH="${PYTHONPATH}" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" "${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model "${CHECKPOINT}" --configs "${DFT_VALID_FILE}" --output "${OUT_ROOT}/stage176_local_l0_rank3_dft_benchmark.json" --variant stage176_local_l0_rank3 --device cuda --default-dtype float32 --start-config 0 --limit-configs 1024 --measure-passes 5 --force-mode autograd --graph-construction-backend matscipy_neighborlist
env PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" PYTHONPATH="${PYTHONPATH}" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" "${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model "${CHECKPOINT}" --configs "${TEACHER_VALID_FILE}" --output "${OUT_ROOT}/stage176_local_l0_rank3_teacher_benchmark.json" --variant stage176_local_l0_rank3 --device cuda --default-dtype float32 --start-config 0 --limit-configs 1024 --measure-passes 5 --force-mode autograd --graph-construction-backend matscipy_neighborlist
env PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" PYTHONPATH="${PYTHONPATH}" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" "${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model "${CHECKPOINT}" --configs "${DFT_VALID_FILE}" --output "${OUT_ROOT}/stage176_local_l0_rank3_scaling_limit32.json" --variant stage176_local_l0_rank3 --device cuda --default-dtype float32 --start-config 0 --limit-configs 32 --measure-passes 5 --force-mode autograd --graph-construction-backend matscipy_neighborlist
env PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" PYTHONPATH="${PYTHONPATH}" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" "${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model "${CHECKPOINT}" --configs "${DFT_VALID_FILE}" --output "${OUT_ROOT}/stage176_local_l0_rank3_scaling_limit128.json" --variant stage176_local_l0_rank3 --device cuda --default-dtype float32 --start-config 0 --limit-configs 128 --measure-passes 5 --force-mode autograd --graph-construction-backend matscipy_neighborlist
env PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" PYTHONPATH="${PYTHONPATH}" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" "${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model "${CHECKPOINT}" --configs "${DFT_VALID_FILE}" --output "${OUT_ROOT}/stage176_local_l0_rank3_scaling_limit512.json" --variant stage176_local_l0_rank3 --device cuda --default-dtype float32 --start-config 0 --limit-configs 512 --measure-passes 5 --force-mode autograd --graph-construction-backend matscipy_neighborlist
env PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" PYTHONPATH="${PYTHONPATH}" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" "${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model "${CHECKPOINT}" --configs "${DFT_VALID_FILE}" --output "${OUT_ROOT}/stage176_local_l0_rank3_scaling_limit1024.json" --variant stage176_local_l0_rank3 --device cuda --default-dtype float32 --start-config 0 --limit-configs 1024 --measure-passes 5 --force-mode autograd --graph-construction-backend matscipy_neighborlist
