#!/bin/bash
#SBATCH --job-name=rtece177-phys
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=03:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece177-phys-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece177-phys-%j.err

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
OUT_ROOT="runs/oc20neb_tace_mace/rtece-stage177-unified-pareto-audit/stage176_physical"
CHECKPOINT="runs/oc20neb_tace_mace/rtece-stage176-production-local-l0-front/train_smoke/rtece_scalar_best.pt"
DFT_VALID_FILE="/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz"
DFT_BENCHMARK="runs/oc20neb_tace_mace/rtece-stage177-unified-pareto-audit/stage176_benchmark/stage176_local_l0_rank3_dft_benchmark.json"
TEACHER_BENCHMARK="runs/oc20neb_tace_mace/rtece-stage177-unified-pareto-audit/stage176_benchmark/stage176_local_l0_rank3_teacher_benchmark.json"
VARIANT="stage176_local_l0_rank3"
mkdir -p "${OUT_ROOT}"
env PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" PYTHONPATH="${PYTHONPATH}" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" "${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/dimer_scan_rtece.py --checkpoint "${CHECKPOINT}" --output-json "${OUT_ROOT}/${VARIANT}_dimer.json" --output-md "${OUT_ROOT}/${VARIANT}_dimer.md" --route "${VARIANT}" --pairs C-N C-O C-H N-H O-H C-C N-N --num-points 24 --min-scale 0.5 --max-scale 5.0 --device cuda --default-dtype float32 --force-mode autograd --neighborlist-backend ase
env PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" PYTHONPATH="${PYTHONPATH}" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" "${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/rattle_relax_rtece.py --checkpoint "${CHECKPOINT}" --configs "${DFT_VALID_FILE}" --output-json "${OUT_ROOT}/${VARIANT}_rattle.json" --output-md "${OUT_ROOT}/${VARIANT}_rattle.md" --route "${VARIANT}" --start-config 58 --limit-configs 8 --rattle-std 0.05 --rattle-seed 20260718 --fmax 0.05 --max-steps 10 --device cuda --default-dtype float32 --force-mode autograd --neighborlist-backend matscipy
env PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" PYTHONPATH="${PYTHONPATH}" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" "${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/summarize_rtece_physical_pareto.py --case "${VARIANT}:${DFT_BENCHMARK}:${TEACHER_BENCHMARK}:${OUT_ROOT}/${VARIANT}_dimer.json:${OUT_ROOT}/${VARIANT}_rattle.json" --max-dft-f-rmse-mev-a 120.0 --max-dft-e-rmse-mev-atom 320.0 --max-dft-f-max-mev-a 2600.0 --max-dft-e-max-mev-atom 760.0 --rattle-focus-label C_or_N --max-focus-rattle-rmsd-a 0.35 --max-rattle-fmax-ev-a 1.00 --output-json "${OUT_ROOT}/${VARIANT}_physical_pareto.json" --output-md "${OUT_ROOT}/${VARIANT}_physical_pareto.md"
