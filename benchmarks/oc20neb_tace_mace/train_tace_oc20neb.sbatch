#!/bin/bash
#SBATCH --job-name=tace-oc20neb
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=03:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/tace-oc20neb-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/tace-oc20neb-%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

TACE_ROOT=${TACE_ROOT:-/home/gengjianrui/bin/tace}
ENV_DIR=${ENV_DIR:-/home/gengjianrui/bin/.venvs/tace-mace-cu126}
RUN_ROOT=${RUN_ROOT:-/home/gengjianrui/bin/tace/runs/oc20neb_tace_mace/${SLURM_JOB_ID}}
CONFIG=${CONFIG:-${TACE_ROOT}/benchmarks/oc20neb_tace_mace/tace_oc20neb.yaml}

mkdir -p /home/gengjianrui/bin/logs "${RUN_ROOT}"
cd "${RUN_ROOT}"
export PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}"
export TACE_USE_CUE=${TACE_USE_CUE:-0}
export TACE_USE_OEQ=${TACE_USE_OEQ:-0}
export TACE_USE_EQT=${TACE_USE_EQT:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-7.0}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo "host=$(hostname)"
echo "python=${ENV_DIR}/bin/python"
"${ENV_DIR}/bin/python" -V
nvidia-smi -L
export HYDRA_FULL_ERROR=1

nvidia-smi dmon -s pucvmte -o T > "nvdmon_job-${SLURM_JOB_ID}_tace.log" &
NVDMON_PID=$!
trap 'kill ${NVDMON_PID} 2>/dev/null || true' EXIT

echo "CUDA probe deferred to training process"

cp "${CONFIG}" "${RUN_ROOT}/tace.yaml"
"${ENV_DIR}/bin/tace-train" -cn tace ${TACE_TRAIN_EXTRA_ARGS:-}
