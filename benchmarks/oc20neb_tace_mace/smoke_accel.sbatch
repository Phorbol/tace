#!/bin/bash
#SBATCH --job-name=tace-accel
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=00:20:00
#SBATCH --output=/home/gengjianrui/bin/logs/tace-accel-%x-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/tace-accel-%x-%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

TACE_ROOT=${TACE_ROOT:-/home/gengjianrui/bin/tace}
ENV_DIR=${ENV_DIR:-/home/gengjianrui/bin/.venvs/tace-mace-cu126}
BACKEND=${BACKEND:-none}
RUN_ROOT=${RUN_ROOT:-/home/gengjianrui/bin/tace/runs/oc20neb_tace_mace/accel-${BACKEND}-${SLURM_JOB_ID}}
CONFIG=${CONFIG:-${TACE_ROOT}/benchmarks/oc20neb_tace_mace/tace_oc20neb.yaml}
mkdir -p /home/gengjianrui/bin/logs "${RUN_ROOT}"
cd "${RUN_ROOT}"
export PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}"
export PATH="${ENV_DIR}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-7.0}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

source /etc/profile.d/modules.sh 2>/dev/null || true

case "${BACKEND}" in
  none)
    export TACE_USE_OEQ=0 TACE_USE_CUE=0 TACE_USE_EQT=0 TACE_USE_COMPILE=0 ;;
  oeq)
    set +u
    module load cuda/12.6.3
    set -u
    NVCC_PATH=$(command -v nvcc || true)
    if [[ -z "${NVCC_PATH}" ]]; then
      echo "nvcc not found after module load cuda/12.6.3" >&2
      exit 2
    fi
    export CUDA_HOME=${CUDA_HOME:-$(cd "$(dirname "${NVCC_PATH}")/.." && pwd)}
    echo "nvcc=${NVCC_PATH} CUDA_HOME=${CUDA_HOME}"
    export TACE_USE_OEQ=1 TACE_USE_CUE=0 TACE_USE_EQT=0 TACE_USE_COMPILE=0 ;;
  cue)
    export TACE_USE_OEQ=0 TACE_USE_CUE=1 TACE_USE_EQT=0 TACE_USE_COMPILE=0 ;;
  compile)
    export TACE_USE_OEQ=0 TACE_USE_CUE=0 TACE_USE_EQT=0 TACE_USE_COMPILE=1 ;;
  eqt)
    export TACE_USE_OEQ=0 TACE_USE_CUE=0 TACE_USE_EQT=1 TACE_USE_COMPILE=0 ;;
  *) echo "unknown BACKEND=${BACKEND}" >&2; exit 2 ;;
esac

echo "host=$(hostname)"
echo "backend=${BACKEND}"
echo "TACE_USE_OEQ=${TACE_USE_OEQ:-0} TACE_USE_CUE=${TACE_USE_CUE:-0} TACE_USE_EQT=${TACE_USE_EQT:-0} TACE_USE_COMPILE=${TACE_USE_COMPILE:-0}"
"${ENV_DIR}/bin/python" -V
nvidia-smi -L
export HYDRA_FULL_ERROR=1

nvidia-smi dmon -s pucvmte -o T > "nvdmon_job-${SLURM_JOB_ID}_accel-${BACKEND}.log" &
NVDMON_PID=$!
trap 'kill ${NVDMON_PID} 2>/dev/null || true' EXIT
cp "${CONFIG}" "${RUN_ROOT}/tace.yaml"
"${ENV_DIR}/bin/tace-train" -cn tace   trainer.max_steps=2   trainer.max_epochs=1   trainer.check_val_every_n_epoch=1   trainer.log_every_n_steps=1   callbacks.early_stopping.monitor=train/loss_epoch   scheduler.extra.monitor=train/loss_epoch   scheduler.extra.frequency=1
