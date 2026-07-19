#!/bin/bash
#SBATCH --job-name=tm-bench
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --output=/home/gengjianrui/bin/logs/tace-mace-bench-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/tace-mace-bench-%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

TACE_ROOT=${TACE_ROOT:-/home/gengjianrui/bin/tace}
ENV_DIR=${ENV_DIR:-/home/gengjianrui/bin/.venvs/tace-mace-cu126}
MACE_PYTHON=${MACE_PYTHON:-/home/sjtu-caoxiaoming/gengjianrui/conda-envs/mace-dpa4-cu126/bin/python}
TACE_PYTHON=${TACE_PYTHON:-${ENV_DIR}/bin/python}
RUN_ROOT=${RUN_ROOT:-/home/gengjianrui/bin/tace/runs/oc20neb_tace_mace/bench-${SLURM_JOB_ID}}
VALID_FILE=${VALID_FILE:-/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz}
MACE_MODEL=${MACE_MODEL:-/home/gengjianrui/worktrees/mace-update-boundary/runs/oc20neb_fullcase200_ef_20k/674543/cueq_hybrid_muon/models/oc20neb_fullcase200_ef20k_cueq_hybrid_muon.model}
TACE_MODEL=${TACE_MODEL:-}
LIMIT_CONFIGS=${LIMIT_CONFIGS:-1024}
BATCH_SIZE=${BATCH_SIZE:-16}
MEASURE_PASSES=${MEASURE_PASSES:-3}
MACE_ACCEL_BACKEND=${MACE_ACCEL_BACKEND:-none}
TACE_ACCEL_BACKEND=${TACE_ACCEL_BACKEND:-none}
TACE_STRICT_LOAD=${TACE_STRICT_LOAD:-1}

mkdir -p /home/gengjianrui/bin/logs "${RUN_ROOT}"
cd "${RUN_ROOT}"
export PYTHONPATH="${TACE_ROOT}:/home/gengjianrui/worktrees/mace-update-boundary:${PYTHONPATH:-}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-7.0}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

if [[ "${TACE_ACCEL_BACKEND}" == "oeq" || "${MACE_ACCEL_BACKEND}" == "oeq" || "${MACE_ACCEL_BACKEND}" == "hybrid" ]]; then
  source /etc/profile.d/modules.sh 2>/dev/null || true
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
fi

echo "host=$(hostname)"
echo "mace_python=${MACE_PYTHON}"
echo "mace_accel_backend=${MACE_ACCEL_BACKEND}"
echo "tace_python=${TACE_PYTHON}"
"${MACE_PYTHON}" -V
"${TACE_PYTHON}" -V
nvidia-smi -L

nvidia-smi dmon -s pucvmte -o T > "nvdmon_job-${SLURM_JOB_ID}_bench.log" &
NVDMON_PID=$!
trap 'kill ${NVDMON_PID} 2>/dev/null || true' EXIT

"${MACE_PYTHON}" "${TACE_ROOT}/benchmarks/oc20neb_tace_mace/benchmark_models.py" \
  --backend mace \
  --model "${MACE_MODEL}" \
  --configs "${VALID_FILE}" \
  --output "${RUN_ROOT}/mace_benchmark.json" \
  --batch-size "${BATCH_SIZE}" \
  --device cuda \
  --default-dtype float32 \
  --limit-configs "${LIMIT_CONFIGS}" \
  --measure-passes "${MEASURE_PASSES}" \
  --mace-accel-backend "${MACE_ACCEL_BACKEND}"

if [[ -n "${TACE_MODEL}" ]]; then
  echo "tace_accel_backend=${TACE_ACCEL_BACKEND} tace_strict_load=${TACE_STRICT_LOAD}"
  "${TACE_PYTHON}" "${TACE_ROOT}/benchmarks/oc20neb_tace_mace/benchmark_models.py" \
    --backend tace \
    --model "${TACE_MODEL}" \
    --configs "${VALID_FILE}" \
    --output "${RUN_ROOT}/tace_benchmark.json" \
    --batch-size "${BATCH_SIZE}" \
    --device cuda \
    --default-dtype float32 \
    --limit-configs "${LIMIT_CONFIGS}" \
    --measure-passes "${MEASURE_PASSES}" \
    --tace-accel-backend "${TACE_ACCEL_BACKEND}" \
    --tace-strict-load "${TACE_STRICT_LOAD}"
else
  echo "TACE_MODEL is empty; ran MACE baseline only. Set TACE_MODEL=/path/to/TACE.ckpt after training."
fi
