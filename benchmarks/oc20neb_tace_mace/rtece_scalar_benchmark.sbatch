#!/bin/bash
#SBATCH --job-name=rtece-bench
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=01:00:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-bench-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-bench-%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

TACE_ROOT=${TACE_ROOT:-/home/gengjianrui/bin/tace}
ENV_DIR=${ENV_DIR:-/home/gengjianrui/bin/.venvs/tace-mace-cu126}
TACE_PYTHON=${TACE_PYTHON:-${ENV_DIR}/bin/python}
MODEL=${MODEL:?MODEL checkpoint path is required}
VARIANT=${VARIANT:-rtece_pair}
CONFIGS=${CONFIGS:-/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz}
OUT_DIR=${OUT_DIR:-${TACE_ROOT}/runs/oc20neb_tace_mace/rtece-bench-${SLURM_JOB_ID}}
LIMIT_CONFIGS_LIST=${LIMIT_CONFIGS_LIST:-1024 2048 4096}
MEASURE_PASSES=${MEASURE_PASSES:-10}
DEFAULT_DTYPE=${DEFAULT_DTYPE:-float32}
FORCE_MODE=${FORCE_MODE:-autograd}

mkdir -p /home/gengjianrui/bin/logs "${OUT_DIR}"
cd "${TACE_ROOT}"
export PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-7.0}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo "host=$(hostname)"
echo "out_dir=${OUT_DIR}"
echo "model=${MODEL}"
echo "force_mode=${FORCE_MODE}"
"${TACE_PYTHON}" -V
nvidia-smi -L

# shellcheck disable=SC2206
limit_args=(${LIMIT_CONFIGS_LIST})
for limit in "${limit_args[@]}"; do
  "${TACE_PYTHON}" "${TACE_ROOT}/benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py" \
    --model "${MODEL}" \
    --configs "${CONFIGS}" \
    --output "${OUT_DIR}/${VARIANT}_limit${limit}_benchmark.json" \
    --variant "${VARIANT}" \
    --limit-configs "${limit}" \
    --measure-passes "${MEASURE_PASSES}" \
    --force-mode "${FORCE_MODE}" \
    --device cuda \
    --default-dtype "${DEFAULT_DTYPE}"
done

echo "rTECE scalar benchmark complete: ${OUT_DIR}"
