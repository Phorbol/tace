#!/bin/bash
#SBATCH --job-name=rtece-prof
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=00:30:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-prof-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-prof-%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

TACE_ROOT=${TACE_ROOT:-/home/gengjianrui/bin/tace}
ENV_DIR=${ENV_DIR:-/home/gengjianrui/bin/.venvs/tace-mace-cu126}
TACE_PYTHON=${TACE_PYTHON:-${ENV_DIR}/bin/python}
MODEL=${MODEL:?MODEL checkpoint path is required}
CONFIGS=${CONFIGS:-/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz}
OUT_DIR=${OUT_DIR:-${TACE_ROOT}/runs/oc20neb_tace_mace/rtece-prof-${SLURM_JOB_ID}}
FORCE_MODE=${FORCE_MODE:-analytic_pair}
START_CONFIG=${START_CONFIG:-0}
LIMIT_CONFIGS=${LIMIT_CONFIGS:-4096}
PROFILE_PASSES=${PROFILE_PASSES:-5}
WARMUP_PASSES=${WARMUP_PASSES:-3}
TOP_OPS=${TOP_OPS:-30}
DEFAULT_DTYPE=${DEFAULT_DTYPE:-float32}

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

"${TACE_PYTHON}" "${TACE_ROOT}/benchmarks/oc20neb_tace_mace/profile_rtece_scalar.py" \
  --model "${MODEL}" \
  --configs "${CONFIGS}" \
  --output "${OUT_DIR}/${FORCE_MODE}_profile.json" \
  --trace-output "${OUT_DIR}/${FORCE_MODE}_trace.json" \
  --force-mode "${FORCE_MODE}" \
  --start-config "${START_CONFIG}" \
  --limit-configs "${LIMIT_CONFIGS}" \
  --warmup-passes "${WARMUP_PASSES}" \
  --profile-passes "${PROFILE_PASSES}" \
  --top-ops "${TOP_OPS}" \
  --device cuda \
  --default-dtype "${DEFAULT_DTYPE}"

echo "rTECE scalar profile complete: ${OUT_DIR}"
