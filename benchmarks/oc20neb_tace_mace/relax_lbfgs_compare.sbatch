#!/bin/bash
#SBATCH --job-name=relax-lbfgs
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=02:00:00
#SBATCH --output=/home/gengjianrui/bin/logs/relax-lbfgs-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/relax-lbfgs-%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

TACE_ROOT=${TACE_ROOT:-/home/gengjianrui/bin/tace}
ENV_DIR=${ENV_DIR:-/home/gengjianrui/bin/.venvs/tace-mace-cu126}
TACE_PYTHON=${TACE_PYTHON:-${ENV_DIR}/bin/python}

# ── models ───────────────────────────────────────────────────────
TEACHER=${TEACHER:-${TACE_ROOT}/runs/oc20neb_tace_mace/675969_resume/checkpoints_epoch/TACE-0-100000-0.1329.ckpt}

# rTECE SOTA accuracy: radial=8, hidden=[24,24], element_density, DFT F MAE=30.19
RTECE=${RTECE:-${TACE_ROOT}/runs/oc20neb_tace_mace/rtece-scalar-678773/rtece_element_density/rtece_scalar_best.pt}

# rTECE speed king: radial=5, hidden=[16,16], element_density, DFT F MAE=39.02
RTECE_FAST=${RTECE_FAST:-${TACE_ROOT}/runs/oc20neb_tace_mace/rtece-stage24-radial5-hidden16/rtece_element_density/rtece_scalar_best.pt}

# ── data ─────────────────────────────────────────────────────────
CONFIGS=${CONFIGS:-${TACE_ROOT}/runs/oc20neb_tace_mace/tece-distill-20260717/teacher_valid.extxyz}
CONFIG_INDEX=${CONFIG_INDEX:-0}

# ── relaxation settings ──────────────────────────────────────────
FMAX=${FMAX:-0.05}
MAX_STEPS=${MAX_STEPS:-300}

# ── output ───────────────────────────────────────────────────────
OUT_DIR=${OUT_DIR:-${TACE_ROOT}/runs/oc20neb_tace_mace/relax-lbfgs-${SLURM_JOB_ID}}

mkdir -p /home/gengjianrui/bin/logs "${OUT_DIR}"
cd "${TACE_ROOT}"
export PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-7.0}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo "host=$(hostname)"
echo "out_dir=${OUT_DIR}"
echo "teacher=${TEACHER}"
echo "rtece=${RTECE}"
echo "configs=${CONFIGS}  index=${CONFIG_INDEX}"
echo "fmax=${FMAX}  max_steps=${MAX_STEPS}"
"${TACE_PYTHON}" -V
nvidia-smi -L

# OpenEquivariance needs CUDA_HOME for JIT compilation
source /etc/profile.d/modules.sh 2>/dev/null || true
set +u
module load cuda/12.6.3 2>/dev/null || true
NVCC_PATH=$(which nvcc 2>/dev/null || echo "/usr/local/cuda-12.6/bin/nvcc")
if [ -x "${NVCC_PATH}" ]; then
  export CUDA_HOME=$(cd "$(dirname "${NVCC_PATH}")/.." && pwd)
else
  export CUDA_HOME=/usr/local/cuda-12.6
fi
export TACE_USE_OEQ=1
set -u
echo "CUDA_HOME=${CUDA_HOME}"

"${TACE_PYTHON}" "${TACE_ROOT}/benchmarks/oc20neb_tace_mace/relax_lbfgs_compare.py" \
  --teacher "${TEACHER}" \
  --rtece "${RTECE}" \
  --rtece-fast "${RTECE_FAST}" \
  --configs "${CONFIGS}" \
  --config-index "${CONFIG_INDEX}" \
  --out-dir "${OUT_DIR}" \
  --fmax "${FMAX}" \
  --max-steps "${MAX_STEPS}" \
  --device cuda \
  --default-dtype float32

echo "relaxation complete → ${OUT_DIR}"
