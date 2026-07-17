#!/bin/bash
#SBATCH --job-name=rtece-scalar
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=03:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-scalar-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-scalar-%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

TACE_ROOT=${TACE_ROOT:-/home/gengjianrui/bin/tace}
ENV_DIR=${ENV_DIR:-/home/gengjianrui/bin/.venvs/tace-mace-cu126}
TACE_PYTHON=${TACE_PYTHON:-${ENV_DIR}/bin/python}
RUN_ROOT=${RUN_ROOT:-${TACE_ROOT}/runs/oc20neb_tace_mace/rtece-scalar-${SLURM_JOB_ID}}
TRAIN_FILE=${TRAIN_FILE:-${TACE_ROOT}/runs/oc20neb_tace_mace/tece-distill-20260717/mixed_train_tw0.75.extxyz}
DFT_VALID_FILE=${DFT_VALID_FILE:-/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz}
TRAIN_VALID_FILE=${TRAIN_VALID_FILE:-${DFT_VALID_FILE}}
TEACHER_VALID_FILE=${TEACHER_VALID_FILE:-${TACE_ROOT}/runs/oc20neb_tace_mace/tece-distill-20260717/teacher_valid.extxyz}
VARIANTS=${VARIANTS:-rtece_pair rtece_element_density rtece_vector_moments rtece_density_quadratic rtece_atomic_moments rtece_edge_sketch8}
LIMIT_CONFIGS=${LIMIT_CONFIGS:-128}
VALID_LIMIT_CONFIGS=${VALID_LIMIT_CONFIGS:-64}
BENCH_LIMIT_CONFIGS=${BENCH_LIMIT_CONFIGS:-1024}
MAX_STEPS=${MAX_STEPS:-200}
LR=${LR:-1e-3}
HIDDEN_CHANNELS=${HIDDEN_CHANNELS:-64,64}
ENERGY_WEIGHT=${ENERGY_WEIGHT:-1.0}
FORCE_WEIGHT=${FORCE_WEIGHT:-10.0}
MEASURE_PASSES=${MEASURE_PASSES:-5}
DEFAULT_DTYPE=${DEFAULT_DTYPE:-float32}
FORCE_MODE=${FORCE_MODE:-autograd}
EVAL_INTERVAL=${EVAL_INTERVAL:-100}
CHECKPOINT_NAME=${CHECKPOINT_NAME:-rtece_scalar_best.pt}

mkdir -p /home/gengjianrui/bin/logs "${RUN_ROOT}"
cd "${TACE_ROOT}"
export PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-7.0}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo "host=$(hostname)"
echo "run_root=${RUN_ROOT}"
echo "train_file=${TRAIN_FILE}"
echo "train_valid_file=${TRAIN_VALID_FILE}"
echo "dft_valid_file=${DFT_VALID_FILE}"
echo "teacher_valid_file=${TEACHER_VALID_FILE}"
echo "hidden_channels=${HIDDEN_CHANNELS}"
echo "energy_weight=${ENERGY_WEIGHT}"
echo "force_weight=${FORCE_WEIGHT}"
echo "force_mode=${FORCE_MODE}"
"${TACE_PYTHON}" -V
nvidia-smi -L

nvidia-smi dmon -s pucvmte -o T > "${RUN_ROOT}/nvdmon_job-${SLURM_JOB_ID}_rtece.log" &
NVDMON_PID=$!
trap 'kill ${NVDMON_PID} 2>/dev/null || true' EXIT

# shellcheck disable=SC2206
variant_args=(${VARIANTS})
for variant in "${variant_args[@]}"; do
  out_dir="${RUN_ROOT}/${variant}"
  mkdir -p "${out_dir}"
  "${TACE_PYTHON}" "${TACE_ROOT}/benchmarks/oc20neb_tace_mace/train_rtece_scalar.py" \
    --variant "${variant}" \
    --train-file "${TRAIN_FILE}" \
    --valid-file "${TRAIN_VALID_FILE}" \
    --output-dir "${out_dir}" \
    --limit-configs "${LIMIT_CONFIGS}" \
    --valid-limit-configs "${VALID_LIMIT_CONFIGS}" \
    --max-steps "${MAX_STEPS}" \
    --lr "${LR}" \
    --hidden-channels "${HIDDEN_CHANNELS}" \
    --energy-weight "${ENERGY_WEIGHT}" \
    --force-weight "${FORCE_WEIGHT}" \
    --device cuda \
    --default-dtype "${DEFAULT_DTYPE}" \
    --eval-interval "${EVAL_INTERVAL}"

  "${TACE_PYTHON}" "${TACE_ROOT}/benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py" \
    --model "${out_dir}/${CHECKPOINT_NAME}" \
    --configs "${DFT_VALID_FILE}" \
    --output "${out_dir}/${variant}_dft_benchmark.json" \
    --variant "${variant}" \
    --limit-configs "${BENCH_LIMIT_CONFIGS}" \
    --measure-passes "${MEASURE_PASSES}" \
    --force-mode "${FORCE_MODE}" \
    --device cuda \
    --default-dtype "${DEFAULT_DTYPE}"

  "${TACE_PYTHON}" "${TACE_ROOT}/benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py" \
    --model "${out_dir}/${CHECKPOINT_NAME}" \
    --configs "${TEACHER_VALID_FILE}" \
    --output "${out_dir}/${variant}_teacher_benchmark.json" \
    --variant "${variant}" \
    --limit-configs "${BENCH_LIMIT_CONFIGS}" \
    --measure-passes "${MEASURE_PASSES}" \
    --force-mode "${FORCE_MODE}" \
    --device cuda \
    --default-dtype "${DEFAULT_DTYPE}"
done

echo "rTECE scalar matrix complete: ${RUN_ROOT}"
