#!/bin/bash
#SBATCH --job-name=tece-distill
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=03:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/tece-distill-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/tece-distill-%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

TACE_ROOT=${TACE_ROOT:-/home/gengjianrui/bin/tace}
ENV_DIR=${ENV_DIR:-/home/gengjianrui/bin/.venvs/tace-mace-cu126}
TACE_PYTHON=${TACE_PYTHON:-${ENV_DIR}/bin/python}
MODE=${MODE:-labels}
VARIANT=${VARIANT:-scalar_fast}
DISTILL_ROOT=${DISTILL_ROOT:-${TACE_ROOT}/runs/oc20neb_tace_mace/tece-distill-${SLURM_JOB_ID}}
TEACHER_MODEL=${TEACHER_MODEL:-${TACE_ROOT}/runs/oc20neb_tace_mace/675969_resume/checkpoints_epoch/TACE-0-100000-0.1329.ckpt}
BASE_CONFIG=${BASE_CONFIG:-${TACE_ROOT}/benchmarks/oc20neb_tace_mace/tace_oc20neb.yaml}
TRAIN_FILE=${TRAIN_FILE:-/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/train.extxyz}
VALID_FILE=${VALID_FILE:-/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz}
BATCH_SIZE=${BATCH_SIZE:-16}
LIMIT_CONFIGS=${LIMIT_CONFIGS:-1024}
MEASURE_PASSES=${MEASURE_PASSES:-3}
TACE_MODEL=${TACE_MODEL:-}
TACE_ACCEL_BACKEND=${TACE_ACCEL_BACKEND:-oeq}
TEACHER_WEIGHT=${TEACHER_WEIGHT:-0.75}
TRAIN_TARGET_FILE=${TRAIN_TARGET_FILE:-${DISTILL_ROOT}/teacher_train.extxyz}
VALID_TARGET_FILE=${VALID_TARGET_FILE:-${DISTILL_ROOT}/teacher_valid.extxyz}
CONFIG_VARIANTS=${CONFIG_VARIANTS:-}

mkdir -p /home/gengjianrui/bin/logs "${DISTILL_ROOT}"
cd "${DISTILL_ROOT}"
export PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-7.0}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export HYDRA_FULL_ERROR=1

if [[ "${TACE_ACCEL_BACKEND}" == "oeq" ]]; then
  source /etc/profile.d/modules.sh 2>/dev/null || true
  set +u
  module load cuda/12.6.3 || true
  set -u
fi

case "${TACE_ACCEL_BACKEND}" in
  oeq)
    export TACE_USE_OEQ=1
    export TACE_USE_CUE=0
    export TACE_USE_EQT=0
    export TACE_USE_COMPILE=0
    ;;
  cue)
    export TACE_USE_OEQ=0
    export TACE_USE_CUE=1
    export TACE_USE_EQT=0
    export TACE_USE_COMPILE=0
    ;;
  eqt)
    export TACE_USE_OEQ=0
    export TACE_USE_CUE=0
    export TACE_USE_EQT=1
    export TACE_USE_COMPILE=0
    ;;
  compile)
    export TACE_USE_OEQ=0
    export TACE_USE_CUE=0
    export TACE_USE_EQT=0
    export TACE_USE_COMPILE=1
    ;;
  none)
    export TACE_USE_OEQ=0
    export TACE_USE_CUE=0
    export TACE_USE_EQT=0
    export TACE_USE_COMPILE=0
    ;;
  *)
    echo "unknown TACE_ACCEL_BACKEND=${TACE_ACCEL_BACKEND}" >&2
    exit 2
    ;;
esac

if [[ "${MODE}" == "labels" ]]; then
  test -f "${TEACHER_MODEL}"
  "${TACE_PYTHON}" "${TACE_ROOT}/benchmarks/oc20neb_tace_mace/distill_tace_labels.py" \
    --model "${TEACHER_MODEL}" \
    --input "${TRAIN_FILE}" \
    --output "${DISTILL_ROOT}/teacher_train.extxyz" \
    --summary "${DISTILL_ROOT}/teacher_train_summary.json" \
    --batch-size "${BATCH_SIZE}" \
    --device cuda
  "${TACE_PYTHON}" "${TACE_ROOT}/benchmarks/oc20neb_tace_mace/distill_tace_labels.py" \
    --model "${TEACHER_MODEL}" \
    --input "${VALID_FILE}" \
    --output "${DISTILL_ROOT}/teacher_valid.extxyz" \
    --summary "${DISTILL_ROOT}/teacher_valid_summary.json" \
    --batch-size "${BATCH_SIZE}" \
    --device cuda
elif [[ "${MODE}" == "mix" ]]; then
  "${TACE_PYTHON}" "${TACE_ROOT}/benchmarks/oc20neb_tace_mace/mix_tece_distill_labels.py" \
    --input "${DISTILL_ROOT}/teacher_train.extxyz" \
    --output "${DISTILL_ROOT}/mixed_train_tw${TEACHER_WEIGHT}.extxyz" \
    --summary "${DISTILL_ROOT}/mixed_train_tw${TEACHER_WEIGHT}_summary.json" \
    --teacher-weight "${TEACHER_WEIGHT}"
  "${TACE_PYTHON}" "${TACE_ROOT}/benchmarks/oc20neb_tace_mace/mix_tece_distill_labels.py" \
    --input "${DISTILL_ROOT}/teacher_valid.extxyz" \
    --output "${DISTILL_ROOT}/mixed_valid_tw${TEACHER_WEIGHT}.extxyz" \
    --summary "${DISTILL_ROOT}/mixed_valid_tw${TEACHER_WEIGHT}_summary.json" \
    --teacher-weight "${TEACHER_WEIGHT}"
elif [[ "${MODE}" == "configs" ]]; then
  config_args=(
    --base-config "${BASE_CONFIG}"
    --train-file "${TRAIN_TARGET_FILE}"
    --valid-file "${VALID_TARGET_FILE}"
    --output-dir "${DISTILL_ROOT}/configs"
  )
  if [[ -n "${CONFIG_VARIANTS}" ]]; then
    # shellcheck disable=SC2206
    variant_args=(${CONFIG_VARIANTS})
    config_args+=(--variants "${variant_args[@]}")
  fi
  "${TACE_PYTHON}" "${TACE_ROOT}/benchmarks/oc20neb_tace_mace/make_tece_student_configs.py" "${config_args[@]}"
elif [[ "${MODE}" == "train" ]]; then
  CONFIG=${CONFIG:-${DISTILL_ROOT}/configs/tace_oc20neb_${VARIANT}.yaml}
  RUN_ROOT=${RUN_ROOT:-${DISTILL_ROOT}/train-${VARIANT}-${SLURM_JOB_ID}}
  mkdir -p "${RUN_ROOT}"
  cd "${RUN_ROOT}"
  cp "${CONFIG}" "${RUN_ROOT}/tace.yaml"
  export TACE_USE_CUE=${TACE_USE_CUE:-0}
  export TACE_USE_OEQ=${TACE_USE_OEQ:-0}
  export TACE_USE_EQT=${TACE_USE_EQT:-0}
  export TACE_USE_COMPILE=${TACE_USE_COMPILE:-0}
  "${ENV_DIR}/bin/tace-train" -cn tace ${TACE_TRAIN_EXTRA_ARGS:-}
elif [[ "${MODE}" == "bench" ]]; then
  test -n "${TACE_MODEL}"
  OUT_DIR=${OUT_DIR:-${DISTILL_ROOT}/bench-${VARIANT}-${SLURM_JOB_ID}}
  mkdir -p "${OUT_DIR}"
  "${TACE_PYTHON}" "${TACE_ROOT}/benchmarks/oc20neb_tace_mace/benchmark_models.py" \
    --backend tace \
    --model "${TACE_MODEL}" \
    --configs "${VALID_FILE}" \
    --output "${OUT_DIR}/${VARIANT}_dft_benchmark.json" \
    --batch-size "${BATCH_SIZE}" \
    --device cuda \
    --default-dtype float32 \
    --limit-configs "${LIMIT_CONFIGS}" \
    --measure-passes "${MEASURE_PASSES}" \
    --tace-accel-backend "${TACE_ACCEL_BACKEND}"
  "${TACE_PYTHON}" "${TACE_ROOT}/benchmarks/oc20neb_tace_mace/benchmark_models.py" \
    --backend tace \
    --model "${TACE_MODEL}" \
    --configs "${DISTILL_ROOT}/teacher_valid.extxyz" \
    --output "${OUT_DIR}/${VARIANT}_teacher_benchmark.json" \
    --batch-size "${BATCH_SIZE}" \
    --device cuda \
    --default-dtype float32 \
    --limit-configs "${LIMIT_CONFIGS}" \
    --measure-passes "${MEASURE_PASSES}" \
    --tace-accel-backend "${TACE_ACCEL_BACKEND}"
else
  echo "unknown MODE=${MODE}; expected labels, mix, configs, train, or bench" >&2
  exit 2
fi
