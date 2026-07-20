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

set -eo pipefail
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
SCALAR_PATH_IDS=${SCALAR_PATH_IDS:-}
LIMIT_CONFIGS=${LIMIT_CONFIGS:-128}
VALID_LIMIT_CONFIGS=${VALID_LIMIT_CONFIGS:-64}
BENCH_LIMIT_CONFIGS=${BENCH_LIMIT_CONFIGS:-1024}
MAX_STEPS=${MAX_STEPS:-200}
LR=${LR:-1e-3}
HIDDEN_CHANNELS=${HIDDEN_CHANNELS:-64,64}
NUM_RADIAL=${NUM_RADIAL:-8}
MOMENT_L_MAX=${MOMENT_L_MAX:-}
SPECIES_BASIS_CHANNELS=${SPECIES_BASIS_CHANNELS:-0}
SPECIES_BASIS_MODE=${SPECIES_BASIS_MODE:-fixed_z_power}
ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS=${ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS:-2}
ATOMIC_CROSS_RADIAL_PROJECTION=${ATOMIC_CROSS_RADIAL_PROJECTION:-fixed_shell_mean}
ATOMIC_CROSS_RADIAL_PROJECTION_FILE=${ATOMIC_CROSS_RADIAL_PROJECTION_FILE:-}
SEED=${SEED:-0}
ENERGY_WEIGHT=${ENERGY_WEIGHT:-1.0}
FORCE_WEIGHT=${FORCE_WEIGHT:-10.0}
FORCE_FOCUS_ELEMENTS=${FORCE_FOCUS_ELEMENTS:-}
FORCE_FOCUS_WEIGHT=${FORCE_FOCUS_WEIGHT:-1.0}
RELATIVE_ENERGY_WEIGHT=${RELATIVE_ENERGY_WEIGHT:-0.0}
RELATIVE_ENERGY_GROUP_KEY=${RELATIVE_ENERGY_GROUP_KEY:-case_id}
RELATIVE_ENERGY_IMAGE_KEY=${RELATIVE_ENERGY_IMAGE_KEY:-source_frame}
USE_SHORT_RANGE_REPULSION=${USE_SHORT_RANGE_REPULSION:-0}
SHORT_RANGE_REPULSION_POTENTIAL=${SHORT_RANGE_REPULSION_POTENTIAL:-softplus_overlap}
LEARNABLE_RADIAL_MIXING=${LEARNABLE_RADIAL_MIXING:-0}
RADIAL_SPECIES_ADAPTER_CHANNELS=${RADIAL_SPECIES_ADAPTER_CHANNELS:-0}
RADIAL_SPECIES_ADAPTER_SCOPE=${RADIAL_SPECIES_ADAPTER_SCOPE:-all}
DESCRIPTOR_CONDITIONER=${DESCRIPTOR_CONDITIONER:-none}
DESCRIPTOR_CONDITIONER_HIDDEN_CHANNELS=${DESCRIPTOR_CONDITIONER_HIDDEN_CHANNELS:-0}
DESCRIPTOR_BOTTLENECK_DIM=${DESCRIPTOR_BOTTLENECK_DIM:-0}
SHORT_RANGE_REPULSION_STRENGTH=${SHORT_RANGE_REPULSION_STRENGTH:-0.0}
SHORT_RANGE_REPULSION_BETA=${SHORT_RANGE_REPULSION_BETA:-10.0}
SHORT_RANGE_REPULSION_RADIUS_SCALE=${SHORT_RANGE_REPULSION_RADIUS_SCALE:-0.75}
MEASURE_PASSES=${MEASURE_PASSES:-5}
DEFAULT_DTYPE=${DEFAULT_DTYPE:-float32}
FORCE_MODE=${FORCE_MODE:-autograd}
EVAL_INTERVAL=${EVAL_INTERVAL:-100}
MIN_EVAL_STEP=${MIN_EVAL_STEP:-0}
CHECKPOINT_NAME=${CHECKPOINT_NAME:-rtece_scalar_best.pt}
TRAINER_BACKEND=${TRAINER_BACKEND:-lightning}
BATCH_SIZE=${BATCH_SIZE:-1}
VALID_BATCH_SIZE=${VALID_BATCH_SIZE:-1}
LR_SCHEDULER=${LR_SCHEDULER:-plateau}
LR_PATIENCE=${LR_PATIENCE:-25}
LR_FACTOR=${LR_FACTOR:-0.5}
LR_WARMUP_STEPS=${LR_WARMUP_STEPS:-0}
EARLY_STOPPING_PATIENCE=${EARLY_STOPPING_PATIENCE:-}
GRADIENT_CLIP_VAL=${GRADIENT_CLIP_VAL:-0.0}

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
echo "num_radial=${NUM_RADIAL}"
echo "moment_l_max=${MOMENT_L_MAX}"
echo "species_basis_channels=${SPECIES_BASIS_CHANNELS}"
echo "species_basis_mode=${SPECIES_BASIS_MODE}"
echo "atomic_cross_radial_sketch_channels=${ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS}"
echo "atomic_cross_radial_projection=${ATOMIC_CROSS_RADIAL_PROJECTION}"
echo "atomic_cross_radial_projection_file=${ATOMIC_CROSS_RADIAL_PROJECTION_FILE}"
echo "scalar_path_ids=${SCALAR_PATH_IDS}"
echo "seed=${SEED}"
echo "energy_weight=${ENERGY_WEIGHT}"
echo "force_weight=${FORCE_WEIGHT}"
echo "force_focus_elements=${FORCE_FOCUS_ELEMENTS}"
echo "force_focus_weight=${FORCE_FOCUS_WEIGHT}"
echo "relative_energy_weight=${RELATIVE_ENERGY_WEIGHT}"
echo "relative_energy_group_key=${RELATIVE_ENERGY_GROUP_KEY}"
echo "relative_energy_image_key=${RELATIVE_ENERGY_IMAGE_KEY}"
echo "use_short_range_repulsion=${USE_SHORT_RANGE_REPULSION}"
echo "short_range_repulsion_potential=${SHORT_RANGE_REPULSION_POTENTIAL}"
echo "learnable_radial_mixing=${LEARNABLE_RADIAL_MIXING}"
echo "radial_species_adapter_channels=${RADIAL_SPECIES_ADAPTER_CHANNELS}"
echo "radial_species_adapter_scope=${RADIAL_SPECIES_ADAPTER_SCOPE}"
echo "descriptor_conditioner=${DESCRIPTOR_CONDITIONER}"
echo "descriptor_conditioner_hidden_channels=${DESCRIPTOR_CONDITIONER_HIDDEN_CHANNELS}"
echo "descriptor_bottleneck_dim=${DESCRIPTOR_BOTTLENECK_DIM}"
echo "short_range_repulsion_strength=${SHORT_RANGE_REPULSION_STRENGTH}"
echo "short_range_repulsion_beta=${SHORT_RANGE_REPULSION_BETA}"
echo "short_range_repulsion_radius_scale=${SHORT_RANGE_REPULSION_RADIUS_SCALE}"
echo "force_mode=${FORCE_MODE}"
echo "trainer_backend=${TRAINER_BACKEND}"
echo "batch_size=${BATCH_SIZE}"
echo "valid_batch_size=${VALID_BATCH_SIZE}"
echo "lr_scheduler=${LR_SCHEDULER}"
echo "lr_warmup_steps=${LR_WARMUP_STEPS}"
echo "eval_interval=${EVAL_INTERVAL}"
echo "min_eval_step=${MIN_EVAL_STEP}"
"${TACE_PYTHON}" -V
nvidia-smi -L

nvidia-smi dmon -s pucvmte -o T > "${RUN_ROOT}/nvdmon_job-${SLURM_JOB_ID}_rtece.log" &
NVDMON_PID=$!
trap 'kill ${NVDMON_PID} 2>/dev/null || true' EXIT

# shellcheck disable=SC2206
variant_args=(${VARIANTS})
scalar_path_args=()
if [[ -n "${SCALAR_PATH_IDS}" ]]; then
  scalar_path_args=(--scalar-path-ids "${SCALAR_PATH_IDS}")
fi
moment_l_max_args=()
if [[ -n "${MOMENT_L_MAX}" ]]; then
  moment_l_max_args=(--moment-l-max "${MOMENT_L_MAX}")
fi
atomic_cross_radial_args=()
if [[ -n "${ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS}" ]]; then
  atomic_cross_radial_args=(
    --atomic-cross-radial-sketch-channels "${ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS}"
    --atomic-cross-radial-projection "${ATOMIC_CROSS_RADIAL_PROJECTION}"
  )
fi
atomic_cross_radial_projection_file_args=()
if [[ -n "${ATOMIC_CROSS_RADIAL_PROJECTION_FILE}" ]]; then
  atomic_cross_radial_projection_file_args=(
    --atomic-cross-radial-projection-file "${ATOMIC_CROSS_RADIAL_PROJECTION_FILE}"
  )
fi
lightning_args=(
  --trainer-backend "${TRAINER_BACKEND}"
  --batch-size "${BATCH_SIZE}"
  --valid-batch-size "${VALID_BATCH_SIZE}"
  --lr-scheduler "${LR_SCHEDULER}"
  --lr-patience "${LR_PATIENCE}"
  --lr-factor "${LR_FACTOR}"
  --lr-warmup-steps "${LR_WARMUP_STEPS}"
  --gradient-clip-val "${GRADIENT_CLIP_VAL}"
)
if [[ -n "${EARLY_STOPPING_PATIENCE}" ]]; then
  lightning_args+=(--early-stopping-patience "${EARLY_STOPPING_PATIENCE}")
fi

force_focus_args=()
if [[ -n "${FORCE_FOCUS_ELEMENTS}" ]]; then
  force_focus_args=(
    --force-focus-elements "${FORCE_FOCUS_ELEMENTS}"
    --force-focus-weight "${FORCE_FOCUS_WEIGHT}"
  )
fi
learnable_radial_args=()
if [[ "${LEARNABLE_RADIAL_MIXING}" == "1" ]]; then
  learnable_radial_args=(--learnable-radial-mixing)
fi
radial_species_adapter_args=()
if [[ "${RADIAL_SPECIES_ADAPTER_CHANNELS}" != "0" ]]; then
  radial_species_adapter_args=(
    --radial-species-adapter-channels "${RADIAL_SPECIES_ADAPTER_CHANNELS}"
    --radial-species-adapter-scope "${RADIAL_SPECIES_ADAPTER_SCOPE}"
  )
fi
descriptor_conditioner_args=()
if [[ "${DESCRIPTOR_CONDITIONER}" != "none" ]]; then
  descriptor_conditioner_args=(
    --descriptor-conditioner "${DESCRIPTOR_CONDITIONER}"
    --descriptor-conditioner-hidden-channels "${DESCRIPTOR_CONDITIONER_HIDDEN_CHANNELS}"
  )
fi
descriptor_bottleneck_args=()
if [[ "${DESCRIPTOR_BOTTLENECK_DIM}" != "0" ]]; then
  descriptor_bottleneck_args=(--descriptor-bottleneck-dim "${DESCRIPTOR_BOTTLENECK_DIM}")
fi
short_range_args=()
if [[ "${USE_SHORT_RANGE_REPULSION}" == "1" ]]; then
  short_range_args=(
    --use-short-range-repulsion
    --short-range-repulsion-potential "${SHORT_RANGE_REPULSION_POTENTIAL}"
    --short-range-repulsion-strength "${SHORT_RANGE_REPULSION_STRENGTH}"
    --short-range-repulsion-beta "${SHORT_RANGE_REPULSION_BETA}"
    --short-range-repulsion-radius-scale "${SHORT_RANGE_REPULSION_RADIUS_SCALE}"
  )
fi
for variant in "${variant_args[@]}"; do
  out_dir="${RUN_ROOT}/${variant}"
  mkdir -p "${out_dir}"
  "${TACE_PYTHON}" -m tace.scripts.rtece_train_scalar \
    "${lightning_args[@]}" \
    --variant "${variant}" \
    "${scalar_path_args[@]}" \
    "${learnable_radial_args[@]}" \
    "${radial_species_adapter_args[@]}" \
    "${descriptor_conditioner_args[@]}" \
    "${descriptor_bottleneck_args[@]}" \
    "${short_range_args[@]}" \
    --train-file "${TRAIN_FILE}" \
    --valid-file "${TRAIN_VALID_FILE}" \
    --output-dir "${out_dir}" \
    --limit-configs "${LIMIT_CONFIGS}" \
    --valid-limit-configs "${VALID_LIMIT_CONFIGS}" \
    --max-steps "${MAX_STEPS}" \
    --lr "${LR}" \
    --hidden-channels "${HIDDEN_CHANNELS}" \
    --num-radial "${NUM_RADIAL}" \
    "${moment_l_max_args[@]}" \
    --species-basis-channels "${SPECIES_BASIS_CHANNELS}" \
    --species-basis-mode "${SPECIES_BASIS_MODE}" \
    "${atomic_cross_radial_args[@]}" \
    "${atomic_cross_radial_projection_file_args[@]}" \
    --seed "${SEED}" \
    --energy-weight "${ENERGY_WEIGHT}" \
    --force-weight "${FORCE_WEIGHT}" \
    "${force_focus_args[@]}" \
    --relative-energy-weight "${RELATIVE_ENERGY_WEIGHT}" \
    --relative-energy-group-key "${RELATIVE_ENERGY_GROUP_KEY}" \
    --relative-energy-image-key "${RELATIVE_ENERGY_IMAGE_KEY}" \
    --device cuda \
    --default-dtype "${DEFAULT_DTYPE}" \
    --eval-interval "${EVAL_INTERVAL}" \
    --min-eval-step "${MIN_EVAL_STEP}"

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
