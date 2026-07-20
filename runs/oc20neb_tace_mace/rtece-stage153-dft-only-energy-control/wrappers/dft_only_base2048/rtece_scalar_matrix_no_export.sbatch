#!/bin/bash
#SBATCH --job-name=rtece153-dft
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=03:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece153-dft-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece153-dft-%j.err

set -eo pipefail

export TACE_ROOT=/home/gengjianrui/bin/tace
export ENV_DIR=/home/gengjianrui/bin/.venvs/tace-mace-cu126
export TACE_PYTHON=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python
export STAGE_ROOT=runs/oc20neb_tace_mace/rtece-stage153-dft-only-energy-control
export SOURCE_TRAIN=runs/oc20neb_tace_mace/tece-distill-20260717/mixed_train_tw0.75.extxyz
export DFT_ONLY_TRAIN=runs/oc20neb_tace_mace/rtece-stage153-dft-only-energy-control/data/dft_only_base2048.extxyz
export DFT_ONLY_SUMMARY=runs/oc20neb_tace_mace/rtece-stage153-dft-only-energy-control/data/dft_only_base2048_summary.json

cd "${TACE_ROOT}"
mkdir -p "${STAGE_ROOT}/data"
"${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/mix_tece_distill_labels.py \
  --input "${SOURCE_TRAIN}" \
  --output "${DFT_ONLY_TRAIN}" \
  --summary "${DFT_ONLY_SUMMARY}" \
  --limit-configs 2048 \
  --teacher-weight 0.0 \
  --target-energy-key energy \
  --target-forces-key forces \
  --dft-energy-key dft_energy \
  --dft-forces-key dft_forces \
  --teacher-prefix stage153_mixed_

export VARIANTS=stage153_l2_dft_only_base2048
export RUN_ROOT=runs/oc20neb_tace_mace/rtece-stage153-dft-only-energy-control/control_runs/dft_only_base2048
export TRAIN_FILE="${DFT_ONLY_TRAIN}"
export TRAIN_VALID_FILE=/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz
export DFT_VALID_FILE=/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz
export TEACHER_VALID_FILE=runs/oc20neb_tace_mace/tece-distill-20260717/teacher_valid.extxyz
export LIMIT_CONFIGS=2048
export VALID_LIMIT_CONFIGS=256
export BENCH_LIMIT_CONFIGS=1024
export MAX_STEPS=20000
export LR=1e-3
export HIDDEN_CHANNELS=64,64
export NUM_RADIAL=12
export MOMENT_L_MAX=2
export SCALAR_PATH_IDS=atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius
export SPECIES_BASIS_CHANNELS=24
export SPECIES_BASIS_MODE=learnable_embedding
export ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS=3
export ATOMIC_CROSS_RADIAL_PROJECTION=learnable
export ENERGY_WEIGHT=1.0
export FORCE_WEIGHT=10.0
export USE_SHORT_RANGE_REPULSION=1
export SHORT_RANGE_REPULSION_POTENTIAL=zbl
export LEARNABLE_RADIAL_MIXING=1
export RADIAL_SPECIES_ADAPTER_CHANNELS=8
export RADIAL_SPECIES_ADAPTER_SCOPE=all
export DESCRIPTOR_CONDITIONER=residual_mlp
export DESCRIPTOR_CONDITIONER_HIDDEN_CHANNELS=32
export DESCRIPTOR_BOTTLENECK_DIM=0
export FORCE_MODE=autograd
export MEASURE_PASSES=5
export DEFAULT_DTYPE=float32
export EVAL_INTERVAL=500
export MIN_EVAL_STEP=2000
export CHECKPOINT_NAME=rtece_scalar_best.pt
export TRAINER_BACKEND=lightning
export BATCH_SIZE=8
export VALID_BATCH_SIZE=16
export LR_SCHEDULER=plateau
export LR_PATIENCE=100
export LR_FACTOR=0.5
export LR_WARMUP_STEPS=500
export EARLY_STOPPING_PATIENCE=400
export GRADIENT_CLIP_VAL=0.0
exec /bin/bash /home/gengjianrui/bin/tace/benchmarks/oc20neb_tace_mace/rtece_scalar_matrix.sbatch
