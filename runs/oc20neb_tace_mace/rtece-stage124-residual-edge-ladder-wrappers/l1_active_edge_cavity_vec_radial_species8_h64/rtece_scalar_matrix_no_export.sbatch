#!/bin/bash
#SBATCH --job-name=rtece-matrix
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=03:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-matrix-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-matrix-%j.err

set -euo pipefail
# SAI policy: pass parameters inside this wrapper, not through Slurm command-line environment export.
export VARIANTS=l1_active_edge_cavity_vec_radial_species8_h64
export RUN_ROOT=/home/gengjianrui/bin/tace/runs/oc20neb_tace_mace/rtece-stage124-residual-edge-ladder/l1_active_edge_cavity_vec_radial_species8_h64
export TRAIN_FILE=runs/oc20neb_tace_mace/tece-distill-20260717/mixed_train_tw0.75.extxyz
export TRAIN_VALID_FILE=/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz
export DFT_VALID_FILE=/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz
export TEACHER_VALID_FILE=runs/oc20neb_tace_mace/tece-distill-20260717/teacher_valid.extxyz
export LIMIT_CONFIGS=2048
export VALID_LIMIT_CONFIGS=256
export BENCH_LIMIT_CONFIGS=1024
export MAX_STEPS=20000
export LR=1e-3
export HIDDEN_CHANNELS=64,64
export NUM_RADIAL=8
export MOMENT_L_MAX=1
export SCALAR_PATH_IDS=atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,edge.cavity.vector_dot,edge.direct.radial
export SPECIES_BASIS_CHANNELS=16
export SPECIES_BASIS_MODE=learnable_embedding
export ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS=3
export ATOMIC_CROSS_RADIAL_PROJECTION=learnable
export FORCE_WEIGHT=10.0
export USE_SHORT_RANGE_REPULSION=1
export SHORT_RANGE_REPULSION_POTENTIAL=zbl
export LEARNABLE_RADIAL_MIXING=1
export RADIAL_SPECIES_ADAPTER_CHANNELS=8
export RADIAL_SPECIES_ADAPTER_SCOPE=edge
export DESCRIPTOR_CONDITIONER=none
export DESCRIPTOR_CONDITIONER_HIDDEN_CHANNELS=0
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
