#!/bin/bash
#SBATCH --job-name=rtece-st180-init
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=01:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-st180-init-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-st180-init-%j.err

set -eo pipefail
set -x
TACE_ROOT=/home/gengjianrui/bin/tace
ENV_DIR=/home/gengjianrui/bin/.venvs/tace-mace-cu126
TACE_PYTHON="${ENV_DIR}/bin/python"
PATH="${ENV_DIR}/bin:${PATH}"
TRAIN_FILE=runs/oc20neb_tace_mace/tece-distill-20260717/mixed_train_tw0.75.extxyz
VALID_FILE=/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz
TEACHER_VALID_FILE=runs/oc20neb_tace_mace/tece-distill-20260717/mixed_valid_tw0.75_regen.extxyz
LIMIT_CONFIGS=2048
VALID_LIMIT_CONFIGS=256
PROJECTION_LIMIT_CONFIGS=256
MAX_STEPS=20000
RESULTS_ROOT=runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/results
DIAGNOSTICS_ROOT=runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/diagnostics
mkdir -p /home/gengjianrui/bin/logs "${RESULTS_ROOT}" "${DIAGNOSTICS_ROOT}"
cd "${TACE_ROOT}"
"${TACE_PYTHON}" -V
nvidia-smi -L

mkdir -p runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/results/renorm_initialized_same_student
PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" python benchmarks/oc20neb_tace_mace/initialize_rtece_from_projection.py --projection-json runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/diagnostics/stage180_projection_diagnostic.json --candidate scratch_same_student --train-file "${TRAIN_FILE}" --output runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/results/renorm_initialized_same_student/rtece_scalar_init.pt --variant stage180_fixed_student_l1_atomic_cross --scalar-path-ids atomic.radial_density,atomic.species_basis_density,atomic.local_l0_lowrank_density,atomic.vector_norm,atomic.vector_cross_radial_dot --limit-configs "${PROJECTION_LIMIT_CONFIGS}" --init-steps 512 --lr 0.001 --energy-weight 1.0 --force-weight 10.0 --num-radial 10 --hidden-channels 64,64 --species-basis-channels 16 --species-basis-mode learnable_embedding --local-l0-chemistry-rank 4 --moment-l-max 1 --atomic-cross-radial-sketch-channels 3 --atomic-cross-radial-projection learnable --device cuda --default-dtype float32 --neighborlist-backend matscipy
