#!/bin/bash
#SBATCH --job-name=rtece-proj136
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=03:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-proj136-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-proj136-%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1
TACE_ROOT=/home/gengjianrui/bin/tace
ENV_DIR=/home/gengjianrui/bin/.venvs/tace-mace-cu126
TACE_PYTHON="${ENV_DIR}/bin/python"
PATH="${ENV_DIR}/bin:${PATH}"
PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}"
TRAIN_CONFIGS=runs/oc20neb_tace_mace/rtece-stage132-broad-teacher-distill/augmented_train_base2048_plus_teacher_rattle512.extxyz
VALID_CONFIGS=/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz
OUT_ROOT=runs/oc20neb_tace_mace/rtece-stage136-l2-projection-diagnostic-smoke16/projection_outputs
TRAIN_OUTPUT=runs/oc20neb_tace_mace/rtece-stage136-l2-projection-diagnostic-smoke16/projection_outputs/stage136_train_mixed16_l2_reference_projection.json
VALID_OUTPUT=runs/oc20neb_tace_mace/rtece-stage136-l2-projection-diagnostic-smoke16/projection_outputs/stage136_valid_dft16_l2_reference_projection.json
LIMIT_CONFIGS=16
NUM_RADIAL=12
SPECIES_BASIS_CHANNELS=24
ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS=3
REFERENCE_PATH_IDS=atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius
mkdir -p /home/gengjianrui/bin/logs "${OUT_ROOT}"
cd "${TACE_ROOT}"
export TACE_PYTHON PATH PYTHONPATH
"${TACE_PYTHON}" -V
nvidia-smi -L

# Stage132 mixed DFT/teacher train-distribution E/F projection
"${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py --configs "${TRAIN_CONFIGS}" --output-json "${TRAIN_OUTPUT}" --reference-path-ids "${REFERENCE_PATH_IDS}" --candidate full_l2_reference:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius --candidate stage132_l1_atomic:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot --candidate drop_quadrupole_cross:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm --candidate drop_quadrupole_norm:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_cross_radial_frobenius --num-radial "${NUM_RADIAL}" --species-basis-channels "${SPECIES_BASIS_CHANNELS}" --atomic-cross-radial-sketch-channels "${ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS}" --limit-configs "${LIMIT_CONFIGS}" --default-dtype float64 --neighborlist-backend matscipy --energy-target-key energy --energy-eval-stride 4 --energy-eval-offset 0 --force-target-key forces --force-eval-stride 4 --force-eval-offset 0

# Pure DFT valid-distribution E/F projection
"${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py --configs "${VALID_CONFIGS}" --output-json "${VALID_OUTPUT}" --reference-path-ids "${REFERENCE_PATH_IDS}" --candidate full_l2_reference:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius --candidate stage132_l1_atomic:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot --candidate drop_quadrupole_cross:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm --candidate drop_quadrupole_norm:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_cross_radial_frobenius --num-radial "${NUM_RADIAL}" --species-basis-channels "${SPECIES_BASIS_CHANNELS}" --atomic-cross-radial-sketch-channels "${ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS}" --limit-configs "${LIMIT_CONFIGS}" --default-dtype float64 --neighborlist-backend matscipy --energy-target-key energy --energy-eval-stride 4 --energy-eval-offset 0 --force-target-key forces --force-eval-stride 4 --force-eval-offset 0
