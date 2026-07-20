#!/bin/bash
#SBATCH --job-name=rtece-proj143
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=03:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-proj143-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-proj143-%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1
TACE_ROOT=/home/gengjianrui/bin/tace
ENV_DIR=/home/gengjianrui/bin/.venvs/tace-mace-cu126
TACE_PYTHON="${ENV_DIR}/bin/python"
PATH="${ENV_DIR}/bin:${PATH}"
PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}"
TRAIN_CONFIGS=runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/weighted_train_base2048_plus_teacher_relax640_forceonly_eanchor.extxyz
VALID_CONFIGS=/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz
OUT_ROOT=runs/oc20neb_tace_mace/rtece-stage143-semantic-active-set/projection_outputs
TRAIN_OUTPUT=runs/oc20neb_tace_mace/rtece-stage143-semantic-active-set/projection_outputs/stage143_train_limit128_semantic_active_set_projection.json
VALID_OUTPUT=runs/oc20neb_tace_mace/rtece-stage143-semantic-active-set/projection_outputs/stage143_valid_limit128_semantic_active_set_projection.json
LIMIT_CONFIGS=128
NUM_RADIAL=12
SPECIES_BASIS_CHANNELS=24
ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS=3
REFERENCE_PATH_IDS=atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot,edge.cavity.quadrupole_frobenius,edge.direct.radial
mkdir -p /home/gengjianrui/bin/logs "${OUT_ROOT}"
cd "${TACE_ROOT}"
export TACE_PYTHON PATH PYTHONPATH
"${TACE_PYTHON}" -V
nvidia-smi -L

# Stage142 force-only teacher-relax deployment-measure projection; force-Jacobian projection is deferred by default
"${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py --configs "${TRAIN_CONFIGS}" --output-json "${TRAIN_OUTPUT}" --reference-path-ids "${REFERENCE_PATH_IDS}" --candidate t2_l0_species_radial:atomic.radial_density,atomic.species_basis_density --candidate t2_l1_atomic_cross:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot --candidate t2_l2_atomic_cross:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius --candidate t3_l2_cavity_vector:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot --candidate t3_l2_cavity_vector_quadrupole_direct:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot,edge.cavity.quadrupole_frobenius,edge.direct.radial --num-radial "${NUM_RADIAL}" --species-basis-channels "${SPECIES_BASIS_CHANNELS}" --atomic-cross-radial-sketch-channels "${ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS}" --limit-configs "${LIMIT_CONFIGS}" --default-dtype float64 --neighborlist-backend matscipy --energy-target-key energy --energy-eval-stride 4 --energy-eval-offset 0

# Pure DFT validation projection for held-out benchmark consistency
"${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py --configs "${VALID_CONFIGS}" --output-json "${VALID_OUTPUT}" --reference-path-ids "${REFERENCE_PATH_IDS}" --candidate t2_l0_species_radial:atomic.radial_density,atomic.species_basis_density --candidate t2_l1_atomic_cross:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot --candidate t2_l2_atomic_cross:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius --candidate t3_l2_cavity_vector:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot --candidate t3_l2_cavity_vector_quadrupole_direct:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot,edge.cavity.quadrupole_frobenius,edge.direct.radial --num-radial "${NUM_RADIAL}" --species-basis-channels "${SPECIES_BASIS_CHANNELS}" --atomic-cross-radial-sketch-channels "${ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS}" --limit-configs "${LIMIT_CONFIGS}" --default-dtype float64 --neighborlist-backend matscipy --energy-target-key energy --energy-eval-stride 4 --energy-eval-offset 0
