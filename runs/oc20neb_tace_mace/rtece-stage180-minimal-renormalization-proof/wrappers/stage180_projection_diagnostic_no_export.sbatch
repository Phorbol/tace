#!/bin/bash
#SBATCH --job-name=rtece-st180-proj
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=01:00:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-st180-proj-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-st180-proj-%j.err

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

mkdir -p runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/diagnostics
PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" python benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py --configs "${TRAIN_FILE}" --output-json runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/diagnostics/stage180_projection_diagnostic.json --reference-path-ids atomic.radial_density,atomic.species_basis_density,atomic.local_l0_lowrank_density,atomic.vector_norm,atomic.vector_cross_radial_dot,edge.cavity.vector_dot,edge.direct.radial --candidate scratch_same_student:atomic.radial_density,atomic.species_basis_density,atomic.local_l0_lowrank_density,atomic.vector_norm,atomic.vector_cross_radial_dot --num-radial 10 --species-basis-channels 16 --reference-atomic-cross-radial-sketch-channels 3 --candidate-atomic-cross-radial-sketch-channels 3 --limit-configs "${PROJECTION_LIMIT_CONFIGS}" --default-dtype float64 --neighborlist-backend matscipy --energy-target-key energy --energy-baseline element_counts --energy-fit-intercept --energy-standardize-features --energy-ridge-grid 1e-12,1e-10,1e-8,1e-6 --energy-split-mode group-loocv --energy-group-key case_id --force-target-key forces --force-component-sample-count 6000 --force-eval-stride 4
