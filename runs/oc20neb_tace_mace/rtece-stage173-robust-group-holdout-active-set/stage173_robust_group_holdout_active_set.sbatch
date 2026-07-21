#!/bin/bash
#SBATCH --job-name=rtece-res173
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=03:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-res173-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-res173-%j.err

set -eo pipefail
set -x
TACE_ROOT=/home/gengjianrui/bin/tace
ENV_DIR=/home/gengjianrui/bin/.venvs/tace-mace-cu126
TACE_PYTHON="${ENV_DIR}/bin/python"
PATH="${ENV_DIR}/bin:${PATH}"
PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}"
CONFIGS=/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz
STAGE165_JSON=runs/oc20neb_tace_mace/rtece-stage165-energy-baseline-triage/stage165_energy_baseline_triage.json
PROJECTION_JSON=runs/oc20neb_tace_mace/rtece-stage173-robust-group-holdout-active-set/projection_outputs/stage173_limit512_robust_group_holdout_active_set_projection.json
LIMIT_CONFIGS=512
NUM_RADIAL=12
SPECIES_BASIS_CHANNELS=24
ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS=3
REFERENCE_PATH_IDS=atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot,edge.cavity.quadrupole_frobenius,edge.cavity.target_vector_projection,edge.cavity.source_vector_projection,edge.cavity.target_quadrupole_projection,edge.cavity.source_quadrupole_projection,edge.direct.radial
mkdir -p /home/gengjianrui/bin/logs "$(dirname "${PROJECTION_JSON}")"
cd "${TACE_ROOT}"
export TACE_PYTHON PATH PYTHONPATH
"${TACE_PYTHON}" -V
nvidia-smi -L

# Stage173: robust group-heldout path projection for Stage170 low-frequency energy residual
"${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py --configs "${CONFIGS}" --output-json "${PROJECTION_JSON}" --reference-path-ids "${REFERENCE_PATH_IDS}" --residual-target-mode stage165-case-offset --stage165-json "${STAGE165_JSON}" --stage165-variant stage157_direct_b32_rel0p25_mixed2048 --residual-group-key case_id --candidate t1_l0_species_radial:atomic.radial_density,atomic.species_basis_density --candidate t1_l1_atomic_cross:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot --candidate t1_l2_atomic_cross:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius --candidate t2_cavity_vector:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot --candidate t2_cavity_quadrupole:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot,edge.cavity.quadrupole_frobenius --candidate t2_edge_frame_projection:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot,edge.cavity.quadrupole_frobenius,edge.cavity.target_vector_projection,edge.cavity.source_vector_projection,edge.cavity.target_quadrupole_projection,edge.cavity.source_quadrupole_projection --candidate t2_edge_frame_plus_direct:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot,edge.cavity.quadrupole_frobenius,edge.cavity.target_vector_projection,edge.cavity.source_vector_projection,edge.cavity.target_quadrupole_projection,edge.cavity.source_quadrupole_projection,edge.direct.radial --num-radial "${NUM_RADIAL}" --species-basis-channels "${SPECIES_BASIS_CHANNELS}" --atomic-cross-radial-sketch-channels "${ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS}" --limit-configs "${LIMIT_CONFIGS}" --default-dtype float64 --neighborlist-backend matscipy --ridge 1e-08 --energy-baseline none --energy-split-mode group-loocv --energy-group-key case_id --energy-ridge-grid 1e-08,1e-06,0.0001,0.01,1,100 --active-set-baseline-candidate t1_l0_species_radial --active-set-energy-weight 1.0 --active-set-force-weight 0.0 --active-set-projection-weight 0.25 --active-set-gain-mode absolute --energy-fit-intercept --energy-standardize-features --active-set-require-energy-gain --active-set-require-beat-intercept
