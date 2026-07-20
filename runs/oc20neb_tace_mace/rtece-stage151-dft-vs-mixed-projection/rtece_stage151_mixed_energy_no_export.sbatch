#!/bin/bash
#SBATCH --job-name=rtece-s151-mixed-e
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --output=runs/oc20neb_tace_mace/rtece-stage151-dft-vs-mixed-projection/logs/mixed-energy-%j.out
#SBATCH --error=runs/oc20neb_tace_mace/rtece-stage151-dft-vs-mixed-projection/logs/mixed-energy-%j.err

set -eo pipefail

cd /home/gengjianrui/bin/tace
mkdir -p runs/oc20neb_tace_mace/rtece-stage151-dft-vs-mixed-projection/projection_outputs
mkdir -p runs/oc20neb_tace_mace/rtece-stage151-dft-vs-mixed-projection/logs

/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py \
  --configs runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/weighted_train_base2048_plus_teacher_relax640_forceonly_eanchor.extxyz \
  --output-json runs/oc20neb_tace_mace/rtece-stage151-dft-vs-mixed-projection/projection_outputs/stage151_mixed_train_limit1024_energy_projection.json \
  --reference-path-ids atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot,edge.cavity.quadrupole_frobenius,edge.direct.radial \
  --candidate t2_l0_species_radial:atomic.radial_density,atomic.species_basis_density \
  --candidate t2_l1_atomic_cross:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot \
  --candidate t2_l2_atomic_cross:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius \
  --candidate t3_l2_cavity_vector:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot \
  --candidate t3_full_reference:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot,edge.cavity.quadrupole_frobenius,edge.direct.radial \
  --auto-candidate-strategy single_delete \
  --auto-candidate-strategy prefix \
  --num-radial 12 \
  --species-basis-channels 24 \
  --atomic-cross-radial-sketch-channels 3 \
  --cutoff 5.0 \
  --limit-configs 1024 \
  --default-dtype float64 \
  --neighborlist-backend matscipy \
  --energy-target-key energy \
  --energy-baseline element_counts \
  --energy-eval-stride 4 \
  --energy-eval-offset 0
