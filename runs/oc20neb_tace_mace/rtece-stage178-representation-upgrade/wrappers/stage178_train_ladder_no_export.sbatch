#!/bin/bash
#SBATCH --job-name=rtece-st178-train
#SBATCH --array=0-2
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=03:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-st178-train-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-st178-train-%j.err

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
MAX_STEPS=20000
RESULTS_ROOT=runs/oc20neb_tace_mace/rtece-stage178-representation-upgrade/results
DIAGNOSTICS_ROOT=runs/oc20neb_tace_mace/rtece-stage178-representation-upgrade/diagnostics
mkdir -p /home/gengjianrui/bin/logs "${RESULTS_ROOT}" "${DIAGNOSTICS_ROOT}"
cd "${TACE_ROOT}"
"${TACE_PYTHON}" -V
nvidia-smi -L

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
echo "Stage178 array task ${TASK_ID}"

if [ "${TASK_ID}" = "0" ]; then
  mkdir -p runs/oc20neb_tace_mace/rtece-stage178-representation-upgrade/results/stage178_l0_local_species
  PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" python -m tace.scripts.rtece_train_scalar --variant stage178_l0_local_species --scalar-path-ids atomic.radial_density,atomic.species_basis_density,atomic.local_l0_lowrank_density --train-file "${TRAIN_FILE}" --valid-file "${VALID_FILE}" --output-dir runs/oc20neb_tace_mace/rtece-stage178-representation-upgrade/results/stage178_l0_local_species --limit-configs "${LIMIT_CONFIGS}" --valid-limit-configs "${VALID_LIMIT_CONFIGS}" --max-steps "${MAX_STEPS}" --trainer-backend lightning --batch-size 32 --num-radial 10 --hidden-channels 64,64 --species-basis-channels 16 --species-basis-mode learnable_embedding --local-l0-chemistry-rank 4 --moment-l-max 0 --atomic-cross-radial-sketch-channels 0 --atomic-cross-radial-projection fixed_shell_mean --lr 0.001 --energy-weight 1.0 --force-weight 10.0 --lr-scheduler plateau --lr-warmup-steps 500 --early-stopping-patience 250 --neighborlist-backend matscipy --device cuda --default-dtype float32 --no-progress-bar
elif [ "${TASK_ID}" = "1" ]; then
  mkdir -p runs/oc20neb_tace_mace/rtece-stage178-representation-upgrade/results/stage178_l1_atomic_cross
  PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" python -m tace.scripts.rtece_train_scalar --variant stage178_l1_atomic_cross --scalar-path-ids atomic.radial_density,atomic.species_basis_density,atomic.local_l0_lowrank_density,atomic.vector_norm,atomic.vector_cross_radial_dot --train-file "${TRAIN_FILE}" --valid-file "${VALID_FILE}" --output-dir runs/oc20neb_tace_mace/rtece-stage178-representation-upgrade/results/stage178_l1_atomic_cross --limit-configs "${LIMIT_CONFIGS}" --valid-limit-configs "${VALID_LIMIT_CONFIGS}" --max-steps "${MAX_STEPS}" --trainer-backend lightning --batch-size 32 --num-radial 10 --hidden-channels 64,64 --species-basis-channels 16 --species-basis-mode learnable_embedding --local-l0-chemistry-rank 4 --moment-l-max 1 --atomic-cross-radial-sketch-channels 3 --atomic-cross-radial-projection learnable --lr 0.001 --energy-weight 1.0 --force-weight 10.0 --lr-scheduler plateau --lr-warmup-steps 500 --early-stopping-patience 250 --neighborlist-backend matscipy --device cuda --default-dtype float32 --no-progress-bar
elif [ "${TASK_ID}" = "2" ]; then
  mkdir -p runs/oc20neb_tace_mace/rtece-stage178-representation-upgrade/results/stage178_t3_minimal_cavity_direct
  PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" python -m tace.scripts.rtece_train_scalar --variant stage178_t3_minimal_cavity_direct --scalar-path-ids atomic.radial_density,atomic.species_basis_density,atomic.local_l0_lowrank_density,atomic.vector_norm,atomic.vector_cross_radial_dot,edge.cavity.vector_dot,edge.direct.radial --train-file "${TRAIN_FILE}" --valid-file "${VALID_FILE}" --output-dir runs/oc20neb_tace_mace/rtece-stage178-representation-upgrade/results/stage178_t3_minimal_cavity_direct --limit-configs "${LIMIT_CONFIGS}" --valid-limit-configs "${VALID_LIMIT_CONFIGS}" --max-steps "${MAX_STEPS}" --trainer-backend lightning --batch-size 32 --num-radial 10 --hidden-channels 64,64 --species-basis-channels 16 --species-basis-mode learnable_embedding --local-l0-chemistry-rank 4 --moment-l-max 1 --atomic-cross-radial-sketch-channels 3 --atomic-cross-radial-projection learnable --lr 0.001 --energy-weight 1.0 --force-weight 10.0 --lr-scheduler plateau --lr-warmup-steps 500 --early-stopping-patience 250 --neighborlist-backend matscipy --device cuda --default-dtype float32 --no-progress-bar
else
  echo "Unknown Stage178 array task ${TASK_ID}"
  exit 2
fi
