#!/bin/bash
#SBATCH --job-name=rtece-st176
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=01:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-st176-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-st176-%j.err

set -eo pipefail
set -x
TACE_ROOT=/home/gengjianrui/bin/tace
ENV_DIR=/home/gengjianrui/bin/.venvs/tace-mace-cu126
TACE_PYTHON="${ENV_DIR}/bin/python"
PATH="${ENV_DIR}/bin:${PATH}"
TRAIN_FILE=runs/oc20neb_tace_mace/tece-distill-20260717/mixed_train_tw0.75.extxyz
VALID_FILE=/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz
TRAIN_OUTPUT_DIR=runs/oc20neb_tace_mace/rtece-stage176-production-local-l0-front/train_smoke
LIMIT_CONFIGS=512
VALID_LIMIT_CONFIGS=64
MAX_STEPS=2000
mkdir -p /home/gengjianrui/bin/logs "${TRAIN_OUTPUT_DIR}"
cd "${TACE_ROOT}"
"${TACE_PYTHON}" -V
nvidia-smi -L

# Stage176: production train smoke for Stage175-supported local L0 rank3 front
PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" python -m tace.scripts.rtece_train_scalar --variant stage176_local_l0_rank3 --scalar-path-ids atomic.radial_density,atomic.local_l0_lowrank_density --local-l0-chemistry-rank 3 --train-file "${TRAIN_FILE}" --valid-file "${VALID_FILE}" --output-dir "${TRAIN_OUTPUT_DIR}" --limit-configs "${LIMIT_CONFIGS}" --valid-limit-configs "${VALID_LIMIT_CONFIGS}" --max-steps "${MAX_STEPS}" --trainer-backend lightning --batch-size 1 --num-radial 8 --hidden-channels 64,64 --lr 0.001 --energy-weight 1.0 --force-weight 10.0 --lr-scheduler plateau --lr-warmup-steps 100 --early-stopping-patience 50 --neighborlist-backend matscipy --device cuda --default-dtype float32 --no-progress-bar
