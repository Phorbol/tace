#!/bin/bash
#SBATCH --job-name=rtece-st181-rmd17-train
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=03:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-st181-rmd17-train-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-st181-rmd17-train-%j.err

set -eo pipefail
set -x
TACE_ROOT=/home/gengjianrui/bin/tace
ENV_DIR=/home/gengjianrui/bin/.venvs/tace-mace-cu126
TACE_PYTHON="${ENV_DIR}/bin/python"
PATH="${ENV_DIR}/bin:${PATH}"
DATASET_ROOT=datasets/3BPA/dataset_3BPA
RMD17_ROOT=datasets/rMD17
RESULTS_ROOT=runs/oc20neb_tace_mace/rtece-stage181-conventional-md/results
DIAGNOSTICS_ROOT=runs/oc20neb_tace_mace/rtece-stage181-conventional-md/diagnostics
mkdir -p /home/gengjianrui/bin/logs "${RESULTS_ROOT}" "${DIAGNOSTICS_ROOT}"
cd "${TACE_ROOT}"
"${TACE_PYTHON}" -V
nvidia-smi -L

mkdir -p runs/oc20neb_tace_mace/rtece-stage181-conventional-md/results/stage181_rmd17_ethanol
PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" "${TACE_PYTHON}" -m tace.scripts.rtece_train_scalar --variant stage181_rmd17_ethanol_l1_local_l0_student --scalar-path-ids atomic.radial_density,atomic.species_basis_density,atomic.local_l0_lowrank_density,atomic.vector_norm,atomic.vector_cross_radial_dot --train-file "${RMD17_ROOT}/converted_extxyz/rmd17_ethanol_train.extxyz" --valid-file "${RMD17_ROOT}/converted_extxyz/rmd17_ethanol_valid.extxyz" --output-dir runs/oc20neb_tace_mace/rtece-stage181-conventional-md/results/stage181_rmd17_ethanol --max-steps 20000 --trainer-backend lightning --batch-size 32 --num-radial 10 --hidden-channels 64,64 --species-basis-channels 16 --species-basis-mode learnable_embedding --local-l0-chemistry-rank 4 --moment-l-max 1 --atomic-cross-radial-sketch-channels 3 --atomic-cross-radial-projection learnable --lr 0.001 --energy-weight 1.0 --force-weight 10.0 --lr-scheduler plateau --lr-warmup-steps 500 --early-stopping-patience 250 --neighborlist-backend matscipy --device cuda --default-dtype float32 --no-progress-bar
