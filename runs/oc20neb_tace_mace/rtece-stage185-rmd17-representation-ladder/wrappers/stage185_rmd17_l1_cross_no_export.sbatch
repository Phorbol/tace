#!/bin/bash
#SBATCH --job-name=rtece-st185-l1_sparse_cross_radial_power_spectrum
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=04:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-st185-l1_sparse_cross_radial_power_spectrum-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-st185-l1_sparse_cross_radial_power_spectrum-%j.err

set -eo pipefail
set -x
TACE_ROOT=/home/gengjianrui/bin/tace
TACE_PYTHON=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python
PATH=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin:${PATH}
mkdir -p /home/gengjianrui/bin/logs
cd ${TACE_ROOT}
${TACE_PYTHON} -V

mkdir -p runs/oc20neb_tace_mace/rtece-stage185-rmd17-representation-ladder/results/stage185_rmd17_l1_cross
PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} -m tace.scripts.rtece_train_scalar --variant stage185_rmd17_l1_cross --scalar-path-ids atomic.radial_density,atomic.species_basis_density,atomic.local_l0_lowrank_density,atomic.vector_norm,atomic.vector_cross_radial_dot --train-file datasets/rMD17/converted_extxyz/rmd17_ethanol_train.extxyz --valid-file datasets/rMD17/converted_extxyz/rmd17_ethanol_valid.extxyz --output-dir runs/oc20neb_tace_mace/rtece-stage185-rmd17-representation-ladder/results/stage185_rmd17_l1_cross --max-steps 20000 --limit-configs 1000 --valid-limit-configs 1000 --trainer-backend lightning --batch-size 32 --num-radial 10 --hidden-channels 64,64 --species-basis-channels 16 --species-basis-mode learnable_embedding --local-l0-chemistry-rank 4 --moment-l-max 1 --atomic-cross-radial-sketch-channels 3 --atomic-cross-radial-projection learnable --energy-key energy --forces-key forces --energy-weight 1.0 --force-weight 10.0 --lr 0.001 --lr-scheduler plateau --lr-warmup-steps 500 --early-stopping-patience 250 --neighborlist-backend matscipy --device cuda --default-dtype float32 --no-progress-bar
