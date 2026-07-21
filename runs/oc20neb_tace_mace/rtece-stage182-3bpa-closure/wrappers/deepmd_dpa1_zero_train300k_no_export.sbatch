#!/bin/bash
#SBATCH --job-name=rtece-st182-deepmd-train
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=03:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-st182-deepmd-train-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-st182-deepmd-train-%j.err

set -eo pipefail
set -x
TACE_ROOT=/home/gengjianrui/bin/tace
TACE_PYTHON=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python
PATH=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin:${PATH}
mkdir -p /home/gengjianrui/bin/logs
cd ${TACE_ROOT}
${TACE_PYTHON} -V

module load deepmd-kit/3.1.2
PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/convert_stage145_deepmd.py --manifest runs/oc20neb_tace_mace/rtece-stage182-3bpa-closure/stage182_manifest.json --row deepmd_dpa1_zero_train300k
cd runs/oc20neb_tace_mace/rtece-stage182-3bpa-closure/results/deepmd_dpa1_zero_train300k
dp --pt train input.json
dp --pt freeze -o frozen_model.pth
