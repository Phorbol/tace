#!/bin/bash
#SBATCH --job-name=rtece-st182-nep-train
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=03:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-st182-nep-train-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-st182-nep-train-%j.err

set -eo pipefail
set -x
TACE_ROOT=/home/gengjianrui/bin/tace
TACE_PYTHON=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python
PATH=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin:${PATH}
mkdir -p /home/gengjianrui/bin/logs
cd ${TACE_ROOT}
${TACE_PYTHON} -V

module load gpumd/4.8-cuda12.4
PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/convert_stage145_nep.py --manifest runs/oc20neb_tace_mace/rtece-stage182-3bpa-closure/stage182_manifest.json --row nep4_train300k
cd runs/oc20neb_tace_mace/rtece-stage182-3bpa-closure/results/nep4_train300k
nep
