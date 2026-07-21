#!/bin/bash
#SBATCH --job-name=rtece-st185-rmd17-prep
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=03:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-st185-rmd17-prep-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-st185-rmd17-prep-%j.err

set -eo pipefail
set -x
TACE_ROOT=/home/gengjianrui/bin/tace
TACE_PYTHON=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python
PATH=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin:${PATH}
mkdir -p /home/gengjianrui/bin/logs
cd ${TACE_ROOT}
${TACE_PYTHON} -V

RMD17_ROOT=datasets/rMD17
RMD17_NPZ_URL=https://ndownloader.figshare.com/files/62265733
RMD17_NPZ=datasets/rMD17/rmd17_ethanol.npz
mkdir -p ${RMD17_ROOT} ${RMD17_ROOT}/converted_extxyz
if [ ! -f ${RMD17_NPZ} ]; then
  echo Missing local rMD17 source npz: ${RMD17_NPZ} >&2
  echo Run this on a login node before submitting this wrapper. >&2
  echo Download URL: ${RMD17_NPZ_URL} >&2
  exit 12
fi
PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/prepare_stage181_conventional_datasets.py rmd17-npz-to-splits --source-npz datasets/rMD17/rmd17_ethanol.npz --output-dir datasets/rMD17/converted_extxyz --molecule ethanol --train-count 1000 --valid-count 1000 --test-count 1000 --summary-json runs/oc20neb_tace_mace/rtece-stage185-rmd17-representation-ladder/diagnostics/rmd17_split_conversion_summary.json
