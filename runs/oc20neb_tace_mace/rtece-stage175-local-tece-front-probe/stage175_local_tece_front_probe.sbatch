#!/bin/bash
#SBATCH --job-name=rtece-res175
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=01:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-res175-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-res175-%j.err

set -eo pipefail
set -x
TACE_ROOT=/home/gengjianrui/bin/tace
ENV_DIR=/home/gengjianrui/bin/.venvs/tace-mace-cu126
TACE_PYTHON="${ENV_DIR}/bin/python"
PATH="${ENV_DIR}/bin:${PATH}"
PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}"
CONFIGS=/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz
STAGE165_JSON=runs/oc20neb_tace_mace/rtece-stage165-energy-baseline-triage/stage165_energy_baseline_triage.json
RESULTS_JSON=runs/oc20neb_tace_mace/rtece-stage175-local-tece-front-probe/stage175_results_summary.json
RESULTS_MD=runs/oc20neb_tace_mace/rtece-stage175-local-tece-front-probe/stage175_results_summary.md
LIMIT_CONFIGS=512
mkdir -p /home/gengjianrui/bin/logs "$(dirname "${RESULTS_JSON}")"
cd "${TACE_ROOT}"
export TACE_PYTHON PATH PYTHONPATH
"${TACE_PYTHON}" -V
nvidia-smi -L

# Stage175: local TECE front feature probe
"${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/analyze_rtece_local_front_features.py --configs "${CONFIGS}" --stage165-json "${STAGE165_JSON}" --stage165-variant stage157_direct_b32_rel0p25_mixed2048 --group-key case_id --limit-configs "${LIMIT_CONFIGS}" --output-json "${RESULTS_JSON}" --output-md "${RESULTS_MD}" --shell-edges 0,1.6,2.6,3.6,5 --energy-ridge-grid 1e-08,1e-06,0.0001,0.01,1,100
