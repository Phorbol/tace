#!/bin/bash
#SBATCH --job-name=rtecephys141
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=03:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtecephys141-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtecephys141-%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

TACE_ROOT=${TACE_ROOT:-/home/gengjianrui/bin/tace}
ENV_DIR=${ENV_DIR:-/home/gengjianrui/bin/.venvs/tace-mace-cu126}
TACE_PYTHON=${TACE_PYTHON:-${ENV_DIR}/bin/python}
VARIANT=l2_active_nrad12_species24_radial_species8_cross3_cond32_fm2p0_ew3_h64
OUT_DIR=runs/oc20neb_tace_mace/rtece-stage141-loss-measure-frontier/physical_triage/l2_active_nrad12_species24_radial_species8_cross3_cond32_fm2p0_ew3_h64
CHECKPOINT=runs/oc20neb_tace_mace/rtece-stage141-loss-measure-frontier/loss_measure_runs/l2_active_nrad12_species24_radial_species8_cross3_cond32_fm2p0_ew3_h64/l2_active_nrad12_species24_radial_species8_cross3_cond32_fm2p0_ew3_h64/rtece_scalar_best.pt
DFT_BENCHMARK=runs/oc20neb_tace_mace/rtece-stage141-loss-measure-frontier/loss_measure_runs/l2_active_nrad12_species24_radial_species8_cross3_cond32_fm2p0_ew3_h64/l2_active_nrad12_species24_radial_species8_cross3_cond32_fm2p0_ew3_h64/l2_active_nrad12_species24_radial_species8_cross3_cond32_fm2p0_ew3_h64_dft_benchmark.json
TEACHER_BENCHMARK=runs/oc20neb_tace_mace/rtece-stage141-loss-measure-frontier/loss_measure_runs/l2_active_nrad12_species24_radial_species8_cross3_cond32_fm2p0_ew3_h64/l2_active_nrad12_species24_radial_species8_cross3_cond32_fm2p0_ew3_h64/l2_active_nrad12_species24_radial_species8_cross3_cond32_fm2p0_ew3_h64_teacher_benchmark.json
CONFIGS=/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz
DIMER_PAIRS="C-N C-O C-H N-H O-H C-C N-N"
DIMER_NUM_POINTS=24
RATTLE_START_CONFIG=58
RATTLE_LIMIT_CONFIGS=8
RATTLE_MAX_STEPS=10
RATTLE_FOCUS_LABEL=C_or_N
MAX_FOCUS_RATTLE_RMSD_A=0.35

mkdir -p /home/gengjianrui/bin/logs "${OUT_DIR}"
cd "${TACE_ROOT}"
export PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST:-7.0}

"${TACE_PYTHON}" -V
nvidia-smi -L

"${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/dimer_scan_rtece.py --checkpoint "${CHECKPOINT}" --output-json "${OUT_DIR}/${VARIANT}_dimer.json" --output-md "${OUT_DIR}/${VARIANT}_dimer.md" --route "${VARIANT}" --pairs ${DIMER_PAIRS} --num-points "${DIMER_NUM_POINTS}" --min-scale 0.5 --max-scale 5.0 --device cuda --default-dtype float32 --force-mode autograd --neighborlist-backend ase

"${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/rattle_relax_rtece.py --checkpoint "${CHECKPOINT}" --configs "${CONFIGS}" --output-json "${OUT_DIR}/${VARIANT}_rattle.json" --output-md "${OUT_DIR}/${VARIANT}_rattle.md" --route "${VARIANT}" --start-config "${RATTLE_START_CONFIG}" --limit-configs "${RATTLE_LIMIT_CONFIGS}" --rattle-std 0.05 --rattle-seed 20260718 --fmax 0.05 --max-steps "${RATTLE_MAX_STEPS}" --device cuda --default-dtype float32 --force-mode autograd --neighborlist-backend matscipy

"${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/summarize_rtece_physical_pareto.py --case "${VARIANT}:${DFT_BENCHMARK}:${TEACHER_BENCHMARK}:${OUT_DIR}/${VARIANT}_dimer.json:${OUT_DIR}/${VARIANT}_rattle.json" --max-dft-f-rmse-mev-a 120.0 --max-dft-e-rmse-mev-atom 320.0 --max-dft-f-max-mev-a 2600.0 --max-dft-e-max-mev-atom 760.0 --rattle-focus-label "${RATTLE_FOCUS_LABEL}" --max-focus-rattle-rmsd-a "${MAX_FOCUS_RATTLE_RMSD_A}" --max-rattle-fmax-ev-a 1.00 --output-json "${OUT_DIR}/${VARIANT}_physical_pareto.json" --output-md "${OUT_DIR}/${VARIANT}_physical_pareto.md"
