#!/bin/bash
#SBATCH --job-name=rtece122-phys
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=01:30:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece122-phys-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece122-phys-%j.err

set -euo pipefail
set -x

export PYTHONUNBUFFERED=1

TACE_ROOT=/home/gengjianrui/bin/tace
ENV_DIR=/home/gengjianrui/bin/.venvs/tace-mace-cu126
TACE_PYTHON=${ENV_DIR}/bin/python

VARIANT=l1_active_radial_species8_h64
RUN_DIR=${TACE_ROOT}/runs/oc20neb_tace_mace/rtece-stage122-radial-species-adapter/${VARIANT}/${VARIANT}
OUT_DIR=${TACE_ROOT}/runs/oc20neb_tace_mace/rtece-stage122-radial-species-adapter/physical-${VARIANT}
CHECKPOINT=${RUN_DIR}/rtece_scalar_best.pt
CONFIGS=${TACE_ROOT}/runs/oc20neb_tace_mace/tece-distill-20260717/teacher_valid.extxyz
DFT_BENCH=${RUN_DIR}/${VARIANT}_dft_benchmark.json
TEACHER_BENCH=${RUN_DIR}/${VARIANT}_teacher_benchmark.json
DIMER_JSON=${OUT_DIR}/${VARIANT}_dimer.json
DIMER_MD=${OUT_DIR}/${VARIANT}_dimer.md
RATTLE_JSON=${OUT_DIR}/${VARIANT}_rattle_relax.json
RATTLE_MD=${OUT_DIR}/${VARIANT}_rattle_relax.md
PHYSICAL_JSON=${OUT_DIR}/${VARIANT}_physical_pareto.json
PHYSICAL_MD=${OUT_DIR}/${VARIANT}_physical_pareto.md

mkdir -p /home/gengjianrui/bin/logs "${OUT_DIR}"
cd "${TACE_ROOT}"
export PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}"
export PATH="${ENV_DIR}/bin:${PATH}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

echo "host=$(hostname)"
echo "variant=${VARIANT}"
echo "checkpoint=${CHECKPOINT}"
echo "out_dir=${OUT_DIR}"
"${TACE_PYTHON}" -V
nvidia-smi -L

"${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/dimer_scan_rtece.py \
  --checkpoint "${CHECKPOINT}" \
  --output-json "${DIMER_JSON}" \
  --output-md "${DIMER_MD}" \
  --route stage122_best_physical \
  --pairs C-N C-O N-H O-H Cu-O \
  --num-points 24 \
  --min-scale 0.5 \
  --max-scale 5.0 \
  --device cuda \
  --default-dtype float32 \
  --force-mode autograd \
  --neighborlist-backend ase

"${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/rattle_relax_rtece.py \
  --checkpoint "${CHECKPOINT}" \
  --configs "${CONFIGS}" \
  --output-json "${RATTLE_JSON}" \
  --output-md "${RATTLE_MD}" \
  --route stage122_best_physical \
  --start-config 0 \
  --limit-configs 8 \
  --rattle-std 0.05 \
  --rattle-seed 20260720 \
  --fmax 0.05 \
  --max-steps 20 \
  --device cuda \
  --default-dtype float32 \
  --force-mode autograd \
  --neighborlist-backend ase

"${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/summarize_rtece_physical_pareto.py \
  --case "${VARIANT}:${DFT_BENCH}:${TEACHER_BENCH}:${DIMER_JSON}:${RATTLE_JSON}" \
  --output-json "${PHYSICAL_JSON}" \
  --output-md "${PHYSICAL_MD}" \
  --rattle-focus-label C_or_N \
  --max-focus-rattle-rmsd-a 0.20 \
  --max-rattle-fmax-ev-a 0.40

echo "stage122 physical diagnostics complete: ${OUT_DIR}"
