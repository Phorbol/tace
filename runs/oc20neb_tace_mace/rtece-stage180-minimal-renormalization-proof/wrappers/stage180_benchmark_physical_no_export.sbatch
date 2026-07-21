#!/bin/bash
#SBATCH --job-name=rtece-st180-diag
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=03:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-st180-diag-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-st180-diag-%j.err

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
PROJECTION_LIMIT_CONFIGS=256
MAX_STEPS=20000
RESULTS_ROOT=runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/results
DIAGNOSTICS_ROOT=runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/diagnostics
mkdir -p /home/gengjianrui/bin/logs "${RESULTS_ROOT}" "${DIAGNOSTICS_ROOT}"
cd "${TACE_ROOT}"
"${TACE_PYTHON}" -V
nvidia-smi -L

mkdir -p runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/diagnostics/scratch_same_student
PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" python benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/results/scratch_same_student/rtece_scalar_best.pt --configs "${VALID_FILE}" --output runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/diagnostics/scratch_same_student/scratch_same_student_dft_benchmark.json --variant scratch_same_student --start-config 0 --limit-configs "${VALID_LIMIT_CONFIGS}" --measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist
PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" python benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/results/scratch_same_student/rtece_scalar_best.pt --configs "${TEACHER_VALID_FILE}" --output runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/diagnostics/scratch_same_student/scratch_same_student_teacher_benchmark.json --variant scratch_same_student --start-config 0 --limit-configs "${VALID_LIMIT_CONFIGS}" --measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist
PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" python benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/results/scratch_same_student/rtece_scalar_best.pt --configs "${VALID_FILE}" --output runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/diagnostics/scratch_same_student/scratch_same_student_scaling_limit32.json --variant scratch_same_student --start-config 0 --limit-configs 32 --measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist
PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" python benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/results/scratch_same_student/rtece_scalar_best.pt --configs "${VALID_FILE}" --output runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/diagnostics/scratch_same_student/scratch_same_student_scaling_limit128.json --variant scratch_same_student --start-config 0 --limit-configs 128 --measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist
PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" python benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/results/scratch_same_student/rtece_scalar_best.pt --configs "${VALID_FILE}" --output runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/diagnostics/scratch_same_student/scratch_same_student_scaling_limit512.json --variant scratch_same_student --start-config 0 --limit-configs 512 --measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist
PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" python benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/results/scratch_same_student/rtece_scalar_best.pt --configs "${VALID_FILE}" --output runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/diagnostics/scratch_same_student/scratch_same_student_scaling_limit1024.json --variant scratch_same_student --start-config 0 --limit-configs 1024 --measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist
PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" python benchmarks/oc20neb_tace_mace/dimer_scan_rtece.py --checkpoint runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/results/scratch_same_student/rtece_scalar_best.pt --output-json runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/diagnostics/scratch_same_student/scratch_same_student_dimer_scan.json --output-md runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/diagnostics/scratch_same_student/scratch_same_student_dimer_scan.md --route scratch_same_student --pairs C-N C-O C-H N-H O-H C-C N-N --num-points 24 --min-scale 0.5 --max-scale 5.0 --device cuda --default-dtype float32 --force-mode autograd --neighborlist-backend ase
PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" python benchmarks/oc20neb_tace_mace/rattle_relax_rtece.py --checkpoint runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/results/scratch_same_student/rtece_scalar_best.pt --configs "${VALID_FILE}" --output-json runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/diagnostics/scratch_same_student/scratch_same_student_rattle_relax.json --output-md runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/diagnostics/scratch_same_student/scratch_same_student_rattle_relax.md --route scratch_same_student --start-config 58 --limit-configs 16 --rattle-std 0.05 --rattle-seed 20260718 --fmax 0.05 --max-steps 10 --device cuda --default-dtype float32 --force-mode autograd --neighborlist-backend matscipy
PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" python benchmarks/oc20neb_tace_mace/summarize_rtece_physical_pareto.py --case scratch_same_student:runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/diagnostics/scratch_same_student/scratch_same_student_dft_benchmark.json:runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/diagnostics/scratch_same_student/scratch_same_student_teacher_benchmark.json:runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/diagnostics/scratch_same_student/scratch_same_student_dimer_scan.json:runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/diagnostics/scratch_same_student/scratch_same_student_rattle_relax.json --max-dft-f-rmse-mev-a 120.0 --max-dft-e-rmse-mev-atom 320.0 --max-dft-f-max-mev-a 2600.0 --max-dft-e-max-mev-atom 760.0 --rattle-focus-label C_or_N --max-focus-rattle-rmsd-a 0.35 --max-rattle-fmax-ev-a 1.00 --output-json runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/diagnostics/scratch_same_student/scratch_same_student_physical_pareto.json --output-md runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof/diagnostics/scratch_same_student/scratch_same_student_physical_pareto.md
