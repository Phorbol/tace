#!/bin/bash
#SBATCH --job-name=rtece-prep129
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=03:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-prep129-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-prep129-%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1
export TACE_ROOT=/home/gengjianrui/bin/tace
export ENV_DIR=/home/gengjianrui/bin/.venvs/tace-mace-cu126
export TACE_PYTHON="${ENV_DIR}/bin/python"
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_CUDA_ARCH_LIST=7.0
mkdir -p /home/gengjianrui/bin/logs
cd "${TACE_ROOT}"
"${TACE_PYTHON}" -V
nvidia-smi -L

# make_stage128_window_rattles
${TACE_PYTHON:-python} benchmarks/oc20neb_tace_mace/make_rattle_distill_configs.py --input /home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz --output runs/oc20neb_tace_mace/rtece-stage129-teacher-rattle-distill/rattled_valid_start58_limit8_copies16_std0p05.extxyz --summary runs/oc20neb_tace_mace/rtece-stage129-teacher-rattle-distill/rattled_valid_start58_limit8_copies16_std0p05_summary.json --start-config 58 --limit-configs 8 --copies-per-config 16 --rattle-std-a 0.05 --seed 20260720

# teacher_label_stage128_window_rattles
${TACE_PYTHON:-python} benchmarks/oc20neb_tace_mace/distill_tace_labels.py --model runs/oc20neb_tace_mace/675969_resume/checkpoints_epoch/TACE-0-100000-0.1329.ckpt --input runs/oc20neb_tace_mace/rtece-stage129-teacher-rattle-distill/rattled_valid_start58_limit8_copies16_std0p05.extxyz --output runs/oc20neb_tace_mace/rtece-stage129-teacher-rattle-distill/teacher_labeled_stage128_window_rattles.extxyz --summary runs/oc20neb_tace_mace/rtece-stage129-teacher-rattle-distill/teacher_labeled_stage128_window_rattles_summary.json --device cuda --default-dtype float32 --reference-prefix source_

# concat_base_and_teacher_rattles
${TACE_PYTHON:-python} benchmarks/oc20neb_tace_mace/concat_extxyz_datasets.py --input runs/oc20neb_tace_mace/tece-distill-20260717/mixed_train_tw0.75.extxyz --source-label base_mixed_train_tw0p75 --input runs/oc20neb_tace_mace/rtece-stage129-teacher-rattle-distill/teacher_labeled_stage128_window_rattles.extxyz --source-label teacher_labeled_stage128_rattles --output runs/oc20neb_tace_mace/rtece-stage129-teacher-rattle-distill/augmented_train_base2048_plus_teacher_rattle128.extxyz --summary runs/oc20neb_tace_mace/rtece-stage129-teacher-rattle-distill/augmented_train_base2048_plus_teacher_rattle128_summary.json
