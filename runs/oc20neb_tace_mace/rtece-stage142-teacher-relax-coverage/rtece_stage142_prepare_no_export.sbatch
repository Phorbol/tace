#!/bin/bash
#SBATCH --job-name=rtece-prep142
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=03:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-prep142-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-prep142-%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1
export TACE_ROOT=/home/gengjianrui/bin/tace
export ENV_DIR=/home/gengjianrui/bin/.venvs/tace-mace-cu126
export TACE_PYTHON="${ENV_DIR}/bin/python"
export PATH="${ENV_DIR}/bin:${PATH}"
export PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}"
export RELAX_MAX_STEPS=4
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_CUDA_ARCH_LIST=7.0
mkdir -p /home/gengjianrui/bin/logs
cd "${TACE_ROOT}"
"${TACE_PYTHON}" -V
nvidia-smi -L

# make_teacher_relax_trajectory_labels
${TACE_PYTHON:-python} benchmarks/oc20neb_tace_mace/make_teacher_relax_distill_configs.py --model runs/oc20neb_tace_mace/675969_resume/checkpoints_epoch/TACE-0-100000-0.1329.ckpt --input /home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz --output runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/teacher_relax_valid_start0_limit64_copies2_std0p05_steps4.extxyz --summary runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/teacher_relax_valid_start0_limit64_copies2_std0p05_steps4_summary.json --start-config 0 --limit-configs 64 --copies-per-config 2 --rattle-std-a 0.05 --seed 20260720 --relax-max-steps ${RELAX_MAX_STEPS:-4} --device cuda --default-dtype float32 --nl-backend matscipy --reference-prefix source_

# concat_base_and_teacher_relax_trajectory
${TACE_PYTHON:-python} benchmarks/oc20neb_tace_mace/concat_extxyz_datasets.py --input runs/oc20neb_tace_mace/tece-distill-20260717/mixed_train_tw0.75.extxyz --source-label base_mixed_train_tw0p75 --input runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/teacher_relax_valid_start0_limit64_copies2_std0p05_steps4.extxyz --source-label teacher_relax_trajectory640 --input-limit 2048 --input-limit -1 --output runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/augmented_train_base2048_plus_teacher_relax640.extxyz --summary runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/augmented_train_base2048_plus_teacher_relax640_summary.json

# apply_dft_anchor_force_only_teacher_relax_weights
${TACE_PYTHON:-python} benchmarks/oc20neb_tace_mace/apply_extxyz_sample_weights.py --input runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/augmented_train_base2048_plus_teacher_relax640.extxyz --output runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/weighted_train_base2048_plus_teacher_relax640_forceonly_eanchor.extxyz --summary runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/weighted_train_base2048_plus_teacher_relax640_forceonly_eanchor_summary.json --source-energy-multiplier base_mixed_train_tw0p75:1.25 --source-energy-multiplier teacher_relax_trajectory640:0.0 --source-force-multiplier teacher_relax_trajectory640:2.0 --normalize-energy-mean
