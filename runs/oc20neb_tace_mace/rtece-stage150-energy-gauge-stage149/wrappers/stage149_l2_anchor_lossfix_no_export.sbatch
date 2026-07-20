#!/bin/bash
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --job-name=rtece150-egauge
#SBATCH --output=runs/oc20neb_tace_mace/rtece-stage150-energy-gauge-stage149/logs/stage149_l2_anchor_lossfix-%j.out
#SBATCH --error=runs/oc20neb_tace_mace/rtece-stage150-energy-gauge-stage149/logs/stage149_l2_anchor_lossfix-%j.err

set -eo pipefail

cd /home/gengjianrui/bin/tace

/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python benchmarks/oc20neb_tace_mace/analyze_rtece_energy_gauge.py \
  --model runs/oc20neb_tace_mace/rtece-stage149-energy-loss-normalization-rerun/corrected_energy_loss_runs/stage149_l2_anchor_lossfix/stage149_l2_anchor_lossfix/rtece_scalar_best.pt \
  --configs /home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz \
  --output runs/oc20neb_tace_mace/rtece-stage150-energy-gauge-stage149/stage149_l2_anchor_lossfix_valid_calib128_eval896_energy_gauge.json \
  --variant stage149_l2_anchor_lossfix \
  --device cuda \
  --default-dtype float32 \
  --calib-start 0 \
  --calib-limit 128 \
  --eval-start 128 \
  --eval-limit 896 \
  --force-mode autograd \
  --graph-construction-backend matscipy_neighborlist
