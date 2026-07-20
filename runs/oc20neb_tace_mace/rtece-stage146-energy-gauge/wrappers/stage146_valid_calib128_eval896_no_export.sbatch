#!/bin/bash
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --job-name=rtece146-egauge
#SBATCH --output=runs/oc20neb_tace_mace/rtece-stage146-energy-gauge/logs/stage146-egauge-%j.out
#SBATCH --error=runs/oc20neb_tace_mace/rtece-stage146-energy-gauge/logs/stage146-egauge-%j.err

set -eo pipefail

cd /home/gengjianrui/bin/tace

/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python benchmarks/oc20neb_tace_mace/analyze_rtece_energy_gauge.py \
  --model runs/oc20neb_tace_mace/rtece-stage144-t3-cavity-vector/t3_cavity_vector_runs/t3_l2_cavity_vector_cond32_h64/t3_l2_cavity_vector_cond32_h64/rtece_scalar_best.pt \
  --configs /home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz \
  --output runs/oc20neb_tace_mace/rtece-stage146-energy-gauge/stage144_valid_calib128_eval896_energy_gauge.json \
  --variant t3_l2_cavity_vector_cond32_h64 \
  --device cuda \
  --default-dtype float32 \
  --calib-start 0 \
  --calib-limit 128 \
  --eval-start 128 \
  --eval-limit 896 \
  --force-mode autograd \
  --graph-construction-backend matscipy_neighborlist
