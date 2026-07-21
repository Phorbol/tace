#!/usr/bin/env bash
#SBATCH --job-name=rtece-desc169
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=03:55:00
#SBATCH --output=logs/rtece-desc169-%j.out
#SBATCH --error=logs/rtece-desc169-%j.err

set -eo pipefail
cd /home/gengjianrui/bin/tace
mkdir -p logs
mkdir -p "runs/oc20neb_tace_mace/rtece-stage169-descriptor-cache"
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python benchmarks/oc20neb_tace_mace/summarize_rtece_stage168_rtece_descriptor_proxy.py --benchmark "runs/oc20neb_tace_mace/rtece-stage157-relative-neb-loss/relative_loss_runs/stage157_direct_b32_rel0p25_mixed2048/stage157_direct_b32_rel0p25_mixed2048_dft_benchmark.json" --configs "/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz" --num-radial 8 --max-configs-per-case 2 --ridge 0.001 --neighborlist-backend "matscipy" --feature-cache-json "runs/oc20neb_tace_mace/rtece-stage169-descriptor-cache/stage169_rtece_descriptor_features.json" --write-feature-cache-json "runs/oc20neb_tace_mace/rtece-stage169-descriptor-cache/stage169_rtece_descriptor_features.json" --output-json "runs/oc20neb_tace_mace/rtece-stage169-descriptor-cache/stage169_rtece_descriptor_proxy.json" --output-md "runs/oc20neb_tace_mace/rtece-stage169-descriptor-cache/stage169_rtece_descriptor_proxy.md"
