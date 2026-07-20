#!/bin/bash
#SBATCH --job-name=deepmd_dpa_like_mixed_smoke-bench
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=00:40:00
#SBATCH --output=logs/deepmd_dpa_like_mixed_smoke-benchmark-%j.out
#SBATCH --error=logs/deepmd_dpa_like_mixed_smoke-benchmark-%j.err

set -eo pipefail
module load deepmd-kit/3.1.2
export STAGE145_ROOT_LABEL="community-baselines-stage145"
cd "$(pwd)"
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python benchmarks/oc20neb_tace_mace/benchmark_stage145_community.py   --engine deepmd   --model-artifact "runs/oc20neb_tace_mace/community-baselines-stage145/deepmd_dpa_like_mixed_smoke/frozen_model.pth"   --configs "/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz"   --output "runs/oc20neb_tace_mace/community-baselines-stage145/deepmd_dpa_like_mixed_smoke/deepmd_dpa_like_mixed_smoke_dft_benchmark.json"   --row-name "deepmd_dpa_like_mixed_smoke"   --limit-configs "1024"   --measure-passes 3   --device cuda
