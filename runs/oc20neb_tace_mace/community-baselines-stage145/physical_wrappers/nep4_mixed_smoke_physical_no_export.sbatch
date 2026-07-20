#!/bin/bash
#SBATCH --job-name=nep4_mixed_smoke-phys
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=00:50:00
#SBATCH --output=logs/nep4_mixed_smoke-physical-%j.out
#SBATCH --error=logs/nep4_mixed_smoke-physical-%j.err

set -eo pipefail
module load gpumd/4.8-cuda12.4
export STAGE145_ROOT_LABEL="community-baselines-stage145"
cd "$(pwd)"
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python benchmarks/oc20neb_tace_mace/physical_stage145_community.py   --engine nep   --model-artifact "runs/oc20neb_tace_mace/community-baselines-stage145/nep4_mixed_smoke/nep.txt"   --configs "/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz"   --output-json "runs/oc20neb_tace_mace/community-baselines-stage145/nep4_mixed_smoke/nep4_mixed_smoke_physical_pareto.json"   --row-name "nep4_mixed_smoke"   --run-dir "runs/oc20neb_tace_mace/community-baselines-stage145/nep4_mixed_smoke/stage145_physical"   --limit-configs 2   --device cuda
