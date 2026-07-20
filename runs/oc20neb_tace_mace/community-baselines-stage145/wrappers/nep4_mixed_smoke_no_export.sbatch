#!/bin/bash
#SBATCH --job-name=nep4_mixed_smoke
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=rush-1o2gpu
#SBATCH --output=logs/nep4_mixed_smoke-%j.out
#SBATCH --error=logs/nep4_mixed_smoke-%j.err

set -euo pipefail
export STAGE145_ROOT_LABEL="community-baselines-stage145"
cd "$(pwd)"
module load gpumd/4.8-cuda12.4
export STAGE145_TRAIN_FILE="runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/weighted_train_base2048_plus_teacher_relax640_forceonly_eanchor.extxyz"
export STAGE145_OUTPUT_DIR="runs/oc20neb_tace_mace/community-baselines-stage145/nep4_mixed_smoke"
export STAGE145_LIMIT_CONFIGS="2688"
python benchmarks/oc20neb_tace_mace/convert_stage145_nep.py --manifest runs/oc20neb_tace_mace/community-baselines-stage145/stage145_manifest.json --row nep4_mixed_smoke
cd "runs/oc20neb_tace_mace/community-baselines-stage145/nep4_mixed_smoke"
nep

