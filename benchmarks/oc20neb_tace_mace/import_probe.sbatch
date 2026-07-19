#!/bin/bash
#SBATCH --job-name=tace-probe
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --output=/home/gengjianrui/bin/logs/tace-import-probe-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/tace-import-probe-%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1
TACE_ROOT=${TACE_ROOT:-/home/gengjianrui/bin/tace}
ENV_DIR=${ENV_DIR:-/home/gengjianrui/bin/.venvs/tace-mace-cu126}
export PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}"
export TACE_USE_CUE=0
export TACE_USE_OEQ=0
export TACE_USE_EQT=0
export TORCH_CUDA_ARCH_LIST=7.0
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
hostname
nvidia-smi -L
"${ENV_DIR}/bin/python" - <<'PYINFO'
import importlib
import time
mods = [
    'torch',
    'ase',
    'torch_geometric',
    'lightning',
    'hydra',
    'omegaconf',
    'tace',
    'tace.dataset.dataloader',
    'tace.lightning.trainer',
    'tace.scripts.train',
]
for name in mods:
    start = time.perf_counter()
    print(f'IMPORT_START {name}', flush=True)
    importlib.import_module(name)
    print(f'IMPORT_DONE {name} {time.perf_counter() - start:.3f}s', flush=True)
print('probe complete', flush=True)
PYINFO
