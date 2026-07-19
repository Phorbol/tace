#!/usr/bin/env bash
set -euo pipefail

# Create an isolated TACE environment that reuses the existing CUDA-enabled
# MACE conda environment packages when available. This avoids reinstalling the
# heavy PyTorch CUDA stack and keeps compatibility with V100/sm70.
BASE_PYTHON=${BASE_PYTHON:-/home/sjtu-caoxiaoming/gengjianrui/conda-envs/mace-dpa4-cu126/bin/python}
ENV_DIR=${ENV_DIR:-/home/gengjianrui/bin/.venvs/tace-mace-cu126}
TACE_ROOT=${TACE_ROOT:-/home/gengjianrui/bin/tace}

"${BASE_PYTHON}" -m venv --system-site-packages "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${ENV_DIR}/bin/python" -m pip install -e "${TACE_ROOT}"
# TACE 0.2.0 calls e3nn.Irreps.regroup(), absent in the MACE reference
# environment e3nn==0.4.4. Keep MACE on its original Python and
# upgrade e3nn only inside this TACE venv.
"${ENV_DIR}/bin/python" -m pip install "e3nn>=0.6.0"
"${ENV_DIR}/bin/python" - <<'PYINFO'
import sys
print('python', sys.version)
for name in ['torch', 'torch_geometric', 'ase', 'e3nn', 'lightning', 'tace']:
    mod = __import__(name)
    print(name, getattr(mod, '__version__', 'unknown'))
import os
import torch
print('torch cuda', torch.version.cuda)
print('torch cuda probe', 'deferred to Slurm compute node')
print('TORCH_CUDA_ARCH_LIST', os.environ.get('TORCH_CUDA_ARCH_LIST', 'set to 7.0 in sbatch scripts'))
PYINFO
