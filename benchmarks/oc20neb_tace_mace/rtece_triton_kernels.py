from __future__ import annotations

# Compatibility shim for historical OC20NEB benchmark scripts.
# The rTECE Triton backend now lives in tace.models.rtece_triton_kernels.
from tace.models.rtece_triton_kernels import *  # noqa: F401,F403
