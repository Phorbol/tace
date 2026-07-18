from __future__ import annotations

# Compatibility shim for historical OC20NEB benchmark scripts.
# The rTECE scalar endpoint now lives in tace.models.rtece_scalar.
from tace.models.rtece_scalar import *  # noqa: F401,F403
