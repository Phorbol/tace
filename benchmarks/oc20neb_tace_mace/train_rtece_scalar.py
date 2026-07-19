#!/usr/bin/env python3
from __future__ import annotations

# Compatibility shim for historical OC20NEB benchmark jobs.
# The maintained rTECE scalar training CLI now lives in tace.scripts.rtece_train_scalar.
from tace.scripts.rtece_train_scalar import *  # noqa: F401,F403
from tace.scripts.rtece_train_scalar import main


if __name__ == "__main__":
    main()
