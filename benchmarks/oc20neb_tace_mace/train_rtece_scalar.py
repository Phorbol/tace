#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from benchmarks.oc20neb_tace_mace.rtece_scalar_model import (
    RTECEScalarConfig,
    RTECEScalarModel,
    build_rtece_config,
)


def save_checkpoint(path: Path, model: RTECEScalarModel, config: RTECEScalarConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(config), "state_dict": model.state_dict()}, path)


def load_checkpoint(
    path: Path,
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> tuple[RTECEScalarModel, RTECEScalarConfig]:
    payload = torch.load(path, map_location=device)
    config = RTECEScalarConfig(**payload["config"])
    model = RTECEScalarModel(config).to(device=device, dtype=dtype)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a scalar-sketched rTECE prototype.")
    parser.add_argument(
        "--variant",
        required=True,
        choices=(
            "rtece_pair",
            "rtece_atomic_moments",
            "rtece_edge_sketch8",
            "rtece_edge_sketch16",
        ),
    )
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-configs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--default-dtype", choices=("float32", "float64"), default="float32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ = build_rtece_config(args.variant)
    raise SystemExit("training body is added in Task 6")


if __name__ == "__main__":
    main()
