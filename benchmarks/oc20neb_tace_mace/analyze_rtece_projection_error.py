#!/usr/bin/env python3
"""Measure descriptor-space projection error between rTECE path-id routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from benchmarks.oc20neb_tace_mace.train_rtece_scalar import atoms_to_rtece_graph
from tace.models.rtece_scalar import (
    RTECEGraph,
    RTECEScalarConfig,
    build_rtece_config_from_path_ids,
    descriptor_dim,
    rtece_descriptors,
    rtece_path_manifest,
)


def _as_float64_matrix(values: torch.Tensor, name: str) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 matrix, got shape {tuple(values.shape)}")
    if values.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one sample")
    return values.detach().to(dtype=torch.float64, device="cpu")


def projection_residual_metrics(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float = 1.0e-12,
) -> dict[str, float | int]:
    """Project target descriptors onto source descriptors and report residual size."""
    x = _as_float64_matrix(source, "source")
    y = _as_float64_matrix(target, "target")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"source/target sample counts differ: {x.shape[0]} vs {y.shape[0]}")
    if x.shape[1] < 1 or y.shape[1] < 1:
        raise ValueError("source and target must both have at least one descriptor column")

    lhs = x.T @ x
    if ridge > 0.0:
        lhs = lhs + float(ridge) * torch.eye(lhs.shape[0], dtype=x.dtype)
    rhs = x.T @ y
    try:
        coeff = torch.linalg.solve(lhs, rhs)
    except RuntimeError:
        coeff = torch.linalg.lstsq(x, y).solution
    residual = y - x @ coeff
    residual_norm = torch.linalg.vector_norm(residual)
    target_norm = torch.linalg.vector_norm(y)
    relative = residual_norm / target_norm.clamp_min(torch.finfo(y.dtype).tiny)
    return {
        "num_samples": int(x.shape[0]),
        "source_dim": int(x.shape[1]),
        "target_dim": int(y.shape[1]),
        "residual_frobenius": float(residual_norm.item()),
        "target_frobenius": float(target_norm.item()),
        "relative_residual": float(relative.item()),
        "ridge": float(ridge),
    }


def _descriptor_matrix(graphs: list[RTECEGraph], config: RTECEScalarConfig) -> torch.Tensor:
    if not graphs:
        raise ValueError("projection diagnostic requires at least one graph")
    return torch.cat([rtece_descriptors(graph, config).detach().cpu() for graph in graphs], dim=0)


def _deleted_path_ids(candidate: RTECEScalarConfig, reference: RTECEScalarConfig) -> list[str]:
    candidate_paths = set(candidate.scalar_path_ids or ())
    return [path_id for path_id in (reference.scalar_path_ids or ()) if path_id not in candidate_paths]


def make_projection_diagnostic_row(
    candidate_name: str,
    *,
    candidate_config: RTECEScalarConfig,
    reference_config: RTECEScalarConfig,
    graphs: list[RTECEGraph],
    ridge: float = 1.0e-12,
) -> dict[str, Any]:
    candidate_descriptors = _descriptor_matrix(graphs, candidate_config)
    reference_descriptors = _descriptor_matrix(graphs, reference_config)
    metrics = projection_residual_metrics(candidate_descriptors, reference_descriptors, ridge=ridge)
    candidate_manifest = rtece_path_manifest(candidate_config)
    reference_manifest = rtece_path_manifest(reference_config)
    row: dict[str, Any] = {
        "candidate": str(candidate_name),
        "num_graphs": int(len(graphs)),
        "candidate_variant": candidate_config.variant,
        "reference_variant": reference_config.variant,
        "candidate_manifest_hash": candidate_manifest["manifest_hash"],
        "reference_manifest_hash": reference_manifest["manifest_hash"],
        "candidate_scalar_path_ids": list(candidate_config.scalar_path_ids or []),
        "reference_scalar_path_ids": list(reference_config.scalar_path_ids or []),
        "deleted_scalar_path_ids": _deleted_path_ids(candidate_config, reference_config),
        "candidate_dim": int(descriptor_dim(candidate_config)),
        "reference_dim": int(descriptor_dim(reference_config)),
    }
    row.update(metrics)
    return row


def _parse_path_ids(value: str) -> tuple[str, ...]:
    path_ids = tuple(part.strip() for part in value.split(",") if part.strip())
    if not path_ids:
        raise argparse.ArgumentTypeError("path id list must not be empty")
    return path_ids


def _parse_candidate(value: str) -> tuple[str, tuple[str, ...]]:
    parts = value.split(":", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("candidate must be name:path_id,path_id")
    return parts[0], _parse_path_ids(parts[1])


def _load_graphs(
    configs: Path,
    *,
    cutoff: float,
    device: torch.device,
    dtype: torch.dtype,
    limit_configs: int,
    neighborlist_backend: str,
) -> list[RTECEGraph]:
    import ase.io

    atoms_list = ase.io.read(str(configs), index=f":{int(limit_configs)}")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    return [
        atoms_to_rtece_graph(
            atoms,
            cutoff=cutoff,
            device=device,
            dtype=dtype,
            neighborlist_backend=neighborlist_backend,
        )
        for atoms in atoms_list
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--reference-path-ids", type=_parse_path_ids, required=True)
    parser.add_argument("--candidate", action="append", type=_parse_candidate, default=[])
    parser.add_argument("--num-radial", type=int, default=8)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--limit-configs", type=int, default=32)
    parser.add_argument("--default-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--neighborlist-backend", choices=("ase", "vesin", "matscipy"), default="matscipy")
    parser.add_argument("--ridge", type=float, default=1.0e-12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float64 if args.default_dtype == "float64" else torch.float32
    device = torch.device("cpu")
    reference_config = build_rtece_config_from_path_ids(
        "rtece_projection_reference",
        args.reference_path_ids,
        cutoff=float(args.cutoff),
        num_radial=int(args.num_radial),
    )
    graphs = _load_graphs(
        args.configs,
        cutoff=float(args.cutoff),
        device=device,
        dtype=dtype,
        limit_configs=int(args.limit_configs),
        neighborlist_backend=args.neighborlist_backend,
    )
    rows = []
    for candidate_name, candidate_path_ids in args.candidate:
        candidate_config = build_rtece_config_from_path_ids(
            f"rtece_projection_{candidate_name}",
            candidate_path_ids,
            cutoff=float(args.cutoff),
            num_radial=int(args.num_radial),
        )
        rows.append(
            make_projection_diagnostic_row(
                candidate_name,
                candidate_config=candidate_config,
                reference_config=reference_config,
                graphs=graphs,
                ridge=float(args.ridge),
            )
        )
    payload = {
        "schema_version": "rtece_projection_diagnostic.v1",
        "configs": str(args.configs),
        "limit_configs": int(args.limit_configs),
        "num_radial": int(args.num_radial),
        "cutoff": float(args.cutoff),
        "reference_path_ids": list(args.reference_path_ids),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output_json)


if __name__ == "__main__":
    main()
