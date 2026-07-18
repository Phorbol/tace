#!/usr/bin/env python3
"""Build sample-weight JSON files for weighted rTECE projection diagnostics."""

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

from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import _load_graphs
from tace.models.rtece_scalar import RTECEGraph, RTECEScalarModel, rtece_descriptors
from tace.models.rtece_workflow import load_checkpoint


def _normalize_weights(weights: torch.Tensor, normalize: str, *, floor: float) -> torch.Tensor:
    values = weights.detach().to(dtype=torch.float64, device="cpu").flatten()
    if values.numel() < 1:
        raise ValueError("sample weights must not be empty")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("sample weights must be finite")
    values = values.clamp_min(float(floor))
    if normalize == "none":
        return values
    if normalize == "mean1":
        mean = float(values.mean().item())
        if mean <= 0.0:
            raise ValueError("cannot mean-normalize zero sample weights")
        return values / mean
    raise ValueError(f"unknown normalization {normalize!r}")


def rtece_head_jacobian_sample_weights(
    model: RTECEScalarModel,
    graphs: list[RTECEGraph],
    *,
    normalize: str = "mean1",
    floor: float = 1.0e-8,
) -> dict[str, Any]:
    """Return per-atom descriptor-sensitivity weights from an rTECE energy head."""
    if not graphs:
        raise ValueError("rtece head-Jacobian weights require at least one graph")
    was_training = model.training
    model.eval()
    raw_parts = []
    for graph in graphs:
        z_scaled = graph.z.to(dtype=graph.pos.dtype, device=graph.pos.device).view(-1, 1)
        z_scaled = z_scaled / float(model.config.max_atomic_number)
        descriptors = rtece_descriptors(graph, model.config).detach().requires_grad_(True)
        atomic_energy = model.energy_head(torch.cat([z_scaled, descriptors], dim=-1)).squeeze(-1)
        descriptor_grad = torch.autograd.grad(
            atomic_energy.sum(),
            descriptors,
            create_graph=False,
            retain_graph=False,
        )[0]
        raw_parts.append(torch.linalg.vector_norm(descriptor_grad, dim=-1).detach().cpu())
    if was_training:
        model.train()
    raw = torch.cat(raw_parts, dim=0).to(dtype=torch.float64)
    weights = _normalize_weights(raw, normalize, floor=float(floor))
    return {
        "schema_version": "rtece_projection_sample_weights.v1",
        "weight_source": "rtece_head_jacobian_l2",
        "normalization": str(normalize),
        "floor": float(floor),
        "num_graphs": int(len(graphs)),
        "num_samples": int(weights.numel()),
        "raw_min": float(raw.min().item()),
        "raw_mean": float(raw.mean().item()),
        "raw_max": float(raw.max().item()),
        "weight_min": float(weights.min().item()),
        "weight_mean": float(weights.mean().item()),
        "weight_max": float(weights.max().item()),
        "sample_weights": [float(value) for value in weights.tolist()],
    }


def write_sample_weight_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    item = dict(payload)
    item.setdefault("schema_version", "rtece_projection_sample_weights.v1")
    target.write_text(json.dumps(item, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--limit-configs", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=None)
    parser.add_argument("--default-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--neighborlist-backend", choices=("ase", "vesin", "matscipy"), default="matscipy")
    parser.add_argument("--normalize", choices=("mean1", "none"), default="mean1")
    parser.add_argument("--floor", type=float, default=1.0e-8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float64 if args.default_dtype == "float64" else torch.float32
    device = torch.device("cpu")
    model, config, metadata = load_checkpoint(args.checkpoint, dtype=dtype, device=device)
    cutoff = float(args.cutoff) if args.cutoff is not None else float(config.cutoff)
    graphs = _load_graphs(
        args.configs,
        cutoff=cutoff,
        device=device,
        dtype=dtype,
        limit_configs=int(args.limit_configs),
        neighborlist_backend=args.neighborlist_backend,
    )
    payload = rtece_head_jacobian_sample_weights(
        model,
        graphs,
        normalize=str(args.normalize),
        floor=float(args.floor),
    )
    payload.update(
        {
            "checkpoint": str(args.checkpoint),
            "configs": str(args.configs),
            "limit_configs": int(args.limit_configs),
            "cutoff": cutoff,
            "dtype": str(dtype).replace("torch.", ""),
            "checkpoint_tece_path_manifest_hash": (metadata.get("tece_path_manifest") or {}).get("manifest_hash"),
            "checkpoint_scalar_path_ids": list(config.scalar_path_ids or []),
        }
    )
    write_sample_weight_json(args.output_json, payload)
    print(args.output_json)


if __name__ == "__main__":
    main()
