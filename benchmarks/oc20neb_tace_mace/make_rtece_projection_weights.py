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


def force_residual_sample_weight_payload(
    predicted_forces: list[torch.Tensor],
    reference_forces: list[torch.Tensor],
    *,
    target_force_source: str,
    normalize: str = "mean1",
    floor: float = 1.0e-8,
) -> dict[str, Any]:
    if not predicted_forces:
        raise ValueError("force-residual weights require at least one force tensor")
    if len(predicted_forces) != len(reference_forces):
        raise ValueError(
            f"predicted/reference force tensor counts differ: {len(predicted_forces)} vs {len(reference_forces)}"
        )
    raw_parts = []
    for predicted, reference in zip(predicted_forces, reference_forces, strict=True):
        pred = predicted.detach().to(dtype=torch.float64, device="cpu")
        ref = reference.detach().to(dtype=torch.float64, device="cpu")
        if pred.shape != ref.shape:
            raise ValueError(f"predicted/reference force shapes differ: {tuple(pred.shape)} vs {tuple(ref.shape)}")
        if pred.ndim != 2 or pred.shape[1] != 3:
            raise ValueError(f"force tensors must have shape [atoms, 3], got {tuple(pred.shape)}")
        raw_parts.append(torch.linalg.vector_norm(pred - ref, dim=-1))
    raw = torch.cat(raw_parts, dim=0).to(dtype=torch.float64)
    weights = _normalize_weights(raw, normalize, floor=float(floor))
    return {
        "schema_version": "rtece_projection_sample_weights.v1",
        "weight_source": f"rtece_force_residual_l2:{target_force_source}",
        "target_force_source": str(target_force_source),
        "normalization": str(normalize),
        "floor": float(floor),
        "num_graphs": int(len(predicted_forces)),
        "num_samples": int(weights.numel()),
        "raw_min": float(raw.min().item()),
        "raw_mean": float(raw.mean().item()),
        "raw_max": float(raw.max().item()),
        "weight_min": float(weights.min().item()),
        "weight_mean": float(weights.mean().item()),
        "weight_max": float(weights.max().item()),
        "sample_weights": [float(value) for value in weights.tolist()],
    }


def rtece_force_residual_sample_weights(
    model: RTECEScalarModel,
    graphs: list[RTECEGraph],
    reference_forces: list[torch.Tensor],
    *,
    target_force_source: str,
    normalize: str = "mean1",
    floor: float = 1.0e-8,
) -> dict[str, Any]:
    was_training = model.training
    model.eval()
    predicted_forces = []
    for graph in graphs:
        predicted_forces.append(model(graph)["forces"].detach().cpu())
    if was_training:
        model.train()
    return force_residual_sample_weight_payload(
        predicted_forces,
        reference_forces,
        target_force_source=target_force_source,
        normalize=normalize,
        floor=float(floor),
    )


def _load_reference_forces(
    configs: Path,
    *,
    force_array: str,
    dtype: torch.dtype,
    limit_configs: int,
) -> list[torch.Tensor]:
    import ase.io

    atoms_list = ase.io.read(str(configs), index=f":{int(limit_configs)}")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    forces = []
    for index, atoms in enumerate(atoms_list):
        if force_array not in atoms.arrays:
            raise ValueError(f"force array {force_array!r} not found in config {index}")
        forces.append(torch.tensor(atoms.arrays[force_array], dtype=dtype, device="cpu"))
    return forces


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
    parser.add_argument(
        "--weight-mode",
        choices=("rtece_head_jacobian_l2", "rtece_force_residual_l2"),
        default="rtece_head_jacobian_l2",
    )
    parser.add_argument("--target-force-array", default="teacher_forces")
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
    if args.weight_mode == "rtece_head_jacobian_l2":
        payload = rtece_head_jacobian_sample_weights(
            model,
            graphs,
            normalize=str(args.normalize),
            floor=float(args.floor),
        )
    elif args.weight_mode == "rtece_force_residual_l2":
        reference_forces = _load_reference_forces(
            args.configs,
            force_array=str(args.target_force_array),
            dtype=dtype,
            limit_configs=int(args.limit_configs),
        )
        payload = rtece_force_residual_sample_weights(
            model,
            graphs,
            reference_forces,
            target_force_source=str(args.target_force_array),
            normalize=str(args.normalize),
            floor=float(args.floor),
        )
    else:
        raise ValueError(f"unknown weight mode {args.weight_mode!r}")
    payload.update(
        {
            "checkpoint": str(args.checkpoint),
            "configs": str(args.configs),
            "limit_configs": int(args.limit_configs),
            "cutoff": cutoff,
            "dtype": str(dtype).replace("torch.", ""),
            "weight_mode": str(args.weight_mode),
            "checkpoint_tece_path_manifest_hash": (metadata.get("tece_path_manifest") or {}).get("manifest_hash"),
            "checkpoint_scalar_path_ids": list(config.scalar_path_ids or []),
        }
    )
    write_sample_weight_json(args.output_json, payload)
    print(args.output_json)


if __name__ == "__main__":
    main()
