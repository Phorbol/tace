#!/usr/bin/env python3
"""Build fixed low-rank radial POD projections for rTECE cross-radial paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tace.interface.ase.rtece_calculator import atoms_to_rtece_graph
from tace.models.rtece_scalar import RTECEGraph, RTECEScalarConfig, compute_atomic_moments


def _accumulate_radial_covariance(
    covariance: torch.Tensor,
    moment: torch.Tensor,
    *,
    atom_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    if moment.ndim < 3:
        raise ValueError("atomic moment tensor must have atom, radial, and component dimensions")
    values = moment.detach().to(dtype=torch.float64).reshape(moment.shape[0], moment.shape[1], -1)
    if atom_weights is not None:
        weights = atom_weights.detach().to(device=values.device, dtype=torch.float64).reshape(-1)
        if weights.shape[0] != values.shape[0]:
            raise ValueError("atom_weights length must match the number of atoms in the moment tensor")
        values = values * weights.clamp_min(0.0).sqrt()[:, None, None]
    return covariance + torch.einsum("arc,asc->rs", values, values)


def compute_atomic_cross_radial_pod_projection(
    graphs: list[RTECEGraph] | tuple[RTECEGraph, ...],
    config: RTECEScalarConfig,
    *,
    num_sketches: int,
    moment: str = "vector",
    atom_weights: list[torch.Tensor] | tuple[torch.Tensor, ...] | None = None,
) -> torch.Tensor:
    if int(num_sketches) <= 0:
        raise ValueError("num_sketches must be positive")
    if int(num_sketches) > int(config.num_radial):
        raise ValueError("num_sketches cannot exceed config.num_radial")
    if moment not in {"vector", "quadrupole", "combined"}:
        raise ValueError("moment must be vector, quadrupole, or combined")

    if atom_weights is not None and len(atom_weights) != len(graphs):
        raise ValueError("atom_weights length must match the number of graphs")

    covariance = torch.zeros((int(config.num_radial), int(config.num_radial)), dtype=torch.float64)
    max_ell = 2 if moment in {"quadrupole", "combined"} else 1
    for graph_index, graph in enumerate(graphs):
        graph_atom_weights = None if atom_weights is None else atom_weights[graph_index]
        moments = compute_atomic_moments(graph, config, max_ell=max_ell)
        if moment in {"vector", "combined"}:
            vector = moments.get("vector")
            if vector is None:
                raise RuntimeError("vector moments were not computed")
            covariance = _accumulate_radial_covariance(covariance, vector, atom_weights=graph_atom_weights)
        if moment in {"quadrupole", "combined"}:
            quadrupole = moments.get("quadrupole")
            if quadrupole is None:
                raise RuntimeError("quadrupole moments were not computed")
            covariance = _accumulate_radial_covariance(covariance, quadrupole, atom_weights=graph_atom_weights)

    covariance = 0.5 * (covariance + covariance.T)
    eigvals, eigvecs = torch.linalg.eigh(covariance)
    order = torch.argsort(eigvals, descending=True)[: int(num_sketches)]
    projection = eigvecs[:, order].T.contiguous()
    for row in range(projection.shape[0]):
        pivot = int(torch.argmax(torch.abs(projection[row])).item())
        if projection[row, pivot] < 0:
            projection[row] = -projection[row]
    return projection


def force_magnitude_atom_weights(atoms_list: list[object] | tuple[object, ...], *, force_key: str) -> tuple[list[torch.Tensor], str]:
    weights = []
    for atoms in atoms_list:
        if force_key not in atoms.arrays:
            raise ValueError(f"force array {force_key!r} is not present in input configs")
        forces = torch.as_tensor(atoms.arrays[force_key], dtype=torch.float64)
        if tuple(forces.shape) != (len(atoms), 3):
            raise ValueError(f"force array {force_key!r} must have shape (num_atoms, 3)")
        weights.append(torch.linalg.norm(forces, dim=1))
    if not weights:
        raise ValueError("at least one config is required to build force-magnitude POD weights")
    all_weights = torch.cat(weights)
    mean_weight = all_weights.mean()
    if not torch.isfinite(mean_weight) or mean_weight <= 0:
        raise ValueError("force-magnitude POD weights must have a positive finite mean")
    normalized = [item / mean_weight for item in weights]
    return normalized, f"force_magnitude:{force_key}:mean1"


def _read_atoms(args: argparse.Namespace) -> list[object]:
    import ase.io

    index = ":" if args.limit_configs is None else f":{int(args.limit_configs)}"
    atoms_list = ase.io.read(str(args.configs), index=index)
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    return atoms_list


def _graphs_from_atoms(atoms_list: list[object], args: argparse.Namespace) -> list[RTECEGraph]:
    dtype = torch.float64 if args.default_dtype == "float64" else torch.float32
    return [
        atoms_to_rtece_graph(
            atoms,
            cutoff=float(args.cutoff),
            device="cpu",
            dtype=dtype,
            neighborlist_backend=args.neighborlist_backend,
        )
        for atoms in atoms_list
    ]


def _read_graphs(args: argparse.Namespace) -> list[RTECEGraph]:
    return _graphs_from_atoms(_read_atoms(args), args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-radial", type=int, default=8)
    parser.add_argument("--num-sketches", type=int, default=2)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--moment", choices=("vector", "quadrupole", "combined"), default="vector")
    parser.add_argument("--weight-mode", choices=("none", "force_magnitude"), default="none")
    parser.add_argument("--force-key", default="forces")
    parser.add_argument("--limit-configs", type=int, default=None)
    parser.add_argument("--neighborlist-backend", choices=("ase", "vesin", "matscipy"), default="matscipy")
    parser.add_argument("--default-dtype", choices=("float32", "float64"), default="float64")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RTECEScalarConfig(variant="atomic_cross_radial_pod", cutoff=args.cutoff, num_radial=args.num_radial)
    atoms_list = _read_atoms(args)
    graphs = _graphs_from_atoms(atoms_list, args)
    atom_weights = None
    weight_source = "none"
    if args.weight_mode == "force_magnitude":
        atom_weights, weight_source = force_magnitude_atom_weights(atoms_list, force_key=args.force_key)
    projection = compute_atomic_cross_radial_pod_projection(
        graphs,
        config,
        num_sketches=args.num_sketches,
        moment=args.moment,
        atom_weights=atom_weights,
    )
    payload = {
        "schema_version": "rtece_atomic_cross_radial_projection.v1",
        "projection_kind": "pod_fixed",
        "moment": args.moment,
        "num_radial": int(args.num_radial),
        "num_sketches": int(args.num_sketches),
        "weight_mode": args.weight_mode,
        "weight_source": weight_source,
        "projection_matrix": projection.tolist(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
