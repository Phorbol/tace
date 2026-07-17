#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from benchmarks.oc20neb_tace_mace.rtece_scalar_model import (
    RTECEGraph,
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


def _energy_and_forces(atoms):
    if "energy" in atoms.info:
        energy = float(atoms.info["energy"])
    elif atoms.calc is not None and "energy" in getattr(atoms.calc, "results", {}):
        energy = float(atoms.calc.results["energy"])
    else:
        energy = float(atoms.get_potential_energy())

    if "forces" in atoms.arrays:
        forces = np.asarray(atoms.arrays["forces"], dtype=np.float64)
    elif atoms.calc is not None and "forces" in getattr(atoms.calc, "results", {}):
        forces = np.asarray(atoms.calc.results["forces"], dtype=np.float64)
    else:
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    return energy, forces


def atoms_to_graph(
    atoms,
    *,
    cutoff: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[RTECEGraph, torch.Tensor, torch.Tensor]:
    from ase.neighborlist import neighbor_list

    src, dst = neighbor_list("ij", atoms, cutoff)
    if len(src) == 0:
        edge_index_np = np.zeros((2, 0), dtype=np.int64)
    else:
        edge_index_np = np.stack([src, dst], axis=0)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long, device=device)
    z = torch.tensor(atoms.numbers, dtype=torch.long, device=device)
    pos = torch.tensor(atoms.positions, dtype=dtype, device=device)
    batch = torch.zeros(len(atoms), dtype=torch.long, device=device)
    energy_value, forces_value = _energy_and_forces(atoms)
    energy = torch.tensor([energy_value], dtype=dtype, device=device)
    forces = torch.tensor(forces_value, dtype=dtype, device=device)
    return RTECEGraph(z=z, pos=pos, edge_index=edge_index, batch=batch), energy, forces


def fit_energy_per_atom_shift(
    samples: list[tuple[RTECEGraph, torch.Tensor, torch.Tensor]],
) -> float:
    if not samples:
        raise ValueError("fit_energy_per_atom_shift requires at least one sample")
    total_energy = 0.0
    total_atoms = 0
    for graph, energy, _ in samples:
        total_energy += float(energy.detach().sum().cpu())
        total_atoms += int(graph.z.numel())
    if total_atoms <= 0:
        raise ValueError("cannot fit energy shift for zero atoms")
    return total_energy / float(total_atoms)


def loss_for_batch(
    model: RTECEScalarModel,
    graph: RTECEGraph,
    ref_energy: torch.Tensor,
    ref_forces: torch.Tensor,
) -> torch.Tensor:
    out = model(graph)
    natoms = graph.z.numel()
    e_loss = ((out["energy"] - ref_energy) / natoms).pow(2).mean()
    f_loss = (out["forces"] - ref_forces).pow(2).mean()
    return e_loss + 10.0 * f_loss


def train_steps(
    model: RTECEScalarModel,
    samples: list[tuple[RTECEGraph, torch.Tensor, torch.Tensor]],
    *,
    max_steps: int,
    lr: float,
) -> dict[str, float | int | None]:
    if not samples:
        raise ValueError("train_steps requires at least one sample")
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    final_loss: float | None = None
    for step in range(max_steps):
        graph, energy, forces = samples[step % len(samples)]
        opt.zero_grad(set_to_none=True)
        loss = loss_for_batch(model, graph, energy, forces)
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().cpu())
    return {"steps": max_steps, "final_loss": final_loss}


def load_samples(
    configs: Path,
    *,
    cutoff: float,
    device: torch.device,
    dtype: torch.dtype,
    limit_configs: int | None,
) -> list[tuple[RTECEGraph, torch.Tensor, torch.Tensor]]:
    import ase.io

    index = ":" if limit_configs is None else f":{int(limit_configs)}"
    atoms_list = ase.io.read(str(configs), index=index)
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    return [atoms_to_graph(atoms, cutoff=cutoff, device=device, dtype=dtype) for atoms in atoms_list]


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
    parser.add_argument("--no-fit-energy-shift", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float64 if args.default_dtype == "float64" else torch.float32
    requested = torch.device(args.device)
    device = requested if requested.type == "cpu" or torch.cuda.is_available() else torch.device("cpu")
    config = build_rtece_config(args.variant)
    samples = load_samples(
        args.train_file,
        cutoff=config.cutoff,
        device=device,
        dtype=dtype,
        limit_configs=args.limit_configs,
    )
    if not args.no_fit_energy_shift:
        config = replace(config, energy_per_atom_shift=fit_energy_per_atom_shift(samples))
    model = RTECEScalarModel(config).to(device=device, dtype=dtype)
    summary = train_steps(model, samples, max_steps=args.max_steps, lr=args.lr)
    summary.update(
        {
            "variant": args.variant,
            "train_file": str(args.train_file),
            "valid_file": str(args.valid_file),
            "train_configs": len(samples),
            "energy_per_atom_shift": config.energy_per_atom_shift,
            "device": str(device),
            "default_dtype": args.default_dtype,
            "checkpoint": str(args.output_dir / "rtece_scalar.pt"),
        }
    )
    save_checkpoint(args.output_dir / "rtece_scalar.pt", model, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
