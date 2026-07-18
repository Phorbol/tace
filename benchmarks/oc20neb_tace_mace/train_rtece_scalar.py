#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from tace.models.rtece_scalar import (
    RTECEGraph,
    RTECEScalarConfig,
    RTECEScalarModel,
    build_rtece_config,
)
from tace.models.rtece_workflow import (
    evaluate_loss,
    load_checkpoint as load_checkpoint_with_metadata,
    loss_for_batch,
    save_checkpoint,
    train_steps,
)


def set_training_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def parse_hidden_channels(value: str) -> tuple[int, ...]:
    channels = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not channels:
        raise ValueError("hidden channel list must contain at least one integer")
    if any(channel < 1 for channel in channels):
        raise ValueError(f"hidden channels must be positive, got {channels}")
    return channels


def load_checkpoint(
    path: str | Path,
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> tuple[RTECEScalarModel, RTECEScalarConfig]:
    model, config, _metadata = load_checkpoint_with_metadata(path, dtype=dtype, device=device)
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
            "rtece_element_density",
            "rtece_density_quadratic",
            "rtece_vector_moments",
            "rtece_atomic_moments",
            "rtece_edge_sketch8",
            "rtece_edge_sketch16",
        ),
    )
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-configs", type=int, default=None)
    parser.add_argument("--valid-limit-configs", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", default="64,64")
    parser.add_argument("--num-radial", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--force-weight", type=float, default=10.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--default-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--no-fit-energy-shift", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--disable-best-checkpoint", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_training_seed(args.seed)
    dtype = torch.float64 if args.default_dtype == "float64" else torch.float32
    requested = torch.device(args.device)
    device = requested if requested.type == "cpu" or torch.cuda.is_available() else torch.device("cpu")
    config = replace(
        build_rtece_config(args.variant),
        hidden_channels=parse_hidden_channels(args.hidden_channels),
        num_radial=int(args.num_radial),
    )
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
    valid_samples = None
    best_checkpoint_path = None
    if not args.disable_best_checkpoint and args.eval_interval > 0:
        valid_samples = load_samples(
            args.valid_file,
            cutoff=config.cutoff,
            device=device,
            dtype=dtype,
            limit_configs=args.valid_limit_configs,
        )
        best_checkpoint_path = args.output_dir / "rtece_scalar_best.pt"
    summary = train_steps(
        model,
        samples,
        max_steps=args.max_steps,
        lr=args.lr,
        valid_samples=valid_samples,
        eval_interval=args.eval_interval,
        best_checkpoint_path=best_checkpoint_path,
        config=config,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
    )
    summary.update(
        {
            "variant": args.variant,
            "train_file": str(args.train_file),
            "valid_file": str(args.valid_file),
            "train_configs": len(samples),
            "valid_configs": len(valid_samples) if valid_samples is not None else 0,
            "energy_per_atom_shift": config.energy_per_atom_shift,
            "hidden_channels": list(config.hidden_channels),
            "num_radial": config.num_radial,
            "seed": int(args.seed),
            "energy_weight": args.energy_weight,
            "force_weight": args.force_weight,
            "device": str(device),
            "default_dtype": args.default_dtype,
            "checkpoint": str(args.output_dir / "rtece_scalar.pt"),
            "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
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
