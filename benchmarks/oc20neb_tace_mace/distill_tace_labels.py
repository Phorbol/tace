#!/usr/bin/env python3
"""Write TACE teacher predictions into an extxyz distillation target file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


def log(message: str) -> None:
    print(f"[distill_tace_labels] {message}", file=sys.stderr, flush=True)


def copy_atoms_with_teacher_labels(
    atoms_list: Sequence,
    *,
    energies: np.ndarray,
    forces: np.ndarray,
    target_energy_key: str,
    target_forces_key: str,
    reference_prefix: str,
) -> list:
    """Return copied atoms with teacher labels as train targets.

    Existing target labels are preserved under ``reference_prefix`` before the
    teacher values overwrite the training keys.
    """

    energy_values = np.asarray(energies, dtype=np.float64).reshape(-1)
    force_values = np.asarray(forces, dtype=np.float64).reshape(-1, 3)
    if len(energy_values) != len(atoms_list):
        raise ValueError(
            f"energy count mismatch: got {len(energy_values)} predictions "
            f"for {len(atoms_list)} configurations"
        )
    expected_force_rows = sum(len(atoms) for atoms in atoms_list)
    if len(force_values) != expected_force_rows:
        raise ValueError(
            f"forces rows mismatch: got {len(force_values)} predictions "
            f"for {expected_force_rows} atoms"
        )

    labeled = []
    force_offset = 0
    for config_idx, atoms in enumerate(atoms_list):
        copied = atoms.copy()
        natoms = len(copied)
        if reference_prefix:
            reference_energy = None
            if target_energy_key in atoms.info:
                reference_energy = atoms.info[target_energy_key]
            elif atoms.calc is not None and target_energy_key in getattr(atoms.calc, "results", {}):
                reference_energy = atoms.calc.results[target_energy_key]
            else:
                try:
                    reference_energy = atoms.get_potential_energy()
                except Exception:
                    reference_energy = None
            if reference_energy is not None:
                copied.info[f"{reference_prefix}{target_energy_key}"] = float(reference_energy)

            reference_forces = None
            if target_forces_key in atoms.arrays:
                reference_forces = atoms.arrays[target_forces_key]
            elif atoms.calc is not None and target_forces_key in getattr(atoms.calc, "results", {}):
                reference_forces = atoms.calc.results[target_forces_key]
            else:
                try:
                    reference_forces = atoms.get_forces()
                except Exception:
                    reference_forces = None
            if reference_forces is not None:
                copied.arrays[f"{reference_prefix}{target_forces_key}"] = np.asarray(
                    reference_forces,
                    dtype=np.float64,
                ).copy()
        copied.info[target_energy_key] = float(energy_values[config_idx])
        copied.arrays[target_forces_key] = force_values[force_offset : force_offset + natoms].copy()
        force_offset += natoms
        labeled.append(copied)
    return labeled


def load_atoms(configs: Path, limit: int | None):
    import ase.io

    index = ":" if limit is None else f":{int(limit)}"
    atoms_list = ase.io.read(str(configs), index=index)
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    if not atoms_list:
        raise ValueError(f"no configurations read from {configs}")
    return atoms_list


def predict_tace(
    *,
    model_path: Path,
    atoms_list: Sequence,
    batch_size: int,
    device_name: str,
    default_dtype: str,
    neighborlist_backend: str,
    strict_load: bool,
    use_ema: bool,
) -> tuple[np.ndarray, np.ndarray]:
    import torch
    from torch_geometric.loader import DataLoader

    from tace.dataset.graph import from_atoms
    from tace.dataset.quantity import KEYS, PROPERTY, KeySpecification, update_keyspec_from_kwargs
    from tace.lightning import load_tace
    from tace.utils._global import DTYPE

    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    dtype = DTYPE[default_dtype]
    torch.set_default_dtype(dtype)
    key_spec = KeySpecification()
    update_keyspec_from_kwargs(key_spec, dict(KEYS))

    log(f"loading teacher {model_path}")
    model = load_tace(str(model_path), str(device), strict=strict_load, use_ema=use_ema)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    target_property = model.get_target_property()
    embedding_property = model.get_embedding_property()
    dataset = [
        from_atoms(
            model.get_torch_element(),
            atoms,
            model.get_cutoff(),
            max_neighbors="inf" if model.get_max_neighbors() is None else model.get_max_neighbors(),
            keyspec=key_spec,
            target_property=target_property,
            embedding_property=embedding_property,
            training=False,
            neighborlist_backend=neighborlist_backend,
        )
        for atoms in atoms_list
    ]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    energies: list[np.ndarray] = []
    forces: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        for target in target_property:
            for requires_grad_p in PROPERTY[target]["requires_grad_with"]:
                if hasattr(batch, requires_grad_p):
                    getattr(batch, requires_grad_p).requires_grad_(True)
        with torch.enable_grad():
            output = model(batch)
        energies.append(output["energy"].detach().cpu().numpy().reshape(-1))
        forces.append(output["forces"].detach().cpu().numpy().reshape(-1, 3))
    return np.concatenate(energies, axis=0), np.concatenate(forces, axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--limit-configs", type=int)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--default-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--target-energy-key", default="energy")
    parser.add_argument("--target-forces-key", default="forces")
    parser.add_argument("--reference-prefix", default="dft_")
    parser.add_argument("--nl-backend", choices=("ase", "vesin", "matscipy"), default="matscipy")
    parser.add_argument("--strict-load", type=int, default=1)
    parser.add_argument("--ema", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    import ase.io

    args = parse_args()
    atoms_list = load_atoms(args.input, args.limit_configs)
    log(f"read {len(atoms_list)} configs from {args.input}")
    energies, forces = predict_tace(
        model_path=args.model,
        atoms_list=atoms_list,
        batch_size=args.batch_size,
        device_name=args.device,
        default_dtype=args.default_dtype,
        neighborlist_backend=args.nl_backend,
        strict_load=bool(args.strict_load),
        use_ema=bool(args.ema),
    )
    labeled = copy_atoms_with_teacher_labels(
        atoms_list,
        energies=energies,
        forces=forces,
        target_energy_key=args.target_energy_key,
        target_forces_key=args.target_forces_key,
        reference_prefix=args.reference_prefix,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ase.io.write(str(args.output), labeled, format="extxyz")
    summary = {
        "model": str(args.model),
        "input": str(args.input),
        "output": str(args.output),
        "configs": len(labeled),
        "atoms": int(sum(len(atoms) for atoms in labeled)),
        "target_energy_key": args.target_energy_key,
        "target_forces_key": args.target_forces_key,
        "reference_prefix": args.reference_prefix,
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
