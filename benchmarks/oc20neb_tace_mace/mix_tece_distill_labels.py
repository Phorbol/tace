#!/usr/bin/env python3
"""Mix teacher labels with preserved DFT labels in TECE distillation extxyz files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def _validate_weight(teacher_weight: float) -> float:
    weight = float(teacher_weight)
    if weight < 0.0 or weight > 1.0:
        raise ValueError(f"teacher_weight must be in [0, 1], got {teacher_weight!r}")
    return weight


def _read_energy(atoms, key: str, config_idx: int, label_name: str) -> float:
    if key in atoms.info:
        return float(atoms.info[key])
    if atoms.calc is not None and key in getattr(atoms.calc, "results", {}):
        return float(atoms.calc.results[key])
    if key == "energy":
        try:
            return float(atoms.get_potential_energy())
        except Exception:
            pass
    raise KeyError(f"configuration {config_idx} missing {label_name} energy key {key!r}")


def _read_forces(atoms, key: str, config_idx: int, label_name: str) -> np.ndarray:
    if key in atoms.arrays:
        return np.asarray(atoms.arrays[key], dtype=np.float64)
    if atoms.calc is not None and key in getattr(atoms.calc, "results", {}):
        return np.asarray(atoms.calc.results[key], dtype=np.float64)
    if key == "forces":
        try:
            return np.asarray(atoms.get_forces(), dtype=np.float64)
        except Exception:
            pass
    raise KeyError(f"configuration {config_idx} missing {label_name} forces key {key!r}")


def mix_atoms_labels(
    atoms_list: Sequence,
    *,
    teacher_weight: float,
    target_energy_key: str = "energy",
    target_forces_key: str = "forces",
    dft_energy_key: str = "dft_energy",
    dft_forces_key: str = "dft_forces",
    teacher_prefix: str = "teacher_",
) -> list:
    """Return atoms whose train targets are teacher/DFT convex combinations.

    ``distill_tace_labels.py`` stores teacher labels in the normal training keys
    and preserves original labels under ``dft_*``. This helper keeps both source
    label sets and rewrites the training keys to
    ``teacher_weight * teacher + (1 - teacher_weight) * dft``.
    """

    weight = _validate_weight(teacher_weight)
    mixed = []
    for config_idx, atoms in enumerate(atoms_list):
        copied = atoms.copy()
        teacher_energy = _read_energy(atoms, target_energy_key, config_idx, "teacher")
        dft_energy = _read_energy(atoms, dft_energy_key, config_idx, "DFT")
        teacher_forces = _read_forces(atoms, target_forces_key, config_idx, "teacher")
        dft_forces = _read_forces(atoms, dft_forces_key, config_idx, "DFT")
        if teacher_forces.shape != dft_forces.shape:
            raise ValueError(
                f"configuration {config_idx} force shape mismatch: "
                f"teacher {teacher_forces.shape}, dft {dft_forces.shape}"
            )

        if teacher_prefix:
            copied.info[f"{teacher_prefix}{target_energy_key}"] = teacher_energy
            copied.arrays[f"{teacher_prefix}{target_forces_key}"] = teacher_forces.copy()
        copied.info[target_energy_key] = weight * teacher_energy + (1.0 - weight) * dft_energy
        copied.arrays[target_forces_key] = weight * teacher_forces + (1.0 - weight) * dft_forces
        mixed.append(copied)
    return mixed


def load_atoms(configs: Path, limit: int | None):
    import ase.io

    index = ":" if limit is None else f":{int(limit)}"
    atoms_list = ase.io.read(str(configs), index=index)
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    return atoms_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--limit-configs", type=int)
    parser.add_argument("--teacher-weight", type=float, default=0.75)
    parser.add_argument("--target-energy-key", default="energy")
    parser.add_argument("--target-forces-key", default="forces")
    parser.add_argument("--dft-energy-key", default="dft_energy")
    parser.add_argument("--dft-forces-key", default="dft_forces")
    parser.add_argument("--teacher-prefix", default="teacher_")
    return parser.parse_args()


def main() -> None:
    import ase.io

    args = parse_args()
    atoms_list = load_atoms(args.input, args.limit_configs)
    mixed = mix_atoms_labels(
        atoms_list,
        teacher_weight=args.teacher_weight,
        target_energy_key=args.target_energy_key,
        target_forces_key=args.target_forces_key,
        dft_energy_key=args.dft_energy_key,
        dft_forces_key=args.dft_forces_key,
        teacher_prefix=args.teacher_prefix,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ase.io.write(str(args.output), mixed, format="extxyz")
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "configs": len(mixed),
        "atoms": int(sum(len(atoms) for atoms in mixed)),
        "teacher_weight": float(args.teacher_weight),
        "dft_weight": float(1.0 - args.teacher_weight),
        "target_energy_key": args.target_energy_key,
        "target_forces_key": args.target_forces_key,
        "dft_energy_key": args.dft_energy_key,
        "dft_forces_key": args.dft_forces_key,
    }
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
