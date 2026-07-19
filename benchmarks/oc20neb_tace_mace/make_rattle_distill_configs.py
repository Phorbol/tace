#!/usr/bin/env python3
"""Generate rattled structures for teacher-label distillation coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.oc20neb_tace_mace.rattle_relax_rtece import deterministic_rattle_atoms, positions_rmsd


def _read_standard_energy(atoms, config_idx: int) -> float:
    if "energy" in atoms.info:
        return float(atoms.info["energy"])
    if atoms.calc is not None and "energy" in getattr(atoms.calc, "results", {}):
        return float(atoms.calc.results["energy"])
    raise KeyError(f"configuration {config_idx} missing standard energy target 'energy'")


def _read_standard_forces(atoms, config_idx: int) -> np.ndarray:
    if "forces" in atoms.arrays:
        return np.asarray(atoms.arrays["forces"], dtype=np.float64)
    if atoms.calc is not None and "forces" in getattr(atoms.calc, "results", {}):
        return np.asarray(atoms.calc.results["forces"], dtype=np.float64)
    raise KeyError(f"configuration {config_idx} missing standard force target 'forces'")


def make_rattle_distill_configs(
    atoms_list: Sequence,
    *,
    copies_per_config: int,
    rattle_std_a: float,
    seed: int,
) -> tuple[list, dict]:
    """Return rattled copies that keep the existing standard training labels.

    The subsequent teacher-label step should overwrite ``energy``/``forces``;
    keeping the current labels here makes the intermediate file valid for the
    same production rTECE training reader and lets ``distill_tace_labels.py``
    preserve the pre-teacher labels under ``dft_*`` when they exist.
    """

    num_copies = int(copies_per_config)
    if num_copies < 1:
        raise ValueError("copies_per_config must be positive")
    if float(rattle_std_a) < 0.0:
        raise ValueError("rattle_std_a must be non-negative")
    if not atoms_list:
        raise ValueError("at least one source configuration is required")

    rattled = []
    rmsd_values: list[float] = []
    total_atoms = 0
    for config_idx, atoms in enumerate(atoms_list):
        energy = _read_standard_energy(atoms, config_idx)
        forces = _read_standard_forces(atoms, config_idx)
        if forces.shape != (len(atoms), 3):
            raise ValueError(
                f"configuration {config_idx} forces must have shape ({len(atoms)}, 3), got {forces.shape}"
            )
        for copy_idx in range(num_copies):
            rattle_seed = int(seed) + config_idx * num_copies + copy_idx
            copied = deterministic_rattle_atoms(atoms, rattle_std_a=float(rattle_std_a), seed=rattle_seed)
            initial_rmsd = positions_rmsd(copied.positions, atoms.positions)
            copied.info["energy"] = energy
            copied.arrays["forces"] = forces.copy()
            copied.info["rattle_source_config_index"] = int(config_idx)
            copied.info["rattle_copy_index"] = int(copy_idx)
            copied.info["rattle_seed"] = int(rattle_seed)
            copied.info["rattle_std_a"] = float(rattle_std_a)
            copied.info["rattle_initial_rmsd_a"] = float(initial_rmsd)
            if "case_id" in atoms.info:
                copied.info["rattle_source_case_id"] = str(atoms.info["case_id"])
            rattled.append(copied)
            rmsd_values.append(float(initial_rmsd))
            total_atoms += len(copied)

    summary = {
        "schema_version": "rtece_rattle_distill_configs.v1",
        "source_configs": int(len(atoms_list)),
        "copies_per_config": int(num_copies),
        "configs": int(len(rattled)),
        "atoms": int(total_atoms),
        "rattle_std_a": float(rattle_std_a),
        "seed": int(seed),
        "mean_initial_rmsd_a": float(np.mean(rmsd_values)) if rmsd_values else None,
        "max_initial_rmsd_a": float(np.max(rmsd_values)) if rmsd_values else None,
    }
    return rattled, summary


def load_atoms(configs: Path, limit: int | None):
    import ase.io

    index = ":" if limit is None else f":{int(limit)}"
    atoms_list = ase.io.read(str(configs), index=index)
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    if not atoms_list:
        raise ValueError(f"no configurations read from {configs}")
    return atoms_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--limit-configs", type=int)
    parser.add_argument("--copies-per-config", type=int, default=1)
    parser.add_argument("--rattle-std-a", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    import ase.io

    args = parse_args()
    source = load_atoms(args.input, args.limit_configs)
    rattled, summary = make_rattle_distill_configs(
        source,
        copies_per_config=args.copies_per_config,
        rattle_std_a=args.rattle_std_a,
        seed=args.seed,
    )
    summary.update({"input": str(args.input), "output": str(args.output)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ase.io.write(str(args.output), rattled, format="extxyz")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
