#!/usr/bin/env python3
"""Generate teacher-labeled relaxation-trajectory configs for rTECE distillation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.oc20neb_tace_mace.rattle_relax_rtece import deterministic_rattle_atoms


def _reference_energy(atoms, target_energy_key: str) -> float | None:
    if target_energy_key in atoms.info:
        return float(atoms.info[target_energy_key])
    if atoms.calc is not None and target_energy_key in getattr(atoms.calc, "results", {}):
        return float(atoms.calc.results[target_energy_key])
    return None


def _reference_forces(atoms, target_forces_key: str) -> np.ndarray | None:
    if target_forces_key in atoms.arrays:
        return np.asarray(atoms.arrays[target_forces_key], dtype=np.float64).copy()
    if atoms.calc is not None and target_forces_key in getattr(atoms.calc, "results", {}):
        return np.asarray(atoms.calc.results[target_forces_key], dtype=np.float64).copy()
    return None


def _copy_teacher_labeled_frame(
    atoms,
    *,
    reference_atoms,
    source_config_index: int,
    copy_index: int,
    step: int,
    rattle_seed: int,
    rattle_std_a: float,
    target_energy_key: str,
    target_forces_key: str,
    reference_prefix: str,
):
    energy = float(atoms.get_potential_energy())
    forces = np.asarray(atoms.get_forces(), dtype=np.float64).reshape(len(atoms), 3)
    copied = atoms.copy()
    copied.info[target_energy_key] = energy
    copied.arrays[target_forces_key] = forces.copy()
    if reference_prefix:
        ref_e = _reference_energy(reference_atoms, target_energy_key)
        if ref_e is not None:
            copied.info[f"{reference_prefix}{target_energy_key}"] = ref_e
        ref_f = _reference_forces(reference_atoms, target_forces_key)
        if ref_f is not None:
            copied.arrays[f"{reference_prefix}{target_forces_key}"] = ref_f
    copied.info["teacher_relax_source_config_index"] = int(source_config_index)
    copied.info["teacher_relax_copy_index"] = int(copy_index)
    copied.info["teacher_relax_step"] = int(step)
    copied.info["teacher_relax_rattle_seed"] = int(rattle_seed)
    copied.info["teacher_relax_rattle_std_a"] = float(rattle_std_a)
    copied.info["teacher_relax_fixed_length"] = True
    return copied


def make_teacher_relax_distill_configs(
    atoms_list: Sequence,
    *,
    calculator,
    source_start_config: int = 0,
    copies_per_config: int = 2,
    rattle_std_a: float = 0.05,
    seed: int = 20260720,
    relax_max_steps: int = 4,
    target_energy_key: str = "energy",
    target_forces_key: str = "forces",
    reference_prefix: str = "source_",
) -> tuple[list, dict[str, Any]]:
    """Return fixed-length LBFGS teacher trajectory frames with E/F train labels.

    The trajectory is intentionally fixed length: for each source config and each
    rattle copy, this records the initial teacher-labeled rattled structure and
    exactly ``relax_max_steps`` LBFGS steps. That gives deterministic dataset
    cardinality for downstream Slurm wrappers and makes stage-to-stage ablations
    comparable.
    """

    if int(source_start_config) < 0:
        raise ValueError("source_start_config must be non-negative")
    if int(copies_per_config) < 1:
        raise ValueError("copies_per_config must be positive")
    if float(rattle_std_a) < 0.0:
        raise ValueError("rattle_std_a must be non-negative")
    if int(relax_max_steps) < 0:
        raise ValueError("relax_max_steps must be non-negative")

    from ase.optimize import LBFGS

    frames: list = []
    records: list[dict[str, Any]] = []
    for local_idx, source_atoms in enumerate(atoms_list):
        source_config_index = int(source_start_config) + int(local_idx)
        for copy_index in range(int(copies_per_config)):
            rattle_seed = int(seed) + source_config_index * 1009 + int(copy_index)
            work = deterministic_rattle_atoms(source_atoms, rattle_std_a=float(rattle_std_a), seed=rattle_seed)
            work.calc = calculator
            frames.append(
                _copy_teacher_labeled_frame(
                    work,
                    reference_atoms=source_atoms,
                    source_config_index=source_config_index,
                    copy_index=copy_index,
                    step=0,
                    rattle_seed=rattle_seed,
                    rattle_std_a=float(rattle_std_a),
                    target_energy_key=target_energy_key,
                    target_forces_key=target_forces_key,
                    reference_prefix=reference_prefix,
                )
            )
            optimizer = LBFGS(work, logfile=None)
            for step in range(1, int(relax_max_steps) + 1):
                optimizer.step()
                frames.append(
                    _copy_teacher_labeled_frame(
                        work,
                        reference_atoms=source_atoms,
                        source_config_index=source_config_index,
                        copy_index=copy_index,
                        step=step,
                        rattle_seed=rattle_seed,
                        rattle_std_a=float(rattle_std_a),
                        target_energy_key=target_energy_key,
                        target_forces_key=target_forces_key,
                        reference_prefix=reference_prefix,
                    )
                )
            records.append(
                {
                    "source_config_index": source_config_index,
                    "copy_index": int(copy_index),
                    "rattle_seed": int(rattle_seed),
                    "frames": int(relax_max_steps) + 1,
                    "atoms": int(len(source_atoms)),
                }
            )

    summary = {
        "schema_version": "teacher_relax_distill_configs.v1",
        "source_configs": int(len(atoms_list)),
        "source_start_config": int(source_start_config),
        "copies_per_config": int(copies_per_config),
        "rattle_std_a": float(rattle_std_a),
        "seed": int(seed),
        "relax_max_steps": int(relax_max_steps),
        "frames_per_copy": int(relax_max_steps) + 1,
        "configs": int(len(frames)),
        "atoms": int(sum(len(frame) for frame in frames)),
        "target_energy_key": str(target_energy_key),
        "target_forces_key": str(target_forces_key),
        "reference_prefix": str(reference_prefix),
        "records": records,
    }
    return frames, summary


def _load_atoms(configs: Path, start: int, limit: int):
    import ase.io

    if int(start) < 0:
        raise ValueError("--start-config must be non-negative")
    if int(limit) < 1:
        raise ValueError("--limit-configs must be positive")
    atoms = ase.io.read(str(configs), index=f"{int(start)}:{int(start) + int(limit)}")
    if not isinstance(atoms, list):
        atoms = [atoms]
    if not atoms:
        raise ValueError(f"no configurations read from {configs}")
    return atoms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--start-config", type=int, default=0)
    parser.add_argument("--limit-configs", type=int, default=32)
    parser.add_argument("--copies-per-config", type=int, default=2)
    parser.add_argument("--rattle-std-a", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--relax-max-steps", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--default-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--nl-backend", choices=("ase", "vesin", "matscipy"), default="matscipy")
    parser.add_argument("--target-energy-key", default="energy")
    parser.add_argument("--target-forces-key", default="forces")
    parser.add_argument("--reference-prefix", default="source_")
    return parser.parse_args()


def main() -> None:
    import ase.io

    args = parse_args()
    atoms_list = _load_atoms(args.input, args.start_config, args.limit_configs)
    from tace.interface.ase import TACEAseCalc

    calculator = TACEAseCalc(
        str(args.model),
        dtype=args.default_dtype,
        device=args.device,
        neighborlist_backend=args.nl_backend,
    )
    frames, summary = make_teacher_relax_distill_configs(
        atoms_list,
        calculator=calculator,
        source_start_config=args.start_config,
        copies_per_config=args.copies_per_config,
        rattle_std_a=args.rattle_std_a,
        seed=args.seed,
        relax_max_steps=args.relax_max_steps,
        target_energy_key=args.target_energy_key,
        target_forces_key=args.target_forces_key,
        reference_prefix=args.reference_prefix,
    )
    summary.update({"model": str(args.model), "input": str(args.input), "output": str(args.output)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ase.io.write(str(args.output), frames, format="extxyz")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
