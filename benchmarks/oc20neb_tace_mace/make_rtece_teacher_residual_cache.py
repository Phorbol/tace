#!/usr/bin/env python3
"""Build schema-versioned teacher residual E/F caches for rTECE distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

UNITS = {"energy": "eV", "forces": "eV/A", "distance": "A"}


def _round_float_array(values: Any, *, decimals: int = 12) -> list:
    arr = np.asarray(values, dtype=np.float64)
    return np.round(arr, decimals=decimals).tolist()


def structure_hash(atoms) -> str:
    payload = {
        "numbers": [int(z) for z in atoms.get_atomic_numbers()],
        "positions_A": _round_float_array(atoms.get_positions()),
        "cell_A": _round_float_array(atoms.get_cell().array),
        "pbc": [bool(v) for v in atoms.get_pbc()],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "sha256:" + hashlib.sha256(blob).hexdigest()


def _read_energy(atoms, *, key: str = "energy") -> float:
    if key in atoms.info:
        return float(atoms.info[key])
    if atoms.calc is not None and key in getattr(atoms.calc, "results", {}):
        return float(atoms.calc.results[key])
    if key == "energy":
        return float(atoms.get_potential_energy())
    raise KeyError(f"atoms object is missing energy key {key!r}")


def _read_forces(atoms, *, key: str = "forces") -> np.ndarray:
    if key in atoms.arrays:
        forces = np.asarray(atoms.arrays[key], dtype=np.float64)
    elif atoms.calc is not None and key in getattr(atoms.calc, "results", {}):
        forces = np.asarray(atoms.calc.results[key], dtype=np.float64)
    elif key == "forces":
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    else:
        raise KeyError(f"atoms object is missing force key {key!r}")
    if forces.shape != (len(atoms), 3):
        raise ValueError(f"force key {key!r} has shape {forces.shape}, expected {(len(atoms), 3)}")
    return forces


def make_teacher_residual_cache_record(
    dft_atoms,
    teacher_atoms,
    *,
    index: int,
    teacher_model_hash: str,
    cutoff: float,
    pbc_convention: str,
    dft_energy_key: str = "energy",
    dft_forces_key: str = "forces",
    teacher_energy_key: str = "energy",
    teacher_forces_key: str = "forces",
) -> dict[str, Any]:
    if not str(teacher_model_hash).strip():
        raise ValueError("teacher_model_hash is required")
    source_hash = structure_hash(dft_atoms)
    teacher_hash = structure_hash(teacher_atoms)
    if teacher_hash != source_hash:
        raise ValueError(f"structure hash mismatch at index {index}: DFT {source_hash} vs teacher {teacher_hash}")
    dft_energy = _read_energy(dft_atoms, key=dft_energy_key)
    teacher_energy = _read_energy(teacher_atoms, key=teacher_energy_key)
    dft_forces = _read_forces(dft_atoms, key=dft_forces_key)
    teacher_forces = _read_forces(teacher_atoms, key=teacher_forces_key)
    force_residual = teacher_forces - dft_forces
    return {
        "schema_version": "rtece_teacher_residual_cache_record.v1",
        "index": int(index),
        "teacher_model_hash": str(teacher_model_hash),
        "units": dict(UNITS),
        "cutoff": float(cutoff),
        "pbc_convention": str(pbc_convention),
        "source_structure_hash": source_hash,
        "teacher_structure_hash": teacher_hash,
        "num_atoms": int(len(dft_atoms)),
        "atomic_numbers": [int(z) for z in dft_atoms.get_atomic_numbers()],
        "dft_energy_eV": float(dft_energy),
        "teacher_energy_eV": float(teacher_energy),
        "energy_residual_eV": float(teacher_energy - dft_energy),
        "force_residual_eV_per_A": force_residual.tolist(),
        "dft_energy_key": str(dft_energy_key),
        "dft_forces_key": str(dft_forces_key),
        "teacher_energy_key": str(teacher_energy_key),
        "teacher_forces_key": str(teacher_forces_key),
    }


def build_teacher_residual_cache_records(
    dft_atoms_list: list,
    teacher_atoms_list: list,
    *,
    teacher_model_hash: str,
    cutoff: float,
    pbc_convention: str,
    dft_energy_key: str = "energy",
    dft_forces_key: str = "forces",
    teacher_energy_key: str = "energy",
    teacher_forces_key: str = "forces",
) -> list[dict[str, Any]]:
    if len(dft_atoms_list) != len(teacher_atoms_list):
        raise ValueError(f"DFT and teacher lists must have the same length, got {len(dft_atoms_list)} and {len(teacher_atoms_list)}")
    return [
        make_teacher_residual_cache_record(
            dft_atoms,
            teacher_atoms,
            index=idx,
            teacher_model_hash=teacher_model_hash,
            cutoff=cutoff,
            pbc_convention=pbc_convention,
            dft_energy_key=dft_energy_key,
            dft_forces_key=dft_forces_key,
            teacher_energy_key=teacher_energy_key,
            teacher_forces_key=teacher_forces_key,
        )
        for idx, (dft_atoms, teacher_atoms) in enumerate(zip(dft_atoms_list, teacher_atoms_list, strict=True))
    ]


def write_teacher_residual_cache_jsonl(records: list[dict[str, Any]], output_jsonl: str | Path) -> dict[str, Any]:
    target = Path(output_jsonl)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    summary = {
        "schema_version": "rtece_teacher_residual_cache_summary.v1",
        "cache_jsonl": str(target),
        "num_records": int(len(records)),
        "teacher_model_hash": records[0]["teacher_model_hash"] if records else None,
        "units": dict(UNITS),
        "energy_residual_rmse_eV": None,
        "force_residual_rmse_eV_per_A": None,
    }
    if records:
        e = np.asarray([row["energy_residual_eV"] for row in records], dtype=np.float64)
        f = np.concatenate([np.asarray(row["force_residual_eV_per_A"], dtype=np.float64).reshape(-1) for row in records])
        summary["energy_residual_rmse_eV"] = float(np.sqrt(np.mean(e * e)))
        summary["force_residual_rmse_eV_per_A"] = float(np.sqrt(np.mean(f * f)))
    summary_path = target.with_suffix(target.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["summary_json"] = str(summary_path)
    return summary


def _read_extxyz(path: Path, *, limit: int | None):
    import ase.io

    index = ":" if limit is None else f":{int(limit)}"
    atoms = ase.io.read(str(path), index=index)
    return atoms if isinstance(atoms, list) else [atoms]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dft-file", type=Path, required=True)
    parser.add_argument("--teacher-file", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--teacher-model-hash", required=True)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--pbc-convention", default="ase_cell_edge_shifts")
    parser.add_argument("--limit-configs", type=int, default=None)
    parser.add_argument("--dft-energy-key", default="energy")
    parser.add_argument("--dft-forces-key", default="forces")
    parser.add_argument("--teacher-energy-key", default="energy")
    parser.add_argument("--teacher-forces-key", default="forces")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dft_atoms = _read_extxyz(args.dft_file, limit=args.limit_configs)
    teacher_atoms = _read_extxyz(args.teacher_file, limit=args.limit_configs)
    records = build_teacher_residual_cache_records(
        dft_atoms,
        teacher_atoms,
        teacher_model_hash=args.teacher_model_hash,
        cutoff=args.cutoff,
        pbc_convention=args.pbc_convention,
        dft_energy_key=args.dft_energy_key,
        dft_forces_key=args.dft_forces_key,
        teacher_energy_key=args.teacher_energy_key,
        teacher_forces_key=args.teacher_forces_key,
    )
    summary = write_teacher_residual_cache_jsonl(records, args.output_jsonl)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
