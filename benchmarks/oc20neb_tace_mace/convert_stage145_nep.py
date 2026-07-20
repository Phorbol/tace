#!/usr/bin/env python3
"""Convert stage145 extxyz data to GPUMD NEP train.xyz format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_atoms(path: Path, limit_configs: int | None):
    import ase.io

    index = ":" if limit_configs is None else f":{int(limit_configs)}"
    atoms_list = ase.io.read(str(path), index=index)
    return atoms_list if isinstance(atoms_list, list) else [atoms_list]


def _type_map(atoms_list) -> list[str]:
    return sorted({symbol for atoms in atoms_list for symbol in atoms.get_chemical_symbols()})


def _get_energy(atoms, key: str) -> float:
    if key in atoms.info:
        return float(atoms.info[key])
    if atoms.calc is not None and key in getattr(atoms.calc, "results", {}):
        return float(atoms.calc.results[key])
    if key == "energy":
        return float(atoms.get_potential_energy())
    raise KeyError(key)


def _get_forces(atoms, key: str) -> np.ndarray:
    if key in atoms.arrays:
        return np.asarray(atoms.arrays[key], dtype=np.float64)
    if atoms.calc is not None and key in getattr(atoms.calc, "results", {}):
        return np.asarray(atoms.calc.results[key], dtype=np.float64)
    if key in {"forces", "force"}:
        return np.asarray(atoms.get_forces(), dtype=np.float64)
    raise KeyError(key)


def _write_gpumd_extxyz(path: Path, atoms_list, *, energy_key: str, forces_key: str) -> None:
    import ase.io

    converted = []
    for atoms in atoms_list:
        item = atoms.copy()
        item.info.clear()
        item.info["energy"] = _get_energy(atoms, energy_key)
        item.arrays["force"] = _get_forces(atoms, forces_key)
        item.set_constraint()
        for key in list(item.arrays):
            if key not in {"numbers", "positions", "force"}:
                del item.arrays[key]
        converted.append(item)
    ase.io.write(path, converted, format="extxyz")


def convert_extxyz_to_nep(
    train_file: str | Path,
    output_dir: str | Path,
    *,
    limit_configs: int | None,
    energy_key: str = "energy",
    forces_key: str = "forces",
    generation: int = 20000,
) -> dict[str, Any]:
    train_path = Path(train_file)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    atoms_list = _read_atoms(train_path, limit_configs)
    if not atoms_list:
        raise ValueError(f"no configurations read from {train_path}")
    missing_energy = []
    missing_forces = []
    for idx, atoms in enumerate(atoms_list):
        try:
            _get_energy(atoms, energy_key)
        except Exception:
            missing_energy.append(idx)
        try:
            _get_forces(atoms, forces_key)
        except Exception:
            missing_forces.append(idx)
    if missing_energy:
        raise KeyError(f"missing energy key {energy_key!r} in configs {missing_energy[:5]}")
    if missing_forces:
        raise KeyError(f"missing forces key {forces_key!r} in configs {missing_forces[:5]}")
    types = _type_map(atoms_list)
    train_xyz = out / "train.xyz"
    _write_gpumd_extxyz(train_xyz, atoms_list, energy_key=energy_key, forces_key=forces_key)
    nep_in = out / "nep.in"
    nep_in.write_text(
        f"type         {len(types)} {' '.join(types)}\n"
        f"generation   {int(generation)}\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": "stage145_nep_conversion.v1",
        "engine": "nep",
        "train_file": str(train_path),
        "output_dir": str(out),
        "train_xyz": str(train_xyz),
        "nep_in": str(nep_in),
        "num_configs": len(atoms_list),
        "num_atoms": int(sum(len(atoms) for atoms in atoms_list)),
        "type_map": types,
        "energy_key": energy_key,
        "forces_key": forces_key,
        "generation": int(generation),
    }
    (out / "conversion_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--row", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.manifest.read_text())
    row = next(row for row in payload["rows"] if row["name"] == args.row)
    summary = convert_extxyz_to_nep(
        payload["train_contract"]["train_file"],
        row["train_dir"],
        limit_configs=payload["train_contract"]["limit_configs"],
        energy_key=payload["train_contract"]["energy_key"],
        forces_key=payload["train_contract"]["forces_key"],
        generation=row["generation"],
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
