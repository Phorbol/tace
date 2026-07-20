#!/usr/bin/env python3
"""Convert stage145 extxyz data to DeepMD-kit numpy systems."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
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
        value = atoms.arrays[key]
    elif atoms.calc is not None and key in getattr(atoms.calc, "results", {}):
        value = atoms.calc.results[key]
    elif key in {"forces", "force"}:
        value = atoms.get_forces()
    else:
        raise KeyError(key)
    forces = np.asarray(value, dtype=np.float64)
    expected = (len(atoms), 3)
    if forces.shape != expected:
        raise ValueError(f"forces key {key!r} has shape {forces.shape}, expected {expected}")
    return forces


def _ordered_groups(atoms_list) -> OrderedDict[tuple[str, ...], list]:
    groups: OrderedDict[tuple[str, ...], list] = OrderedDict()
    for atoms in atoms_list:
        groups.setdefault(tuple(atoms.get_chemical_symbols()), []).append(atoms)
    return groups


def _system_name(index: int, total: int) -> str:
    return "mixed" if total == 1 else f"mixed_{index:03d}"


def _write_system_arrays(
    system_dir: Path,
    atoms_list,
    type_to_id: dict[str, int],
    type_map: list[str],
    *,
    energy_key: str,
    forces_key: str,
) -> dict[str, Any]:
    set_dir = system_dir / "set.000"
    set_dir.mkdir(parents=True, exist_ok=True)
    coords = []
    boxes = []
    energies = []
    forces = []
    symbols = atoms_list[0].get_chemical_symbols()
    natoms = len(atoms_list[0])
    for atoms in atoms_list:
        if atoms.get_chemical_symbols() != symbols:
            raise ValueError(f"system {system_dir.name} received mixed atom ordering")
        coords.append(np.asarray(atoms.positions, dtype=np.float64).reshape(-1))
        boxes.append(np.asarray(atoms.cell.array, dtype=np.float64).reshape(-1))
        energies.append(_get_energy(atoms, energy_key))
        forces.append(_get_forces(atoms, forces_key).reshape(-1))
    np.save(set_dir / "coord.npy", np.asarray(coords, dtype=np.float64))
    np.save(set_dir / "box.npy", np.asarray(boxes, dtype=np.float64))
    np.save(set_dir / "energy.npy", np.asarray(energies, dtype=np.float64))
    np.save(set_dir / "force.npy", np.asarray(forces, dtype=np.float64))
    type_ids = [type_to_id[symbol] for symbol in symbols]
    (system_dir / "type.raw").write_text("\n".join(str(value) for value in type_ids) + "\n", encoding="utf-8")
    (system_dir / "type_map.raw").write_text("\n".join(type_map) + "\n", encoding="utf-8")
    return {"name": system_dir.name, "num_configs": len(atoms_list), "num_atoms_per_config": natoms}


def _deepmd_input(type_map: list[str], systems: list[str], *, stop_batch: int) -> dict[str, Any]:
    return {
        "model": {
            "type_map": type_map,
            "descriptor": {
                "type": "se_atten",
                "rcut": 5.0,
                "rcut_smth": 4.5,
                "sel": 128,
                "neuron": [16, 32, 64],
                "axis_neuron": 16,
                "attn": 16,
                "attn_layer": 0,
                "set_davg_zero": True,
            },
            "fitting_net": {"neuron": [64, 64, 64], "resnet_dt": True},
        },
        "learning_rate": {"type": "exp", "start_lr": 0.001, "stop_lr": 1.0e-8, "decay_steps": 5000},
        "loss": {
            "type": "ener",
            "start_pref_e": 0.02,
            "limit_pref_e": 1.0,
            "start_pref_f": 1000.0,
            "limit_pref_f": 1.0,
        },
        "training": {
            "training_data": {"systems": systems, "batch_size": "auto"},
            "validation_data": {"systems": systems, "batch_size": "auto", "numb_btch": 1},
            "numb_steps": int(stop_batch),
            "seed": 145,
            "disp_file": "lcurve.out",
            "disp_freq": 100,
            "save_freq": 1000,
        },
    }


def convert_extxyz_to_deepmd(
    train_file: str | Path,
    output_dir: str | Path,
    *,
    limit_configs: int | None,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stop_batch: int = 20000,
) -> dict[str, Any]:
    train_path = Path(train_file)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    atoms_list = _read_atoms(train_path, limit_configs)
    if not atoms_list:
        raise ValueError(f"no configurations read from {train_path}")
    types = _type_map(atoms_list)
    type_to_id = {symbol: idx for idx, symbol in enumerate(types)}
    groups = _ordered_groups(atoms_list)
    systems = []
    system_summaries = []
    for index, group_atoms in enumerate(groups.values()):
        name = _system_name(index, len(groups))
        systems.append(name)
        system_summaries.append(
            _write_system_arrays(
                out / name, group_atoms, type_to_id, types, energy_key=energy_key, forces_key=forces_key
            )
        )
    (out / "type_map.raw").write_text("\n".join(types) + "\n", encoding="utf-8")
    input_payload = _deepmd_input(types, systems, stop_batch=stop_batch)
    (out / "input.json").write_text(json.dumps(input_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "stage145_deepmd_conversion.v1",
        "engine": "deepmd",
        "descriptor_label": "dpa1_zero_attention",
        "train_file": str(train_path),
        "output_dir": str(out),
        "num_configs": len(atoms_list),
        "num_atoms": int(sum(len(atoms) for atoms in atoms_list)),
        "num_systems": len(systems),
        "systems": systems,
        "system_summaries": system_summaries,
        "type_map": types,
        "energy_key": energy_key,
        "forces_key": forces_key,
        "stop_batch": int(stop_batch),
        "split_by_atom_order": len(systems) > 1,
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
    summary = convert_extxyz_to_deepmd(
        payload["train_contract"]["train_file"],
        row["train_dir"],
        limit_configs=payload["train_contract"]["limit_configs"],
        energy_key=payload["train_contract"]["energy_key"],
        forces_key=payload["train_contract"]["forces_key"],
        stop_batch=row["stop_batch"],
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
