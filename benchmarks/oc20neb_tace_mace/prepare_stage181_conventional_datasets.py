#!/usr/bin/env python3
"""Prepare conventional molecular MD datasets for Stage181 rTECE validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

KCAL_MOL_TO_EV = 0.0433641153

THREE_BPA_FILES: dict[str, dict[str, str]] = {
    "iso_atoms": {"filename": "iso_atoms.xyz", "role": "isolated_atom_reference"},
    "train_300K": {"filename": "train_300K.xyz", "role": "train_id_300K"},
    "train_mixedT": {"filename": "train_mixedT.xyz", "role": "train_mixed_temperature"},
    "test_300K": {"filename": "test_300K.xyz", "role": "id_test_300K"},
    "test_600K": {"filename": "test_600K.xyz", "role": "medium_temperature_ood"},
    "test_1200K": {"filename": "test_1200K.xyz", "role": "strong_temperature_ood"},
    "test_dih": {"filename": "test_dih.xyz", "role": "dihedral_pes_ood"},
}


def _require_key(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in data.files:
        raise KeyError(f"rMD17 npz is missing required key {key!r}; found {data.files}")
    return data[key]


def convert_rmd17_npz_to_extxyz(
    source_npz: str | Path,
    output_extxyz: str | Path,
    *,
    molecule: str | None = None,
    limit_configs: int | None = None,
) -> dict[str, Any]:
    """Convert one rMD17 npz file from kcal/mol units to ASE extxyz in eV units."""
    from ase import Atoms
    from ase.io import write

    source = Path(source_npz)
    out = Path(output_extxyz)
    if limit_configs is not None and int(limit_configs) < 1:
        raise ValueError("limit_configs must be positive when provided")
    with np.load(source) as data:
        z = np.asarray(_require_key(data, "nuclear_charges"), dtype=np.int64)
        coords = np.asarray(_require_key(data, "coords"), dtype=np.float64)
        energies = np.asarray(_require_key(data, "energies"), dtype=np.float64).reshape(-1)
        forces = np.asarray(_require_key(data, "forces"), dtype=np.float64)

    if coords.ndim != 3 or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape (n_configs, n_atoms, 3), got {coords.shape}")
    if forces.shape != coords.shape:
        raise ValueError(f"forces shape {forces.shape} must match coords shape {coords.shape}")
    if energies.shape[0] != coords.shape[0]:
        raise ValueError(f"energies length {energies.shape[0]} must match coords configs {coords.shape[0]}")
    if z.shape[0] != coords.shape[1]:
        raise ValueError(f"nuclear_charges length {z.shape[0]} must match coords atoms {coords.shape[1]}")

    count = coords.shape[0] if limit_configs is None else min(int(limit_configs), int(coords.shape[0]))
    mol = str(molecule or source.stem.removeprefix("rmd17_") or "unknown")
    atoms_list = []
    for idx in range(count):
        atoms = Atoms(numbers=z, positions=coords[idx])
        atoms.info["energy"] = float(energies[idx] * KCAL_MOL_TO_EV)
        atoms.info["dataset"] = "rMD17"
        atoms.info["molecule"] = mol
        atoms.info["source_frame"] = int(idx)
        atoms.info["energy_units"] = "eV"
        atoms.info["forces_units"] = "eV/A"
        atoms.arrays["forces"] = np.asarray(forces[idx] * KCAL_MOL_TO_EV, dtype=np.float64)
        atoms_list.append(atoms)

    out.parent.mkdir(parents=True, exist_ok=True)
    write(str(out), atoms_list, format="extxyz")
    return {
        "schema_version": "rtece_stage181_rmd17_conversion.v1",
        "dataset": "rMD17",
        "molecule": mol,
        "source_npz": str(source),
        "output_extxyz": str(out),
        "num_configs": int(count),
        "num_atoms": int(z.shape[0]),
        "source_units": {"energy": "kcal/mol", "forces": "kcal/mol/A", "distance": "A"},
        "target_units": {"energy": "eV", "forces": "eV/A", "distance": "A"},
        "conversion_factor_energy": float(KCAL_MOL_TO_EV),
        "conversion_factor_forces": float(KCAL_MOL_TO_EV),
        "labels": {"energy_key": "energy", "forces_key": "forces"},
    }


def make_3bpa_dataset_manifest(root: str | Path) -> dict[str, Any]:
    dataset_root = Path(root)
    splits = {}
    for split, spec in THREE_BPA_FILES.items():
        path = dataset_root / spec["filename"]
        splits[split] = {
            "filename": spec["filename"],
            "path": str(path),
            "role": spec["role"],
            "exists": bool(path.exists()),
        }
    return {
        "schema_version": "rtece_stage181_3bpa_dataset_manifest.v1",
        "dataset": "3BPA",
        "root": str(dataset_root),
        "source": {
            "repository": "https://github.com/davkovacs/BOTNet-datasets",
            "archive_url": "https://github.com/davkovacs/BOTNet-datasets/raw/refs/heads/main/dataset_3BPA.tar.gz",
        },
        "units": {"energy": "eV", "forces": "eV/A", "distance": "A"},
        "labels": {"energy_key": "energy", "forces_key": "forces"},
        "splits": splits,
    }


def _has_energy(atoms: Any) -> bool:
    if "energy" in getattr(atoms, "info", {}):
        return True
    try:
        float(atoms.get_potential_energy())
    except Exception:
        return False
    return True


def _has_forces(atoms: Any) -> bool:
    if "forces" in getattr(atoms, "arrays", {}):
        forces = np.asarray(atoms.arrays["forces"])
        return forces.ndim == 2 and forces.shape[1] == 3 and forces.shape[0] == len(atoms)
    try:
        forces = np.asarray(atoms.get_forces())
    except Exception:
        return False
    return forces.ndim == 2 and forces.shape[1] == 3 and forces.shape[0] == len(atoms)


def validate_3bpa_dataset(root: str | Path, *, max_configs_per_split: int | None = 4) -> dict[str, Any]:
    """Read 3BPA extxyz splits and verify the label/unit contract used by Stage181."""
    from ase.io import read

    dataset_root = Path(root)
    if max_configs_per_split is not None and int(max_configs_per_split) < 1:
        raise ValueError("max_configs_per_split must be positive when provided")

    split_reports: dict[str, dict[str, Any]] = {}
    failed: list[str] = []
    index = ":" if max_configs_per_split is None else f":{int(max_configs_per_split)}"
    for split, spec in THREE_BPA_FILES.items():
        path = dataset_root / spec["filename"]
        report: dict[str, Any] = {
            "filename": spec["filename"],
            "path": str(path),
            "role": spec["role"],
            "exists": bool(path.exists()),
            "num_configs_checked": 0,
            "num_atoms_first": 0,
            "has_energy": False,
            "has_forces": False,
        }
        if not path.exists():
            failed.append(f"{split}:missing_file")
            split_reports[split] = report
            continue
        try:
            frames = read(str(path), index=index)
        except Exception as exc:
            report["read_error"] = str(exc)
            failed.append(f"{split}:read_error")
            split_reports[split] = report
            continue
        if not isinstance(frames, list):
            frames = [frames]
        report["num_configs_checked"] = len(frames)
        if not frames:
            failed.append(f"{split}:empty")
            split_reports[split] = report
            continue
        first = frames[0]
        report["num_atoms_first"] = len(first)
        report["has_energy"] = _has_energy(first)
        report["has_forces"] = _has_forces(first)
        if not report["has_energy"]:
            failed.append(f"{split}:missing_energy")
        if split != "iso_atoms" and not report["has_forces"]:
            failed.append(f"{split}:missing_forces")
        split_reports[split] = report

    return {
        "schema_version": "rtece_stage181_3bpa_validation.v1",
        "dataset": "3BPA",
        "root": str(dataset_root),
        "contract_pass": not failed,
        "failed_checks": failed,
        "units": {"energy": "eV", "forces": "eV/A", "distance": "A"},
        "labels": {"energy_key": "energy", "forces_key": "forces"},
        "splits": split_reports,
    }


def make_rmd17_download_manifest(root: str | Path) -> dict[str, Any]:
    dataset_root = Path(root)
    molecules = [
        "aspirin",
        "azobenzene",
        "benzene",
        "ethanol",
        "malonaldehyde",
        "naphthalene",
        "paracetamol",
        "salicylic",
        "toluene",
        "uracil",
    ]
    return {
        "schema_version": "rtece_stage181_rmd17_download_manifest.v1",
        "dataset": "rMD17",
        "root": str(dataset_root),
        "source": {
            "figshare_article": "12672038",
            "version": 4,
            "archive_url": "https://figshare.com/ndownloader/articles/12672038/versions/4",
        },
        "source_units": {"energy": "kcal/mol", "forces": "kcal/mol/A", "distance": "A"},
        "target_units": {"energy": "eV", "forces": "eV/A", "distance": "A"},
        "conversion_factor_energy": float(KCAL_MOL_TO_EV),
        "conversion_factor_forces": float(KCAL_MOL_TO_EV),
        "molecules": [
            {
                "name": molecule,
                "npz": str(dataset_root / f"rmd17_{molecule}.npz"),
                "converted_extxyz": str(dataset_root / "converted_extxyz" / f"rmd17_{molecule}.extxyz"),
            }
            for molecule in molecules
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_rmd17 = sub.add_parser("rmd17-npz-to-extxyz")
    p_rmd17.add_argument("--source-npz", type=Path, required=True)
    p_rmd17.add_argument("--output-extxyz", type=Path, required=True)
    p_rmd17.add_argument("--molecule", default=None)
    p_rmd17.add_argument("--limit-configs", type=int, default=None)
    p_rmd17.add_argument("--summary-json", type=Path, default=None)
    p_3bpa = sub.add_parser("3bpa-manifest")
    p_3bpa.add_argument("--root", type=Path, required=True)
    p_3bpa.add_argument("--output-json", type=Path, required=True)
    p_3bpa_validate = sub.add_parser("3bpa-validate")
    p_3bpa_validate.add_argument("--root", type=Path, required=True)
    p_3bpa_validate.add_argument("--output-json", type=Path, default=None)
    p_3bpa_validate.add_argument("--max-configs-per-split", type=int, default=4)
    p_manifest = sub.add_parser("rmd17-manifest")
    p_manifest.add_argument("--root", type=Path, required=True)
    p_manifest.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "rmd17-npz-to-extxyz":
        payload = convert_rmd17_npz_to_extxyz(
            args.source_npz,
            args.output_extxyz,
            molecule=args.molecule,
            limit_configs=args.limit_configs,
        )
        if args.summary_json is not None:
            args.summary_json.parent.mkdir(parents=True, exist_ok=True)
            args.summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "3bpa-manifest":
        payload = make_3bpa_dataset_manifest(args.root)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "3bpa-validate":
        payload = validate_3bpa_dataset(args.root, max_configs_per_split=args.max_configs_per_split)
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))
    elif args.command == "rmd17-manifest":
        payload = make_rmd17_download_manifest(args.root)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
