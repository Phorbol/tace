from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np

from tace.models.rtece_protocol import sha256_file, sha256_json, write_canonical_json


STAGE186_DATA_SCHEMA = "rtece_stage186_data_manifest.v1"
_NAMED_TEST_FILES = (
    "test_300K.xyz",
    "test_600K.xyz",
    "test_1200K.xyz",
    "test_dih.xyz",
)


def choose_validation_window(source_hash: str, *, split_seed: int = 186) -> int:
    token = f"{int(split_seed)}|{source_hash}".encode("ascii")
    return int(hashlib.sha256(token).hexdigest(), 16) % 5


def split_stage186_indices(
    num_frames: int,
    source_hash: str,
    *,
    split_seed: int = 186,
    window_size: int = 100,
    embargo: int = 5,
) -> dict[str, list[int] | int]:
    if int(num_frames) != 500:
        raise ValueError("Stage186 3BPA protocol requires exactly 500 source frames")
    if int(window_size) != 100 or int(embargo) != 5:
        raise ValueError("Stage186 3BPA protocol requires window_size=100 and embargo=5")
    window = choose_validation_window(source_hash, split_seed=int(split_seed))
    valid_start = window * int(window_size)
    valid_stop = valid_start + int(window_size)
    validation_indices = list(range(valid_start, valid_stop))
    embargo_indices = [
        *range(max(0, valid_start - int(embargo)), valid_start),
        *range(valid_stop, min(int(num_frames), valid_stop + int(embargo))),
    ]
    excluded = set(validation_indices) | set(embargo_indices)
    training_indices = [index for index in range(int(num_frames)) if index not in excluded]
    return {
        "validation_window": int(window),
        "validation_start": int(valid_start),
        "validation_stop": int(valid_stop),
        "training_indices": training_indices,
        "validation_indices": validation_indices,
        "embargo_indices": embargo_indices,
    }


def _read_energy(atoms: Any, index: int) -> float:
    if "energy" in atoms.info:
        value = atoms.info["energy"]
    elif atoms.calc is not None and "energy" in getattr(atoms.calc, "results", {}):
        value = atoms.calc.results["energy"]
    else:
        raise KeyError(f"training frame {index} is missing energy")
    energy = float(value)
    if not math.isfinite(energy):
        raise ValueError(f"training frame {index} has non-finite energy")
    return energy


def _read_forces(atoms: Any, index: int) -> np.ndarray:
    if "forces" in atoms.arrays:
        values = atoms.arrays["forces"]
    elif atoms.calc is not None and "forces" in getattr(atoms.calc, "results", {}):
        values = atoms.calc.results["forces"]
    else:
        raise KeyError(f"training frame {index} is missing forces")
    forces = np.asarray(values, dtype=np.float64)
    if forces.shape != (len(atoms), 3) or not np.isfinite(forces).all():
        raise ValueError(f"training frame {index} has invalid forces")
    return forces


def _fit_training_e0(
    frames: list[Any],
    training_indices: list[int],
    *,
    ridge: float = 1.0e-12,
) -> tuple[dict[int, float], dict[str, Any]]:
    elements = sorted(
        {int(z) for index in training_indices for z in frames[index].numbers.tolist()}
    )
    design = np.asarray(
        [
            [float(np.count_nonzero(frames[index].numbers == z)) for z in elements]
            for index in training_indices
        ],
        dtype=np.float64,
    )
    targets = np.asarray(
        [_read_energy(frames[index], index) for index in training_indices],
        dtype=np.float64,
    )
    weights = np.asarray(
        [float(frames[index].info.get("energy_weight", 1.0)) for index in training_indices],
        dtype=np.float64,
    )
    if not np.isfinite(weights).all() or np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("energy weights must be finite, non-negative, and have positive sum")
    weighted_design = design * weights[:, None]
    lhs = design.T @ weighted_design
    if float(ridge) > 0.0:
        lhs = lhs + float(ridge) * np.eye(lhs.shape[0], dtype=np.float64)
    rhs = design.T @ (weights * targets)
    sqrt_weights = np.sqrt(weights)
    rank = int(np.linalg.matrix_rank(design * sqrt_weights[:, None]))
    if rank < len(elements):
        values = np.linalg.lstsq(
            design * sqrt_weights[:, None],
            targets * sqrt_weights,
            rcond=None,
        )[0]
        solver = "weighted_minimum_norm_lstsq_rank_deficient"
    else:
        solver = "weighted_normal_equation_solve"
        try:
            values = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            values = np.linalg.lstsq(
                design * sqrt_weights[:, None],
                targets * sqrt_weights,
                rcond=None,
            )[0]
            solver = "weighted_lstsq_fallback"
    residuals = design @ values - targets
    weighted_rmse = math.sqrt(float(np.sum(weights * residuals**2) / weights.sum()))
    atomic_energies = {
        int(z): float(value) for z, value in zip(elements, values, strict=True)
    }
    metadata = {
        "source": "training_indices_only",
        "method": "tace_weighted_least_squares",
        "solver": solver,
        "ridge": float(ridge),
        "solver_rank": rank,
        "elementwise_e0_identifiable": bool(rank == len(elements)),
        "num_elements": len(elements),
        "num_training_frames": len(training_indices),
        "weighted_residual_rmse_ev": float(weighted_rmse),
    }
    return atomic_energies, metadata


def _training_normalization(
    frames: list[Any],
    training_indices: list[int],
    atomic_energies: dict[int, float],
) -> dict[str, float | str]:
    residual_energy_per_atom = []
    force_components = []
    for index in training_indices:
        atoms = frames[index]
        baseline = sum(atomic_energies[int(z)] for z in atoms.numbers.tolist())
        residual_energy_per_atom.append(
            (_read_energy(atoms, index) - baseline) / float(len(atoms))
        )
        force_components.append(_read_forces(atoms, index).reshape(-1))
    energy_scale = max(float(np.std(residual_energy_per_atom)) * 1000.0, 1.0)
    all_force_components = np.concatenate(force_components)
    force_scale = max(
        math.sqrt(float(np.mean(all_force_components**2))) * 1000.0,
        1.0,
    )
    return {
        "source": "training_indices_only",
        "energy_definition": "std_e0_subtracted_energy_per_atom",
        "force_definition": "rms_force_components",
        "energy_scale_mev_atom": float(energy_scale),
        "force_scale_mev_a": float(force_scale),
        "energy_floor_mev_atom": 1.0,
        "force_floor_mev_a": 1.0,
    }


def materialize_stage186_3bpa_split(
    dataset_root: str | Path,
    output_root: str | Path,
    *,
    split_seed: int = 186,
) -> dict[str, Any]:
    import ase.io

    dataset_root = Path(dataset_root)
    output_root = Path(output_root)
    source_path = dataset_root / "train_300K.xyz"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    named_test_paths = {name: dataset_root / name for name in _NAMED_TEST_FILES}
    missing_tests = [name for name, path in named_test_paths.items() if not path.is_file()]
    if missing_tests:
        raise FileNotFoundError(dataset_root / missing_tests[0])

    source_hash = sha256_file(source_path)
    test_hashes_before = {
        name: sha256_file(path) for name, path in named_test_paths.items()
    }
    frames = ase.io.read(str(source_path), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    split = split_stage186_indices(
        len(frames), source_hash, split_seed=int(split_seed)
    )
    training_indices = list(split["training_indices"])
    validation_indices = list(split["validation_indices"])
    atomic_energies, e0_fit = _fit_training_e0(frames, training_indices)
    normalization = _training_normalization(
        frames, training_indices, atomic_energies
    )

    output_root.mkdir(parents=True, exist_ok=True)
    for index, atoms in enumerate(frames):
        atoms.info["stage186_source_index"] = int(index)
    train_path = output_root / "train.extxyz"
    valid_path = output_root / "valid.extxyz"
    ase.io.write(train_path, [frames[index] for index in training_indices], format="extxyz")
    ase.io.write(valid_path, [frames[index] for index in validation_indices], format="extxyz")

    e0_payload: dict[str, Any] = {
        "schema_version": "rtece_stage186_e0.v1",
        "source": "training_indices_only",
        "source_file_hash": source_hash,
        "training_indices": training_indices,
        "atomic_energies_ev": atomic_energies,
        **e0_fit,
    }
    e0_payload["e0_hash"] = sha256_json(e0_payload)
    e0_path = output_root / "e0.json"
    write_canonical_json(e0_path, e0_payload)

    test_hashes_after = {
        name: sha256_file(path) for name, path in named_test_paths.items()
    }
    touched = [
        name for name in _NAMED_TEST_FILES if test_hashes_before[name] != test_hashes_after[name]
    ]
    manifest: dict[str, Any] = {
        "schema_version": STAGE186_DATA_SCHEMA,
        "source_split": "train_300K.xyz",
        "source_file_hash": source_hash,
        "source_num_frames": len(frames),
        "split_seed": int(split_seed),
        **split,
        "named_test_file_hashes": test_hashes_before,
        "test_files_touched": touched,
        "units": {"distance": "A", "energy": "eV", "forces": "eV/A"},
        "e0_fit": e0_fit,
        "normalization": normalization,
        "artifacts": {
            "train": {"path": "train.extxyz", "sha256": sha256_file(train_path)},
            "valid": {"path": "valid.extxyz", "sha256": sha256_file(valid_path)},
            "e0": {"path": "e0.json", "sha256": sha256_file(e0_path)},
        },
    }
    manifest["data_manifest_hash"] = sha256_json(manifest)
    write_canonical_json(output_root / "data_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=186)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = materialize_stage186_3bpa_split(
        args.dataset_root,
        args.output_root,
        split_seed=int(args.split_seed),
    )
    print(manifest["data_manifest_hash"])


if __name__ == "__main__":
    main()
