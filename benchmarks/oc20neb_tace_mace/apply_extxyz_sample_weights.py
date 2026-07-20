#!/usr/bin/env python3
"""Apply per-configuration Sobolev/deployment weights to an extxyz dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np


def _read_frames(path: Path):
    import ase.io

    frames = ase.io.read(str(path), index=":")
    if not isinstance(frames, list):
        frames = [frames]
    return frames


def _energy_value(atoms, *, index: int) -> float:
    if "energy" in atoms.info:
        return float(atoms.info["energy"])
    if atoms.calc is not None and "energy" in getattr(atoms.calc, "results", {}):
        return float(atoms.calc.results["energy"])
    try:
        return float(atoms.get_potential_energy())
    except Exception as exc:  # noqa: BLE001 - convert ASE calculator detail into dataset context.
        raise KeyError(f"configuration {index} is missing energy") from exc


def _forces_array(atoms, *, index: int) -> np.ndarray:
    if "forces" in atoms.arrays:
        forces = np.asarray(atoms.arrays["forces"], dtype=np.float64)
    elif atoms.calc is not None and "forces" in getattr(atoms.calc, "results", {}):
        forces = np.asarray(atoms.calc.results["forces"], dtype=np.float64)
    else:
        raise KeyError(f"configuration {index} is missing forces")
    if forces.shape != (len(atoms), 3):
        raise ValueError(f"configuration {index} forces shape must be {(len(atoms), 3)}, got {forces.shape}")
    return forces


def _parse_multiplier(value: str) -> tuple[str, float]:
    if ":" not in value:
        raise ValueError(f"multiplier must use label:value syntax, got {value!r}")
    label, raw = value.split(":", 1)
    label = label.strip()
    if not label:
        raise ValueError("multiplier label must not be empty")
    multiplier = float(raw)
    if multiplier < 0.0:
        raise ValueError("multipliers must be non-negative")
    return label, multiplier


def _normalize_to_mean_one(weights: np.ndarray) -> np.ndarray:
    mean = float(np.mean(weights))
    if not np.isfinite(mean) or mean <= 0.0:
        raise ValueError("cannot normalize weights with non-positive or non-finite mean")
    return weights / mean


def apply_extxyz_sample_weights(
    *,
    input_path: str | Path,
    output_path: str | Path,
    summary_path: str | Path | None = None,
    source_force_multipliers: Mapping[str, float] | None = None,
    source_energy_multipliers: Mapping[str, float] | None = None,
    force_tail_quantile: float | None = None,
    force_tail_multiplier: float = 1.0,
    normalize_force_mean: bool = True,
    normalize_energy_mean: bool = False,
    energy_weight: float = 1.0,
) -> dict:
    """Write ``energy_weight`` and ``forces_weight`` info fields to every frame.

    ``rtece_concat_source`` labels can receive source-specific energy and force
    multipliers. A force-tail multiplier can also be applied to configurations
    whose maximum atomic force norm is at or above the requested dataset
    quantile.
    """
    import ase.io

    input_file = Path(input_path)
    output_file = Path(output_path)
    frames = _read_frames(input_file)
    if not frames:
        raise ValueError("input dataset is empty")
    if float(energy_weight) < 0.0:
        raise ValueError("energy_weight must be non-negative")
    if float(force_tail_multiplier) < 0.0:
        raise ValueError("force_tail_multiplier must be non-negative")
    if force_tail_quantile is not None and not (0.0 <= float(force_tail_quantile) <= 1.0):
        raise ValueError("force_tail_quantile must be in [0, 1]")

    source_multipliers = {str(k): float(v) for k, v in dict(source_force_multipliers or {}).items()}
    source_energy = {str(k): float(v) for k, v in dict(source_energy_multipliers or {}).items()}
    for label, multiplier in source_multipliers.items():
        if multiplier < 0.0:
            raise ValueError(f"source force multiplier for {label!r} must be non-negative")
    for label, multiplier in source_energy.items():
        if multiplier < 0.0:
            raise ValueError(f"source energy multiplier for {label!r} must be non-negative")

    max_force_norms = np.asarray(
        [float(np.linalg.norm(_forces_array(atoms, index=i), axis=1).max()) for i, atoms in enumerate(frames)],
        dtype=np.float64,
    )
    tail_threshold = None
    tail_mask = np.zeros(len(frames), dtype=bool)
    if force_tail_quantile is not None:
        tail_threshold = float(np.quantile(max_force_norms, float(force_tail_quantile)))
        tail_mask = max_force_norms >= tail_threshold

    raw_force_weights = np.ones(len(frames), dtype=np.float64)
    raw_energy_weights = np.full(len(frames), float(energy_weight), dtype=np.float64)
    source_counts: dict[str, int] = {}
    for i, atoms in enumerate(frames):
        source = str(atoms.info.get("rtece_concat_source", ""))
        if source:
            source_counts[source] = source_counts.get(source, 0) + 1
        if source in source_multipliers:
            raw_force_weights[i] *= source_multipliers[source]
        if source in source_energy:
            raw_energy_weights[i] *= source_energy[source]
        if tail_mask[i]:
            raw_force_weights[i] *= float(force_tail_multiplier)

    force_weights = _normalize_to_mean_one(raw_force_weights) if normalize_force_mean else raw_force_weights
    energy_weights = _normalize_to_mean_one(raw_energy_weights) if normalize_energy_mean else raw_energy_weights

    weighted_frames = []
    for index, (atoms, e_weight, f_weight, max_force) in enumerate(
        zip(frames, energy_weights, force_weights, max_force_norms, strict=True)
    ):
        copied = atoms.copy()
        copied.info["energy"] = _energy_value(atoms, index=index)
        copied.arrays["forces"] = _forces_array(atoms, index=index).copy()
        copied.info["energy_weight"] = float(e_weight)
        copied.info["forces_weight"] = float(f_weight)
        copied.info["rtece_weight_max_force_norm_ev_a"] = float(max_force)
        weighted_frames.append(copied)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    if output_file.exists():
        output_file.unlink()
    ase.io.write(str(output_file), weighted_frames, format="extxyz")

    summary = {
        "schema_version": "rtece_extxyz_sample_weights.v1",
        "input": str(input_file),
        "output": str(output_file),
        "configs": int(len(weighted_frames)),
        "atoms": int(sum(len(atoms) for atoms in weighted_frames)),
        "energy_weight": float(energy_weight),
        "normalize_energy_mean": bool(normalize_energy_mean),
        "source_energy_multipliers": source_energy,
        "raw_energy_weight_mean": float(np.mean(raw_energy_weights)),
        "energy_weight_mean": float(np.mean(energy_weights)),
        "energy_weight_min": float(np.min(energy_weights)),
        "energy_weight_max": float(np.max(energy_weights)),
        "force_tail_quantile": None if force_tail_quantile is None else float(force_tail_quantile),
        "force_tail_threshold_ev_a": tail_threshold,
        "force_tail_multiplier": float(force_tail_multiplier),
        "normalize_force_mean": bool(normalize_force_mean),
        "source_force_multipliers": source_multipliers,
        "source_counts": source_counts,
        "raw_force_weight_mean": float(np.mean(raw_force_weights)),
        "force_weight_mean": float(np.mean(force_weights)),
        "force_weight_min": float(np.min(force_weights)),
        "force_weight_max": float(np.max(force_weights)),
        "max_force_norm_ev_a_min": float(np.min(max_force_norms)),
        "max_force_norm_ev_a_max": float(np.max(max_force_norms)),
        "tail_config_count": int(tail_mask.sum()),
    }
    if summary_path is not None:
        summary_file = Path(summary_path)
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--source-force-multiplier", action="append", default=[], help="Source force multiplier as label:value; can be repeated.")
    parser.add_argument("--source-energy-multiplier", action="append", default=[], help="Source energy multiplier as label:value; can be repeated.")
    parser.add_argument("--force-tail-quantile", type=float, default=None)
    parser.add_argument("--force-tail-multiplier", type=float, default=1.0)
    parser.add_argument("--no-normalize-force-mean", action="store_true")
    parser.add_argument("--normalize-energy-mean", action="store_true")
    parser.add_argument("--energy-weight", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    multipliers = dict(_parse_multiplier(value) for value in args.source_force_multiplier)
    energy_multipliers = dict(_parse_multiplier(value) for value in args.source_energy_multiplier)
    summary = apply_extxyz_sample_weights(
        input_path=args.input,
        output_path=args.output,
        summary_path=args.summary,
        source_force_multipliers=multipliers,
        source_energy_multipliers=energy_multipliers,
        force_tail_quantile=args.force_tail_quantile,
        force_tail_multiplier=args.force_tail_multiplier,
        normalize_force_mean=not args.no_normalize_force_mean,
        normalize_energy_mean=bool(args.normalize_energy_mean),
        energy_weight=args.energy_weight,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
