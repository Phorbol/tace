#!/usr/bin/env python3
"""Concatenate extxyz datasets while recording per-configuration source provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def _read_atoms(path: Path):
    import ase.io

    atoms_list = ase.io.read(str(path), index=":")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    return atoms_list


def concat_extxyz_datasets(
    *,
    inputs: Sequence[str | Path],
    output: str | Path,
    summary_path: str | Path | None = None,
    source_labels: Sequence[str] | None = None,
) -> dict:
    import ase.io

    input_paths = [Path(value) for value in inputs]
    if not input_paths:
        raise ValueError("at least one input dataset is required")
    labels = list(source_labels or [path.stem for path in input_paths])
    if len(labels) != len(input_paths):
        raise ValueError("source_labels length must match inputs length")

    merged = []
    sources = []
    for path, label in zip(input_paths, labels, strict=True):
        atoms_list = _read_atoms(path)
        if not atoms_list:
            raise ValueError(f"input dataset is empty: {path}")
        source_start = len(merged)
        for source_index, atoms in enumerate(atoms_list):
            copied = atoms.copy()
            copied.info["rtece_concat_source"] = str(label)
            copied.info["rtece_concat_source_file"] = str(path)
            copied.info["rtece_concat_source_index"] = int(source_index)
            merged.append(copied)
        sources.append(
            {
                "label": str(label),
                "path": str(path),
                "configs": int(len(atoms_list)),
                "atoms": int(sum(len(atoms) for atoms in atoms_list)),
                "output_start_index": int(source_start),
                "output_stop_index": int(len(merged)),
            }
        )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ase.io.write(str(output_path), merged, format="extxyz")
    summary = {
        "schema_version": "rtece_concat_extxyz.v1",
        "output": str(output_path),
        "configs": int(len(merged)),
        "atoms": int(sum(len(atoms) for atoms in merged)),
        "sources": sources,
    }
    if summary_path is not None:
        summary_file = Path(summary_path)
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--source-label", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = args.source_label if args.source_label else None
    summary = concat_extxyz_datasets(
        inputs=args.input,
        output=args.output,
        summary_path=args.summary,
        source_labels=labels,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
