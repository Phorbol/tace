#!/usr/bin/env python3
"""Stratify rTECE projection sample weights by element and lightweight focus groups."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
from math import ceil
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _as_float_list(values: Iterable[float]) -> list[float]:
    result = [float(value) for value in values]
    if not result:
        raise ValueError("weights must not be empty")
    if any(value < 0.0 for value in result):
        raise ValueError("weights must be non-negative")
    return result


def _group_row(
    label: str,
    indices: list[int],
    weights: list[float],
    *,
    total_weight: float,
    num_samples: int,
    top_indices: set[int],
    top_weight_total: float,
) -> dict[str, Any]:
    group_weights = [weights[index] for index in indices]
    weight_sum = float(sum(group_weights))
    top_weight_sum = float(sum(weights[index] for index in indices if index in top_indices))
    return {
        "label": str(label),
        "count": int(len(indices)),
        "sample_fraction": float(len(indices) / num_samples) if num_samples else 0.0,
        "weight_sum": weight_sum,
        "weight_fraction": float(weight_sum / total_weight) if total_weight > 0.0 else 0.0,
        "mean_weight": float(weight_sum / len(indices)) if indices else 0.0,
        "max_weight": float(max(group_weights)) if group_weights else 0.0,
        "top_count": int(sum(1 for index in indices if index in top_indices)),
        "top_weight_sum": top_weight_sum,
        "top_weight_fraction": float(top_weight_sum / top_weight_total) if top_weight_total > 0.0 else 0.0,
    }


def _summary_rows(labels: list[str], weights: list[float], *, top_indices: set[int], top_weight_total: float) -> list[dict[str, Any]]:
    total_weight = float(sum(weights))
    by_label: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        by_label.setdefault(str(label), []).append(index)
    rows = [
        _group_row(
            label,
            indices,
            weights,
            total_weight=total_weight,
            num_samples=len(weights),
            top_indices=top_indices,
            top_weight_total=top_weight_total,
        )
        for label, indices in by_label.items()
    ]
    return sorted(rows, key=lambda row: (-float(row["weight_fraction"]), str(row["label"])))


def _focus_masks(symbols: list[str]) -> dict[str, list[int]]:
    c_or_n = {"C", "N"}
    chno = {"C", "H", "N", "O"}
    return {
        "C_or_N": [index for index, symbol in enumerate(symbols) if symbol in c_or_n],
        "CHNO": [index for index, symbol in enumerate(symbols) if symbol in chno],
        "not_CHNO": [index for index, symbol in enumerate(symbols) if symbol not in chno],
    }


def stratify_symbol_weights(
    *,
    symbols: list[str],
    weights: Iterable[float],
    top_fraction: float = 0.1,
) -> dict[str, Any]:
    values = _as_float_list(weights)
    if len(symbols) != len(values):
        raise ValueError(f"symbols/weights length mismatch: {len(symbols)} vs {len(values)}")
    if not 0.0 < float(top_fraction) <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    total_weight = float(sum(values))
    top_count = max(1, int(ceil(len(values) * float(top_fraction))))
    ranked = sorted(range(len(values)), key=lambda index: values[index], reverse=True)
    top_indices = set(ranked[:top_count])
    top_weight_total = float(sum(values[index] for index in top_indices))

    focus_rows = [
        _group_row(
            label,
            indices,
            values,
            total_weight=total_weight,
            num_samples=len(values),
            top_indices=top_indices,
            top_weight_total=top_weight_total,
        )
        for label, indices in _focus_masks(symbols).items()
    ]
    focus_rows.sort(key=lambda row: (-float(row["weight_fraction"]), str(row["label"])))

    return {
        "schema_version": "rtece_projection_weight_stratification.v1",
        "num_samples": int(len(values)),
        "total_weight": total_weight,
        "top_fraction": float(top_fraction),
        "top_count": int(top_count),
        "top_weight_total": top_weight_total,
        "elements": _summary_rows(symbols, values, top_indices=top_indices, top_weight_total=top_weight_total),
        "focus_groups": focus_rows,
    }


def load_symbols_from_extxyz(configs: Path, *, limit_configs: int) -> list[str]:
    import ase.io

    atoms_list = ase.io.read(str(configs), index=f":{int(limit_configs)}")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    symbols: list[str] = []
    for atoms in atoms_list:
        symbols.extend(atoms.get_chemical_symbols())
    return symbols


def load_sample_weights(path: Path) -> tuple[list[float], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "sample_weights" not in payload:
        raise ValueError("sample-weight JSON must be an object containing sample_weights")
    return _as_float_list(payload["sample_weights"]), payload


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# rTECE Projection Weight Stratification",
        "",
        f"- weight source: `{payload.get('weight_source', 'NA')}`",
        f"- samples: {payload['num_samples']}",
        f"- top fraction: {payload['top_fraction']}",
        "",
        "## Focus Groups",
        "",
        "| group | count | sample frac | weight frac | mean weight | max weight | top weight frac |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["focus_groups"]:
        lines.append(
            "| {label} | {count} | {sample_fraction:.3f} | {weight_fraction:.3f} | {mean_weight:.3f} | {max_weight:.3f} | {top_weight_fraction:.3f} |".format(
                **row
            )
        )
    lines.extend([
        "",
        "## Elements",
        "",
        "| element | count | sample frac | weight frac | mean weight | max weight | top weight frac |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for row in payload["elements"]:
        lines.append(
            "| {label} | {count} | {sample_fraction:.3f} | {weight_fraction:.3f} | {mean_weight:.3f} | {max_weight:.3f} | {top_weight_fraction:.3f} |".format(
                **row
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--sample-weight-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--limit-configs", type=int, required=True)
    parser.add_argument("--top-fraction", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights, weight_payload = load_sample_weights(args.sample_weight_json)
    symbols = load_symbols_from_extxyz(args.configs, limit_configs=int(args.limit_configs))
    payload = stratify_symbol_weights(symbols=symbols, weights=weights, top_fraction=float(args.top_fraction))
    payload.update(
        {
            "configs": str(args.configs),
            "limit_configs": int(args.limit_configs),
            "sample_weight_json": str(args.sample_weight_json),
            "weight_source": weight_payload.get("weight_source"),
            "source_weight_min": weight_payload.get("weight_min"),
            "source_weight_mean": weight_payload.get("weight_mean"),
            "source_weight_max": weight_payload.get("weight_max"),
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(format_markdown(payload), encoding="utf-8")
    print(args.output_json)
    if args.output_md is not None:
        print(args.output_md)


if __name__ == "__main__":
    main()
