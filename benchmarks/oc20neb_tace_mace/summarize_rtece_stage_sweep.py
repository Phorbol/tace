#!/usr/bin/env python3
"""Collect rTECE stage-sweep benchmark rows into RMSE-first Pareto summaries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.oc20neb_tace_mace.summarize_tece_distill import pareto_front_rows


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _benchmark_sort_key(path: Path) -> tuple[int, str]:
    name = path.name
    limit = -1
    marker = "_limit"
    if marker in name:
        tail = name.split(marker, 1)[1]
        digits = ""
        for char in tail:
            if char.isdigit():
                digits += char
            else:
                break
        if digits:
            limit = int(digits)
    return (limit, str(path))


def _latest_json_sort_key(path: Path) -> tuple[float, int, str]:
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (mtime, len(path.name), str(path))


def _find_row_json(row_name: str, roots: list[str | Path], patterns: list[str]) -> Path | None:
    candidates: list[Path] = []
    for root_value in roots:
        root = Path(root_value)
        if not root.exists():
            continue
        for pattern in patterns:
            candidates.extend(path for path in root.glob(pattern.format(row=row_name)) if path.is_file())
    if not candidates:
        return None
    return sorted(set(candidates), key=_latest_json_sort_key, reverse=True)[0]


def find_row_force_stratification(row_name: str, diagnostic_roots: list[str | Path]) -> Path | None:
    return _find_row_json(
        row_name,
        diagnostic_roots,
        [
            "{row}/**/*force_stratification.json",
            "{row}/**/*stratification.json",
            "**/{row}*force_stratification.json",
            "**/{row}*stratification.json",
        ],
    )


def find_row_physical_pareto(row_name: str, diagnostic_roots: list[str | Path]) -> Path | None:
    return _find_row_json(
        row_name,
        diagnostic_roots,
        [
            "{row}/**/*physical_pareto*.json",
            "**/{row}*physical_pareto*.json",
        ],
    )


def find_row_benchmark(row_name: str, benchmark_roots: list[str | Path]) -> Path | None:
    candidates: list[Path] = []
    for root_value in benchmark_roots:
        root = Path(root_value)
        if not root.exists():
            continue
        patterns = [
            f"{row_name}/{row_name}_limit*_benchmark.json",
            f"{row_name}/**/{row_name}_limit*_benchmark.json",
            f"{row_name}/{row_name}_dft_benchmark.json",
            f"{row_name}/**/{row_name}_dft_benchmark.json",
            f"**/{row_name}_limit*_benchmark.json",
            f"**/{row_name}_dft_benchmark.json",
        ]
        for pattern in patterns:
            candidates.extend(path for path in root.glob(pattern) if path.is_file())
    if not candidates:
        return None
    unique = sorted(set(candidates), key=_benchmark_sort_key, reverse=True)
    return unique[0]


def _path_ids_from_row(row: dict[str, Any]) -> list[str]:
    value = row.get("scalar_path_ids")
    if isinstance(value, str):
        return [item for item in value.split(",") if item]
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def row_from_index_and_benchmark(
    index_row: dict[str, Any],
    *,
    benchmark_path: Path | None,
) -> dict[str, Any]:
    name = str(index_row["name"])
    path_ids = _path_ids_from_row(index_row)
    item: dict[str, Any] = {
        "name": name,
        "variant": name,
        "train_variant": index_row.get("train_variant") or index_row.get("variant") or name,
        "status": "missing_benchmark",
        "benchmark_path": None,
        "hidden_channels": index_row.get("hidden_channels"),
        "moment_l_max": index_row.get("moment_l_max"),
        "scalar_path_ids": path_ids,
        "tece_axes": index_row.get("tece_axes", []),
        "stage_basis": index_row.get("stage_basis"),
        "atoms_per_second": None,
        "configs_per_second": None,
        "seconds_per_pass": None,
        "configs": None,
        "atoms": None,
        "num_parameters": index_row.get("num_parameters"),
        "num_parameters_estimate": index_row.get("num_parameters_estimate"),
        "num_parameters_source": "index" if index_row.get("num_parameters") is not None else ("index_estimate" if index_row.get("num_parameters_estimate") is not None else None),
        "dft_f_rmse_mev_a": None,
        "dft_f_mae_mev_a": None,
        "dft_f_max_abs_mev_a": None,
        "dft_e_rmse_mev_atom": None,
        "dft_e_mae_mev_atom": None,
        "dft_e_bias_mev_atom": None,
        "dft_e_max_abs_mev_atom": None,
        "tece_path_manifest_hash": None,
        "force_stratification_path": None,
        "focus_force_label": None,
        "focus_atom_fraction": None,
        "focus_f_rmse_mev_a": None,
        "focus_f_mae_mev_a": None,
        "focus_f_max_abs_mev_a": None,
        "force_selection_score_mev_a": None,
        "physical_pareto_path": None,
        "physical_score": None,
        "physical_gate_pass": None,
        "benchmark_gate_pass": None,
        "dimer_gate_pass": None,
        "rattle_gate_pass": None,
        "dimer_short_repulsive_fraction": None,
        "rattle_focus_label": None,
        "focus_rattle_final_rmsd_a": None,
    }
    if benchmark_path is None:
        return item
    benchmark = load_json(benchmark_path)
    manifest = benchmark.get("tece_path_manifest") or benchmark.get("tece_architecture_path_manifest")
    item.update(
        {
            "status": "benchmark_found",
            "benchmark_path": str(benchmark_path),
            "variant": str(benchmark.get("variant") or item.get("train_variant") or name),
            "train_variant": str(benchmark.get("variant") or item.get("train_variant") or name),
            "hidden_channels": benchmark.get("hidden_channels", item["hidden_channels"]),
            "moment_l_max": benchmark.get("moment_l_max", item["moment_l_max"]),
            "atoms_per_second": _finite_float(benchmark.get("atoms_per_second")),
            "configs_per_second": _finite_float(benchmark.get("configs_per_second")),
            "seconds_per_pass": _finite_float(benchmark.get("seconds_per_pass")),
            "configs": benchmark.get("configs"),
            "atoms": benchmark.get("atoms"),
            "num_parameters": benchmark.get("num_parameters", item.get("num_parameters")),
            "num_parameters_source": "benchmark" if benchmark.get("num_parameters") is not None else item.get("num_parameters_source"),
            "dft_f_rmse_mev_a": _finite_float(benchmark.get("rmse_f_mev_a")),
            "dft_f_mae_mev_a": _finite_float(benchmark.get("mae_f_mev_a")),
            "dft_f_max_abs_mev_a": _finite_float(benchmark.get("max_abs_f_mev_a")),
            "dft_e_rmse_mev_atom": _finite_float(benchmark.get("rmse_e_mev_atom")),
            "dft_e_mae_mev_atom": _finite_float(benchmark.get("mae_e_mev_atom")),
            "dft_e_bias_mev_atom": _finite_float(benchmark.get("bias_e_mev_atom") or benchmark.get("mean_signed_e_mev_atom")),
            "dft_e_max_abs_mev_atom": _finite_float(benchmark.get("max_abs_e_mev_atom")),
            "force_mode": benchmark.get("force_mode"),
            "graph_construction_backend": benchmark.get("graph_construction_backend"),
            "peak_allocated_mb": _finite_float(benchmark.get("peak_allocated_mb")),
            "tece_path_manifest_hash": manifest.get("manifest_hash") if isinstance(manifest, dict) else None,
        }
    )
    if isinstance(manifest, dict) and manifest.get("scalar_paths"):
        item["scalar_path_ids"] = [str(path.get("id")) for path in manifest["scalar_paths"] if path.get("id")]
    return item


def _focus_group(summary: dict[str, Any], label: str) -> dict[str, Any] | None:
    for row in summary.get("focus_groups") or []:
        if str(row.get("label")) == str(label):
            return row
    return None


def enrich_row_with_force_stratification(item: dict[str, Any], path: Path | None) -> None:
    if path is None:
        return
    payload = load_json(path)
    focus_label = str(payload.get("selection_focus_label") or "C_or_N")
    focus = _focus_group(payload, focus_label) or {}
    item.update(
        {
            "force_stratification_path": str(path),
            "focus_force_label": focus_label,
            "focus_atom_fraction": _finite_float(focus.get("atom_fraction")),
            "focus_f_rmse_mev_a": _finite_float(focus.get("rmse_f_mev_a")),
            "focus_f_mae_mev_a": _finite_float(focus.get("mae_f_mev_a")),
            "focus_f_max_abs_mev_a": _finite_float(focus.get("max_abs_f_mev_a")),
            "force_selection_score_mev_a": _finite_float(payload.get("selection_score_mev_a")),
        }
    )


def _physical_row_for_variant(payload: dict[str, Any], variant: str) -> dict[str, Any]:
    rows = payload.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if str(row.get("variant") or row.get("name")) == str(variant):
                return row
        return rows[0] if rows else {}
    return payload


def enrich_row_with_physical_pareto(item: dict[str, Any], path: Path | None) -> None:
    if path is None:
        return
    payload = load_json(path)
    row = _physical_row_for_variant(payload, str(item.get("variant") or item.get("name")))
    item.update(
        {
            "physical_pareto_path": str(path),
            "physical_score": _finite_float(row.get("physical_score")),
            "physical_gate_pass": row.get("physical_gate_pass"),
            "benchmark_gate_pass": row.get("benchmark_gate_pass"),
            "dimer_gate_pass": row.get("dimer_gate_pass"),
            "rattle_gate_pass": row.get("rattle_gate_pass"),
            "dimer_short_repulsive_fraction": _finite_float(row.get("dimer_short_repulsive_fraction")),
            "rattle_focus_label": row.get("rattle_focus_label"),
            "focus_rattle_final_rmsd_a": _finite_float(row.get("focus_rattle_final_rmsd_a")),
        }
    )


def collect_stage_sweep_results(
    index_path: str | Path,
    *,
    benchmark_roots: list[str | Path],
    diagnostic_roots: list[str | Path] | None = None,
) -> dict[str, Any]:
    index = load_json(index_path)
    rows = []
    diagnostic_roots = list(diagnostic_roots or [])
    for index_row in index.get("rows", []):
        name = str(index_row["name"])
        benchmark = find_row_benchmark(name, benchmark_roots)
        row = row_from_index_and_benchmark(index_row, benchmark_path=benchmark)
        enrich_row_with_force_stratification(row, find_row_force_stratification(name, diagnostic_roots))
        enrich_row_with_physical_pareto(row, find_row_physical_pareto(name, diagnostic_roots))
        rows.append(row)
    missing = [row["name"] for row in rows if row["status"] != "benchmark_found"]
    return {
        "schema_version": "rtece_stage_sweep_summary.v1",
        "source_index": str(index_path),
        "row_set": index.get("row_set"),
        "primary_error_metric": "dft_f_rmse_mev_a",
        "primary_throughput_metric": "atoms_per_second",
        "review_basis": [
            "TECE_design_space: hardware Pareto frontier over physical error vs atom/s and memory",
            "rTECE_review: record force RMSE, high-force tail/max error, energy RMSE/bias/max separately",
            "rTECE_review: keep dimer/rattle/element-stratified physical probes explicit rather than collapsing to one MAE",
        ],
        "rows": rows,
        "missing_benchmark_rows": missing,
        "dft_force_rmse_pareto_front": pareto_front_rows(rows, error_key="dft_f_rmse_mev_a"),
        "dft_force_mae_pareto_front": pareto_front_rows(rows, error_key="dft_f_mae_mev_a"),
    }


def _fmt(value: Any, *, digits: int = 3) -> str:
    number = _finite_float(value)
    if number is None:
        return "NA"
    if abs(number) >= 1.0e5 or (abs(number) > 0 and abs(number) < 1.0e-2):
        return f"{number:.{digits}e}"
    return f"{number:.{digits}f}"


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# rTECE Stage Sweep Summary: {payload.get('row_set')}",
        "",
        "Primary metric: DFT force RMSE vs atoms/s. Missing benchmark rows are kept explicit.",
        "",
        "| row | status | params | TECE axes | atoms/s | F RMSE | F MAE | F max | E RMSE | E MAE | E bias | E max |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload.get("rows", []):
        lines.append(
            "| {name} | {status} | {params} | {axes} | {atoms} | {frmse} | {fmae} | {fmax} | {ermse} | {emae} | {ebias} | {emax} |".format(
                name=row.get("name"),
                status=row.get("status"),
                params=_fmt(row.get("num_parameters") or row.get("num_parameters_estimate"), digits=0),
                axes=", ".join(str(axis) for axis in (row.get("tece_axes") or row.get("scalar_path_ids") or [])) or "NA",
                atoms=_fmt(row.get("atoms_per_second")),
                frmse=_fmt(row.get("dft_f_rmse_mev_a")),
                fmae=_fmt(row.get("dft_f_mae_mev_a")),
                fmax=_fmt(row.get("dft_f_max_abs_mev_a")),
                ermse=_fmt(row.get("dft_e_rmse_mev_atom")),
                emae=_fmt(row.get("dft_e_mae_mev_atom")),
                ebias=_fmt(row.get("dft_e_bias_mev_atom")),
                emax=_fmt(row.get("dft_e_max_abs_mev_atom")),
            )
        )
    missing = payload.get("missing_benchmark_rows", [])
    diagnostic_rows = [row for row in payload.get("rows", []) if row.get("force_stratification_path") or row.get("physical_pareto_path")]
    if diagnostic_rows:
        lines.extend([
            "",
            "## Physical Diagnostics",
            "",
            "| row | focus | focus F RMSE | focus F max | physical score | physical gate | dimer gate | rattle gate |",
            "|---|---|---:|---:|---:|---|---|---|",
        ])
        for row in diagnostic_rows:
            lines.append(
                "| {name} | {focus} | {frmse} | {fmax} | {score} | {pgate} | {dgate} | {rgate} |".format(
                    name=row.get("name"),
                    focus=row.get("focus_force_label") or row.get("rattle_focus_label") or "NA",
                    frmse=_fmt(row.get("focus_f_rmse_mev_a")),
                    fmax=_fmt(row.get("focus_f_max_abs_mev_a")),
                    score=_fmt(row.get("physical_score")),
                    pgate=row.get("physical_gate_pass"),
                    dgate=row.get("dimer_gate_pass"),
                    rgate=row.get("rattle_gate_pass"),
                )
            )
    lines.extend(["", "## Missing Benchmark Rows", ""])
    if missing:
        lines.extend(f"- `{name}`" for name in missing)
    else:
        lines.append("None")
    lines.extend(["", "## DFT Force RMSE Pareto Front", ""])
    front = payload.get("dft_force_rmse_pareto_front", [])
    if front:
        for row in front:
            lines.append(f"- `{row.get('name')}`: atoms/s={_fmt(row.get('atoms_per_second'))}, F RMSE={_fmt(row.get('dft_f_rmse_mev_a'))}")
    else:
        lines.append("No complete rows yet.")
    lines.extend(["", "## Review Notes", ""])
    lines.extend(f"- {item}" for item in payload.get("review_basis", []))
    return "\n".join(lines) + "\n"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, action="append", default=[])
    parser.add_argument("--diagnostic-root", type=Path, action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = collect_stage_sweep_results(args.index, benchmark_roots=args.benchmark_root, diagnostic_roots=args.diagnostic_root)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(format_markdown(payload), encoding="utf-8")
    print(args.output_md)
    print(args.output_json)


if __name__ == "__main__":
    main()
