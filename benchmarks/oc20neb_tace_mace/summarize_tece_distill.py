#!/usr/bin/env python3
"""Summarize reduced TECE/TACE distillation benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def make_student_row(
    variant: str,
    *,
    dft_benchmark: dict[str, Any],
    teacher_benchmark: dict[str, Any],
) -> dict[str, Any]:
    return {
        "variant": variant,
        "atoms_per_second": dft_benchmark.get("atoms_per_second"),
        "configs_per_second": dft_benchmark.get("configs_per_second"),
        "force_mode": dft_benchmark.get("force_mode", "autograd"),
        "graph_construction_backend": dft_benchmark.get("graph_construction_backend"),
        "graph_update_backend": dft_benchmark.get("graph_update_backend"),
        "hidden_channels": dft_benchmark.get("hidden_channels"),
        "num_radial": dft_benchmark.get("num_radial"),
        "seconds_per_pass": dft_benchmark.get("seconds_per_pass"),
        "peak_allocated_mb": dft_benchmark.get("peak_allocated_mb"),
        "peak_reserved_mb": dft_benchmark.get("peak_reserved_mb"),
        "num_parameters": dft_benchmark.get("num_parameters"),
        "dft_e_mae_mev_atom": dft_benchmark.get("mae_e_mev_atom"),
        "dft_f_mae_mev_a": dft_benchmark.get("mae_f_mev_a"),
        "teacher_e_mae_mev_atom": teacher_benchmark.get("mae_e_mev_atom"),
        "teacher_f_mae_mev_a": teacher_benchmark.get("mae_f_mev_a"),
        "dft_benchmark": dft_benchmark.get("model"),
        "teacher_benchmark": teacher_benchmark.get("model"),
    }


def pareto_front_rows(
    rows: list[dict[str, Any]],
    *,
    error_key: str,
    throughput_key: str = "atoms_per_second",
) -> list[dict[str, Any]]:
    valid = [
        row
        for row in rows
        if row.get(throughput_key) is not None and row.get(error_key) is not None
    ]
    front = []
    for row in valid:
        row_speed = float(row[throughput_key])
        row_error = float(row[error_key])
        dominated = False
        for other in valid:
            if other is row:
                continue
            other_speed = float(other[throughput_key])
            other_error = float(other[error_key])
            no_worse = other_speed >= row_speed and other_error <= row_error
            strictly_better = other_speed > row_speed or other_error < row_error
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(row)
    return sorted(
        front,
        key=lambda row: (
            -float(row.get(throughput_key) or 0.0),
            float(row.get(error_key) or 1.0e30),
        ),
    )


def rank_student_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row.get("atoms_per_second") or 0.0),
            float(row.get("teacher_f_mae_mev_a") or 1.0e30),
            float(row.get("dft_f_mae_mev_a") or 1.0e30),
        ),
    )


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def append_front_section(lines: list[str], title: str, rows: list[dict[str, Any]], error_key: str) -> None:
    front = pareto_front_rows(rows, error_key=error_key)
    if not front:
        return
    lines.extend([
        "",
        title,
        "",
        "| variant | graph backend | force mode | atoms/s | DFT F MAE | teacher F MAE | params |",
        "|---|---|---|---:|---:|---:|---:|",
    ])
    for row in front:
        lines.append(
            "| {variant} | {graph_backend} | {force_mode} | {atoms} | {df} | {tf} | {params} |".format(
                variant=row["variant"],
                graph_backend=fmt(row.get("graph_construction_backend")),
                force_mode=fmt(row.get("force_mode")),
                atoms=fmt(row.get("atoms_per_second")),
                df=fmt(row.get("dft_f_mae_mev_a")),
                tf=fmt(row.get("teacher_f_mae_mev_a")),
                params=fmt(row.get("num_parameters"), digits=0),
            )
        )


def format_markdown(rows: list[dict[str, Any]], *, baselines: list[dict[str, Any]]) -> str:
    lines = [
        "# TECE Distillation Matrix Summary",
        "",
        "## Students",
        "",
        "| variant | graph backend | force mode | atoms/s | configs/s | peak alloc MB | params | teacher E MAE | teacher F MAE | DFT E MAE | DFT F MAE |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {graph_backend} | {force_mode} | {atoms} | {configs} | {mem} | {params} | {te} | {tf} | {de} | {df} |".format(
                variant=row["variant"],
                graph_backend=fmt(row.get("graph_construction_backend")),
                force_mode=fmt(row.get("force_mode")),
                atoms=fmt(row.get("atoms_per_second")),
                configs=fmt(row.get("configs_per_second")),
                mem=fmt(row.get("peak_allocated_mb")),
                params=fmt(row.get("num_parameters"), digits=0),
                te=fmt(row.get("teacher_e_mae_mev_atom")),
                tf=fmt(row.get("teacher_f_mae_mev_a")),
                de=fmt(row.get("dft_e_mae_mev_atom")),
                df=fmt(row.get("dft_f_mae_mev_a")),
            )
        )
    append_front_section(lines, "## DFT Force Pareto Front", rows, "dft_f_mae_mev_a")
    append_front_section(lines, "## Teacher Force Pareto Front", rows, "teacher_f_mae_mev_a")
    if baselines:
        lines.extend([
            "",
            "## Baselines",
            "",
            "| name | atoms/s | peak alloc MB | params | DFT E MAE | DFT F MAE |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for base in baselines:
            lines.append(
                "| {name} | {atoms} | {mem} | {params} | {e} | {f} |".format(
                    name=base["name"],
                    atoms=fmt(base.get("atoms_per_second")),
                    mem=fmt(base.get("peak_allocated_mb")),
                    params=fmt(base.get("num_parameters"), digits=0),
                    e=fmt(base.get("mae_e_mev_atom")),
                    f=fmt(base.get("mae_f_mev_a")),
                )
            )
    lines.extend([
        "",
        "## TECE Source-Document Review",
        "",
        "- Projection: did reducing persistent angular state improve real throughput and memory?",
        "- Renormalization/distillation: is teacher-label error low enough to justify the deleted paths?",
        "- Hardware cost: did atoms/s and peak memory improve, not just parameter count?",
        "- Next priority: architecture projection, distillation loss, fusion/export, or benchmark methodology.",
        "",
    ])
    return "\n".join(lines)


def parse_student_arg(value: str) -> tuple[str, Path, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("student must be variant:dft_benchmark.json:teacher_benchmark.json")
    return parts[0], Path(parts[1]), Path(parts[2])


def parse_baseline_arg(value: str) -> tuple[str, Path]:
    parts = value.split(":", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("baseline must be name:benchmark.json")
    return parts[0], Path(parts[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", action="append", type=parse_student_arg, default=[])
    parser.add_argument("--baseline", action="append", type=parse_baseline_arg, default=[])
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = rank_student_rows([
        make_student_row(
            variant,
            dft_benchmark=load_json(dft_path),
            teacher_benchmark=load_json(teacher_path),
        )
        for variant, dft_path, teacher_path in args.student
    ])
    baselines = []
    for name, path in args.baseline:
        item = load_json(path)
        item["name"] = name
        baselines.append(item)
    payload = {
        "students": rows,
        "dft_force_pareto_front": pareto_front_rows(rows, error_key="dft_f_mae_mev_a"),
        "teacher_force_pareto_front": pareto_front_rows(rows, error_key="teacher_f_mae_mev_a"),
        "baselines": baselines,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(format_markdown(rows, baselines=baselines), encoding="utf-8")
    print(args.output_md)
    print(args.output_json)


if __name__ == "__main__":
    main()
