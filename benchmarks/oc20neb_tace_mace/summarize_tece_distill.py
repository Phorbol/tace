#!/usr/bin/env python3
"""Summarize reduced TECE/TACE distillation benchmark results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tace.models.rtece_scalar import RTECEScalarConfig, build_rtece_config, rtece_path_manifest, rtece_route_contract


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _rtece_config_from_benchmark(variant: str, benchmark: dict[str, Any]) -> RTECEScalarConfig | None:
    model_variant = benchmark.get("variant")
    force_mode = benchmark.get("force_mode", "autograd")
    if model_variant is None:
        if variant.startswith("rtece_pair") or "pair" in variant:
            model_variant = "rtece_pair"
        elif "cavity" in variant:
            if "radial" in variant:
                model_variant = "rtece_cavity_radial_edge_sketch14"
            else:
                model_variant = "rtece_cavity_edge_sketch8"
        elif "element" in variant or "radial" in variant or "element" in force_mode:
            model_variant = "rtece_element_density"
        elif "quadratic" in variant:
            model_variant = "rtece_density_quadratic"
        elif "vector" in variant:
            model_variant = "rtece_vector_moments"
        elif "species" in variant:
            model_variant = "rtece_species_basis4"
        elif "cavity" in variant:
            model_variant = "rtece_cavity_edge_sketch8"
    if model_variant is None:
        return None
    try:
        config = build_rtece_config(str(model_variant))
    except ValueError:
        return None
    hidden = benchmark.get("hidden_channels") or config.hidden_channels
    if isinstance(hidden, list):
        hidden = tuple(int(x) for x in hidden)
    num_radial = int(benchmark.get("num_radial") or config.num_radial)
    atomic_energies = benchmark.get("atomic_energies")
    if atomic_energies is not None:
        atomic_energies = {int(k): float(v) for k, v in atomic_energies.items()}
    return RTECEScalarConfig(
        variant=config.variant,
        cutoff=float(benchmark.get("cutoff") or config.cutoff),
        num_radial=num_radial,
        hidden_channels=hidden,
        max_atomic_number=int(benchmark.get("max_atomic_number") or config.max_atomic_number),
        use_element_density=config.use_element_density,
        use_density_quadratic=config.use_density_quadratic,
        use_vector_moments=config.use_vector_moments,
        use_atomic_moments=config.use_atomic_moments,
        species_basis_channels=config.species_basis_channels,
        num_edge_sketches=config.num_edge_sketches,
        use_cavity_edge_sketches=config.use_cavity_edge_sketches,
        radial_edge_sketch_channels=config.radial_edge_sketch_channels,
        energy_per_atom_shift=float(benchmark.get("energy_per_atom_shift") or config.energy_per_atom_shift),
        atomic_energies=atomic_energies,
    )


def make_student_row(
    variant: str,
    *,
    dft_benchmark: dict[str, Any],
    teacher_benchmark: dict[str, Any],
) -> dict[str, Any]:
    force_mode = dft_benchmark.get("force_mode", "autograd")
    graph_construction_backend = dft_benchmark.get("graph_construction_backend")
    graph_update_backend = dft_benchmark.get("graph_update_backend")
    config = _rtece_config_from_benchmark(variant, dft_benchmark)
    route = None
    path_manifest = None
    if config is not None:
        route = rtece_route_contract(
            config,
            force_mode=force_mode,
            graph_construction_backend=graph_construction_backend,
            graph_update_backend=graph_update_backend,
        )
        path_manifest = rtece_path_manifest(
            config,
            force_mode=force_mode,
            graph_construction_backend=graph_construction_backend,
            graph_update_backend=graph_update_backend,
        )
    return {
        "variant": variant,
        "atoms_per_second": dft_benchmark.get("atoms_per_second"),
        "configs_per_second": dft_benchmark.get("configs_per_second"),
        "force_mode": force_mode,
        "graph_construction_backend": graph_construction_backend,
        "graph_update_backend": graph_update_backend,
        "tece_route": route,
        "tece_path_manifest": path_manifest,
        "tece_path_manifest_hash": path_manifest.get("manifest_hash") if path_manifest else None,
        "hidden_channels": dft_benchmark.get("hidden_channels"),
        "num_radial": dft_benchmark.get("num_radial"),
        "energy_per_atom_shift": dft_benchmark.get("energy_per_atom_shift"),
        "atomic_energies": dft_benchmark.get("atomic_energies"),
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


def _finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def _sorted_unique_strings(values: list[Any]) -> list[str]:
    return sorted({str(value) for value in values if value is not None})


def _join_limited(values: list[str], *, limit: int = 6) -> str:
    if not values:
        return "NA"
    if len(values) <= limit:
        return ", ".join(values)
    shown = values[:limit]
    return ", ".join(shown) + f", +{len(values) - limit} more"


def load_projection_diagnostic_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        payload = load_json(path)
        for row in payload.get("rows", []):
            item = dict(row)
            item["projection_diagnostic"] = str(path)
            rows.append(item)
    return rows


def _projection_residual_value(row: dict[str, Any] | None) -> float:
    if row is None or row.get("relative_residual") is None:
        return 1.0e30
    try:
        return float(row["relative_residual"])
    except (TypeError, ValueError):
        return 1.0e30


def _projection_rows_by_manifest(projection_rows: list[dict[str, Any]] | None) -> dict[str, dict[str, Any]]:
    by_manifest: dict[str, dict[str, Any]] = {}
    for row in projection_rows or []:
        manifest_hash = row.get("candidate_manifest_hash")
        if manifest_hash is None:
            continue
        key = str(manifest_hash)
        current = by_manifest.get(key)
        if current is None or _projection_residual_value(row) < _projection_residual_value(current):
            by_manifest[key] = row
    return by_manifest


def manifest_group_rows(
    rows: list[dict[str, Any]],
    *,
    projection_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    projection_by_manifest = _projection_rows_by_manifest(projection_rows)
    for row in rows:
        manifest_hash = row.get("tece_path_manifest_hash") or f"missing:{row.get('variant', 'unknown')}"
        grouped.setdefault(str(manifest_hash), []).append(row)

    groups = []
    for manifest_hash, group_rows in grouped.items():
        manifest = group_rows[0].get("tece_path_manifest") or {}
        route = manifest.get("route") or group_rows[0].get("tece_route") or {}
        scalar_paths = manifest.get("scalar_paths") or []
        atoms_values = _finite_values(group_rows, "atoms_per_second")
        dft_force_values = _finite_values(group_rows, "dft_f_mae_mev_a")
        teacher_force_values = _finite_values(group_rows, "teacher_f_mae_mev_a")
        projection = projection_by_manifest.get(manifest_hash)
        groups.append({
            "manifest_hash": manifest_hash,
            "semantic_tier": route.get("semantic_tier"),
            "descriptor_family": route.get("descriptor_family"),
            "variants": _sorted_unique_strings([row.get("variant") for row in group_rows]),
            "row_count": len(group_rows),
            "retained_tece_groups": _sorted_unique_strings(
                list(manifest.get("retained_tece_groups") or route.get("retained_tece_groups") or [])
            ),
            "deleted_tece_groups": _sorted_unique_strings(
                list(manifest.get("deleted_tece_groups") or route.get("deleted_tece_groups") or [])
            ),
            "scalar_path_ids": _sorted_unique_strings([path.get("id") for path in scalar_paths]),
            "cost_groups": _sorted_unique_strings([path.get("cost_group") for path in scalar_paths]),
            "force_modes": _sorted_unique_strings([row.get("force_mode") for row in group_rows]),
            "graph_backends": _sorted_unique_strings([row.get("graph_construction_backend") for row in group_rows]),
            "best_atoms_per_second": max(atoms_values) if atoms_values else None,
            "best_dft_f_mae_mev_a": min(dft_force_values) if dft_force_values else None,
            "best_teacher_f_mae_mev_a": min(teacher_force_values) if teacher_force_values else None,
            "projection_relative_residual": projection.get("relative_residual") if projection else None,
            "projection_deleted_scalar_path_ids": list(projection.get("deleted_scalar_path_ids") or []) if projection else [],
            "projection_num_samples": projection.get("num_samples") if projection else None,
            "projection_candidate": projection.get("candidate") if projection else None,
        })
    return sorted(
        groups,
        key=lambda group: (
            -float(group.get("best_atoms_per_second") or 0.0),
            float(group.get("best_dft_f_mae_mev_a") or 1.0e30),
            str(group.get("manifest_hash")),
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
        "| variant | TECE route | manifest | graph backend | force mode | atoms/s | DFT F MAE | teacher F MAE | params |",
        "|---|---|---|---|---|---:|---:|---:|---:|",
    ])
    for row in front:
        lines.append(
            "| {variant} | {route} | {manifest} | {graph_backend} | {force_mode} | {atoms} | {df} | {tf} | {params} |".format(
                variant=row["variant"],
                route=fmt((row.get("tece_route") or {}).get("semantic_tier")),
                manifest=fmt(row.get("tece_path_manifest_hash")),
                graph_backend=fmt(row.get("graph_construction_backend")),
                force_mode=fmt(row.get("force_mode")),
                atoms=fmt(row.get("atoms_per_second")),
                df=fmt(row.get("dft_f_mae_mev_a")),
                tf=fmt(row.get("teacher_f_mae_mev_a")),
                params=fmt(row.get("num_parameters"), digits=0),
            )
        )


def append_manifest_group_section(
    lines: list[str],
    rows: list[dict[str, Any]],
    *,
    projection_rows: list[dict[str, Any]] | None = None,
) -> None:
    groups = manifest_group_rows(rows, projection_rows=projection_rows)
    if not groups:
        return
    lines.extend([
        "",
        "## Manifest Groups",
        "",
        "| manifest | TECE route | variants | retained groups | scalar paths | projection residual | deleted projection paths | projection samples | best atoms/s | best DFT F MAE | best teacher F MAE | rows |",
        "|---|---|---|---|---|---:|---|---:|---:|---:|---:|---:|",
    ])
    for group in groups:
        lines.append(
            "| {manifest} | {route} | {variants} | {retained} | {paths} | {projection} | {deleted_projection} | {projection_samples} | {atoms} | {df} | {tf} | {rows} |".format(
                manifest=fmt(group.get("manifest_hash")),
                route=fmt(group.get("semantic_tier")),
                variants=_join_limited(group.get("variants") or [], limit=4),
                retained=_join_limited(group.get("retained_tece_groups") or [], limit=4),
                paths=_join_limited(group.get("scalar_path_ids") or [], limit=5),
                projection=fmt(group.get("projection_relative_residual")),
                deleted_projection=_join_limited(group.get("projection_deleted_scalar_path_ids") or [], limit=4),
                projection_samples=fmt(group.get("projection_num_samples"), digits=0),
                atoms=fmt(group.get("best_atoms_per_second")),
                df=fmt(group.get("best_dft_f_mae_mev_a")),
                tf=fmt(group.get("best_teacher_f_mae_mev_a")),
                rows=fmt(group.get("row_count"), digits=0),
            )
        )


def format_markdown(
    rows: list[dict[str, Any]],
    *,
    baselines: list[dict[str, Any]],
    projection_rows: list[dict[str, Any]] | None = None,
) -> str:
    lines = [
        "# TECE Distillation Matrix Summary",
        "",
        "## Students",
        "",
        "| variant | TECE route | manifest | graph backend | force mode | atoms/s | configs/s | peak alloc MB | params | teacher E MAE | teacher F MAE | DFT E MAE | DFT F MAE |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {route} | {manifest} | {graph_backend} | {force_mode} | {atoms} | {configs} | {mem} | {params} | {te} | {tf} | {de} | {df} |".format(
                variant=row["variant"],
                route=fmt((row.get("tece_route") or {}).get("semantic_tier")),
                manifest=fmt(row.get("tece_path_manifest_hash")),
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
    append_manifest_group_section(lines, rows, projection_rows=projection_rows)
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
    parser.add_argument("--projection-diagnostic", action="append", type=Path, default=[])
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
    projection_rows = load_projection_diagnostic_rows(args.projection_diagnostic)
    payload = {
        "students": rows,
        "projection_diagnostics": projection_rows,
        "manifest_groups": manifest_group_rows(rows, projection_rows=projection_rows),
        "dft_force_pareto_front": pareto_front_rows(rows, error_key="dft_f_mae_mev_a"),
        "teacher_force_pareto_front": pareto_front_rows(rows, error_key="teacher_f_mae_mev_a"),
        "baselines": baselines,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(format_markdown(rows, baselines=baselines, projection_rows=projection_rows), encoding="utf-8")
    print(args.output_md)
    print(args.output_json)


if __name__ == "__main__":
    main()
