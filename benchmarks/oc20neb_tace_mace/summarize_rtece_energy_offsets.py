#!/usr/bin/env python3
"""Summarize rTECE raw-energy error into gauge, relative-shape, and force terms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _metric(payload: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = payload.get(key)
        if value is not None:
            return float(value)
    return None


def _safe_removed_fraction(raw_rmse: float | None, corrected_rmse: float | None) -> float | None:
    if raw_rmse is None or corrected_rmse is None or raw_rmse <= 0.0:
        return None
    return float(1.0 - corrected_rmse / raw_rmse)


def _top_group_offsets(payload: Mapping[str, Any], *, top_k: int) -> list[dict[str, float | str]]:
    offsets = payload.get("group_mean_offsets_eV_per_atom") or {}
    if not isinstance(offsets, Mapping):
        return []
    rows = []
    for group, value in offsets.items():
        offset_mev = float(value) * 1000.0
        rows.append(
            {
                "group": str(group),
                "offset_mev_atom": offset_mev,
                "abs_offset_mev_atom": abs(offset_mev),
            }
        )
    return sorted(rows, key=lambda row: (-float(row["abs_offset_mev_atom"]), str(row["group"])))[: max(int(top_k), 0)]


def summarize_energy_offset_benchmarks(
    benchmarks: Sequence[Mapping[str, Any]],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    rows = []
    for spec in benchmarks:
        path = Path(str(spec["path"]))
        payload = _read_json(path)
        raw_rmse = _metric(payload, "raw_rmse_mev_atom", "rmse_e_mev_atom")
        group_rmse = _metric(payload, "group_mean_offset_rmse_mev_atom")
        row = {
            "name": str(spec.get("name") or path.stem),
            "path": str(path),
            "configs": payload.get("configs"),
            "atoms": payload.get("atoms"),
            "raw_e_rmse_mev_atom": raw_rmse,
            "raw_e_mae_mev_atom": _metric(payload, "raw_mae_mev_atom", "mae_e_mev_atom"),
            "raw_e_max_mev_atom": _metric(payload, "raw_max_abs_mev_atom", "max_abs_e_mev_atom"),
            "global_offset_e_rmse_mev_atom": _metric(payload, "global_offset_rmse_mev_atom"),
            "group_offset_e_rmse_mev_atom": group_rmse,
            "first_anchor_e_rmse_mev_atom": _metric(payload, "first_image_anchor_rmse_mev_atom"),
            "relative_image_e_rmse_mev_atom": _metric(payload, "relative_image_rmse_mev_atom"),
            "barrier_e_rmse_mev_atom": _metric(payload, "barrier_rmse_mev_atom"),
            "force_rmse_mev_a": _metric(payload, "rmse_f_mev_a"),
            "force_mae_mev_a": _metric(payload, "mae_f_mev_a"),
            "force_max_mev_a": _metric(payload, "max_abs_f_mev_a"),
            "atoms_per_second": _metric(payload, "atoms_per_second", "atoms_per_s"),
            "num_groups": payload.get("num_groups"),
            "raw_rmse_removed_by_group_offset_fraction": _safe_removed_fraction(raw_rmse, group_rmse),
            "top_group_offsets_mev_atom": _top_group_offsets(payload, top_k=top_k),
        }
        rows.append(row)
    return {
        "schema_version": "rtece_energy_offset_summary.v1",
        "primary_energy_question": "raw_energy_rmse_vs_case_group_offset_and_relative_neb_shape",
        "rows": rows,
    }


def _fmt(value: object, *, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_offsets(offsets: Sequence[Mapping[str, Any]]) -> str:
    if not offsets:
        return "NA"
    return ", ".join(f"{row['group']}:{float(row['offset_mev_atom']):+.3f}" for row in offsets)


def render_energy_offset_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# rTECE Energy Offset Summary",
        "",
        "This separates raw absolute energy error from case/group offsets, relative NEB shape, force error, and throughput.",
        "",
        "| row | raw E RMSE | E MAE | E max | group-offset E RMSE | removed by group offset | first-anchor E RMSE | rel image RMSE | barrier RMSE | F RMSE | F max | atoms/s | top case offsets meV/atom |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.get("rows", []):
        removed = row.get("raw_rmse_removed_by_group_offset_fraction")
        lines.append(
            "| {name} | {raw} | {mae} | {emax} | {group} | {removed} | {anchor} | {rel} | {barrier} | {frmse} | {fmax} | {atoms_s} | {offsets} |".format(
                name=row["name"],
                raw=_fmt(row.get("raw_e_rmse_mev_atom")),
                mae=_fmt(row.get("raw_e_mae_mev_atom")),
                emax=_fmt(row.get("raw_e_max_mev_atom")),
                group=_fmt(row.get("group_offset_e_rmse_mev_atom")),
                removed="NA" if removed is None else f"{float(removed) * 100.0:.1f}%",
                anchor=_fmt(row.get("first_anchor_e_rmse_mev_atom")),
                rel=_fmt(row.get("relative_image_e_rmse_mev_atom")),
                barrier=_fmt(row.get("barrier_e_rmse_mev_atom")),
                frmse=_fmt(row.get("force_rmse_mev_a")),
                fmax=_fmt(row.get("force_max_mev_a")),
                atoms_s=_fmt(row.get("atoms_per_second"), digits=0),
                offsets=_fmt_offsets(row.get("top_group_offsets_mev_atom") or []),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation rule: if group-offset RMSE is much smaller than raw E RMSE, the dominant raw-energy error is a case/slab/adsorbate baseline term; representation changes must still be checked against relative-image/barrier and force RMSE/max.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="append", required=True, help="NAME=PATH or PATH")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    return parser.parse_args()


def _parse_benchmark(value: str) -> dict[str, str]:
    if "=" in value:
        name, path = value.split("=", 1)
        return {"name": name, "path": path}
    path = Path(value)
    return {"name": path.stem, "path": str(path)}


def main() -> None:
    args = parse_args()
    summary = summarize_energy_offset_benchmarks([_parse_benchmark(item) for item in args.benchmark], top_k=args.top_k)
    text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text, encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_energy_offset_markdown(summary), encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
