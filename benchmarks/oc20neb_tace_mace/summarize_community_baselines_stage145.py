#!/usr/bin/env python3
"""Summarize Stage145 community baseline artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def _metric(payload: dict[str, Any] | None, key: str) -> float | None:
    if not payload:
        return None
    value = payload.get(key)
    return float(value) if value is not None else None


def summarize_stage145(manifest: Mapping[str, Any], output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    training_status = _read_json(root / "stage145_training_status.json") or {}
    training_by_name = {
        str(row["name"]): row
        for row in training_status.get("rows", [])
        if isinstance(row, Mapping) and "name" in row
    }
    rows = []
    for row in manifest.get("rows", []):
        train_dir = Path(row["train_dir"])
        row_name = row["name"]
        conversion = _read_json(train_dir / "conversion_summary.json")
        dft = _read_json(train_dir / f"{row_name}_dft_benchmark.json")
        physical = _read_json(train_dir / f"{row_name}_physical_pareto.json")
        training = training_by_name.get(str(row_name), {})
        rows.append(
            {
                "name": row["name"],
                "engine": row["engine"],
                "descriptor_label": row.get("descriptor_label"),
                "conversion_status": "found" if conversion else "missing",
                "training_status": training.get("training_status", "missing"),
                "latest_training_step": training.get("latest_step"),
                "training_target_step": training.get("target_step"),
                "training_f_rmse_val": training.get("latest_rmse_f_val"),
                "training_f_rmse_train": training.get("latest_rmse_f_train"),
                "training_e_rmse_val": training.get("latest_rmse_e_val"),
                "training_e_rmse_train": training.get("latest_rmse_e_train"),
                "dft_benchmark_status": "found" if dft else "missing",
                "physical_status": "found" if physical else "missing",
                "num_configs": conversion.get("num_configs") if conversion else None,
                "num_atoms": conversion.get("num_atoms") if conversion else None,
                "num_systems": conversion.get("num_systems") if conversion else None,
                "type_map": conversion.get("type_map") if conversion else None,
                "dft_f_rmse_mev_a": _metric(dft, "rmse_f_mev_a"),
                "dft_e_rmse_mev_atom": _metric(dft, "rmse_e_mev_atom"),
                "dft_f_mae_mev_a": _metric(dft, "mae_f_mev_a"),
                "dft_e_mae_mev_atom": _metric(dft, "mae_e_mev_atom"),
                "dft_f_max_mev_a": _metric(dft, "max_abs_f_mev_a"),
                "dft_e_max_mev_atom": _metric(dft, "max_abs_e_mev_atom"),
                "dft_e_bias_mev_atom": _metric(dft, "bias_e_mev_atom"),
                "atoms_per_second": _metric(dft, "atoms_per_second") or _metric(dft, "atoms_per_s"),
            }
        )
    summary = {
        "schema_version": "community_baselines_stage145_results.v1",
        "stage": "community_baselines_stage145",
        "primary_ranking_metric": "dft_f_rmse_mev_a",
        "rows": rows,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "stage145_results_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "stage145_results_summary.md").write_text(render_stage145_markdown(summary), encoding="utf-8")
    return summary


def _fmt(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_stage145_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Stage145 Community Baselines Summary",
        "",
        "Primary ranking metric: DFT force RMSE.",
        "",
        "| row | engine | conversion | training | DFT F RMSE | DFT E RMSE | atoms/s | physical |",
        "|---|---|---|---|---:|---:|---:|---|",
    ]
    for row in summary.get("rows", []):
        lines.append(
            "| {name} | {engine} | {conversion_status} | {training_status} | {f_rmse} | {e_rmse} | {atoms_s} | {physical_status} |".format(
                name=row["name"],
                engine=row["engine"],
                conversion_status=row["conversion_status"],
                training_status=row.get("training_status", "missing"),
                f_rmse=_fmt(row.get("dft_f_rmse_mev_a")),
                e_rmse=_fmt(row.get("dft_e_rmse_mev_atom")),
                atoms_s=_fmt(row.get("atoms_per_second")),
                physical_status=row["physical_status"],
            )
        )
    lines.append("")
    lines.append(
        "Missing values are explicit because conversion, training, benchmark, and physical triage complete at different times."
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("runs/oc20neb_tace_mace/community-baselines-stage145/stage145_manifest.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/oc20neb_tace_mace/community-baselines-stage145"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    print(json.dumps(summarize_stage145(manifest, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
