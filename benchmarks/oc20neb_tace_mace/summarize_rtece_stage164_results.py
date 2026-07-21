#!/usr/bin/env python3
"""Summarize Stage164 force-protected uv results against the Stage157 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

DEFAULT_BASELINE_DFT = Path(
    "runs/oc20neb_tace_mace/rtece-stage157-relative-neb-loss/relative_loss_runs/"
    "stage157_direct_b32_rel0p25_mixed2048/stage157_direct_b32_rel0p25_mixed2048_dft_benchmark.json"
)
DEFAULT_MANIFEST = Path("runs/oc20neb_tace_mace/rtece-stage164-force-protected-uv/stage164_manifest.json")
DEFAULT_OUTPUT_JSON = Path("runs/oc20neb_tace_mace/rtece-stage164-force-protected-uv/stage164_results_summary.json")
DEFAULT_OUTPUT_MD = Path("runs/oc20neb_tace_mace/rtece-stage164-force-protected-uv/stage164_results_summary.md")


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def _metric(payload: Mapping[str, Any] | None, key: str) -> float | None:
    if not payload:
        return None
    value = payload.get(key)
    return float(value) if value is not None else None


def _delta(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None:
        return None
    return float(candidate - baseline)


def _relative_regression(candidate: float | None, baseline: float | None) -> float | None:
    if candidate is None or baseline is None or baseline <= 0.0:
        return None
    return max(0.0, float(candidate - baseline) / max(abs(float(baseline)), 1.0e-12))


def _row_dir(manifest: Mapping[str, Any], variant: str) -> Path:
    run_root = Path(str((manifest.get("artifacts") or {}).get("run_root", "")))
    base = run_root / variant
    nested = base / variant
    return nested if nested.exists() else base


def summarize_stage164_results(
    manifest: Mapping[str, Any],
    *,
    baseline_dft_benchmark: str | Path = DEFAULT_BASELINE_DFT,
) -> dict[str, Any]:
    baseline_path = Path(baseline_dft_benchmark)
    baseline = _read_json(baseline_path)
    gate = manifest.get("active_set_gate") or {}
    force_gate = float(gate.get("max_force_regression_fraction", 0.10))
    rows = []
    for source_row in manifest.get("rows", []):
        variant = str(source_row.get("variant") or source_row.get("name"))
        run_dir = _row_dir(manifest, variant)
        dft_path = run_dir / f"{variant}_dft_benchmark.json"
        teacher_path = run_dir / f"{variant}_teacher_benchmark.json"
        train_path = run_dir / "train_summary.json"
        dft = _read_json(dft_path)
        teacher = _read_json(teacher_path)
        train = _read_json(train_path)
        raw_e = _metric(dft, "rmse_e_mev_atom")
        baseline_raw_e = _metric(baseline, "rmse_e_mev_atom")
        f_rmse = _metric(dft, "rmse_f_mev_a")
        baseline_f_rmse = _metric(baseline, "rmse_f_mev_a")
        force_regression = _relative_regression(f_rmse, baseline_f_rmse)
        artifact_status = "complete" if dft else "pending"
        row = {
            "variant": variant,
            "artifact_status": artifact_status,
            "train_summary_status": "found" if train else "missing",
            "dft_benchmark_status": "found" if dft else "missing",
            "teacher_benchmark_status": "found" if teacher else "missing",
            "train_summary_path": str(train_path),
            "dft_benchmark_path": str(dft_path),
            "teacher_benchmark_path": str(teacher_path),
            "baseline_dft_benchmark_path": str(baseline_path),
            "active_set_force_gate": force_gate,
            "baseline_dft_e_rmse_mev_atom": baseline_raw_e,
            "baseline_dft_group_offset_rmse_mev_atom": _metric(baseline, "group_mean_offset_rmse_mev_atom"),
            "baseline_dft_relative_image_rmse_mev_atom": _metric(baseline, "relative_image_rmse_mev_atom"),
            "baseline_dft_barrier_rmse_mev_atom": _metric(baseline, "barrier_rmse_mev_atom"),
            "baseline_dft_f_rmse_mev_a": baseline_f_rmse,
            "baseline_dft_f_max_mev_a": _metric(baseline, "max_abs_f_mev_a"),
            "baseline_atoms_per_second": _metric(baseline, "atoms_per_second"),
            "dft_e_rmse_mev_atom": raw_e,
            "dft_e_mae_mev_atom": _metric(dft, "mae_e_mev_atom"),
            "dft_e_max_mev_atom": _metric(dft, "max_abs_e_mev_atom"),
            "dft_group_offset_rmse_mev_atom": _metric(dft, "group_mean_offset_rmse_mev_atom"),
            "dft_first_anchor_rmse_mev_atom": _metric(dft, "first_image_anchor_rmse_mev_atom"),
            "dft_relative_image_rmse_mev_atom": _metric(dft, "relative_image_rmse_mev_atom"),
            "dft_barrier_rmse_mev_atom": _metric(dft, "barrier_rmse_mev_atom"),
            "dft_f_rmse_mev_a": f_rmse,
            "dft_f_mae_mev_a": _metric(dft, "mae_f_mev_a"),
            "dft_f_max_mev_a": _metric(dft, "max_abs_f_mev_a"),
            "atoms_per_second": _metric(dft, "atoms_per_second"),
            "dft_e_rmse_delta_mev_atom": _delta(raw_e, baseline_raw_e),
            "dft_group_offset_rmse_delta_mev_atom": _delta(_metric(dft, "group_mean_offset_rmse_mev_atom"), _metric(baseline, "group_mean_offset_rmse_mev_atom")),
            "dft_relative_image_rmse_delta_mev_atom": _delta(_metric(dft, "relative_image_rmse_mev_atom"), _metric(baseline, "relative_image_rmse_mev_atom")),
            "dft_barrier_rmse_delta_mev_atom": _delta(_metric(dft, "barrier_rmse_mev_atom"), _metric(baseline, "barrier_rmse_mev_atom")),
            "dft_f_rmse_delta_mev_a": _delta(f_rmse, baseline_f_rmse),
            "dft_f_max_delta_mev_a": _delta(_metric(dft, "max_abs_f_mev_a"), _metric(baseline, "max_abs_f_mev_a")),
            "atoms_per_second_delta": _delta(_metric(dft, "atoms_per_second"), _metric(baseline, "atoms_per_second")),
            "force_rmse_regression_fraction": force_regression,
            "force_gate_passed": None if force_regression is None else bool(force_regression <= force_gate + 1.0e-12),
            "energy_rmse_improved": None if raw_e is None or baseline_raw_e is None else bool(raw_e < baseline_raw_e),
        }
        rows.append(row)
    return {
        "schema_version": "rtece_stage164_results_summary.v1",
        "stage": "stage164_force_protected_uv_training",
        "baseline_variant": "stage157_direct_b32_rel0p25_mixed2048",
        "force_gate": force_gate,
        "rows": rows,
    }


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_stage164_results_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Stage164 Results Summary",
        "",
        f"Baseline: `{summary.get('baseline_variant')}`. Force gate: sampled/benchmark F RMSE regression <= `{summary.get('force_gate')}`.",
        "",
        "| variant | status | E RMSE | dE RMSE | group-offset RMSE | rel image RMSE | barrier RMSE | F RMSE | dF RMSE | force regression | force gate | atoms/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for row in summary.get("rows", []):
        lines.append(
            "| {variant} | {status} | {e} | {de} | {group} | {rel} | {barrier} | {f} | {df} | {freg} | {fgate} | {atoms} |".format(
                variant=row.get("variant"),
                status=row.get("artifact_status"),
                e=_fmt(row.get("dft_e_rmse_mev_atom")),
                de=_fmt(row.get("dft_e_rmse_delta_mev_atom")),
                group=_fmt(row.get("dft_group_offset_rmse_mev_atom")),
                rel=_fmt(row.get("dft_relative_image_rmse_mev_atom")),
                barrier=_fmt(row.get("dft_barrier_rmse_mev_atom")),
                f=_fmt(row.get("dft_f_rmse_mev_a")),
                df=_fmt(row.get("dft_f_rmse_delta_mev_a")),
                freg=_fmt(row.get("force_rmse_regression_fraction")),
                fgate=_fmt(row.get("force_gate_passed")),
                atoms=_fmt(row.get("atoms_per_second"), digits=0),
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--baseline-dft-benchmark", type=Path, default=DEFAULT_BASELINE_DFT)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    summary = summarize_stage164_results(manifest, baseline_dft_benchmark=args.baseline_dft_benchmark)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_stage164_results_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
