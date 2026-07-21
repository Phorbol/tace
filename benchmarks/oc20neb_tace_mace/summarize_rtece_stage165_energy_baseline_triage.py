#!/usr/bin/env python3
"""Stage165: diagnose low-frequency raw-energy baseline residuals.

This script intentionally treats case/group offsets as a diagnostic, not as a
production correction. Per-case IDs are not deployable model inputs; the point is
quantifying how much raw E RMSE lives in low-frequency case/site/adsorbate
baseline modes that a lightweight rTECE student must represent through symmetry-
preserving scalar paths, explicit baselines, or distillation.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_INPUTS = [
    Path("runs/oc20neb_tace_mace/rtece-stage157-relative-neb-loss/relative_loss_runs/stage157_direct_b32_rel0_mixed2048/stage157_direct_b32_rel0_mixed2048_dft_benchmark.json"),
    Path("runs/oc20neb_tace_mace/rtece-stage157-relative-neb-loss/relative_loss_runs/stage157_direct_b32_rel0p25_mixed2048/stage157_direct_b32_rel0p25_mixed2048_dft_benchmark.json"),
    Path("runs/oc20neb_tace_mace/rtece-stage157-relative-neb-loss/relative_loss_runs/stage157_direct_b32_rel1p0_mixed2048/stage157_direct_b32_rel1p0_mixed2048_dft_benchmark.json"),
]
DEFAULT_OUTPUT_JSON = Path("runs/oc20neb_tace_mace/rtece-stage165-energy-baseline-triage/stage165_energy_baseline_triage.json")
DEFAULT_OUTPUT_MD = Path("runs/oc20neb_tace_mace/rtece-stage165-energy-baseline-triage/stage165_energy_baseline_triage.md")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _variant_from_payload(path: Path, payload: Mapping[str, Any]) -> str:
    value = payload.get("variant")
    if value:
        return str(value)
    return path.stem.replace("_dft_benchmark", "")


def case_family(case_id: str) -> str:
    text = str(case_id)
    if "_id_" in text:
        return text.split("_id_", 1)[0]
    if "-" in text:
        return text.split("-", 1)[0]
    if "_" in text:
        return text.split("_", 1)[0]
    return text or "unknown"


def _top_case_offsets(offsets: Mapping[str, Any], *, top_k: int) -> list[dict[str, Any]]:
    rows = []
    for case_id, offset in offsets.items():
        offset_ev = _float_or_none(offset)
        if offset_ev is None:
            continue
        rows.append({
            "case_id": str(case_id),
            "family": case_family(str(case_id)),
            "offset_eV_per_atom": offset_ev,
            "abs_offset_mev_atom": abs(offset_ev) * 1000.0,
        })
    rows.sort(key=lambda row: (-float(row["abs_offset_mev_atom"]), row["case_id"]))
    return rows[: max(0, int(top_k))]


def _family_summary(offsets: Mapping[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for case_id, offset in offsets.items():
        offset_ev = _float_or_none(offset)
        if offset_ev is None:
            continue
        grouped[case_family(str(case_id))].append(offset_ev * 1000.0)
    rows = []
    for family, values in grouped.items():
        arr = values
        abs_values = [abs(v) for v in arr]
        rmse = math.sqrt(sum(v * v for v in arr) / len(arr)) if arr else 0.0
        rows.append({
            "family": family,
            "count": len(arr),
            "mean_offset_mev_atom": sum(arr) / len(arr) if arr else 0.0,
            "mae_offset_mev_atom": sum(abs_values) / len(abs_values) if abs_values else 0.0,
            "rmse_offset_mev_atom": rmse,
            "max_abs_offset_mev_atom": max(abs_values) if abs_values else 0.0,
        })
    rows.sort(key=lambda row: (-float(row["rmse_offset_mev_atom"]), str(row["family"])))
    return rows


def _explained_fraction(raw_rmse: float | None, group_rmse: float | None) -> float | None:
    if raw_rmse is None or group_rmse is None or raw_rmse <= 0.0:
        return None
    return 1.0 - (group_rmse * group_rmse) / (raw_rmse * raw_rmse)


def summarize_stage165_energy_baseline_triage(
    benchmark_paths: Sequence[str | Path] | None = None,
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    paths = [Path(p) for p in (benchmark_paths or DEFAULT_INPUTS)]
    rows = []
    for path in paths:
        payload = _read_json(path)
        offsets = payload.get("group_mean_offsets_eV_per_atom") or {}
        if not isinstance(offsets, Mapping):
            offsets = {}
        raw_rmse = _float_or_none(payload.get("raw_rmse_mev_atom", payload.get("rmse_e_mev_atom")))
        group_rmse = _float_or_none(payload.get("group_mean_offset_rmse_mev_atom"))
        top_offsets = _top_case_offsets(offsets, top_k=top_k)
        max_case = top_offsets[0]["abs_offset_mev_atom"] if top_offsets else None
        rows.append({
            "variant": _variant_from_payload(path, payload),
            "benchmark_path": str(path),
            "raw_e_rmse_mev_atom": raw_rmse,
            "raw_e_mae_mev_atom": _float_or_none(payload.get("raw_mae_mev_atom", payload.get("mae_e_mev_atom"))),
            "raw_e_max_abs_mev_atom": _float_or_none(payload.get("raw_max_abs_mev_atom", payload.get("max_abs_e_mev_atom"))),
            "group_offset_rmse_mev_atom": group_rmse,
            "group_offset_mae_mev_atom": _float_or_none(payload.get("group_mean_offset_mae_mev_atom")),
            "group_offset_max_abs_mev_atom": _float_or_none(payload.get("group_mean_offset_max_abs_mev_atom")),
            "relative_image_rmse_mev_atom": _float_or_none(payload.get("relative_image_rmse_mev_atom")),
            "barrier_rmse_mev_atom": _float_or_none(payload.get("barrier_rmse_mev_atom")),
            "force_rmse_mev_a": _float_or_none(payload.get("rmse_f_mev_a")),
            "case_count": len(offsets),
            "case_offset_explained_raw_rmse_fraction": _explained_fraction(raw_rmse, group_rmse),
            "max_case_offset_abs_mev_atom": max_case,
            "top_case_offsets": top_offsets,
            "family_offset_summary": _family_summary(offsets),
        })
    return {
        "schema_version": "rtece_stage165_energy_baseline_triage.v1",
        "diagnostic_semantics": "case offsets are an oracle diagnostic, not a deployable correction or model input",
        "document_alignment": {
            "TECE_design_space": "separate architecture/truncation error from optimization/distillation error under E/F/V/H metrics",
            "rTECE_review": "raw absolute E remains required, but NEB relative energy and group-offset decompositions must be reported beside it",
        },
        "rows": rows,
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_stage165_energy_baseline_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Stage165 Energy Baseline Triage",
        "",
        "This treats per-case/group offsets as an oracle diagnostic, not a deployable correction or model input.",
        "The goal is to decide whether the bad raw energy needs a symmetry-preserving low-frequency baseline path, stronger front-loaded scalar representation, or true distillation rather than a larger final head.",
        "",
        "| variant | raw E RMSE | group-offset RMSE | explained raw RMSE | relative image RMSE | barrier RMSE | F RMSE | cases | max case offset |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("rows", []):
        explained = row.get("case_offset_explained_raw_rmse_fraction")
        lines.append(
            "| {variant} | {raw} | {group} | {explained} | {rel} | {barrier} | {force} | {cases} | {max_case} |".format(
                variant=row.get("variant"),
                raw=_fmt(row.get("raw_e_rmse_mev_atom")),
                group=_fmt(row.get("group_offset_rmse_mev_atom")),
                explained=_fmt(explained),
                rel=_fmt(row.get("relative_image_rmse_mev_atom")),
                barrier=_fmt(row.get("barrier_rmse_mev_atom")),
                force=_fmt(row.get("force_rmse_mev_a")),
                cases=_fmt(row.get("case_count"), digits=0),
                max_case=_fmt(row.get("max_case_offset_abs_mev_atom")),
            )
        )
    lines.append("")
    lines.append("## Largest Case Offsets")
    lines.append("")
    for row in summary.get("rows", []):
        lines.append(f"### {row.get('variant')}")
        lines.append("")
        lines.append("| case | family | offset | abs offset |")
        lines.append("|---|---|---:|---:|")
        for item in row.get("top_case_offsets", [])[:5]:
            lines.append(
                "| {case} | {family} | {offset} | {abs_offset} |".format(
                    case=item.get("case_id"),
                    family=item.get("family"),
                    offset=_fmt(item.get("offset_eV_per_atom"), digits=6),
                    abs_offset=_fmt(item.get("abs_offset_mev_atom")),
                )
            )
        lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("If the explained raw-RMSE fraction remains near one across relative-loss variants, the energy problem is dominated by low-frequency case/site/adsorbate baseline modes. That points to deployable baseline features or TECE semantic scalar paths, not per-case IDs and not final-head widening.")
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="append", type=Path, dest="benchmarks", help="DFT benchmark JSON; repeatable. Defaults to Stage157 rel0/rel0p25/rel1p0.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_stage165_energy_baseline_triage(args.benchmarks or DEFAULT_INPUTS, top_k=args.top_k)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_stage165_energy_baseline_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
