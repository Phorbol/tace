#!/usr/bin/env python3
"""Summarize rTECE benchmark, dimer, and rattle probes into physical Pareto rows."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_ratio(value: float | None, reference: float) -> float:
    if value is None:
        return 1.0e6
    if float(reference) <= 0.0:
        raise ValueError("score reference values must be positive")
    return float(value) / float(reference)


def _focus_group(summary: dict[str, Any], label: str) -> dict[str, Any] | None:
    for row in summary.get("focus_groups", []) or []:
        if row.get("label") == label:
            return row
    return None


def dimer_gate_summary(dimer_scan: dict[str, Any]) -> dict[str, Any]:
    pair_summaries = list(dimer_scan.get("pair_summaries") or [])
    total = len(pair_summaries)
    repulsive_count = 0
    nonfinite_count = 0
    short_forces: list[float] = []
    short_energy_lifts: list[float] = []
    for item in pair_summaries:
        summary = item.get("summary") or {}
        if bool(summary.get("short_force_repulsive")):
            repulsive_count += 1
        if bool(summary.get("has_nonfinite")):
            nonfinite_count += 1
        short_force = _finite_float(summary.get("short_force_parallel_ev_a"))
        if short_force is not None:
            short_forces.append(short_force)
        short_de = _finite_float(summary.get("short_minus_long_energy_eV"))
        if short_de is not None:
            short_energy_lifts.append(short_de)
    fraction = float(repulsive_count / total) if total else 0.0
    return {
        "dimer_num_pairs": int(total),
        "dimer_short_repulsive_count": int(repulsive_count),
        "dimer_short_repulsive_fraction": fraction,
        "dimer_nonfinite_pair_count": int(nonfinite_count),
        "dimer_gate_pass": bool(total > 0 and repulsive_count == total and nonfinite_count == 0),
        "dimer_min_short_force_ev_a": float(min(short_forces)) if short_forces else None,
        "dimer_max_abs_short_force_ev_a": float(max(abs(value) for value in short_forces)) if short_forces else None,
        "dimer_min_short_energy_lift_eV": float(min(short_energy_lifts)) if short_energy_lifts else None,
    }


def rattle_gate_summary(rattle_relax: dict[str, Any]) -> dict[str, Any]:
    summary = rattle_relax.get("summary") or {}
    cn = _focus_group(summary, "C_or_N") or {}
    chno_no_cn = _focus_group(summary, "CHNO_no_CN") or {}
    return {
        "rattle_converged_fraction": _finite_float(summary.get("converged_fraction")),
        "rattle_mean_final_rmsd_a": _finite_float(summary.get("mean_final_rmsd_a")),
        "rattle_max_fmax_ev_a": _finite_float(summary.get("max_fmax_ev_a")),
        "cn_rattle_final_rmsd_a": _finite_float(cn.get("mean_final_rmsd_a")),
        "cn_rattle_max_fmax_ev_a": _finite_float(cn.get("max_fmax_ev_a")),
        "chno_no_cn_rattle_final_rmsd_a": _finite_float(chno_no_cn.get("mean_final_rmsd_a")),
    }


def make_physical_pareto_row(
    variant: str,
    *,
    dft_benchmark: dict[str, Any],
    teacher_benchmark: dict[str, Any] | None = None,
    dimer_scan: dict[str, Any],
    rattle_relax: dict[str, Any],
    max_dft_f_mae_mev_a: float = 35.0,
    max_cn_rattle_rmsd_a: float = 0.20,
    max_rattle_fmax_ev_a: float = 0.40,
    dimer_energy_penalty_scale_ev: float = 0.05,
) -> dict[str, Any]:
    teacher_benchmark = teacher_benchmark or {}
    dft_f_mae = _finite_float(dft_benchmark.get("mae_f_mev_a"))
    dimer = dimer_gate_summary(dimer_scan)
    rattle = rattle_gate_summary(rattle_relax)
    benchmark_gate_pass = bool(dft_f_mae is not None and dft_f_mae <= float(max_dft_f_mae_mev_a))
    rattle_gate_pass = bool(
        rattle["cn_rattle_final_rmsd_a"] is not None
        and rattle["rattle_max_fmax_ev_a"] is not None
        and rattle["cn_rattle_final_rmsd_a"] <= float(max_cn_rattle_rmsd_a)
        and rattle["rattle_max_fmax_ev_a"] <= float(max_rattle_fmax_ev_a)
    )
    dimer_fail_fraction = 1.0 - float(dimer["dimer_short_repulsive_fraction"])
    nonfinite_penalty = float(dimer["dimer_nonfinite_pair_count"])
    min_short_energy_lift = _finite_float(dimer.get("dimer_min_short_energy_lift_eV"))
    dimer_energy_shape_penalty = max(0.0, -min_short_energy_lift) if min_short_energy_lift is not None else 0.0
    physical_score = (
        _safe_ratio(dft_f_mae, max_dft_f_mae_mev_a)
        + _safe_ratio(rattle["cn_rattle_final_rmsd_a"], max_cn_rattle_rmsd_a)
        + _safe_ratio(rattle["rattle_max_fmax_ev_a"], max_rattle_fmax_ev_a)
        + 2.0 * dimer_fail_fraction
        + nonfinite_penalty
        + _safe_ratio(dimer_energy_shape_penalty, dimer_energy_penalty_scale_ev)
    )
    manifest = dft_benchmark.get("tece_architecture_path_manifest") or dft_benchmark.get("tece_path_manifest") or {}
    manifest_hash = (
        dft_benchmark.get("tece_architecture_path_manifest_hash")
        or dft_benchmark.get("tece_path_manifest_hash")
        or (manifest.get("manifest_hash") if isinstance(manifest, dict) else None)
    )
    row = {
        "schema_version": "rtece_physical_pareto_row.v1",
        "variant": str(variant),
        "atoms_per_second": _finite_float(dft_benchmark.get("atoms_per_second")),
        "num_parameters": dft_benchmark.get("num_parameters"),
        "force_mode": dft_benchmark.get("force_mode"),
        "tece_path_manifest_hash": manifest_hash,
        "dft_e_mae_mev_atom": _finite_float(dft_benchmark.get("mae_e_mev_atom")),
        "dft_f_mae_mev_a": dft_f_mae,
        "dft_f_rmse_mev_a": _finite_float(dft_benchmark.get("rmse_f_mev_a")),
        "teacher_f_mae_mev_a": _finite_float(teacher_benchmark.get("mae_f_mev_a")),
        "teacher_f_rmse_mev_a": _finite_float(teacher_benchmark.get("rmse_f_mev_a")),
        "max_dft_f_mae_mev_a": float(max_dft_f_mae_mev_a),
        "max_cn_rattle_rmsd_a": float(max_cn_rattle_rmsd_a),
        "max_rattle_fmax_ev_a": float(max_rattle_fmax_ev_a),
        "benchmark_gate_pass": benchmark_gate_pass,
        "rattle_gate_pass": rattle_gate_pass,
        "dimer_energy_shape_penalty": float(dimer_energy_shape_penalty),
        "dimer_energy_penalty_scale_ev": float(dimer_energy_penalty_scale_ev),
        "physical_score": float(physical_score),
    }
    row.update(dimer)
    row.update(rattle)
    row["physical_gate_pass"] = bool(row["benchmark_gate_pass"] and row["dimer_gate_pass"] and row["rattle_gate_pass"])
    return row


def physical_pareto_front_rows(
    rows: list[dict[str, Any]],
    *,
    score_key: str = "physical_score",
    throughput_key: str = "atoms_per_second",
) -> list[dict[str, Any]]:
    valid = [row for row in rows if _finite_float(row.get(score_key)) is not None and _finite_float(row.get(throughput_key)) is not None]
    front = []
    for row in valid:
        speed = float(row[throughput_key])
        score = float(row[score_key])
        dominated = False
        for other in valid:
            if other is row:
                continue
            other_speed = float(other[throughput_key])
            other_score = float(other[score_key])
            no_worse = other_speed >= speed and other_score <= score
            strictly_better = other_speed > speed or other_score < score
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(row)
    return sorted(front, key=lambda row: (-float(row[throughput_key]), float(row[score_key]), str(row.get("variant"))))


def rank_physical_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            not bool(row.get("physical_gate_pass")),
            float(row.get("physical_score") or 1.0e30),
            -float(row.get("atoms_per_second") or 0.0),
            str(row.get("variant")),
        ),
    )


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_markdown(rows: list[dict[str, Any]], front: list[dict[str, Any]]) -> str:
    lines = [
        "# rTECE Physical Pareto Summary",
        "",
        "## Ranked Rows",
        "",
        "| variant | gate | score | atoms/s | DFT F MAE | dimer repulsive | C/N RMSD | max fmax | params |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {variant} | {gate} | {score} | {atoms} | {df} | {dimer} | {cn} | {fmax} | {params} |".format(
                variant=row.get("variant"),
                gate=fmt(row.get("physical_gate_pass")),
                score=fmt(row.get("physical_score")),
                atoms=fmt(row.get("atoms_per_second")),
                df=fmt(row.get("dft_f_mae_mev_a")),
                dimer=fmt(row.get("dimer_short_repulsive_fraction")),
                cn=fmt(row.get("cn_rattle_final_rmsd_a")),
                fmax=fmt(row.get("rattle_max_fmax_ev_a")),
                params=fmt(row.get("num_parameters")),
            )
        )
    if front:
        lines.extend([
            "",
            "## Physical Pareto Front",
            "",
            "| variant | score | atoms/s | DFT F MAE | C/N RMSD | max fmax |",
            "|---|---:|---:|---:|---:|---:|",
        ])
        for row in front:
            lines.append(
                "| {variant} | {score} | {atoms} | {df} | {cn} | {fmax} |".format(
                    variant=row.get("variant"),
                    score=fmt(row.get("physical_score")),
                    atoms=fmt(row.get("atoms_per_second")),
                    df=fmt(row.get("dft_f_mae_mev_a")),
                    cn=fmt(row.get("cn_rattle_final_rmsd_a")),
                    fmax=fmt(row.get("rattle_max_fmax_ev_a")),
                )
            )
    lines.extend([
        "",
        "## Gate Definition",
        "",
        "- `physical_score` is lower-is-better: normalized DFT force MAE + normalized C/N rattle RMSD + normalized rattle max fmax + dimer force/nonfinite penalties + short-range dimer energy-shape penalty.",
        "- `physical_gate_pass` requires benchmark, dimer, and rattle gates to pass at the configured thresholds.",
        "- This score is a checkpoint-selection aid, not a replacement for the TECE path manifest or full Pareto table.",
        "",
    ])
    return "\n".join(lines)


def parse_case_arg(value: str) -> tuple[str, Path, Path, Path, Path]:
    parts = value.split(":", 4)
    if len(parts) != 5:
        raise argparse.ArgumentTypeError("case must be variant:dft.json:teacher.json:dimer.json:rattle.json")
    return parts[0], Path(parts[1]), Path(parts[2]), Path(parts[3]), Path(parts[4])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case_arg, default=[])
    parser.add_argument("--max-dft-f-mae-mev-a", type=float, default=35.0)
    parser.add_argument("--max-cn-rattle-rmsd-a", type=float, default=0.20)
    parser.add_argument("--max-rattle-fmax-ev-a", type=float, default=0.40)
    parser.add_argument("--dimer-energy-penalty-scale-ev", type=float, default=0.05)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = rank_physical_rows([
        make_physical_pareto_row(
            variant,
            dft_benchmark=load_json(dft_path),
            teacher_benchmark=load_json(teacher_path),
            dimer_scan=load_json(dimer_path),
            rattle_relax=load_json(rattle_path),
            max_dft_f_mae_mev_a=args.max_dft_f_mae_mev_a,
            max_cn_rattle_rmsd_a=args.max_cn_rattle_rmsd_a,
            max_rattle_fmax_ev_a=args.max_rattle_fmax_ev_a,
            dimer_energy_penalty_scale_ev=args.dimer_energy_penalty_scale_ev,
        )
        for variant, dft_path, teacher_path, dimer_path, rattle_path in args.case
    ])
    front = physical_pareto_front_rows(rows)
    payload = {
        "schema_version": "rtece_physical_pareto_summary.v1",
        "thresholds": {
            "max_dft_f_mae_mev_a": args.max_dft_f_mae_mev_a,
            "max_cn_rattle_rmsd_a": args.max_cn_rattle_rmsd_a,
            "max_rattle_fmax_ev_a": args.max_rattle_fmax_ev_a,
            "dimer_energy_penalty_scale_ev": args.dimer_energy_penalty_scale_ev,
        },
        "rows": rows,
        "physical_pareto_front": front,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(format_markdown(rows, front), encoding="utf-8")
    print(args.output_md)
    print(args.output_json)


if __name__ == "__main__":
    main()
