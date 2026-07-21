#!/usr/bin/env python3
"""Build the Stage178 benchmark analysis summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_STAGE178_ROOT = Path("runs/oc20neb_tace_mace/rtece-stage178-representation-upgrade")
DEFAULT_STAGE177_AUDIT = Path("runs/oc20neb_tace_mace/rtece-stage177-unified-pareto-audit/stage177_pareto_audit.json")


def _read_json(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _metric(payload: dict[str, Any] | None, key: str) -> float | None:
    if payload is None:
        return None
    return _as_float(payload.get(key))


def _candidate_dirs(stage178_root: Path) -> list[Path]:
    diagnostics = stage178_root / "diagnostics"
    if not diagnostics.exists():
        return []
    return sorted(path for path in diagnostics.iterdir() if path.is_dir())


def _first_row(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    rows = payload.get("rows")
    if not rows:
        return None
    return dict(rows[0])


def _memory_mb(payload: dict[str, Any] | None, allocated: bool) -> float | None:
    if payload is None:
        return None
    key = "peak_allocated_mb" if allocated else "peak_reserved_mb"
    fallback = "cuda_max_memory_allocated_mb" if allocated else "cuda_max_memory_reserved_mb"
    return _metric(payload, key) or _metric(payload, fallback)


def _scaling_rows(candidate_dir: Path, variant: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(candidate_dir.glob(f"{variant}_scaling_limit*.json")):
        payload = _read_json(path)
        if payload is None:
            continue
        limit_text = path.stem.rsplit("limit", 1)[-1]
        try:
            limit_configs = int(limit_text)
        except ValueError:
            limit_configs = None
        rows.append(
            {
                "limit_configs": limit_configs,
                "configs": payload.get("configs"),
                "atoms": payload.get("atoms"),
                "atoms_per_second": _metric(payload, "atoms_per_second"),
                "seconds_per_pass": _metric(payload, "seconds_per_pass"),
                "peak_allocated_mb": _memory_mb(payload, allocated=True),
                "peak_reserved_mb": _memory_mb(payload, allocated=False),
            }
        )
    return rows


def _row_from_candidate(stage178_root: Path, candidate_dir: Path) -> dict[str, Any]:
    variant = candidate_dir.name
    dft = _read_json(candidate_dir / f"{variant}_dft_benchmark.json")
    teacher = _read_json(candidate_dir / f"{variant}_teacher_benchmark.json")
    physical = _first_row(_read_json(candidate_dir / f"{variant}_physical_pareto.json")) or {}
    train = _read_json(stage178_root / "results" / variant / "train_summary.json") or {}
    scaling = _scaling_rows(candidate_dir, variant)
    scaling_1024 = next((row for row in scaling if row.get("limit_configs") == 1024), None)
    if scaling_1024 is None and scaling:
        scaling_1024 = scaling[-1]

    return {
        "variant": variant,
        "num_parameters": (dft or {}).get("num_parameters"),
        "train_steps": train.get("steps"),
        "best_step": train.get("best_step"),
        "best_valid_loss": train.get("best_valid_loss"),
        "dft_e_rmse_mev_atom": _metric(dft, "rmse_e_mev_atom"),
        "dft_e_mae_mev_atom": _metric(dft, "mae_e_mev_atom"),
        "dft_e_max_mev_atom": _metric(dft, "max_abs_e_mev_atom"),
        "dft_f_rmse_mev_a": _metric(dft, "rmse_f_mev_a"),
        "dft_f_mae_mev_a": _metric(dft, "mae_f_mev_a"),
        "dft_f_max_mev_a": _metric(dft, "max_abs_f_mev_a"),
        "dft_relative_image_rmse_mev_atom": _metric(dft, "relative_image_rmse_mev_atom"),
        "dft_barrier_rmse_mev_atom": _metric(dft, "barrier_rmse_mev_atom"),
        "teacher_e_rmse_mev_atom": _metric(teacher, "rmse_e_mev_atom"),
        "teacher_f_rmse_mev_a": _metric(teacher, "rmse_f_mev_a"),
        "atoms_per_second_valid256": _metric(dft, "atoms_per_second"),
        "seconds_per_pass_valid256": _metric(dft, "seconds_per_pass"),
        "peak_allocated_mb_valid256": _memory_mb(dft, allocated=True),
        "peak_reserved_mb_valid256": _memory_mb(dft, allocated=False),
        "atoms_per_second_limit1024": None if scaling_1024 is None else scaling_1024.get("atoms_per_second"),
        "peak_allocated_mb_limit1024": None if scaling_1024 is None else scaling_1024.get("peak_allocated_mb"),
        "peak_reserved_mb_limit1024": None if scaling_1024 is None else scaling_1024.get("peak_reserved_mb"),
        "legacy_physical_gate_pass": physical.get("physical_gate_pass"),
        "legacy_rattle_gate_pass": physical.get("rattle_gate_pass"),
        "legacy_dimer_gate_pass": physical.get("dimer_gate_pass"),
        "rattle_relax_decision": "continuous_metric_only",
        "excluded_by_rattle_gate": False,
        "focus_rattle_final_rmsd_a": physical.get("focus_rattle_final_rmsd_a"),
        "focus_rattle_max_fmax_ev_a": physical.get("focus_rattle_max_fmax_ev_a"),
        "physical_score_legacy": physical.get("physical_score"),
        "scaling": scaling,
    }


def _sort_key_error(row: dict[str, Any]) -> tuple[float, float, float]:
    return (
        float("inf") if row.get("dft_f_rmse_mev_a") is None else float(row["dft_f_rmse_mev_a"]),
        float("inf") if row.get("dft_e_rmse_mev_atom") is None else float(row["dft_e_rmse_mev_atom"]),
        float("inf") if row.get("dft_f_max_mev_a") is None else float(row["dft_f_max_mev_a"]),
    )


def _sort_key_throughput(row: dict[str, Any]) -> float:
    value = row.get("atoms_per_second_limit1024") or row.get("atoms_per_second_valid256")
    return -float(value or 0.0)


def _dominates(a: dict[str, Any], b: dict[str, Any]) -> bool:
    metrics = (
        ("dft_f_rmse_mev_a", False),
        ("dft_e_rmse_mev_atom", False),
        ("dft_f_max_mev_a", False),
        ("atoms_per_second_limit1024", True),
    )
    better_or_equal = True
    strictly_better = False
    for key, higher_is_better in metrics:
        av = a.get(key)
        bv = b.get(key)
        if av is None or bv is None:
            return False
        avf = float(av)
        bvf = float(bv)
        if higher_is_better:
            better_or_equal = better_or_equal and avf >= bvf
            strictly_better = strictly_better or avf > bvf
        else:
            better_or_equal = better_or_equal and avf <= bvf
            strictly_better = strictly_better or avf < bvf
    return better_or_equal and strictly_better


def _pareto(rows: list[dict[str, Any]]) -> list[str]:
    return [
        row["variant"]
        for row in rows
        if not any(other is not row and _dominates(other, row) for other in rows)
    ]


def _prior_rows(stage177_audit: Path) -> list[dict[str, Any]]:
    payload = _read_json(stage177_audit)
    if payload is None:
        return []
    return list(payload.get("rows") or [])


def build_stage178_analysis(
    *,
    stage178_root: str | Path = DEFAULT_STAGE178_ROOT,
    stage177_audit: str | Path = DEFAULT_STAGE177_AUDIT,
) -> dict[str, Any]:
    root = Path(stage178_root)
    rows = [_row_from_candidate(root, candidate_dir) for candidate_dir in _candidate_dirs(root)]
    rows = sorted(rows, key=lambda row: row["variant"])
    error_sorted = sorted(rows, key=_sort_key_error)
    throughput_sorted = sorted(rows, key=_sort_key_throughput)
    prior = _prior_rows(Path(stage177_audit))
    return {
        "schema_version": "rtece_stage178_benchmark_analysis.v1",
        "stage": "stage178_representation_upgrade_after_local_l0_endpoint",
        "stage178_root": str(root),
        "rattle_relax_policy": "continuous_rmsd_after_relax_not_binary_gate",
        "primary_error_order": ["dft_f_rmse_mev_a", "dft_e_rmse_mev_atom", "dft_f_max_mev_a"],
        "rows": rows,
        "best_by_error": error_sorted[0] if error_sorted else None,
        "best_by_throughput": throughput_sorted[0] if throughput_sorted else None,
        "stage178_error_throughput_pareto_variants": _pareto(rows),
        "prior_stage177_rows": prior,
        "interpretation": [
            "Stage178 tests front-loaded representation capacity, not final-head widening.",
            "Rattle-relax is interpreted as continuous relax RMSD and force-tail evidence; legacy boolean flags are retained only as provenance.",
            "The L0 local/species front is the current Stage178 best by DFT F RMSE, E RMSE, F max, and throughput.",
            "The L1/T3 increments improve the legacy rattle-relax flag and force MAE, but worsen E RMSE, F RMSE, and force max tails in this training recipe.",
            "Stage177 NEP/DPA rows remain smoke baselines under non-matching throughput protocols, so Stage178 cannot claim superiority over production NEP/DPA yet.",
        ],
        "next_actions": [
            "Replace boolean rattle gating in future reports with continuous RMSD/force-tail thresholds selected per deployment task.",
            "Run matched NEP/DPA1-0-layer training and throughput on the same data, same hardware scope, and same physical tests.",
            "Use Stage178 evidence to prioritize renormalized initialization or teacher residual projection before adding more edge paths.",
            "Promote peak_allocated_mb and peak_reserved_mb into the unified Pareto audit schema.",
        ],
    }


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):.{digits}f}"
    return str(value)


def render_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Stage178 Benchmark Analysis",
        "",
        "This is a stage-level evidence summary, not a final Pareto-front claim.",
        "",
        f"Rattle-relax policy: `{analysis['rattle_relax_policy']}`.",
        "",
        "## Stage178 Rows",
        "",
        "| variant | params | F RMSE | E RMSE | F max | E max | rel-img RMSE | barrier RMSE | atoms/s 256 | atoms/s 1024 | peak alloc 1024 MB | rattle RMSD A | rattle max F eV/A |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["variant"]),
                    _fmt(row.get("num_parameters"), 0),
                    _fmt(row.get("dft_f_rmse_mev_a")),
                    _fmt(row.get("dft_e_rmse_mev_atom")),
                    _fmt(row.get("dft_f_max_mev_a")),
                    _fmt(row.get("dft_e_max_mev_atom")),
                    _fmt(row.get("dft_relative_image_rmse_mev_atom")),
                    _fmt(row.get("dft_barrier_rmse_mev_atom")),
                    _fmt(row.get("atoms_per_second_valid256"), 0),
                    _fmt(row.get("atoms_per_second_limit1024"), 0),
                    _fmt(row.get("peak_allocated_mb_limit1024")),
                    _fmt(row.get("focus_rattle_final_rmsd_a")),
                    _fmt(row.get("focus_rattle_max_fmax_ev_a")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Training",
            "",
            "| variant | steps | best step | best valid loss |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in analysis["rows"]:
        lines.append(
            f"| {row['variant']} | {_fmt(row.get('train_steps'), 0)} | {_fmt(row.get('best_step'), 0)} | {_fmt(row.get('best_valid_loss'), 6)} |"
        )
    lines.extend(
        [
            "",
            "## Prior Stage177 Context",
            "",
            "| candidate | family | F RMSE | E RMSE | F max | E max | atoms/s | note |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in analysis.get("prior_stage177_rows", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("candidate")),
                    str(row.get("family")),
                    _fmt(row.get("dft_f_rmse_mev_a")),
                    _fmt(row.get("dft_e_rmse_mev_atom")),
                    _fmt(row.get("dft_f_max_mev_a")),
                    _fmt(row.get("dft_e_max_mev_atom")),
                    _fmt(row.get("atoms_per_second"), 0),
                    str(row.get("throughput_status") or ""),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Interpretation", ""])
    for item in analysis["interpretation"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Next Actions", ""])
    for item in analysis["next_actions"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def write_stage178_analysis(
    *,
    stage178_root: str | Path = DEFAULT_STAGE178_ROOT,
    stage177_audit: str | Path = DEFAULT_STAGE177_AUDIT,
) -> dict[str, Any]:
    root = Path(stage178_root)
    analysis = build_stage178_analysis(stage178_root=root, stage177_audit=stage177_audit)
    json_path = root / "stage178_benchmark_analysis.json"
    md_path = root / "stage178_benchmark_analysis.md"
    json_path.write_text(json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(analysis), encoding="utf-8")
    return {"analysis": analysis, "json_path": str(json_path), "md_path": str(md_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage178-root", type=Path, default=DEFAULT_STAGE178_ROOT)
    parser.add_argument("--stage177-audit", type=Path, default=DEFAULT_STAGE177_AUDIT)
    args = parser.parse_args()
    result = write_stage178_analysis(stage178_root=args.stage178_root, stage177_audit=args.stage177_audit)
    print(json.dumps({key: value for key, value in result.items() if key != "analysis"}, indent=2))


if __name__ == "__main__":
    main()
