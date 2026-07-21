#!/usr/bin/env python3
"""Create a unified Stage177 Pareto audit from existing lightweight artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_ROOT = Path("runs/oc20neb_tace_mace/rtece-stage177-unified-pareto-audit")
DEFAULT_STAGE176_ROOT = Path("runs/oc20neb_tace_mace/rtece-stage176-production-local-l0-front")
DEFAULT_STAGE157_METRICS = Path("runs/oc20neb_tace_mace/rtece-stage157-relative-neb-loss/stage157_metrics.json")
DEFAULT_STAGE145_SUMMARY = Path("runs/oc20neb_tace_mace/community-baselines-stage145/stage145_results_summary.json")
DEFAULT_DFT_VALID_FILE = Path(
    "/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/"
    "runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz"
)
DEFAULT_TEACHER_VALID_FILE = Path("runs/oc20neb_tace_mace/tece-distill-20260717/teacher_valid.extxyz")


def _read_json(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        return None
    return float(value)


def _stage176_row(stage176_root: Path, audit_root: Path) -> dict[str, Any]:
    manifest = _read_json(stage176_root / "stage176_manifest.json") or {}
    train_summary = _read_json(stage176_root / "train_smoke" / "train_summary.json") or {}
    model_config = manifest.get("model_config") or {}
    route = ((manifest.get("model_manifest") or {}).get("route") or {})
    variant = str(model_config.get("variant") or train_summary.get("variant") or "stage176_local_l0_rank3")
    benchmark = _read_json(audit_root / "stage176_benchmark" / f"{variant}_dft_benchmark.json")
    teacher_benchmark = _read_json(audit_root / "stage176_benchmark" / f"{variant}_teacher_benchmark.json")
    physical = _read_json(audit_root / "stage176_physical" / f"{variant}_physical_pareto.json")
    physical_row = None
    if physical is not None and physical.get("rows"):
        physical_row = physical["rows"][0]
    scaling_limits = []
    for limit in (32, 128, 512, 1024):
        scaling = _read_json(audit_root / "stage176_benchmark" / f"{variant}_scaling_limit{limit}.json")
        if scaling is not None:
            scaling_limits.append(
                {
                    "limit_configs": int(limit),
                    "configs": scaling.get("configs"),
                    "atoms": scaling.get("atoms"),
                    "atoms_per_second": scaling.get("atoms_per_second"),
                    "seconds_per_pass": scaling.get("seconds_per_pass"),
                    "cuda_max_memory_allocated_mb": scaling.get("cuda_max_memory_allocated_mb"),
                }
            )
    gaps = []
    if benchmark is None or teacher_benchmark is None:
        gaps.append("dft_teacher_benchmark")
    if physical_row is None:
        gaps.extend(["dimer_scan", "rattle_relax"])
    if len(scaling_limits) < 4:
        gaps.append("atom_count_throughput_scaling")
    if not scaling_limits or any(row.get("cuda_max_memory_allocated_mb") is None for row in scaling_limits):
        gaps.append("memory_scaling")
    return {
        "candidate": variant,
        "family": "rtece",
        "descriptor_label": "local_l0_lowrank_density",
        "semantic_tier": route.get("semantic_tier"),
        "descriptor_dim": route.get("descriptor_dim"),
        "tece_retained_groups": route.get("retained_tece_groups", []),
        "benchmark_status": "found" if benchmark is not None and teacher_benchmark is not None else "missing_full_dft_teacher_benchmark",
        "physics_status": "found" if physical_row is not None else "missing_dimer_and_rattle_relax",
        "throughput_status": "atom_count_scaling_found_memory_missing" if scaling_limits else "missing_atom_count_scaling",
        "training_status": "train_smoke_completed" if train_summary else "train_smoke_missing",
        "best_step": train_summary.get("best_step"),
        "best_valid_loss": train_summary.get("best_valid_loss"),
        "train_steps": train_summary.get("steps"),
        "num_parameters": (benchmark or {}).get("num_parameters"),
        "dft_e_rmse_mev_atom": _metric(benchmark or {}, "rmse_e_mev_atom"),
        "dft_e_mae_mev_atom": _metric(benchmark or {}, "mae_e_mev_atom"),
        "dft_e_max_mev_atom": _metric(benchmark or {}, "max_abs_e_mev_atom"),
        "dft_f_rmse_mev_a": _metric(benchmark or {}, "rmse_f_mev_a"),
        "dft_f_mae_mev_a": _metric(benchmark or {}, "mae_f_mev_a"),
        "dft_f_max_mev_a": _metric(benchmark or {}, "max_abs_f_mev_a"),
        "dft_relative_image_rmse_mev_atom": _metric(benchmark or {}, "relative_image_rmse_mev_atom"),
        "dft_barrier_rmse_mev_atom": _metric(benchmark or {}, "barrier_rmse_mev_atom"),
        "teacher_e_rmse_mev_atom": _metric(teacher_benchmark or {}, "rmse_e_mev_atom"),
        "teacher_f_rmse_mev_a": _metric(teacher_benchmark or {}, "rmse_f_mev_a"),
        "atoms_per_second": _metric(benchmark or {}, "atoms_per_second"),
        "physical_gate_pass": None if physical_row is None else bool(physical_row.get("physical_gate_pass")),
        "physical_score": None if physical_row is None else physical_row.get("physical_score"),
        "dimer_gate_pass": None if physical_row is None else bool(physical_row.get("dimer_gate_pass")),
        "rattle_gate_pass": None if physical_row is None else bool(physical_row.get("rattle_gate_pass")),
        "focus_rattle_final_rmsd_a": None if physical_row is None else physical_row.get("focus_rattle_final_rmsd_a"),
        "focus_rattle_max_fmax_ev_a": None if physical_row is None else physical_row.get("focus_rattle_max_fmax_ev_a"),
        "scaling": scaling_limits,
        "rankable_now": False,
        "evidence_gaps": gaps,
    }


def _stage157_rows(stage157_metrics: Path) -> list[dict[str, Any]]:
    payload = _read_json(stage157_metrics)
    if payload is None:
        return []
    selected = []
    for row in payload.get("rows", []):
        if row.get("variant") == "stage157_direct_b32_rel0p25_mixed2048":
            selected.append(row)
    if not selected:
        selected = list(payload.get("rows", []))[:1]
    rows: list[dict[str, Any]] = []
    for row in selected:
        dft = row.get("dft") or {}
        rows.append(
            {
                "candidate": str(row.get("variant")),
                "family": "rtece",
                "descriptor_label": "stage157_direct_t3_cavity_vector",
                "semantic_tier": "T3_direct_edge_relational_scalar",
                "descriptor_dim": None,
                "tece_retained_groups": ["direct_edge_relational_scalar", "zbl_short_range_prior"],
                "benchmark_status": "found",
                "physics_status": "missing_or_not_in_stage157_summary",
                "throughput_status": "prebuilt_graph_model_only",
                "training_status": "completed",
                "best_step": row.get("best_step"),
                "best_valid_loss": row.get("best_valid_loss"),
                "num_parameters": row.get("num_parameters"),
                "dft_e_rmse_mev_atom": _metric(dft, "rmse_e_mev_atom"),
                "dft_e_mae_mev_atom": _metric(dft, "mae_e_mev_atom"),
                "dft_e_max_mev_atom": _metric(dft, "max_abs_e_mev_atom"),
                "dft_f_rmse_mev_a": _metric(dft, "rmse_f_mev_a"),
                "dft_f_mae_mev_a": _metric(dft, "mae_f_mev_a"),
                "dft_f_max_mev_a": _metric(dft, "max_abs_f_mev_a"),
                "dft_relative_image_rmse_mev_atom": _metric(dft, "relative_image_rmse_mev_atom"),
                "dft_barrier_rmse_mev_atom": _metric(dft, "barrier_rmse_mev_atom"),
                "atoms_per_second": _metric(dft, "atoms_per_second"),
                "rankable_now": False,
                "evidence_gaps": ["physical_triage", "matched_community_throughput_protocol"],
            }
        )
    return rows


def _stage145_rows(stage145_summary: Path) -> list[dict[str, Any]]:
    payload = _read_json(stage145_summary)
    if payload is None:
        return []
    rows: list[dict[str, Any]] = []
    for row in payload.get("rows", []):
        name = str(row.get("name"))
        rows.append(
            {
                "candidate": name,
                "family": str(row.get("engine") or "community"),
                "descriptor_label": row.get("descriptor_label"),
                "semantic_tier": "community_baseline",
                "descriptor_dim": None,
                "tece_retained_groups": [],
                "benchmark_status": str(row.get("dft_benchmark_status") or "found"),
                "physics_status": str(row.get("physical_status") or "missing"),
                "throughput_status": "community_engine_wall_time",
                "training_status": row.get("training_status"),
                "best_step": row.get("latest_training_step"),
                "num_parameters": row.get("num_parameters"),
                "dft_e_rmse_mev_atom": _metric(row, "dft_e_rmse_mev_atom"),
                "dft_e_mae_mev_atom": _metric(row, "dft_e_mae_mev_atom"),
                "dft_e_max_mev_atom": _metric(row, "dft_e_max_mev_atom"),
                "dft_f_rmse_mev_a": _metric(row, "dft_f_rmse_mev_a"),
                "dft_f_mae_mev_a": _metric(row, "dft_f_mae_mev_a"),
                "dft_f_max_mev_a": _metric(row, "dft_f_max_mev_a"),
                "dft_relative_image_rmse_mev_atom": _metric(row, "dft_relative_image_rmse_mev_atom"),
                "dft_barrier_rmse_mev_atom": _metric(row, "dft_barrier_rmse_mev_atom"),
                "atoms_per_second": _metric(row, "atoms_per_second"),
                "rankable_now": row.get("physical_status") == "found" and row.get("dft_benchmark_status", "found") == "found",
                "evidence_gaps": ["matched_rtece_throughput_protocol"],
            }
        )
    return rows


def _next_required_actions(stage176_root: Path, stage176_row: dict[str, Any]) -> list[dict[str, Any]]:
    best_checkpoint = stage176_root / "train_smoke" / "rtece_scalar_best.pt"
    gaps = set(stage176_row.get("evidence_gaps") or [])
    actions: list[dict[str, Any]] = []
    if "dft_teacher_benchmark" in gaps:
        actions.append(
            {
                "id": "stage176_dft_teacher_benchmark",
                "priority": "P0",
                "status": "wrapper_materialized",
                "reason": "Stage176 only has a train-smoke loss; it needs DFT and teacher E/F RMSE, MAE, max error, and relative NEB metrics.",
                "checkpoint": str(best_checkpoint),
            }
        )
    if "dimer_scan" in gaps:
        actions.append(
            {
                "id": "stage176_dimer_scan",
                "priority": "P0",
                "status": "wrapper_materialized",
                "reason": "Physical smoothness and short-range behavior must be checked before ranking a new front representation.",
            }
        )
    if "rattle_relax" in gaps:
        actions.append(
            {
                "id": "stage176_rattle_relax",
                "priority": "P0",
                "status": "wrapper_materialized",
                "reason": "C/N adsorbate RMSD and force-tail behavior remain external physical stress tests, not training gates.",
            }
        )
    if "atom_count_throughput_scaling" in gaps or "memory_scaling" in gaps:
        actions.append(
            {
                "id": "stage176_atom_count_throughput_scaling",
                "priority": "P0",
                "status": "memory_metrics_incomplete" if "memory_scaling" in gaps else "wrapper_materialized",
                "reason": "TECE design-space ranking requires hardware cost across atom counts and memory; current Stage176 CUDA memory fields are missing.",
            }
        )
    if stage176_row.get("benchmark_status") == "found" and not stage176_row.get("physical_gate_pass", False):
        actions.append(
            {
                "id": "stage178_representation_upgrade_after_local_l0_endpoint",
                "priority": "P0",
                "status": "design_required",
                "reason": "Stage176 proves a fast local-L0 endpoint but fails force/energy/physical gates; next algorithm step should add document-grounded representation capacity, not final-head width.",
            }
        )
    actions.append(
        {
            "id": "community_baseline_protocol_alignment",
            "priority": "P1",
            "status": "analysis_required",
            "reason": "Stage145 community atoms/s, Stage157 prebuilt-graph atoms/s, and Stage176 ASE/autograd atoms/s are not directly comparable yet.",
        }
    )
    return actions


def make_stage177_audit(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    stage176_root: str | Path = DEFAULT_STAGE176_ROOT,
    stage157_metrics: str | Path = DEFAULT_STAGE157_METRICS,
    stage145_summary: str | Path = DEFAULT_STAGE145_SUMMARY,
) -> dict[str, Any]:
    root = Path(output_root)
    s176 = Path(stage176_root)
    rows = [_stage176_row(s176, root)]
    rows.extend(_stage157_rows(Path(stage157_metrics)))
    rows.extend(_stage145_rows(Path(stage145_summary)))
    caveats = [
        "non_matching_throughput_protocols",
        "rankable_now_requires_matching_data_metric_physics_and_hardware_scope",
    ]
    if rows[0].get("benchmark_status") != "found":
        caveats.append("stage176_full_benchmark_missing")
    if any(str(row.get("physics_status")) != "found" for row in rows if row.get("family") == "rtece"):
        caveats.append("physical_triage_not_complete_for_all_rtece_rows")
    return {
        "schema_version": "rtece_stage177_unified_pareto_audit.v1",
        "stage": "stage177_unified_pareto_audit",
        "output_root": str(root),
        "primary_ranking_metric": "dft_f_rmse_mev_a",
        "secondary_metrics": [
            "dft_e_rmse_mev_atom",
            "dft_e_max_mev_atom",
            "dft_f_max_mev_a",
            "dft_relative_image_rmse_mev_atom",
            "dft_barrier_rmse_mev_atom",
            "atoms_per_second",
            "physical_status",
        ],
        "comparison_caveats": caveats,
        "source_artifacts": {
            "stage176_manifest": str(s176 / "stage176_manifest.json"),
            "stage176_train_summary": str(s176 / "train_smoke" / "train_summary.json"),
            "stage157_metrics": str(stage157_metrics),
            "stage145_summary": str(stage145_summary),
            "dft_valid_file": str(DEFAULT_DFT_VALID_FILE),
            "teacher_valid_file": str(DEFAULT_TEACHER_VALID_FILE),
        },
        "rows": rows,
        "next_required_actions": _next_required_actions(s176, rows[0]),
        "artifacts": {
            "json": str(root / "stage177_pareto_audit.json"),
            "markdown": str(root / "stage177_pareto_audit.md"),
            "stage176_benchmark_wrapper": str(root / "wrappers" / "stage176_benchmark_no_export.sbatch"),
            "stage176_physical_wrapper": str(root / "wrappers" / "stage176_physical_no_export.sbatch"),
        },
    }


def audit_stage177_payload(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload.get("rows") or []
    row_names = {row.get("candidate") for row in rows}
    required_rows = {
        "stage176_local_l0_rank3",
        "stage157_direct_b32_rel0p25_mixed2048",
        "nep4_mixed_smoke",
        "deepmd_dpa_like_mixed_smoke",
    }
    action_ids = {action.get("id") for action in payload.get("next_required_actions") or []}
    stage176_row = next((row for row in rows if row.get("candidate") == "stage176_local_l0_rank3"), {})
    gaps = set(stage176_row.get("evidence_gaps") or [])
    required_actions = {"community_baseline_protocol_alignment"}
    if "dft_teacher_benchmark" in gaps:
        required_actions.add("stage176_dft_teacher_benchmark")
    if "dimer_scan" in gaps:
        required_actions.add("stage176_dimer_scan")
    if "rattle_relax" in gaps:
        required_actions.add("stage176_rattle_relax")
    if "atom_count_throughput_scaling" in gaps or "memory_scaling" in gaps:
        required_actions.add("stage176_atom_count_throughput_scaling")
    if stage176_row.get("benchmark_status") == "found" and not stage176_row.get("physical_gate_pass", False):
        required_actions.add("stage178_representation_upgrade_after_local_l0_endpoint")
    checks = {
        "schema_version": payload.get("schema_version") == "rtece_stage177_unified_pareto_audit.v1",
        "required_rows": required_rows.issubset(row_names),
        "required_actions": required_actions.issubset(action_ids),
        "rmse_primary": payload.get("primary_ranking_metric") == "dft_f_rmse_mev_a",
        "stage176_not_rankable": any(
            row.get("candidate") == "stage176_local_l0_rank3" and row.get("rankable_now") is False for row in rows
        ),
        "comparison_caveats": "non_matching_throughput_protocols" in payload.get("comparison_caveats", []),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "schema_version": "rtece_stage177_unified_pareto_audit_audit.v1",
        "contract_pass": not failed,
        "failed_checks": failed,
        "checks": checks,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage177 Unified Pareto Audit",
        "",
        "This is an evidence audit, not a final Pareto-front claim.",
        "",
        "| candidate | family | rankable | F RMSE | E RMSE | F max | E max | rel-img RMSE | barrier RMSE | atoms/s | benchmark | physical | throughput |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for row in payload.get("rows", []):
        def fmt(value: Any) -> str:
            if value is None:
                return "missing"
            if isinstance(value, float):
                return f"{value:.3f}"
            return str(value)

        lines.append(
            "| {candidate} | {family} | {rankable} | {f_rmse} | {e_rmse} | {f_max} | {e_max} | {rel} | {barrier} | {atoms} | {bench} | {phys} | {thr} |".format(
                candidate=row.get("candidate"),
                family=row.get("family"),
                rankable=row.get("rankable_now"),
                f_rmse=fmt(row.get("dft_f_rmse_mev_a")),
                e_rmse=fmt(row.get("dft_e_rmse_mev_atom")),
                f_max=fmt(row.get("dft_f_max_mev_a")),
                e_max=fmt(row.get("dft_e_max_mev_atom")),
                rel=fmt(row.get("dft_relative_image_rmse_mev_atom")),
                barrier=fmt(row.get("dft_barrier_rmse_mev_atom")),
                atoms=fmt(row.get("atoms_per_second")),
                bench=row.get("benchmark_status"),
                phys=row.get("physics_status"),
                thr=row.get("throughput_status"),
            )
        )
    lines.extend(["", "## Caveats", ""])
    for caveat in payload.get("comparison_caveats", []):
        lines.append(f"- `{caveat}`")
    lines.extend(["", "## Next Required Actions", ""])
    for action in payload.get("next_required_actions", []):
        lines.append(f"- `{action['id']}` ({action['priority']}): {action['reason']}")
    return "\n".join(lines) + "\n"


def _wrapper_header(job_name: str) -> str:
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=03:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/{job_name}-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/{job_name}-%j.err

set -eo pipefail
set -x
PYTHONUNBUFFERED=1
TACE_ROOT=${{TACE_ROOT:-/home/gengjianrui/bin/tace}}
ENV_DIR=${{ENV_DIR:-/home/gengjianrui/bin/.venvs/tace-mace-cu126}}
TACE_PYTHON=${{TACE_PYTHON:-${{ENV_DIR}}/bin/python}}
PYTHONPATH=${{TACE_ROOT}}:${{PYTHONPATH:-}}
PATH=${{ENV_DIR}}/bin:${{PATH}}
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
TORCH_CUDA_ARCH_LIST=${{TORCH_CUDA_ARCH_LIST:-7.0}}
mkdir -p /home/gengjianrui/bin/logs
cd "${{TACE_ROOT}}"
env PYTHONUNBUFFERED="${{PYTHONUNBUFFERED}}" PYTHONPATH="${{PYTHONPATH}}" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${{TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD}}" TORCH_CUDA_ARCH_LIST="${{TORCH_CUDA_ARCH_LIST}}" "${{TACE_PYTHON}}" -V
nvidia-smi -L
"""


def _stage176_checkpoint(payload: dict[str, Any]) -> str:
    train_summary = payload.get("source_artifacts", {}).get("stage176_train_summary")
    if train_summary:
        return str(Path(train_summary).parent / "rtece_scalar_best.pt")
    return str(DEFAULT_STAGE176_ROOT / "train_smoke" / "rtece_scalar_best.pt")


def _write_stage176_benchmark_wrapper(payload: dict[str, Any], path: Path) -> None:
    source = payload.get("source_artifacts", {})
    checkpoint = _stage176_checkpoint(payload)
    dft_valid = source.get("dft_valid_file", str(DEFAULT_DFT_VALID_FILE))
    teacher_valid = source.get("teacher_valid_file", str(DEFAULT_TEACHER_VALID_FILE))
    out_root = Path(payload["output_root"]) / "stage176_benchmark"
    variant = "stage176_local_l0_rank3"
    commands = [
        _wrapper_header("rtece177-bench"),
        f'OUT_ROOT="{out_root}"\n',
        f'CHECKPOINT="{checkpoint}"\n',
        f'DFT_VALID_FILE="{dft_valid}"\n',
        f'TEACHER_VALID_FILE="{teacher_valid}"\n',
        'mkdir -p "${OUT_ROOT}"\n',
        'env PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" PYTHONPATH="${PYTHONPATH}" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" "${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py '
        '--model "${CHECKPOINT}" --configs "${DFT_VALID_FILE}" '
        f'--output "${{OUT_ROOT}}/{variant}_dft_benchmark.json" --variant {variant} '
        '--device cuda --default-dtype float32 --start-config 0 --limit-configs 1024 '
        '--measure-passes 5 --force-mode autograd --graph-construction-backend matscipy_neighborlist\n',
        'env PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" PYTHONPATH="${PYTHONPATH}" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" "${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py '
        '--model "${CHECKPOINT}" --configs "${TEACHER_VALID_FILE}" '
        f'--output "${{OUT_ROOT}}/{variant}_teacher_benchmark.json" --variant {variant} '
        '--device cuda --default-dtype float32 --start-config 0 --limit-configs 1024 '
        '--measure-passes 5 --force-mode autograd --graph-construction-backend matscipy_neighborlist\n',
    ]
    for limit in (32, 128, 512, 1024):
        commands.append(
            'env PYTHONUNBUFFERED="${PYTHONUNBUFFERED}" PYTHONPATH="${PYTHONPATH}" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD}" TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}" "${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py '
            '--model "${CHECKPOINT}" --configs "${DFT_VALID_FILE}" '
            f'--output "${{OUT_ROOT}}/{variant}_scaling_limit{limit}.json" --variant {variant} '
            f'--device cuda --default-dtype float32 --start-config 0 --limit-configs {limit} '
            '--measure-passes 5 --force-mode autograd --graph-construction-backend matscipy_neighborlist\n'
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(commands), encoding="utf-8")


def _write_stage176_physical_wrapper(payload: dict[str, Any], path: Path) -> None:
    source = payload.get("source_artifacts", {})
    checkpoint = _stage176_checkpoint(payload)
    dft_valid = source.get("dft_valid_file", str(DEFAULT_DFT_VALID_FILE))
    out_root = Path(payload["output_root"]) / "stage176_physical"
    benchmark_root = Path(payload["output_root"]) / "stage176_benchmark"
    variant = "stage176_local_l0_rank3"
    text = _wrapper_header("rtece177-phys") + f'''OUT_ROOT="{out_root}"
CHECKPOINT="{checkpoint}"
DFT_VALID_FILE="{dft_valid}"
DFT_BENCHMARK="{benchmark_root}/{variant}_dft_benchmark.json"
TEACHER_BENCHMARK="{benchmark_root}/{variant}_teacher_benchmark.json"
VARIANT="{variant}"
mkdir -p "${{OUT_ROOT}}"
env PYTHONUNBUFFERED="${{PYTHONUNBUFFERED}}" PYTHONPATH="${{PYTHONPATH}}" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${{TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD}}" TORCH_CUDA_ARCH_LIST="${{TORCH_CUDA_ARCH_LIST}}" "${{TACE_PYTHON}}" benchmarks/oc20neb_tace_mace/dimer_scan_rtece.py --checkpoint "${{CHECKPOINT}}" --output-json "${{OUT_ROOT}}/${{VARIANT}}_dimer.json" --output-md "${{OUT_ROOT}}/${{VARIANT}}_dimer.md" --route "${{VARIANT}}" --pairs C-N C-O C-H N-H O-H C-C N-N --num-points 24 --min-scale 0.5 --max-scale 5.0 --device cuda --default-dtype float32 --force-mode autograd --neighborlist-backend ase
env PYTHONUNBUFFERED="${{PYTHONUNBUFFERED}}" PYTHONPATH="${{PYTHONPATH}}" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${{TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD}}" TORCH_CUDA_ARCH_LIST="${{TORCH_CUDA_ARCH_LIST}}" "${{TACE_PYTHON}}" benchmarks/oc20neb_tace_mace/rattle_relax_rtece.py --checkpoint "${{CHECKPOINT}}" --configs "${{DFT_VALID_FILE}}" --output-json "${{OUT_ROOT}}/${{VARIANT}}_rattle.json" --output-md "${{OUT_ROOT}}/${{VARIANT}}_rattle.md" --route "${{VARIANT}}" --start-config 58 --limit-configs 8 --rattle-std 0.05 --rattle-seed 20260718 --fmax 0.05 --max-steps 10 --device cuda --default-dtype float32 --force-mode autograd --neighborlist-backend matscipy
env PYTHONUNBUFFERED="${{PYTHONUNBUFFERED}}" PYTHONPATH="${{PYTHONPATH}}" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD="${{TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD}}" TORCH_CUDA_ARCH_LIST="${{TORCH_CUDA_ARCH_LIST}}" "${{TACE_PYTHON}}" benchmarks/oc20neb_tace_mace/summarize_rtece_physical_pareto.py --case "${{VARIANT}}:${{DFT_BENCHMARK}}:${{TEACHER_BENCHMARK}}:${{OUT_ROOT}}/${{VARIANT}}_dimer.json:${{OUT_ROOT}}/${{VARIANT}}_rattle.json" --max-dft-f-rmse-mev-a 120.0 --max-dft-e-rmse-mev-atom 320.0 --max-dft-f-max-mev-a 2600.0 --max-dft-e-max-mev-atom 760.0 --rattle-focus-label C_or_N --max-focus-rattle-rmsd-a 0.35 --max-rattle-fmax-ev-a 1.00 --output-json "${{OUT_ROOT}}/${{VARIANT}}_physical_pareto.json" --output-md "${{OUT_ROOT}}/${{VARIANT}}_physical_pareto.md"
'''
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def materialize_stage177(payload: dict[str, Any]) -> dict[str, Any]:
    json_path = Path(payload["artifacts"]["json"])
    markdown_path = Path(payload["artifacts"]["markdown"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(payload), encoding="utf-8")
    benchmark_wrapper = Path(payload["artifacts"]["stage176_benchmark_wrapper"])
    physical_wrapper = Path(payload["artifacts"]["stage176_physical_wrapper"])
    _write_stage176_benchmark_wrapper(payload, benchmark_wrapper)
    _write_stage176_physical_wrapper(payload, physical_wrapper)
    return {
        "json": str(json_path),
        "markdown": str(markdown_path),
        "wrappers": {
            "stage176_benchmark": str(benchmark_wrapper),
            "stage176_physical": str(physical_wrapper),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage176-root", type=Path, default=DEFAULT_STAGE176_ROOT)
    parser.add_argument("--stage157-metrics", type=Path, default=DEFAULT_STAGE157_METRICS)
    parser.add_argument("--stage145-summary", type=Path, default=DEFAULT_STAGE145_SUMMARY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = make_stage177_audit(
        output_root=args.output_root,
        stage176_root=args.stage176_root,
        stage157_metrics=args.stage157_metrics,
        stage145_summary=args.stage145_summary,
    )
    materialize_stage177(payload)
    print(json.dumps(audit_stage177_payload(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
