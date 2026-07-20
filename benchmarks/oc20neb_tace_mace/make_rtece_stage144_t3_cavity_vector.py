#!/usr/bin/env python3
"""Create Stage144 minimal T3 cavity-vector training wrappers."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.oc20neb_tace_mace.make_rtece_stage132_broad_teacher_distill import (
    DEFAULT_DFT_VALID,
    DEFAULT_TEACHER_VALID,
)
from benchmarks.oc20neb_tace_mace.make_rtece_stage143_semantic_active_set import DEFAULT_STAGE142_ROOT

DEFAULT_TRAIN_FILE = DEFAULT_STAGE142_ROOT / "weighted_train_base2048_plus_teacher_relax640_forceonly_eanchor.extxyz"
DEFAULT_STAGE143_RESULTS = Path("runs/oc20neb_tace_mace/rtece-stage143-semantic-active-set/stage143_results_summary.json")

STAGE144_VARIANT = "t3_l2_cavity_vector_cond32_h64"

T3_CAVITY_VECTOR_PATH_IDS = (
    "atomic.radial_density",
    "atomic.species_basis_density",
    "atomic.vector_norm",
    "atomic.vector_cross_radial_dot",
    "atomic.quadrupole_norm",
    "atomic.quadrupole_cross_radial_frobenius",
    "edge.cavity.vector_dot",
)


def _join_paths(path_ids: tuple[str, ...] | list[str]) -> str:
    return ",".join(str(path_id) for path_id in path_ids)


def _stage144_rows() -> list[dict[str, Any]]:
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import _row, _with_parameter_estimates

    row = _row(
        STAGE144_VARIANT,
        scalar_path_ids=_join_paths(T3_CAVITY_VECTOR_PATH_IDS),
        moment_l_max=2,
        learnable_radial_mixing=True,
        radial_species_adapter_channels=8,
        radial_species_adapter_scope="all",
        short_range_repulsion_potential="zbl",
        hidden_channels="64,64",
        num_radial=12,
        species_basis_channels=24,
        species_basis_mode="learnable_embedding",
        atomic_cross_radial_sketch_channels=3,
        atomic_cross_radial_projection="learnable",
        descriptor_conditioner="residual_mlp",
        descriptor_conditioner_hidden_channels=32,
        descriptor_bottleneck_dim=0,
        tece_axes=(
            "stage144_t3_cavity_vector",
            "stage143_active_set_guided",
            "minimal_edge_relational_scalar_increment",
            "cavity_vector_total_m0_sketch",
            "fixed_stage140_141_l2_conditioned_anchor",
            "front_representation_capacity_not_wider_head",
        ),
    )
    row["tece_tier"] = "T3_rtece_edge_relational_minimal"
    row["stage144_isolated_increment"] = "edge.cavity.vector_dot"
    row["stage143_active_set_source"] = str(DEFAULT_STAGE143_RESULTS)
    row["stage_basis"] = "stage144_minimal_t3_cavity_vector"
    row["review_basis"] = (
        "Stage143 descriptor projection showed that edge.cavity.vector_dot reduces the remaining L2 atomic "
        "descriptor residual. Stage144 isolates that single edge-relational scalar path against the fixed "
        "Stage140/141 L2 conditioned atomic anchor before adding cavity quadrupole or direct radial edge paths."
    )
    return _with_parameter_estimates([row])


def make_stage144_manifest(
    *,
    output_root: str | Path,
    train_file: str | Path = DEFAULT_TRAIN_FILE,
    train_valid_file: str | Path = DEFAULT_DFT_VALID,
    dft_valid_file: str | Path = DEFAULT_DFT_VALID,
    teacher_valid_file: str | Path = DEFAULT_TEACHER_VALID,
    limit_configs: int = 2688,
    valid_limit_configs: int = 256,
    bench_limit_configs: int = 1024,
    max_steps: int = 20000,
    lr_warmup_steps: int = 500,
    early_stopping_patience: int = 400,
) -> dict[str, Any]:
    if int(limit_configs) < 1:
        raise ValueError("limit_configs must be positive")
    if int(valid_limit_configs) < 1:
        raise ValueError("valid_limit_configs must be positive")
    if int(bench_limit_configs) < 1:
        raise ValueError("bench_limit_configs must be positive")

    root = Path(output_root)
    artifacts = {
        "wrapper_root": str(root / "t3_cavity_vector_wrappers"),
        "run_root": str(root / "t3_cavity_vector_runs"),
        "physical_wrapper_root": str(root / "physical_triage_wrappers"),
        "physical_run_root": str(root / "physical_triage"),
        "manifest": str(root / "stage144_manifest.json"),
        "manifest_audit": str(root / "stage144_manifest_audit.json"),
        "stage_plan": str(root / "stage144_plan.md"),
        "stage143_results": str(DEFAULT_STAGE143_RESULTS),
    }
    return {
        "schema_version": "rtece_stage144_t3_cavity_vector.v1",
        "stage": "stage144_t3_cavity_vector",
        "row_set": "stage144-t3-cavity-vector",
        "distillation_semantics": "stage143_active_set_guided_minimal_t3_edge_relational_training",
        "comparison_question": (
            "With the Stage140/141 L2 conditioned atomic anchor and Stage142 force-only teacher-relax training "
            "measure fixed, does adding exactly one edge.cavity.vector_dot architecture increment create a real "
            "RMSE/max-error/physical-tail Pareto movement that justifies the first true rTECE edge-relational cost?"
        ),
        "train_file": str(train_file),
        "train_valid_file": str(train_valid_file),
        "dft_valid_file": str(dft_valid_file),
        "teacher_valid_file": str(teacher_valid_file),
        "limit_configs": int(limit_configs),
        "valid_limit_configs": int(valid_limit_configs),
        "bench_limit_configs": int(bench_limit_configs),
        "max_steps": int(max_steps),
        "lr_warmup_steps": int(lr_warmup_steps),
        "early_stopping_patience": int(early_stopping_patience),
        "stage143_active_set_source": str(DEFAULT_STAGE143_RESULTS),
        "rows": [{"variant": str(row["name"]), **row} for row in _stage144_rows()],
        "artifacts": artifacts,
        "review_basis": [
            "TECE_design_space: rTECE is the middle tier with sparse edge-relational scalar ECE sketches after immediate scalarization.",
            "TECE_design_space: test one controlled semantic path increment and charge real hardware throughput, not just parameter count.",
            "rTECE_review: report E/F RMSE and max errors separately, then dimer/rattle physical probes; do not use MAE alone.",
            "Stage142: more same-window teacher-relax coverage did not close the tail, so do not add more data before checking operator-basis value.",
            "Stage143: edge.cavity.vector_dot is the minimal T3 active-set step; cavity quadrupole/direct radial remain deferred.",
        ],
    }


def audit_stage144_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    rows = list(payload.get("rows") or [])
    variants = [str(row.get("variant") or row.get("name")) for row in rows]
    row = rows[0] if rows else {}
    path_ids = tuple(part.strip() for part in str(row.get("scalar_path_ids") or "").split(",") if part.strip())
    checks = {
        "schema_version": payload.get("schema_version") == "rtece_stage144_t3_cavity_vector.v1",
        "stage": payload.get("stage") == "stage144_t3_cavity_vector",
        "row_set": payload.get("row_set") == "stage144-t3-cavity-vector",
        "distillation_semantics": payload.get("distillation_semantics")
        == "stage143_active_set_guided_minimal_t3_edge_relational_training",
        "single_variant": variants == [STAGE144_VARIANT],
        "exact_path_lattice": path_ids == T3_CAVITY_VECTOR_PATH_IDS,
        "only_minimal_edge_increment": "edge.cavity.vector_dot" in path_ids
        and "edge.cavity.quadrupole_frobenius" not in path_ids
        and "edge.direct.radial" not in path_ids,
        "fixed_l2_conditioned_front": int(row.get("moment_l_max", -1)) == 2
        and row.get("hidden_channels") == "64,64"
        and int(row.get("num_radial", -1)) == 12
        and int(row.get("species_basis_channels", -1)) == 24
        and row.get("species_basis_mode") == "learnable_embedding"
        and int(row.get("radial_species_adapter_channels", -1)) == 8
        and int(row.get("atomic_cross_radial_sketch_channels", -1)) == 3
        and row.get("atomic_cross_radial_projection") == "learnable"
        and row.get("descriptor_conditioner") == "residual_mlp"
        and int(row.get("descriptor_conditioner_hidden_channels", -1)) == 32
        and int(row.get("descriptor_bottleneck_dim", -1)) == 0,
        "zbl_contract": row.get("short_range_repulsion_potential") == "zbl",
        "stage143_source": str(payload.get("stage143_active_set_source", "")).endswith("stage143_results_summary.json"),
        "training_contract": int(payload.get("max_steps", 0)) > 0
        and int(payload.get("lr_warmup_steps", 0)) == 500
        and int(payload.get("early_stopping_patience", 0)) == 400,
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "schema_version": "rtece_stage144_manifest_audit.v1",
        "contract_pass": not failed,
        "failed_checks": failed,
        "checks": checks,
        "checked": {
            "variants": variants,
            "path_ids": list(path_ids),
            "train_file": payload.get("train_file"),
        },
    }


def _stage144_physical_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
    cases = []
    run_root = Path(payload["artifacts"]["run_root"])
    for row in payload["rows"]:
        variant = str(row["variant"])
        train_variant = str(row.get("train_variant") or variant)
        row_dir = run_root / variant / train_variant
        cases.append(
            {
                "variant": variant,
                "checkpoint": str(row_dir / "rtece_scalar_best.pt"),
                "dft_benchmark": str(row_dir / f"{train_variant}_dft_benchmark.json"),
                "teacher_benchmark": str(row_dir / f"{train_variant}_teacher_benchmark.json"),
                "selection_basis": "Stage144 minimal T3 edge.cavity.vector_dot increment from Stage143 semantic active set",
            }
        )
    return cases


def materialize_stage144(payload: dict[str, Any]) -> dict[str, Any]:
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import write_pareto_sweep
    from benchmarks.oc20neb_tace_mace.make_rtece_physical_triage import write_physical_triage_wrappers

    audit = audit_stage144_manifest(payload)
    if not audit["contract_pass"]:
        raise ValueError(f"Stage144 manifest audit failed: {audit['failed_checks']}")

    artifacts = payload["artifacts"]
    sweep = write_pareto_sweep(
        artifacts["wrapper_root"],
        run_root=artifacts["run_root"],
        train_file=payload["train_file"],
        train_valid_file=payload["train_valid_file"],
        dft_valid_file=payload["dft_valid_file"],
        teacher_valid_file=payload["teacher_valid_file"],
        rows=_stage144_rows(),
        row_set="stage144-t3-cavity-vector",
        limit_configs=int(payload["limit_configs"]),
        valid_limit_configs=int(payload["valid_limit_configs"]),
        bench_limit_configs=int(payload["bench_limit_configs"]),
        max_steps=int(payload["max_steps"]),
        batch_size=8,
        valid_batch_size=16,
        lr_warmup_steps=int(payload["lr_warmup_steps"]),
        early_stopping_patience=int(payload["early_stopping_patience"]),
        preflight_extxyz=False,
    )
    physical = write_physical_triage_wrappers(
        artifacts["physical_wrapper_root"],
        run_root=artifacts["physical_run_root"],
        configs=payload["dft_valid_file"],
        cases=_stage144_physical_cases(payload),
        stage="stage144_physical_triage",
        source="runs/oc20neb_tace_mace/rtece-stage144-t3-cavity-vector/stage144_plan.md",
        design_basis=(
            "Evaluate whether the minimal Stage143-selected edge.cavity.vector_dot T3 rTECE increment improves "
            "DFT RMSE/max tails or dimer/rattle physical probes before paying for cavity quadrupole/direct edge paths."
        ),
        job_name="rtece-phys144",
        rattle_start_config=58,
        rattle_limit_configs=8,
    )
    materialized = dict(payload)
    materialized["train_wrappers"] = [row["wrapper"] for row in sweep["rows"]]
    materialized["physical_wrappers"] = [row["wrapper"] for row in physical["cases"]]
    materialized["physical_cases"] = physical["cases"]
    write_manifest_files(materialized)
    return materialized


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage144 T3 Cavity Vector Plan",
        "",
        f"- stage: `{payload['stage']}`",
        f"- row set: `{payload['row_set']}`",
        f"- semantics: `{payload['distillation_semantics']}`",
        f"- train file: `{payload['train_file']}`",
        f"- train limit: `{payload['limit_configs']}` configs",
        f"- Stage143 source: `{payload['stage143_active_set_source']}`",
        "",
        "## Question",
        "",
        payload["comparison_question"],
        "",
        "## Rows",
        "",
        "| variant | params | representation params | readout params | isolated increment | scalar paths |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['variant']} | {row.get('num_parameters_estimate')} | "
            f"{row.get('representation_parameters_estimate')} | {row.get('readout_parameters_estimate')} | "
            f"{row.get('stage144_isolated_increment')} | {row.get('scalar_path_ids')} |"
        )
    lines.extend(["", "## Review Basis", ""])
    lines.extend(f"- {item}" for item in payload.get("review_basis", []))
    lines.append("")
    return "\n".join(lines)


def format_audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Stage144 Manifest Contract Audit",
        "",
        f"- contract pass: `{audit.get('contract_pass')}`",
        f"- failed checks: {', '.join(audit.get('failed_checks') or []) if audit.get('failed_checks') else 'None'}",
        "",
        "| check | pass |",
        "|---|---:|",
    ]
    for name, ok in sorted((audit.get("checks") or {}).items()):
        lines.append(f"| {name} | {ok} |")
    lines.append("")
    return "\n".join(lines)


def write_manifest_files(payload: dict[str, Any]) -> None:
    artifacts = payload["artifacts"]
    audit = audit_stage144_manifest(payload)
    for key in ("manifest", "manifest_audit", "stage_plan"):
        Path(artifacts[key]).parent.mkdir(parents=True, exist_ok=True)
    Path(artifacts["manifest"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(artifacts["manifest_audit"]).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(artifacts["stage_plan"]).write_text(format_markdown(payload), encoding="utf-8")
    Path(artifacts["manifest_audit"]).with_suffix(".md").write_text(format_audit_markdown(audit), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-file", default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--train-valid-file", default=DEFAULT_DFT_VALID)
    parser.add_argument("--dft-valid-file", default=DEFAULT_DFT_VALID)
    parser.add_argument("--teacher-valid-file", default=DEFAULT_TEACHER_VALID)
    parser.add_argument("--limit-configs", type=int, default=2688)
    parser.add_argument("--valid-limit-configs", type=int, default=256)
    parser.add_argument("--bench-limit-configs", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--early-stopping-patience", type=int, default=400)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = make_stage144_manifest(
        output_root=args.output_root,
        train_file=args.train_file,
        train_valid_file=args.train_valid_file,
        dft_valid_file=args.dft_valid_file,
        teacher_valid_file=args.teacher_valid_file,
        limit_configs=args.limit_configs,
        valid_limit_configs=args.valid_limit_configs,
        bench_limit_configs=args.bench_limit_configs,
        max_steps=args.max_steps,
        lr_warmup_steps=args.lr_warmup_steps,
        early_stopping_patience=args.early_stopping_patience,
    )
    materialized = materialize_stage144(payload)
    write_manifest_files(materialized)
    print(json.dumps({"stage": materialized["stage"], "artifacts": materialized["artifacts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
