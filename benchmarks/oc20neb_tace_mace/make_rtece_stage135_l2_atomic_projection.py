#!/usr/bin/env python3
"""Create Stage135 L2 atomic projection wrappers."""

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

DEFAULT_STAGE132_ROOT = Path("runs/oc20neb_tace_mace/rtece-stage132-broad-teacher-distill")
DEFAULT_TRAIN_FILE = DEFAULT_STAGE132_ROOT / "augmented_train_base2048_plus_teacher_rattle512.extxyz"
DEFAULT_STAGE132_RESULTS = DEFAULT_STAGE132_ROOT / "stage132_results_summary.json"
STAGE135_VARIANTS = ("l2_active_nrad12_species24_radial_species8_cross3_h64",)


def _stage135_rows() -> list[dict[str, Any]]:
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import _row, _with_parameter_estimates

    l2_active_backbone = (
        "atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,"
        "atomic.vector_cross_radial_dot,atomic.quadrupole_norm,"
        "atomic.quadrupole_cross_radial_frobenius"
    )
    row = _row(
        "l2_active_nrad12_species24_radial_species8_cross3_h64",
        scalar_path_ids=l2_active_backbone,
        moment_l_max=2,
        learnable_radial_mixing=True,
        radial_species_adapter_channels=8,
        radial_species_adapter_scope="all",
        short_range_repulsion_potential="zbl",
        tece_axes=(
            "stage135_l2_atomic_projection",
            "stage132_broad_teacher_anchor",
            "stage127_local_cross_species_anchor",
            "hardware_cost_conditioned_active_set",
            "controlled_L_A_axis",
            "atomic_quadrupole_scalar_paths",
            "cross_radial_invariants",
            "trainable_cross_radial_projection",
            "trainable_edge_species_radial_basis",
            "low_rank_neighbor_species_basis",
            "trainable_species_basis",
            "projection_error_vs_distillation_error_check",
        ),
        hidden_channels="64,64",
        num_radial=12,
        species_basis_channels=24,
        species_basis_mode="learnable_embedding",
        atomic_cross_radial_sketch_channels=3,
        atomic_cross_radial_projection="learnable",
        descriptor_bottleneck_dim=0,
    )
    row["stage_basis"] = "stage135_l2_atomic_projection"
    row["capacity_allocation"] = "controlled_L_A_2_atomic_front_not_wider_head"
    row["stage132_source"] = "runs/oc20neb_tace_mace/rtece-stage132-broad-teacher-distill/stage132_results_summary.json"
    row["stage134_source"] = "runs/oc20neb_tace_mace/rtece-stage134-balanced-teacher-relax/stage134_results_summary.json"
    row["review_basis"] = (
        "Stage133/134 showed trajectory-label and source-weight changes did not recover a clean RMSE/physical Pareto row. "
        "Stage135 therefore keeps the Stage132 broad-teacher-rattle data and 64,64 scalar head fixed, then increases only "
        "the TECE atomic angular projection from L_A=1 to L_A=2 by adding quadrupole scalar paths. This tests whether the "
        "remaining force-tail/physical error is representation/projection error before adding edge-relational cost."
    )
    return _with_parameter_estimates([row])


def make_stage135_manifest(
    *,
    output_root: str | Path,
    train_file: str | Path = DEFAULT_TRAIN_FILE,
    train_valid_file: str | Path = DEFAULT_DFT_VALID,
    dft_valid_file: str | Path = DEFAULT_DFT_VALID,
    teacher_valid_file: str | Path = DEFAULT_TEACHER_VALID,
    limit_configs: int = 2560,
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
        "wrapper_root": str(root / "l2_atomic_projection_wrappers"),
        "run_root": str(root / "l2_atomic_projection_runs"),
        "physical_wrapper_root": str(root / "physical_triage_wrappers"),
        "physical_run_root": str(root / "physical_triage"),
        "stage132_results": str(DEFAULT_STAGE132_RESULTS),
    }
    return {
        "schema_version": "rtece_stage135_l2_atomic_projection.v1",
        "stage": "stage135_l2_atomic_projection",
        "row_set": "stage135-l2-atomic-projection",
        "distillation_semantics": "fixed_stage132_broad_teacher_rattle_train_l2_atomic_projection",
        "comparison_question": (
            "With Stage132 broad-teacher-rattle data, fixed ZBL, fixed 64,64 head, and no edge sketch, does controlled "
            "L_A=2 atomic quadrupole projection reduce projection error, DFT E/F RMSE/max tails, and physical rattle failure "
            "enough to justify its throughput cost relative to the Stage132 L_A=1 atomic anchor?"
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
        "rows": [{"variant": str(row["name"]), **row} for row in _stage135_rows()],
        "artifacts": artifacts,
        "review_basis": [
            "TECE_design_space: prioritize a fused moment pass with L_A=2 or 3, immediate scalarization, sparse high-nu scalar paths, no persistent high-l node state, and a small scalar head.",
            "rTECE_review: treat current scalar rTECE as a promising endpoint, but separate projection/representation error from label and weighting heuristics before further kernel work.",
            "Stage132: current L_A=1 atomic front is the throughput/RMSE anchor but still fails the rattle physical gate.",
            "Stage133/134: teacher-relax trajectory and source balancing did not recover a clean benchmark/physical Pareto row, so the next isolated variable is representation bandwidth.",
        ],
    }


def audit_stage135_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    rows = list(payload.get("rows") or [])
    variants = [str(row.get("variant") or row.get("name")) for row in rows]
    scalar_paths = [str(row.get("scalar_path_ids") or "") for row in rows]
    checks = {
        "schema_version": payload.get("schema_version") == "rtece_stage135_l2_atomic_projection.v1",
        "stage": payload.get("stage") == "stage135_l2_atomic_projection",
        "distillation_semantics": payload.get("distillation_semantics") == "fixed_stage132_broad_teacher_rattle_train_l2_atomic_projection",
        "single_l2_atomic_row": variants == list(STAGE135_VARIANTS),
        "moment_l_max_2": len(rows) == 1 and int(rows[0].get("moment_l_max", -1)) == 2,
        "quadrupole_paths_present": len(scalar_paths) == 1 and "atomic.quadrupole_norm" in scalar_paths[0] and "atomic.quadrupole_cross_radial_frobenius" in scalar_paths[0],
        "no_edge_paths": all("edge." not in paths for paths in scalar_paths),
        "fixed_head_contract": len(rows) == 1 and rows[0].get("hidden_channels") == "64,64",
        "stage132_train_reuse": "teacher_rattle512" in str(payload.get("train_file", "")) or str(payload.get("train_file")) == "stage132_augmented.extxyz",
        "training_contract": int(payload.get("max_steps", 0)) == 20000 and int(payload.get("lr_warmup_steps", 0)) == 500 and int(payload.get("early_stopping_patience", 0)) == 400,
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "schema_version": "rtece_stage135_manifest_audit.v1",
        "contract_pass": not failed,
        "failed_checks": failed,
        "checks": checks,
        "checked": {
            "variants": variants,
            "row_set": payload.get("row_set"),
            "train_file": payload.get("train_file"),
            "scalar_path_ids": scalar_paths,
        },
    }


def _stage135_physical_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
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
                "selection_basis": "stage135 controlled L_A=2 atomic quadrupole projection ablation on fixed Stage132 broad-teacher-rattle data",
            }
        )
    return cases


def materialize_stage135(payload: dict[str, Any]) -> dict[str, Any]:
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import write_pareto_sweep
    from benchmarks.oc20neb_tace_mace.make_rtece_physical_triage import write_physical_triage_wrappers

    artifacts = payload["artifacts"]
    sweep = write_pareto_sweep(
        artifacts["wrapper_root"],
        run_root=artifacts["run_root"],
        train_file=payload["train_file"],
        train_valid_file=payload["train_valid_file"],
        dft_valid_file=payload["dft_valid_file"],
        teacher_valid_file=payload["teacher_valid_file"],
        rows=_stage135_rows(),
        row_set="stage135-l2-atomic-projection",
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
        cases=_stage135_physical_cases(payload),
        stage="stage135_physical_triage",
        source="runs/oc20neb_tace_mace/rtece-stage135-l2-atomic-projection/stage135_interpretation.md",
        design_basis=(
            "Evaluate whether controlled L_A=2 atomic quadrupole scalar paths improve force-tail and physical robustness "
            "on the fixed Stage132 broad-teacher-rattle training distribution before adding edge-relational cost."
        ),
        job_name="rtece-phys135",
        rattle_start_config=58,
        rattle_limit_configs=8,
    )
    materialized = dict(payload)
    materialized["train_wrappers"] = [row["wrapper"] for row in sweep["rows"]]
    materialized["physical_wrappers"] = [row["wrapper"] for row in physical["cases"]]
    materialized["physical_cases"] = physical["cases"]
    return materialized


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage135 L2 Atomic Projection Plan",
        "",
        f"- stage: `{payload['stage']}`",
        f"- row set: `{payload['row_set']}`",
        f"- semantics: `{payload['distillation_semantics']}`",
        f"- train file: `{payload['train_file']}`",
        f"- train limit: `{payload['limit_configs']}` configs",
        "",
        "## Question",
        "",
        str(payload["comparison_question"]),
        "",
        "## Rows",
        "",
        "| variant | params | representation params | scalar paths | role |",
        "|---|---:|---:|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['variant']} | {row.get('num_parameters_estimate')} | "
            f"{row.get('representation_parameters_estimate')} | {row.get('scalar_path_ids')} | "
            f"{row.get('capacity_allocation', '')} |"
        )
    lines.extend(["", "## Review Basis", ""])
    lines.extend(f"- {item}" for item in payload.get("review_basis", []))
    lines.append("")
    return "\n".join(lines)


def format_audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Stage135 Manifest Contract Audit",
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


def write_manifest_files(payload: dict[str, Any], *, output_json: Path, output_md: Path, audit_json: Path | None, audit_md: Path | None) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(format_markdown(payload), encoding="utf-8")
    audit = audit_stage135_manifest(payload)
    if audit_json is not None:
        audit_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if audit_md is not None:
        audit_md.write_text(format_audit_markdown(audit), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-file", default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument("--train-valid-file", default=DEFAULT_DFT_VALID)
    parser.add_argument("--dft-valid-file", default=DEFAULT_DFT_VALID)
    parser.add_argument("--teacher-valid-file", default=DEFAULT_TEACHER_VALID)
    parser.add_argument("--limit-configs", type=int, default=2560)
    parser.add_argument("--valid-limit-configs", type=int, default=256)
    parser.add_argument("--bench-limit-configs", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--early-stopping-patience", type=int, default=400)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--audit-md", type=Path)
    parser.add_argument("--materialize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = make_stage135_manifest(
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
    if args.materialize:
        payload = materialize_stage135(payload)
    write_manifest_files(payload, output_json=args.output_json, output_md=args.output_md, audit_json=args.audit_json, audit_md=args.audit_md)
    print(json.dumps({"stage": payload["stage"], "rows": len(payload["rows"]), "output_json": str(args.output_json)}, sort_keys=True))


if __name__ == "__main__":
    main()
