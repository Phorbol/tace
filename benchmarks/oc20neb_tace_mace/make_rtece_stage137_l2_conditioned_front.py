#!/usr/bin/env python3
"""Create Stage137 L2 conditioned-front wrappers."""

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
DEFAULT_STAGE135_RESULTS = Path("runs/oc20neb_tace_mace/rtece-stage135-l2-atomic-projection/stage135_results_summary.json")
DEFAULT_STAGE136_RESULTS = Path("runs/oc20neb_tace_mace/rtece-stage136-l2-projection-diagnostic/stage136_results_summary.json")

STAGE137_VARIANTS = (
    "l2_active_nrad12_species24_radial_species8_cross3_bneck32_h64",
    "l2_active_nrad12_species24_radial_species8_cross3_cond32_h64",
)

L2_ACTIVE_BACKBONE = (
    "atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,"
    "atomic.vector_cross_radial_dot,atomic.quadrupole_norm,"
    "atomic.quadrupole_cross_radial_frobenius"
)


def _base_l2_kwargs() -> dict[str, Any]:
    return {
        "scalar_path_ids": L2_ACTIVE_BACKBONE,
        "moment_l_max": 2,
        "learnable_radial_mixing": True,
        "radial_species_adapter_channels": 8,
        "radial_species_adapter_scope": "all",
        "short_range_repulsion_potential": "zbl",
        "hidden_channels": "64,64",
        "num_radial": 12,
        "species_basis_channels": 24,
        "species_basis_mode": "learnable_embedding",
        "atomic_cross_radial_sketch_channels": 3,
        "atomic_cross_radial_projection": "learnable",
    }


def _stage137_rows() -> list[dict[str, Any]]:
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import _row

    shared_axes = (
        "stage137_l2_conditioned_front",
        "stage136_force_projection_guided",
        "stage132_broad_teacher_anchor",
        "stage135_l2_atomic_reference",
        "controlled_L_A_axis",
        "atomic_quadrupole_scalar_paths",
        "cross_radial_invariants",
        "trainable_cross_radial_projection",
        "trainable_edge_species_radial_basis",
        "low_rank_neighbor_species_basis",
        "trainable_species_basis",
        "front_representation_capacity_not_wider_head",
    )
    rows = [
        _row(
            "l2_active_nrad12_species24_radial_species8_cross3_bneck32_h64",
            **_base_l2_kwargs(),
            descriptor_bottleneck_dim=32,
            descriptor_conditioner="none",
            descriptor_conditioner_hidden_channels=0,
            tece_axes=(*shared_axes, "descriptor_bottleneck", "front_low_rank_path_mixer"),
        ),
        _row(
            "l2_active_nrad12_species24_radial_species8_cross3_cond32_h64",
            **_base_l2_kwargs(),
            descriptor_bottleneck_dim=0,
            descriptor_conditioner="residual_mlp",
            descriptor_conditioner_hidden_channels=32,
            tece_axes=(*shared_axes, "scalar_descriptor_conditioning", "zero_init_residual_path_conditioner"),
        ),
    ]
    for row in rows:
        row["stage_basis"] = "stage137_l2_conditioned_front"
        row["capacity_allocation"] = "stage136_force_projection_guided_l2_front_conditioning_not_wider_head"
        row["stage132_source"] = str(DEFAULT_STAGE132_RESULTS)
        row["stage135_source"] = str(DEFAULT_STAGE135_RESULTS)
        row["stage136_source"] = str(DEFAULT_STAGE136_RESULTS)
        row["review_basis"] = (
            "Stage136 showed the L2 atomic quadrupole paths reduce held-out force projection RMSE/max relative to the "
            "Stage132 L1 active basis, while Stage135 showed naive full-L2 training worsens DFT F RMSE/max and throughput. "
            "Stage137 therefore keeps the Stage132 broad-teacher-rattle data, fixed 64,64 head, ZBL baseline, and no edge "
            "sketch, then tests whether a low-rank descriptor mixer or zero-initialized residual descriptor conditioner can "
            "make the force-relevant L2 information trainable without turning this into blind head widening."
        )
    return _with_stage137_parameter_estimates(rows)


def _stage137_descriptor_dim(row: dict[str, Any]) -> int:
    num_radial = int(row.get("num_radial", 12))
    species_basis_channels = int(row.get("species_basis_channels", 0))
    cross_channels = int(row.get("atomic_cross_radial_sketch_channels", 0))
    cross_pairs = cross_channels * (cross_channels - 1) // 2
    dim = 0
    for path_id in tuple(part.strip() for part in str(row.get("scalar_path_ids") or "").split(",") if part.strip()):
        if path_id in {"atomic.radial_density", "atomic.vector_norm", "atomic.quadrupole_norm"}:
            dim += num_radial
        elif path_id == "atomic.species_basis_density":
            dim += num_radial * species_basis_channels
        elif path_id in {"atomic.vector_cross_radial_dot", "atomic.quadrupole_cross_radial_frobenius"}:
            dim += cross_pairs
        else:
            raise ValueError(f"unsupported Stage137 scalar path {path_id!r}")
    return dim


def _linear_parameters(in_features: int, out_features: int, *, bias: bool = True) -> int:
    return int(in_features) * int(out_features) + (int(out_features) if bias else 0)


def _stage137_parameter_estimates(row: dict[str, Any]) -> dict[str, int]:
    descriptor = _stage137_descriptor_dim(row)
    hidden_channels = tuple(int(part.strip()) for part in str(row.get("hidden_channels", "64,64")).split(",") if part.strip())
    max_atomic_number = 100
    num_radial = int(row.get("num_radial", 12))
    species_basis_channels = int(row.get("species_basis_channels", 0))
    adapter_channels = int(row.get("radial_species_adapter_channels", 0))
    cross_channels = int(row.get("atomic_cross_radial_sketch_channels", 0))

    representation = 0
    if row.get("learnable_radial_mixing"):
        representation += num_radial * num_radial
    if adapter_channels:
        representation += 2 * (max_atomic_number + 1) * adapter_channels
        representation += num_radial * (2 * adapter_channels)
    if species_basis_channels and row.get("species_basis_mode") == "learnable_embedding":
        representation += (max_atomic_number + 1) * species_basis_channels
    if row.get("atomic_cross_radial_projection") == "learnable" and cross_channels:
        representation += cross_channels * num_radial

    bottleneck_dim = int(row.get("descriptor_bottleneck_dim") or 0)
    readout_dim = descriptor
    if bottleneck_dim:
        representation += _linear_parameters(descriptor, bottleneck_dim, bias=True)
        readout_dim = bottleneck_dim

    if row.get("descriptor_conditioner") != "none":
        hidden = int(row.get("descriptor_conditioner_hidden_channels") or 0)
        if hidden <= 0:
            raise ValueError("descriptor conditioner hidden channels must be positive")
        representation += _linear_parameters(descriptor, hidden, bias=True)
        representation += _linear_parameters(hidden, descriptor, bias=True)

    prev = readout_dim + 1
    readout = 0
    for hidden in hidden_channels:
        readout += _linear_parameters(prev, hidden, bias=True)
        prev = hidden
    readout += _linear_parameters(prev, 1, bias=True)
    return {
        "descriptor_dim_estimate": int(descriptor),
        "num_parameters_estimate": int(representation + readout),
        "representation_parameters_estimate": int(representation),
        "readout_parameters_estimate": int(readout),
    }


def _with_stage137_parameter_estimates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    estimated = []
    for row in rows:
        item = dict(row)
        item.update(_stage137_parameter_estimates(item))
        estimated.append(item)
    return estimated


def make_stage137_manifest(
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
        "wrapper_root": str(root / "l2_conditioned_front_wrappers"),
        "run_root": str(root / "l2_conditioned_front_runs"),
        "physical_wrapper_root": str(root / "physical_triage_wrappers"),
        "physical_run_root": str(root / "physical_triage"),
        "stage132_results": str(DEFAULT_STAGE132_RESULTS),
        "stage135_results": str(DEFAULT_STAGE135_RESULTS),
        "stage136_results": str(DEFAULT_STAGE136_RESULTS),
    }
    return {
        "schema_version": "rtece_stage137_l2_conditioned_front.v1",
        "stage": "stage137_l2_conditioned_front",
        "row_set": "stage137-l2-conditioned-front",
        "distillation_semantics": "stage136_force_projection_guided_l2_low_rank_conditioning",
        "comparison_question": (
            "Stage136 force projection says L2 atomic quadrupole descriptors contain useful force-response information, "
            "but Stage135 naive full-L2 training is not Pareto. On the fixed Stage132 broad-teacher-rattle data and fixed "
            "64,64 head, can low-rank descriptor mixing or zero-initialized residual descriptor conditioning convert that "
            "force projection signal into lower DFT E/F RMSE/max or better rattle physical robustness without edge-sketch cost?"
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
        "rows": [{"variant": str(row["name"]), **row} for row in _stage137_rows()],
        "artifacts": artifacts,
        "review_basis": [
            "TECE_design_space: keep the degradation axis explicit as L_A=2 immediate scalarization, sparse scalar paths, no persistent high-l node state, no attention, and no per-edge CxC matrices.",
            "TECE_design_space: parameter growth should enter the representation/path compiler, not just the final scalar head.",
            "rTECE_review: compare E/F RMSE and max tails separately from physical probes; do not use a single MAE gate as the research objective.",
            "Stage120b/121: generic bottlenecks on older L2/cavity rows were not Pareto; Stage137 retests bottleneck/conditioning only on the Stage132/135 active atomic basis isolated by Stage136 projection evidence.",
            "Stage125: front representation capacity can improve RMSE when rank is allocated to active paths, but bigger ranks are not monotonic; Stage137 keeps the matrix narrow and diagnostic.",
        ],
    }


def audit_stage137_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    rows = list(payload.get("rows") or [])
    variants = [str(row.get("variant") or row.get("name")) for row in rows]
    scalar_paths = [str(row.get("scalar_path_ids") or "") for row in rows]
    checks = {
        "schema_version": payload.get("schema_version") == "rtece_stage137_l2_conditioned_front.v1",
        "stage": payload.get("stage") == "stage137_l2_conditioned_front",
        "distillation_semantics": payload.get("distillation_semantics") == "stage136_force_projection_guided_l2_low_rank_conditioning",
        "two_l2_front_rows": variants == list(STAGE137_VARIANTS),
        "moment_l_max_2": len(rows) == 2 and all(int(row.get("moment_l_max", -1)) == 2 for row in rows),
        "quadrupole_paths_present": len(rows) == 2 and all(
            "atomic.quadrupole_norm" in paths and "atomic.quadrupole_cross_radial_frobenius" in paths
            for paths in scalar_paths
        ),
        "no_edge_paths": all("edge." not in paths for paths in scalar_paths),
        "fixed_head_contract": len(rows) == 2 and all(row.get("hidden_channels") == "64,64" for row in rows),
        "bottleneck_row_contract": len(rows) == 2
        and int(rows[0].get("descriptor_bottleneck_dim", -1)) == 32
        and rows[0].get("descriptor_conditioner") == "none",
        "conditioner_row_contract": len(rows) == 2
        and int(rows[1].get("descriptor_bottleneck_dim", -1)) == 0
        and rows[1].get("descriptor_conditioner") == "residual_mlp"
        and int(rows[1].get("descriptor_conditioner_hidden_channels", -1)) == 32,
        "stage132_train_reuse": "teacher_rattle512" in str(payload.get("train_file", "")) or str(payload.get("train_file")) == "stage132_augmented.extxyz",
        "training_contract": int(payload.get("max_steps", 0)) == 20000
        and int(payload.get("lr_warmup_steps", 0)) == 500
        and int(payload.get("early_stopping_patience", 0)) == 400,
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "schema_version": "rtece_stage137_manifest_audit.v1",
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


def _stage137_physical_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
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
                "selection_basis": "Stage137 Stage136-force-projection-guided L2 low-rank/conditioned-front ablation",
            }
        )
    return cases


def materialize_stage137(payload: dict[str, Any]) -> dict[str, Any]:
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
        rows=_stage137_rows(),
        row_set="stage137-l2-conditioned-front",
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
        cases=_stage137_physical_cases(payload),
        stage="stage137_physical_triage",
        source="runs/oc20neb_tace_mace/rtece-stage137-l2-conditioned-front/stage137_plan.md",
        design_basis=(
            "Evaluate whether Stage136 force-projection-guided L2 front conditioning improves force-tail and physical "
            "robustness on the fixed Stage132 broad-teacher-rattle training distribution before adding edge-relational cost."
        ),
        job_name="rtece-phys137",
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
        "# Stage137 L2 Conditioned Front Plan",
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
        "| variant | params | representation params | readout params | front control | scalar paths |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in payload["rows"]:
        front_control = ""
        if int(row.get("descriptor_bottleneck_dim") or 0):
            front_control = f"bottleneck={row['descriptor_bottleneck_dim']}"
        elif row.get("descriptor_conditioner") != "none":
            front_control = f"conditioner={row['descriptor_conditioner']}:{row['descriptor_conditioner_hidden_channels']}"
        lines.append(
            f"| {row['variant']} | {row.get('num_parameters_estimate')} | "
            f"{row.get('representation_parameters_estimate')} | {row.get('readout_parameters_estimate')} | "
            f"{front_control} | {row.get('scalar_path_ids')} |"
        )
    lines.extend(["", "## Review Basis", ""])
    lines.extend(f"- {item}" for item in payload.get("review_basis", []))
    lines.append("")
    return "\n".join(lines)


def format_audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Stage137 Manifest Contract Audit",
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


def write_manifest_files(
    payload: dict[str, Any],
    *,
    output_json: Path,
    output_md: Path,
    audit_json: Path | None,
    audit_md: Path | None,
) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(format_markdown(payload), encoding="utf-8")
    audit = audit_stage137_manifest(payload)
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
    payload = make_stage137_manifest(
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
        payload = materialize_stage137(payload)
    write_manifest_files(
        payload,
        output_json=args.output_json,
        output_md=args.output_md,
        audit_json=args.audit_json,
        audit_md=args.audit_md,
    )
    print(json.dumps({"stage": payload["stage"], "rows": len(payload["rows"]), "output_json": str(args.output_json)}, sort_keys=True))


if __name__ == "__main__":
    main()
