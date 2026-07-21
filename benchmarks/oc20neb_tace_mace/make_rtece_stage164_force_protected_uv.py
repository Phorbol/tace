#!/usr/bin/env python3
"""Create Stage164 force-protected uv training wrappers."""

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
from benchmarks.oc20neb_tace_mace.submit_rtece_scalar_matrix import write_rtece_matrix_wrapper

DEFAULT_TRAIN_FILE = Path("runs/oc20neb_tace_mace/tece-distill-20260717/mixed_train_tw0.75.extxyz")
DEFAULT_STAGE163_RESULTS = Path(
    "runs/oc20neb_tace_mace/rtece-stage163-force-protected-active-set/stage163_force_protected_active_set.json"
)

STAGE164_VARIANT = "stage164_uv_force_gate_rel0p25_b32"
STAGE157_BASE_PATH_IDS = (
    "atomic.radial_density",
    "atomic.species_basis_density",
    "atomic.vector_norm",
    "atomic.vector_cross_radial_dot",
    "atomic.quadrupole_norm",
    "atomic.quadrupole_cross_radial_frobenius",
    "edge.cavity.vector_dot",
    "edge.direct.radial",
)
STAGE164_UV_PATH_IDS = (
    *STAGE157_BASE_PATH_IDS,
    "edge.cavity.target_vector_projection",
    "edge.cavity.source_vector_projection",
)


def _join_paths(path_ids: tuple[str, ...] | list[str]) -> str:
    return ",".join(str(path_id) for path_id in path_ids)


def _stage164_row() -> dict[str, Any]:
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import _row, _with_parameter_estimates

    row = _row(
        STAGE164_VARIANT,
        scalar_path_ids=_join_paths(STAGE164_UV_PATH_IDS),
        moment_l_max=2,
        learnable_radial_mixing=True,
        radial_species_adapter_channels=8,
        radial_species_adapter_scope="all",
        short_range_repulsion_potential="zbl",
        hidden_channels="64,64",
        num_radial=12,
        species_basis_channels=24,
        species_basis_mode="learnable_embedding",
        atomic_cross_radial_sketch_channels=6,
        atomic_cross_radial_projection="learnable",
        descriptor_conditioner="residual_mlp",
        descriptor_conditioner_hidden_channels=32,
        descriptor_bottleneck_dim=0,
        tece_axes=(
            "stage164_force_protected_active_set",
            "stage157_fixed_training_contract",
            "stage163_uv_only_gate",
            "edge_cavity_frame_projection_uv",
            "front_representation_capacity_not_wider_head",
        ),
    )
    row["tece_tier"] = "T3_rtece_edge_frame_relational_uv"
    row["stage_basis"] = "stage164_force_protected_uv"
    row["stage163_promoted_gate"] = "force_regression<=0.10_and_energy_gain"
    row["stage164_isolated_increment"] = "edge.cavity.target/source_vector_projection"
    row["stage163_active_set_source"] = str(DEFAULT_STAGE163_RESULTS)
    row["review_basis"] = (
        "Stage163 accepts uv only under a 10% sampled-force regression gate; Stage162 says raw energy error is "
        "dominated by case/slab/adsorbate baseline, so the next trainable step should test this edge-frame "
        "scalar increment under the Stage157 relative-energy and force-loss contract."
    )
    return _with_parameter_estimates([row])[0]


def make_stage164_manifest(
    *,
    output_root: str | Path,
    train_file: str | Path = DEFAULT_TRAIN_FILE,
    train_valid_file: str | Path = DEFAULT_DFT_VALID,
    dft_valid_file: str | Path = DEFAULT_DFT_VALID,
    teacher_valid_file: str | Path = DEFAULT_TEACHER_VALID,
    limit_configs: int = 2048,
    valid_limit_configs: int = 256,
    bench_limit_configs: int = 1024,
    max_steps: int = 20000,
    lr_warmup_steps: int = 500,
    early_stopping_patience: int = 400,
    active_set_force_gate: float = 0.10,
    relative_energy_weight: float = 0.25,
) -> dict[str, Any]:
    if int(limit_configs) < 1:
        raise ValueError("limit_configs must be positive")
    if int(valid_limit_configs) < 1:
        raise ValueError("valid_limit_configs must be positive")
    if int(bench_limit_configs) < 1:
        raise ValueError("bench_limit_configs must be positive")

    root = Path(output_root)
    artifacts = {
        "wrapper_root": str(root / "uv_force_protected_wrappers"),
        "run_root": str(root / "uv_force_protected_runs"),
        "manifest": str(root / "stage164_manifest.json"),
        "manifest_audit": str(root / "stage164_manifest_audit.json"),
        "stage_plan": str(root / "stage164_plan.md"),
        "stage163_results": str(DEFAULT_STAGE163_RESULTS),
    }
    return {
        "schema_version": "rtece_stage164_force_protected_uv.v1",
        "stage": "stage164_force_protected_uv_training",
        "row_set": "stage164-force-protected-uv",
        "distillation_semantics": "stage163_force_protected_uv_training",
        "comparison_question": (
            "Does the Stage163 force-protected uv edge-frame scalar increment repair the Stage162 raw-energy "
            "case/slab/adsorbate baseline without losing the Stage157 force RMSE/max and relative NEB metrics?"
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
        "relative_energy_weight": float(relative_energy_weight),
        "relative_energy_group_key": "case_id",
        "relative_energy_image_key": "source_frame",
        "active_set_gate": {
            "source": str(DEFAULT_STAGE163_RESULTS),
            "max_force_regression_fraction": float(active_set_force_gate),
            "require_energy_gain": True,
            "promoted_candidate": "uv",
            "rejected_candidates": ["uv_uqu", "uqu"],
        },
        "rows": [{"variant": STAGE164_VARIANT, **_stage164_row()}],
        "artifacts": artifacts,
        "review_basis": [
            "TECE_design_space: promote semantic TECE paths only by physical error gain under hardware cost, not by descriptor residual alone.",
            "rTECE_review: keep E/F RMSE and max errors, energy gauge, and physical probes explicit.",
            "Stage162: raw E RMSE is mostly case/slab/adsorbate offset, not a simple global E0 bug.",
            "Stage163: uv is the only candidate passing a 10% sampled-force regression gate with positive energy gain.",
        ],
    }


def audit_stage164_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    rows = list(payload.get("rows") or [])
    row = rows[0] if rows else {}
    path_ids = tuple(part.strip() for part in str(row.get("scalar_path_ids") or "").split(",") if part.strip())
    checks = {
        "schema_version": payload.get("schema_version") == "rtece_stage164_force_protected_uv.v1",
        "stage": payload.get("stage") == "stage164_force_protected_uv_training",
        "row_set": payload.get("row_set") == "stage164-force-protected-uv",
        "single_variant": [str(r.get("variant") or r.get("name")) for r in rows] == [STAGE164_VARIANT],
        "exact_uv_path_lattice": path_ids == STAGE164_UV_PATH_IDS,
        "no_quadrupole_frame_projection": "edge.cavity.target_quadrupole_projection" not in path_ids
        and "edge.cavity.source_quadrupole_projection" not in path_ids,
        "stage157_training_contract": int(payload.get("max_steps", 0)) == 20000
        and int(payload.get("lr_warmup_steps", 0)) == 500
        and int(payload.get("early_stopping_patience", 0)) == 400
        and abs(float(payload.get("relative_energy_weight", -1.0)) - 0.25) < 1.0e-12,
        "force_gate_contract": abs(float((payload.get("active_set_gate") or {}).get("max_force_regression_fraction", -1.0)) - 0.10) < 1.0e-12
        and bool((payload.get("active_set_gate") or {}).get("require_energy_gain")) is True,
        "fixed_stage157_backbone": int(row.get("moment_l_max", -1)) == 2
        and row.get("hidden_channels") == "64,64"
        and int(row.get("num_radial", -1)) == 12
        and int(row.get("species_basis_channels", -1)) == 24
        and row.get("species_basis_mode") == "learnable_embedding"
        and int(row.get("radial_species_adapter_channels", -1)) == 8
        and int(row.get("atomic_cross_radial_sketch_channels", -1)) == 6
        and row.get("atomic_cross_radial_projection") == "learnable"
        and row.get("descriptor_conditioner") == "residual_mlp"
        and int(row.get("descriptor_conditioner_hidden_channels", -1)) == 32
        and row.get("short_range_repulsion_potential") == "zbl",
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "schema_version": "rtece_stage164_manifest_audit.v1",
        "contract_pass": not failed,
        "failed_checks": failed,
        "checks": checks,
        "checked": {
            "path_ids": list(path_ids),
            "train_file": payload.get("train_file"),
            "active_set_gate": payload.get("active_set_gate"),
        },
    }


def _write_stage164_wrapper(payload: dict[str, Any], row: dict[str, Any]) -> Path:
    wrapper_dir = Path(payload["artifacts"]["wrapper_root"]) / str(row["variant"])
    run_root = f"{str(payload['artifacts']['run_root']).rstrip('/')}/{row['variant']}"
    return write_rtece_matrix_wrapper(
        wrapper_dir,
        variants=str(row["variant"]),
        run_root=run_root,
        train_file=str(payload["train_file"]),
        train_valid_file=str(payload["train_valid_file"]),
        dft_valid_file=str(payload["dft_valid_file"]),
        teacher_valid_file=str(payload["teacher_valid_file"]),
        limit_configs=int(payload["limit_configs"]),
        valid_limit_configs=int(payload["valid_limit_configs"]),
        bench_limit_configs=int(payload["bench_limit_configs"]),
        max_steps=int(payload["max_steps"]),
        lr="1e-3",
        hidden_channels=str(row["hidden_channels"]),
        num_radial=int(row["num_radial"]),
        moment_l_max=int(row["moment_l_max"]),
        scalar_path_ids=str(row["scalar_path_ids"]),
        species_basis_channels=int(row["species_basis_channels"]),
        species_basis_mode=str(row["species_basis_mode"]),
        atomic_cross_radial_sketch_channels=int(row["atomic_cross_radial_sketch_channels"]),
        atomic_cross_radial_projection=str(row["atomic_cross_radial_projection"]),
        force_weight=10.0,
        relative_energy_weight=float(payload["relative_energy_weight"]),
        relative_energy_group_key=str(payload["relative_energy_group_key"]),
        relative_energy_image_key=str(payload["relative_energy_image_key"]),
        use_short_range_repulsion=True,
        short_range_repulsion_potential=str(row["short_range_repulsion_potential"]),
        learnable_radial_mixing=True,
        radial_species_adapter_channels=int(row["radial_species_adapter_channels"]),
        radial_species_adapter_scope=str(row["radial_species_adapter_scope"]),
        descriptor_conditioner=str(row["descriptor_conditioner"]),
        descriptor_conditioner_hidden_channels=int(row["descriptor_conditioner_hidden_channels"]),
        descriptor_bottleneck_dim=int(row["descriptor_bottleneck_dim"]),
        short_range_repulsion_strength=0.0,
        short_range_repulsion_beta=10.0,
        short_range_repulsion_radius_scale=0.75,
        force_mode="autograd",
        measure_passes=5,
        default_dtype="float32",
        eval_interval=512,
        min_eval_step=0,
        checkpoint_name="rtece_scalar_best.pt",
        trainer_backend="lightning",
        batch_size=32,
        valid_batch_size=32,
        lr_scheduler="plateau",
        lr_patience=25,
        lr_factor=0.5,
        lr_warmup_steps=int(payload["lr_warmup_steps"]),
        early_stopping_patience=int(payload["early_stopping_patience"]),
        gradient_clip_val=10.0,
    )


def materialize_stage164(payload: dict[str, Any]) -> dict[str, Any]:
    audit = audit_stage164_manifest(payload)
    if not audit["contract_pass"]:
        raise ValueError(f"Stage164 manifest audit failed: {audit['failed_checks']}")
    materialized = dict(payload)
    train_wrappers = []
    for row in materialized["rows"]:
        wrapper = _write_stage164_wrapper(materialized, row)
        row["wrapper"] = str(wrapper)
        row["sbatch_command"] = ["sbatch", str(wrapper)]
        train_wrappers.append(str(wrapper))
    materialized["train_wrappers"] = train_wrappers
    write_manifest_files(materialized)
    return materialized


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage164 Force-Protected UV Training Plan",
        "",
        f"- stage: `{payload['stage']}`",
        f"- row set: `{payload['row_set']}`",
        f"- semantics: `{payload['distillation_semantics']}`",
        f"- train file: `{payload['train_file']}`",
        f"- Stage163 source: `{payload['active_set_gate']['source']}`",
        f"- force-protected gate: sampled force regression <= `{payload['active_set_gate']['max_force_regression_fraction']}` and positive energy gain",
        f"- relative energy weight: `{payload['relative_energy_weight']}`",
        "",
        "## Question",
        "",
        payload["comparison_question"],
        "",
        "## Rows",
        "",
        "| variant | params | gate | isolated increment | scalar paths |",
        "|---|---:|---|---|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['variant']} | {row.get('num_parameters_estimate')} | {row.get('stage163_promoted_gate')} | "
            f"{row.get('stage164_isolated_increment')} | {row.get('scalar_path_ids')} |"
        )
    lines.extend(["", "## Review Basis", ""])
    lines.extend(f"- {item}" for item in payload.get("review_basis", []))
    lines.append("")
    return "\n".join(lines)


def format_audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Stage164 Manifest Contract Audit",
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
    audit = audit_stage164_manifest(payload)
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
    parser.add_argument("--limit-configs", type=int, default=2048)
    parser.add_argument("--valid-limit-configs", type=int, default=256)
    parser.add_argument("--bench-limit-configs", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--early-stopping-patience", type=int, default=400)
    parser.add_argument("--active-set-force-gate", type=float, default=0.10)
    parser.add_argument("--relative-energy-weight", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = make_stage164_manifest(
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
        active_set_force_gate=args.active_set_force_gate,
        relative_energy_weight=args.relative_energy_weight,
    )
    materialized = materialize_stage164(payload)
    print(json.dumps({"stage": materialized["stage"], "artifacts": materialized["artifacts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
