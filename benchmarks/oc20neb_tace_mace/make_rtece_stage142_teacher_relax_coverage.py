#!/usr/bin/env python3
"""Create Stage142 teacher-relax coverage distillation wrappers."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.oc20neb_tace_mace.make_rtece_stage133_teacher_relax_distill import (
    DEFAULT_BASE_TRAIN,
    DEFAULT_SOURCE_CONFIGS,
    DEFAULT_TEACHER_MODEL,
    DEFAULT_TEACHER_VALID,
    _std_token,
)
from benchmarks.oc20neb_tace_mace.make_rtece_stage137_l2_conditioned_front import _stage137_rows

STAGE142_VARIANTS = ("l2_active_nrad12_species24_radial_species8_cross3_cond32_h64",)


def _command(parts: list[str | Path | int | float]) -> str:
    return " ".join(str(part) for part in parts if str(part) != "")


def _shell_export(name: str, value: str | Path | int | float) -> str:
    return f"export {name}={shlex.quote(str(value))}"


def _stage142_rows() -> list[dict[str, Any]]:
    by_name = {str(row["name"]): dict(row) for row in _stage137_rows()}
    rows: list[dict[str, Any]] = []
    for name in STAGE142_VARIANTS:
        row = dict(by_name[name])
        row["stage_basis"] = "stage142_teacher_relax_coverage"
        row["distillation_test_role"] = (
            "fixed Stage140/141 L2 conditioned-front architecture; only teacher-relax deployment coverage changes"
        )
        row["stage142_source_basis"] = (
            "Stage140 showed force-only teacher-relax plus weighted E0/source balance can repair the Stage139 energy drift. "
            "Stage141 showed loss-measure tweaks alone do not close the rattle fmax tail. Stage142 therefore keeps the "
            "architecture and weights fixed enough to test deployment-measure coverage before spending new representation cost."
        )
        axes = list(row.get("tece_axes") or [])
        for axis in (
            "stage142_teacher_relax_coverage",
            "force_only_teacher_relax_distillation",
            "dft_energy_anchor",
            "deployment_measure_coverage_check",
            "architecture_fixed_distillation_ablation",
        ):
            if axis not in axes:
                axes.append(axis)
        row["tece_axes"] = tuple(axes)
        rows.append(row)
    return rows


def make_stage142_manifest(
    *,
    output_root: str | Path,
    base_train: str | Path = DEFAULT_BASE_TRAIN,
    source_configs: str | Path = DEFAULT_SOURCE_CONFIGS,
    teacher_model: str | Path = DEFAULT_TEACHER_MODEL,
    train_valid_file: str | Path = DEFAULT_SOURCE_CONFIGS,
    dft_valid_file: str | Path = DEFAULT_SOURCE_CONFIGS,
    teacher_valid_file: str | Path = DEFAULT_TEACHER_VALID,
    source_start_config: int = 0,
    source_limit_configs: int = 64,
    copies_per_config: int = 2,
    rattle_std_a: float = 0.05,
    seed: int = 20260720,
    relax_max_steps: int = 4,
    base_limit_configs: int = 2048,
    base_energy_multiplier: float = 1.25,
    teacher_force_multiplier: float = 2.0,
    valid_limit_configs: int = 256,
    bench_limit_configs: int = 1024,
    max_steps: int = 20000,
    lr_warmup_steps: int = 500,
    early_stopping_patience: int = 400,
) -> dict[str, Any]:
    if int(source_start_config) < 0:
        raise ValueError("source_start_config must be non-negative")
    if int(source_limit_configs) < 1:
        raise ValueError("source_limit_configs must be positive")
    if int(copies_per_config) < 1:
        raise ValueError("copies_per_config must be positive")
    if int(relax_max_steps) < 0:
        raise ValueError("relax_max_steps must be non-negative")
    if float(rattle_std_a) < 0.0:
        raise ValueError("rattle_std_a must be non-negative")
    if float(base_energy_multiplier) < 0.0:
        raise ValueError("base_energy_multiplier must be non-negative")
    if float(teacher_force_multiplier) < 0.0:
        raise ValueError("teacher_force_multiplier must be non-negative")

    root = Path(output_root)
    trajectory_frame_count = int(source_limit_configs) * int(copies_per_config) * (int(relax_max_steps) + 1)
    teacher_label = f"teacher_relax_trajectory{trajectory_frame_count}"
    base_label = "base_mixed_train_tw0p75"
    trajectory_name = (
        f"teacher_relax_valid_start{int(source_start_config)}_limit{int(source_limit_configs)}_"
        f"copies{int(copies_per_config)}_std{_std_token(rattle_std_a)}_steps{int(relax_max_steps)}.extxyz"
    )
    augmented_name = f"augmented_train_base{int(base_limit_configs)}_plus_teacher_relax{trajectory_frame_count}.extxyz"
    weighted_name = f"weighted_train_base{int(base_limit_configs)}_plus_teacher_relax{trajectory_frame_count}_forceonly_eanchor.extxyz"
    artifacts = {
        "teacher_relax_trajectory": str(root / trajectory_name),
        "teacher_relax_summary": str(root / trajectory_name.replace(".extxyz", "_summary.json")),
        "augmented_train": str(root / augmented_name),
        "augmented_train_summary": str(root / augmented_name.replace(".extxyz", "_summary.json")),
        "weighted_train": str(root / weighted_name),
        "weighted_train_summary": str(root / weighted_name.replace(".extxyz", "_summary.json")),
        "prep_wrapper": str(root / "rtece_stage142_prepare_no_export.sbatch"),
        "wrapper_root": str(root / "teacher_relax_coverage_wrappers"),
        "run_root": str(root / "teacher_relax_coverage_runs"),
        "physical_wrapper_root": str(root / "physical_triage_wrappers"),
        "physical_run_root": str(root / "physical_triage"),
    }
    py = "${TACE_PYTHON:-python}"
    augmented_limit = int(base_limit_configs) + int(trajectory_frame_count)
    weight_policy = {
        "schema_version": "rtece_stage142_force_only_source_balance.v1",
        "source_energy_multipliers": {base_label: float(base_energy_multiplier), teacher_label: 0.0},
        "source_force_multipliers": {teacher_label: float(teacher_force_multiplier)},
        "teacher_energy_multiplier_is_fixed_zero": True,
        "normalize_energy_mean": True,
        "normalize_force_mean": True,
        "purpose": (
            "Keep DFT base structures as the only energy-zero/PES anchor while using broader teacher relaxation "
            "trajectories as force/Sobolev deployment constraints."
        ),
    }
    steps = [
        {
            "id": "make_teacher_relax_trajectory_labels",
            "command": _command([
                py,
                "benchmarks/oc20neb_tace_mace/make_teacher_relax_distill_configs.py",
                "--model", teacher_model,
                "--input", source_configs,
                "--output", artifacts["teacher_relax_trajectory"],
                "--summary", artifacts["teacher_relax_summary"],
                "--start-config", int(source_start_config),
                "--limit-configs", int(source_limit_configs),
                "--copies-per-config", int(copies_per_config),
                "--rattle-std-a", float(rattle_std_a),
                "--seed", int(seed),
                "--relax-max-steps", "${RELAX_MAX_STEPS:-" + str(int(relax_max_steps)) + "}",
                "--device", "cuda",
                "--default-dtype", "float32",
                "--nl-backend", "matscipy",
                "--reference-prefix", "source_",
            ]),
        },
        {
            "id": "concat_base_and_teacher_relax_trajectory",
            "command": _command([
                py,
                "benchmarks/oc20neb_tace_mace/concat_extxyz_datasets.py",
                "--input", base_train,
                "--source-label", base_label,
                "--input", artifacts["teacher_relax_trajectory"],
                "--source-label", teacher_label,
                "--input-limit", int(base_limit_configs),
                "--input-limit", -1,
                "--output", artifacts["augmented_train"],
                "--summary", artifacts["augmented_train_summary"],
            ]),
        },
        {
            "id": "apply_dft_anchor_force_only_teacher_relax_weights",
            "command": _command([
                py,
                "benchmarks/oc20neb_tace_mace/apply_extxyz_sample_weights.py",
                "--input", artifacts["augmented_train"],
                "--output", artifacts["weighted_train"],
                "--summary", artifacts["weighted_train_summary"],
                "--source-energy-multiplier", f"{base_label}:{float(base_energy_multiplier)}",
                "--source-energy-multiplier", f"{teacher_label}:0.0",
                "--source-force-multiplier", f"{teacher_label}:{float(teacher_force_multiplier)}",
                "--normalize-energy-mean",
            ]),
        },
    ]
    return {
        "schema_version": "rtece_stage142_teacher_relax_coverage.v1",
        "stage": "stage142_teacher_relax_coverage",
        "row_set": "stage142-teacher-relax-coverage",
        "distillation_semantics": "dft_energy_anchor_plus_force_only_teacher_relax_coverage",
        "comparison_question": (
            "With the Stage140/141 L2 conditioned rTECE architecture fixed, does broader teacher-relax "
            "deployment-measure coverage reduce the rattle force-tail/fmax failure while keeping DFT energy comparability "
            "through force-only teacher labels and DFT-only energy anchoring?"
        ),
        "base_train": str(base_train),
        "source_configs": str(source_configs),
        "teacher_model": str(teacher_model),
        "train_valid_file": str(train_valid_file),
        "dft_valid_file": str(dft_valid_file),
        "teacher_valid_file": str(teacher_valid_file),
        "source_start_config": int(source_start_config),
        "source_limit_configs": int(source_limit_configs),
        "copies_per_config": int(copies_per_config),
        "rattle_std_a": float(rattle_std_a),
        "seed": int(seed),
        "relax_max_steps": int(relax_max_steps),
        "frames_per_copy": int(relax_max_steps) + 1,
        "trajectory_frame_count": int(trajectory_frame_count),
        "base_limit_configs": int(base_limit_configs),
        "augmented_limit_configs": int(augmented_limit),
        "valid_limit_configs": int(valid_limit_configs),
        "bench_limit_configs": int(bench_limit_configs),
        "max_steps": int(max_steps),
        "lr_warmup_steps": int(lr_warmup_steps),
        "early_stopping_patience": int(early_stopping_patience),
        "weight_policy": weight_policy,
        "rows": [{"variant": str(row["name"]), **row} for row in _stage142_rows()],
        "artifacts": artifacts,
        "steps": steps,
        "review_basis": [
            "TECE_design_space: separate projection/architecture error from optimization/distillation error before changing paths.",
            "TECE_design_space: deployment Sobolev measure should include force and local-curvature trajectory stability, not only train/valid RMSE.",
            "rTECE_review: teacher distillation must preserve explicit source semantics, E/F max errors, and physical dimer/rattle probes.",
            "Stage140: force-only teacher-relax with weighted E0s repaired the Stage139 energy drift and became the physical-rattle reference.",
            "Stage141: loss weighting alone did not close rattle fmax, so coverage is the next controlled variable before edge/L/head complexity.",
        ],
    }


def audit_stage142_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    commands = "\n".join(str(step.get("command", "")) for step in payload.get("steps") or [])
    rows = list(payload.get("rows") or [])
    variants = [str(row.get("variant") or row.get("name")) for row in rows]
    policy = payload.get("weight_policy") or {}
    teacher_label = f"teacher_relax_trajectory{int(payload.get('trajectory_frame_count', 0))}"
    checks = {
        "schema_version": payload.get("schema_version") == "rtece_stage142_teacher_relax_coverage.v1",
        "stage": payload.get("stage") == "stage142_teacher_relax_coverage",
        "distillation_semantics": payload.get("distillation_semantics") == "dft_energy_anchor_plus_force_only_teacher_relax_coverage",
        "trajectory_count": int(payload.get("trajectory_frame_count", 0)) == int(payload.get("source_limit_configs", 0)) * int(payload.get("copies_per_config", 0)) * (int(payload.get("relax_max_steps", -1)) + 1),
        "coverage_larger_than_stage140": int(payload.get("trajectory_frame_count", 0)) > 320,
        "single_cond32_row": variants == list(STAGE142_VARIANTS),
        "moment_l_max_2_cond32": len(rows) == 1 and int(rows[0].get("moment_l_max", -1)) == 2 and rows[0].get("descriptor_conditioner") == "residual_mlp" and int(rows[0].get("descriptor_conditioner_hidden_channels", -1)) == 32,
        "teacher_energy_multiplier_fixed_zero": bool(policy.get("teacher_energy_multiplier_is_fixed_zero")) and (policy.get("source_energy_multipliers") or {}).get(teacher_label) == 0.0 and f"--source-energy-multiplier {teacher_label}:0.0" in commands,
        "base_energy_anchor_present": "base_mixed_train_tw0p75" in (policy.get("source_energy_multipliers") or {}) and "--normalize-energy-mean" in commands,
        "teacher_force_multiplier_present": (policy.get("source_force_multipliers") or {}).get(teacher_label, 0.0) > 0.0 and "--source-force-multiplier" in commands,
        "uses_teacher_relax_generator": "make_teacher_relax_distill_configs.py" in commands and "--reference-prefix source_" in commands,
        "uses_source_weight_tool": "apply_extxyz_sample_weights.py" in commands and "--source-energy-multiplier" in commands,
        "training_contract": int(payload.get("max_steps", 0)) == 20000 and int(payload.get("lr_warmup_steps", 0)) == 500 and int(payload.get("early_stopping_patience", 0)) == 400,
        "no_forbidden_sbatch_flags": all(flag not in commands for flag in ("--export", "--mem", "--cpus-per-task")),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "schema_version": "rtece_stage142_manifest_audit.v1",
        "contract_pass": not failed,
        "failed_checks": failed,
        "checks": checks,
        "checked": {
            "variants": variants,
            "trajectory_frame_count": payload.get("trajectory_frame_count"),
            "weight_policy": policy,
            "row_set": payload.get("row_set"),
        },
    }


def format_markdown(payload: dict[str, Any]) -> str:
    policy = payload["weight_policy"]
    lines = [
        "# Stage142 Teacher-Relax Coverage Plan",
        "",
        f"- stage: `{payload['stage']}`",
        f"- row set: `{payload['row_set']}`",
        f"- semantics: `{payload['distillation_semantics']}`",
        f"- source window: `{payload['source_start_config']}:{payload['source_start_config'] + payload['source_limit_configs']}`",
        f"- teacher trajectory frames: `{payload['trajectory_frame_count']}` configs",
        f"- augmented train limit: `{payload['augmented_limit_configs']}` configs",
        f"- source energy multipliers: `{policy['source_energy_multipliers']}`",
        f"- source force multipliers: `{policy['source_force_multipliers']}`",
        "",
        "## Question",
        "",
        str(payload["comparison_question"]),
        "",
        "## Rows",
        "",
        "| variant | params | representation params | scalar paths |",
        "|---|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['variant']} | {row.get('num_parameters_estimate')} | "
            f"{row.get('representation_parameters_estimate')} | {row.get('scalar_path_ids')} |"
        )
    lines.extend(["", "## Steps", ""])
    for step in payload["steps"]:
        lines.extend([f"### {step['id']}", "", "```bash", step["command"], "```", ""])
    lines.extend(["## Review Basis", ""])
    lines.extend(f"- {item}" for item in payload.get("review_basis", []))
    lines.append("")
    return "\n".join(lines)


def format_audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Stage142 Manifest Contract Audit",
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


def write_prepare_wrapper(path: str | Path, payload: dict[str, Any]) -> Path:
    wrapper = Path(path)
    env_dir = "/home/gengjianrui/bin/.venvs/tace-mace-cu126"
    body = [
        "#!/bin/bash",
        "#SBATCH --job-name=rtece-prep142",
        "#SBATCH --partition=16V100",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --gpus-per-node=1",
        "#SBATCH --qos=flood-1o2gpu",
        "#SBATCH --time=03:55:00",
        "#SBATCH --output=/home/gengjianrui/bin/logs/rtece-prep142-%j.out",
        "#SBATCH --error=/home/gengjianrui/bin/logs/rtece-prep142-%j.err",
        "",
        "set -euo pipefail",
        "set -x",
        _shell_export("PYTHONUNBUFFERED", 1),
        _shell_export("TACE_ROOT", "/home/gengjianrui/bin/tace"),
        _shell_export("ENV_DIR", env_dir),
        'export TACE_PYTHON="${ENV_DIR}/bin/python"',
        'export PATH="${ENV_DIR}/bin:${PATH}"',
        'export PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}"',
        _shell_export("RELAX_MAX_STEPS", int(payload["relax_max_steps"])),
        _shell_export("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", 1),
        _shell_export("TORCH_CUDA_ARCH_LIST", "7.0"),
        "mkdir -p /home/gengjianrui/bin/logs",
        'cd "${TACE_ROOT}"',
        '"${TACE_PYTHON}" -V',
        "nvidia-smi -L",
        "",
    ]
    for step in payload["steps"]:
        body.extend([f"# {step['id']}", step["command"], ""])
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("\n".join(body), encoding="utf-8")
    return wrapper


def _stage142_physical_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
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
                "selection_basis": "stage142 force-only teacher-relax coverage ablation on fixed cond32 L2 architecture",
            }
        )
    return cases


def materialize_stage142(payload: dict[str, Any]) -> dict[str, Any]:
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import write_pareto_sweep
    from benchmarks.oc20neb_tace_mace.make_rtece_physical_triage import write_physical_triage_wrappers

    artifacts = payload["artifacts"]
    root = Path(artifacts["prep_wrapper"]).parent
    root.mkdir(parents=True, exist_ok=True)
    write_prepare_wrapper(artifacts["prep_wrapper"], payload)
    sweep = write_pareto_sweep(
        artifacts["wrapper_root"],
        run_root=artifacts["run_root"],
        train_file=artifacts["weighted_train"],
        train_valid_file=payload["train_valid_file"],
        dft_valid_file=payload["dft_valid_file"],
        teacher_valid_file=payload["teacher_valid_file"],
        rows=_stage142_rows(),
        row_set="stage142-teacher-relax-coverage",
        limit_configs=int(payload["augmented_limit_configs"]),
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
        cases=_stage142_physical_cases(payload),
        stage="stage142_physical_triage",
        source="runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/stage142_plan.md",
        design_basis=(
            "Evaluate whether broader force-only teacher relaxation trajectory coverage repairs the rattle fmax tail "
            "for the fixed Stage140/141 L2 conditioned architecture before changing representation paths."
        ),
        job_name="rtece-phys142",
        rattle_start_config=58,
        rattle_limit_configs=8,
    )
    materialized = dict(payload)
    materialized["train_wrappers"] = [row["wrapper"] for row in sweep["rows"]]
    materialized["physical_wrappers"] = [row["wrapper"] for row in physical["cases"]]
    materialized["physical_cases"] = physical["cases"]
    return materialized


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
    audit = audit_stage142_manifest(payload)
    if audit_json is not None:
        audit_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if audit_md is not None:
        audit_md.write_text(format_audit_markdown(audit), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-train", default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--source-configs", default=DEFAULT_SOURCE_CONFIGS)
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--train-valid-file", default=DEFAULT_SOURCE_CONFIGS)
    parser.add_argument("--dft-valid-file", default=DEFAULT_SOURCE_CONFIGS)
    parser.add_argument("--teacher-valid-file", default=DEFAULT_TEACHER_VALID)
    parser.add_argument("--source-start-config", type=int, default=0)
    parser.add_argument("--source-limit-configs", type=int, default=64)
    parser.add_argument("--copies-per-config", type=int, default=2)
    parser.add_argument("--rattle-std-a", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--relax-max-steps", type=int, default=4)
    parser.add_argument("--base-limit-configs", type=int, default=2048)
    parser.add_argument("--base-energy-multiplier", type=float, default=1.25)
    parser.add_argument("--teacher-force-multiplier", type=float, default=2.0)
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
    payload = make_stage142_manifest(
        output_root=args.output_root,
        base_train=args.base_train,
        source_configs=args.source_configs,
        teacher_model=args.teacher_model,
        train_valid_file=args.train_valid_file,
        dft_valid_file=args.dft_valid_file,
        teacher_valid_file=args.teacher_valid_file,
        source_start_config=args.source_start_config,
        source_limit_configs=args.source_limit_configs,
        copies_per_config=args.copies_per_config,
        rattle_std_a=args.rattle_std_a,
        seed=args.seed,
        relax_max_steps=args.relax_max_steps,
        base_limit_configs=args.base_limit_configs,
        base_energy_multiplier=args.base_energy_multiplier,
        teacher_force_multiplier=args.teacher_force_multiplier,
        valid_limit_configs=args.valid_limit_configs,
        bench_limit_configs=args.bench_limit_configs,
        max_steps=args.max_steps,
        lr_warmup_steps=args.lr_warmup_steps,
        early_stopping_patience=args.early_stopping_patience,
    )
    if args.materialize:
        payload = materialize_stage142(payload)
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
