#!/usr/bin/env python3
"""Create a Stage117 teacher-rattle distillation experiment manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _std_token(value: float) -> str:
    text = f"{float(value):g}".replace(".", "p").replace("-", "m")
    return text


def _command(parts: list[str | Path | int | float]) -> str:
    return " ".join(str(part) for part in parts if str(part) != "")


def make_stage117_manifest(
    *,
    output_root: str | Path,
    base_train: str | Path,
    teacher_model: str | Path,
    valid_file: str | Path,
    dft_valid_file: str | Path,
    source_limit_configs: int,
    copies_per_config: int,
    rattle_std_a: float,
    seed: int,
    row_set: str = "capacity-ladder-stage116",
    bench_limit_configs: int = 1024,
    max_steps: int = 20000,
    lr_warmup_steps: int = 500,
) -> dict[str, Any]:
    root = Path(output_root)
    source_limit = int(source_limit_configs)
    copies = int(copies_per_config)
    if source_limit < 1:
        raise ValueError("source_limit_configs must be positive")
    if copies < 1:
        raise ValueError("copies_per_config must be positive")
    if float(rattle_std_a) < 0.0:
        raise ValueError("rattle_std_a must be non-negative")
    warmup_steps = int(lr_warmup_steps)
    if warmup_steps < 0:
        raise ValueError("lr_warmup_steps must be non-negative")

    rattle_name = f"rattled_source{source_limit}_copies{copies}_std{_std_token(rattle_std_a)}.extxyz"
    artifacts = {
        "rattled_configs": str(root / rattle_name),
        "rattled_summary": str(root / rattle_name.replace(".extxyz", "_summary.json")),
        "teacher_labeled_rattles": str(root / "teacher_labeled_rattles.extxyz"),
        "teacher_labeled_summary": str(root / "teacher_labeled_rattles_summary.json"),
        "augmented_train": str(root / "augmented_train_base_plus_teacher_rattles.extxyz"),
        "augmented_train_summary": str(root / "augmented_train_base_plus_teacher_rattles_summary.json"),
        "wrapper_root": str(root / "capacity_ladder_wrappers"),
        "run_root": str(root / "capacity_ladder_runs"),
    }
    py = "${TACE_PYTHON:-python}"
    steps = [
        {
            "id": "make_rattled_geometries",
            "command": _command([
                py,
                "benchmarks/oc20neb_tace_mace/make_rattle_distill_configs.py",
                "--input", base_train,
                "--output", artifacts["rattled_configs"],
                "--summary", artifacts["rattled_summary"],
                "--limit-configs", source_limit,
                "--copies-per-config", copies,
                "--rattle-std-a", rattle_std_a,
                "--seed", int(seed),
            ]),
        },
        {
            "id": "teacher_label_rattles",
            "command": _command([
                py,
                "benchmarks/oc20neb_tace_mace/distill_tace_labels.py",
                "--model", teacher_model,
                "--input", artifacts["rattled_configs"],
                "--output", artifacts["teacher_labeled_rattles"],
                "--summary", artifacts["teacher_labeled_summary"],
                "--device", "cuda",
                "--default-dtype", "float32",
                "--reference-prefix", "source_",
            ]),
        },
        {
            "id": "concat_base_and_teacher_rattles",
            "command": _command([
                py,
                "benchmarks/oc20neb_tace_mace/concat_extxyz_datasets.py",
                "--input", base_train,
                "--source-label", "base_mixed_train",
                "--input", artifacts["teacher_labeled_rattles"],
                "--source-label", "teacher_labeled_rattles",
                "--output", artifacts["augmented_train"],
                "--summary", artifacts["augmented_train_summary"],
            ]),
        },
        {
            "id": "make_capacity_ladder_wrappers",
            "command": _command([
                py,
                "benchmarks/oc20neb_tace_mace/make_rtece_pareto_sweep.py",
                "--output-dir", artifacts["wrapper_root"],
                "--run-root", artifacts["run_root"],
                "--train-file", artifacts["augmented_train"],
                "--train-valid-file", valid_file,
                "--dft-valid-file", dft_valid_file,
                "--teacher-valid-file", valid_file,
                "--row-set", row_set,
                "--limit-configs", source_limit + source_limit * copies,
                "--valid-limit-configs", 256,
                "--bench-limit-configs", int(bench_limit_configs),
                "--max-steps", int(max_steps),
                "--lr-warmup-steps", warmup_steps,
                "--preflight-extxyz",
            ]),
        },
    ]
    return {
        "schema_version": "rtece_stage117_rattle_distill_plan.v1",
        "stage": "stage117_teacher_rattle_distill_capacity_ladder",
        "row_set": str(row_set),
        "base_train": str(base_train),
        "teacher_model": str(teacher_model),
        "valid_file": str(valid_file),
        "dft_valid_file": str(dft_valid_file),
        "source_limit_configs": source_limit,
        "copies_per_config": copies,
        "rattle_std_a": float(rattle_std_a),
        "seed": int(seed),
        "lr_warmup_steps": warmup_steps,
        "distillation_semantics": "teacher_fake_labels_on_rattled_geometries",
        "rattle_label_policy": "pure_teacher_targets_preserve_source_labels_as_metadata",
        "comparison_question": "If Stage116 capacity alone does not fix force tails, test whether teacher-labeled local rattle coverage repairs student generalization without C/N-specific architecture.",
        "review_basis": [
            "TECE_design_space: deployment distribution and teacher-conditioned distillation separate teacher error, projection error, and distillation error",
            "rTECE_review: add teacher E/F distillation and trajectory/rattle coverage before element-specialized hacks",
            "Stage115 diagnostics: L0/L1 have similar C/N high-force tails, so data coverage vs representation must be separated",
        ],
        "artifacts": artifacts,
        "steps": steps,
        "queue_policy": "manifest_only_no_sbatch_submission",
    }


def audit_stage117_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    steps = payload.get("steps") or []
    commands = "\n".join(str(step.get("command", "")) for step in steps)
    step_ids = [str(step.get("id", "")) for step in steps]
    expected_augmented_limit = int(payload.get("source_limit_configs", 0)) * (1 + int(payload.get("copies_per_config", 0)))
    lr_warmup_steps = int(payload.get("lr_warmup_steps", -1))

    checks = {
        "schema_version": payload.get("schema_version") == "rtece_stage117_rattle_distill_plan.v1",
        "distillation_semantics": payload.get("distillation_semantics") == "teacher_fake_labels_on_rattled_geometries",
        "rattle_label_policy": payload.get("rattle_label_policy") == "pure_teacher_targets_preserve_source_labels_as_metadata",
        "queue_policy": payload.get("queue_policy") == "manifest_only_no_sbatch_submission",
        "required_steps": step_ids == [
            "make_rattled_geometries",
            "teacher_label_rattles",
            "concat_base_and_teacher_rattles",
            "make_capacity_ladder_wrappers",
        ],
        "no_sbatch_export": "--export" not in commands,
        "no_mix_tece_distill_labels": "mix_tece_distill_labels.py" not in commands,
        "teacher_label_step": "distill_tace_labels.py" in commands and "--reference-prefix source_" in commands,
        "capacity_ladder_step": "make_rtece_pareto_sweep.py" in commands
        and "--row-set capacity-ladder-stage116" in commands
        and "--preflight-extxyz" in commands,
        "augmented_limit_configs": f"--limit-configs {expected_augmented_limit}" in commands,
        "lr_warmup_contract": lr_warmup_steps >= 0 and f"--lr-warmup-steps {lr_warmup_steps}" in commands,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "schema_version": "rtece_stage117_manifest_audit.v1",
        "source_stage": payload.get("stage"),
        "contract_pass": not failed,
        "failed_checks": failed,
        "checks": checks,
        "checked": {
            "row_set": payload.get("row_set"),
            "source_limit_configs": int(payload.get("source_limit_configs", 0)),
            "copies_per_config": int(payload.get("copies_per_config", 0)),
            "augmented_limit_configs": int(expected_augmented_limit),
            "lr_warmup_steps": int(lr_warmup_steps),
            "queue_policy": payload.get("queue_policy"),
            "distillation_semantics": payload.get("distillation_semantics"),
            "rattle_label_policy": payload.get("rattle_label_policy"),
        },
        "review_basis": [
            "Stage117 must test teacher fake labels on displaced geometries, not mixed source labels for geometries whose DFT targets do not exist.",
            "Stage117 must keep the Stage116 capacity ladder fixed so data coverage and scalar-head capacity remain separable from representation changes.",
            "SAI wrappers and manifest commands must avoid sbatch --export to prevent cancellation before Python starts.",
        ],
    }


def format_audit_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage117 Manifest Contract Audit",
        "",
        f"- contract pass: `{payload.get('contract_pass')}`",
        f"- failed checks: {', '.join(payload.get('failed_checks') or []) if payload.get('failed_checks') else 'None'}",
        "",
        "| check | pass |",
        "|---|---:|",
    ]
    for name, passed in sorted((payload.get("checks") or {}).items()):
        lines.append(f"| {name} | {passed} |")
    lines.extend(["", "## Review Basis", ""])
    lines.extend(f"- {item}" for item in payload.get("review_basis", []))
    return "\n".join(lines) + "\n"


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage117 Teacher-Rattle Distillation Plan",
        "",
        f"- stage: `{payload['stage']}`",
        f"- row set: `{payload['row_set']}`",
        f"- semantics: `{payload['distillation_semantics']}`",
        f"- rattle label policy: `{payload['rattle_label_policy']}`",
        "",
        "## Steps",
        "",
    ]
    for step in payload["steps"]:
        lines.append(f"### {step['id']}")
        lines.append("")
        lines.append("```bash")
        lines.append(step["command"])
        lines.append("```")
        lines.append("")
    lines.extend(["## Review Basis", ""])
    lines.extend(f"- {item}" for item in payload.get("review_basis", []))
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-train", type=Path, required=True)
    parser.add_argument("--teacher-model", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--dft-valid-file", type=Path, required=True)
    parser.add_argument("--source-limit-configs", type=int, default=2048)
    parser.add_argument("--copies-per-config", type=int, default=1)
    parser.add_argument("--rattle-std-a", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--row-set", default="capacity-ladder-stage116")
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--output-audit-json", type=Path)
    parser.add_argument("--output-audit-md", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = make_stage117_manifest(
        output_root=args.output_root,
        base_train=args.base_train,
        teacher_model=args.teacher_model,
        valid_file=args.valid_file,
        dft_valid_file=args.dft_valid_file,
        source_limit_configs=args.source_limit_configs,
        copies_per_config=args.copies_per_config,
        rattle_std_a=args.rattle_std_a,
        seed=args.seed,
        row_set=args.row_set,
        lr_warmup_steps=args.lr_warmup_steps,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(format_markdown(payload), encoding="utf-8")
    audit = audit_stage117_manifest(payload)
    if args.output_audit_json is not None:
        args.output_audit_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_audit_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_audit_md is not None:
        args.output_audit_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_audit_md.write_text(format_audit_markdown(audit), encoding="utf-8")
    print(args.output_md)
    print(args.output_json)
    if args.output_audit_md is not None:
        print(args.output_audit_md)
    if args.output_audit_json is not None:
        print(args.output_audit_json)
    if not audit["contract_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
