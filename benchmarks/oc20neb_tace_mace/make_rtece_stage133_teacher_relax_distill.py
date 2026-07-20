#!/usr/bin/env python3
"""Create Stage133 teacher-relax trajectory distillation wrappers."""

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

DEFAULT_BASE_TRAIN = "runs/oc20neb_tace_mace/tece-distill-20260717/mixed_train_tw0.75.extxyz"
DEFAULT_SOURCE_CONFIGS = "/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz"
DEFAULT_TEACHER_VALID = "runs/oc20neb_tace_mace/tece-distill-20260717/teacher_valid.extxyz"
DEFAULT_TEACHER_MODEL = "runs/oc20neb_tace_mace/675969_resume/checkpoints_epoch/TACE-0-100000-0.1329.ckpt"

STAGE133_VARIANTS = ("l1_active_nrad12_species24_radial_species8_cross3_h64",)


def _std_token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def _command(parts: list[str | Path | int | float]) -> str:
    return " ".join(str(part) for part in parts if str(part) != "")


def _shell_export(name: str, value: str | Path | int | float) -> str:
    return f"export {name}={shlex.quote(str(value))}"


def _stage133_rows() -> list[dict[str, Any]]:
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import stage129_current_pareto_rows

    by_name = {str(row["name"]): dict(row) for row in stage129_current_pareto_rows()}
    rows = []
    for name in STAGE133_VARIANTS:
        row = dict(by_name[name])
        row["stage_basis"] = "stage133_teacher_relax_trajectory_distill"
        row["distillation_test_role"] = (
            "fixed stage129/stage132 atomic Pareto backbone; only teacher PES relaxation trajectory coverage changes"
        )
        row["stage133_source_basis"] = (
            "Stage132 broad single-point teacher-rattle labels reduced some force-tail indicators but worsened RMSE and "
            "did not close the rattle-relax physical gate. Stage133 therefore moves from independent fake labels to "
            "short teacher LBFGS trajectories on the same broad deployment window, testing distillation-error and "
            "deployment-manifold coverage before adding edge/Lmax representation axes."
        )
        axes = list(row.get("tece_axes") or [])
        for axis in (
            "stage133_teacher_relax_trajectory_distillation",
            "teacher_pes_deployment_manifold",
            "projection_error_vs_distillation_error_check",
            "fixed_atomic_architecture_distillation_ablation",
        ):
            if axis not in axes:
                axes.append(axis)
        row["tece_axes"] = tuple(axes)
        rows.append(row)
    return rows


def make_stage133_manifest(
    *,
    output_root: str | Path,
    base_train: str | Path = DEFAULT_BASE_TRAIN,
    source_configs: str | Path = DEFAULT_SOURCE_CONFIGS,
    teacher_model: str | Path = DEFAULT_TEACHER_MODEL,
    train_valid_file: str | Path = DEFAULT_SOURCE_CONFIGS,
    dft_valid_file: str | Path = DEFAULT_SOURCE_CONFIGS,
    teacher_valid_file: str | Path = DEFAULT_TEACHER_VALID,
    source_start_config: int = 0,
    source_limit_configs: int = 32,
    copies_per_config: int = 2,
    rattle_std_a: float = 0.05,
    seed: int = 20260720,
    relax_max_steps: int = 4,
    base_limit_configs: int = 2048,
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

    root = Path(output_root)
    trajectory_frame_count = int(source_limit_configs) * int(copies_per_config) * (int(relax_max_steps) + 1)
    trajectory_name = (
        f"teacher_relax_valid_start{int(source_start_config)}_limit{int(source_limit_configs)}_"
        f"copies{int(copies_per_config)}_std{_std_token(rattle_std_a)}_steps{int(relax_max_steps)}.extxyz"
    )
    augmented_name = f"augmented_train_base{int(base_limit_configs)}_plus_teacher_relax{trajectory_frame_count}.extxyz"
    artifacts = {
        "teacher_relax_trajectory": str(root / trajectory_name),
        "teacher_relax_summary": str(root / trajectory_name.replace(".extxyz", "_summary.json")),
        "augmented_train": str(root / augmented_name),
        "augmented_train_summary": str(root / augmented_name.replace(".extxyz", "_summary.json")),
        "prep_wrapper": str(root / "rtece_stage133_prepare_no_export.sbatch"),
        "wrapper_root": str(root / "teacher_relax_wrappers"),
        "run_root": str(root / "teacher_relax_runs"),
        "physical_wrapper_root": str(root / "physical_triage_wrappers"),
        "physical_run_root": str(root / "physical_triage"),
    }
    py = "${TACE_PYTHON:-python}"
    augmented_limit = int(base_limit_configs) + int(trajectory_frame_count)
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
                "--source-label", "base_mixed_train_tw0p75",
                "--input", artifacts["teacher_relax_trajectory"],
                "--source-label", f"teacher_relax_trajectory{trajectory_frame_count}",
                "--input-limit", int(base_limit_configs),
                "--input-limit", -1,
                "--output", artifacts["augmented_train"],
                "--summary", artifacts["augmented_train_summary"],
            ]),
        },
    ]
    return {
        "schema_version": "rtece_stage133_teacher_relax_distill.v1",
        "stage": "stage133_teacher_relax_trajectory_distill",
        "row_set": "stage133-teacher-relax-distill",
        "distillation_semantics": "teacher_energy_force_labels_on_teacher_lbfgs_relaxation_trajectory",
        "comparison_question": (
            "Does short teacher PES relaxation manifold coverage reduce distillation error and physical force-tail "
            "failures for the fixed current-Pareto atomic rTECE front, separating projection error from distillation "
            "error before adding Lmax or edge-relational representation axes?"
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
        "rows": [{"variant": str(row["name"]), **row} for row in _stage133_rows()],
        "artifacts": artifacts,
        "steps": steps,
        "review_basis": [
            "TECE_design_space: deployment Sobolev metrics and projection/distillation error separation should drive row priority.",
            "TECE_design_space: full distillation should follow path projection, not uncontrolled head widening.",
            "rTECE_review: teacher E/F/V distillation and physical tests are P1 for a real rTECE, while kernel-only speedups are lower priority.",
            "Stage132: broader single-point teacher fake labels did not produce a clean RMSE/throughput/physical Pareto improvement.",
        ],
    }


def audit_stage133_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    commands = "\n".join(str(step.get("command", "")) for step in payload.get("steps") or [])
    variants = [str(row.get("variant") or row.get("name")) for row in payload.get("rows") or []]
    checks = {
        "schema_version": payload.get("schema_version") == "rtece_stage133_teacher_relax_distill.v1",
        "stage": payload.get("stage") == "stage133_teacher_relax_trajectory_distill",
        "distillation_semantics": payload.get("distillation_semantics") == "teacher_energy_force_labels_on_teacher_lbfgs_relaxation_trajectory",
        "broad_window": int(payload.get("source_start_config", -1)) == 0 and int(payload.get("source_limit_configs", 0)) >= 32,
        "trajectory_count": int(payload.get("trajectory_frame_count", 0)) == int(payload.get("source_limit_configs", 0)) * int(payload.get("copies_per_config", 0)) * (int(payload.get("relax_max_steps", -1)) + 1),
        "augmented_limit_configs": int(payload.get("augmented_limit_configs", 0)) == int(payload.get("base_limit_configs", 0)) + int(payload.get("trajectory_frame_count", 0)),
        "single_fixed_atomic_row": variants == list(STAGE133_VARIANTS),
        "uses_teacher_relax_generator": "make_teacher_relax_distill_configs.py" in commands and "--reference-prefix source_" in commands,
        "base_subset_before_trajectory": f"--input-limit {int(payload.get('base_limit_configs', 0))} --input-limit -1" in commands,
        "training_contract": int(payload.get("max_steps", 0)) == 20000 and int(payload.get("lr_warmup_steps", 0)) == 500 and int(payload.get("early_stopping_patience", 0)) == 400,
        "no_forbidden_sbatch_flags": all(flag not in commands for flag in ("--export", "--mem", "--cpus-per-task")),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "schema_version": "rtece_stage133_manifest_audit.v1",
        "contract_pass": not failed,
        "failed_checks": failed,
        "checks": checks,
        "checked": {
            "variants": variants,
            "source_start_config": payload.get("source_start_config"),
            "source_limit_configs": payload.get("source_limit_configs"),
            "copies_per_config": payload.get("copies_per_config"),
            "relax_max_steps": payload.get("relax_max_steps"),
            "trajectory_frame_count": payload.get("trajectory_frame_count"),
            "row_set": payload.get("row_set"),
        },
    }


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage133 Teacher-Relax Trajectory Distillation Plan",
        "",
        f"- stage: `{payload['stage']}`",
        f"- row set: `{payload['row_set']}`",
        f"- semantics: `{payload['distillation_semantics']}`",
        f"- source window: `{payload['source_start_config']}:{payload['source_start_config'] + payload['source_limit_configs']}`",
        f"- teacher trajectory frames: `{payload['trajectory_frame_count']}` configs",
        f"- augmented train limit: `{payload['augmented_limit_configs']}` configs",
        "",
        "## Question",
        "",
        str(payload["comparison_question"]),
        "",
        "## Rows",
        "",
        "| variant | params | representation params | role |",
        "|---|---:|---:|---|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['variant']} | {row.get('num_parameters_estimate')} | "
            f"{row.get('representation_parameters_estimate')} | {row.get('distillation_test_role', '')} |"
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
        "# Stage133 Manifest Contract Audit",
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
        "#SBATCH --job-name=rtece-prep133",
        "#SBATCH --partition=16V100",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --gpus-per-node=1",
        "#SBATCH --qos=flood-1o2gpu",
        "#SBATCH --time=03:55:00",
        "#SBATCH --output=/home/gengjianrui/bin/logs/rtece-prep133-%j.out",
        "#SBATCH --error=/home/gengjianrui/bin/logs/rtece-prep133-%j.err",
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


def _stage133_physical_cases(payload: dict[str, Any]) -> list[dict[str, Any]]:
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
                "selection_basis": "stage133 fixed-architecture teacher-relax trajectory distillation ablation",
            }
        )
    return cases


def materialize_stage133(payload: dict[str, Any]) -> dict[str, Any]:
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import write_pareto_sweep
    from benchmarks.oc20neb_tace_mace.make_rtece_physical_triage import write_physical_triage_wrappers

    artifacts = payload["artifacts"]
    root = Path(artifacts["prep_wrapper"]).parent
    root.mkdir(parents=True, exist_ok=True)
    write_prepare_wrapper(artifacts["prep_wrapper"], payload)
    sweep = write_pareto_sweep(
        artifacts["wrapper_root"],
        run_root=artifacts["run_root"],
        train_file=artifacts["augmented_train"],
        train_valid_file=payload["train_valid_file"],
        dft_valid_file=payload["dft_valid_file"],
        teacher_valid_file=payload["teacher_valid_file"],
        rows=_stage133_rows(),
        row_set="stage133-teacher-relax-distill",
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
        cases=_stage133_physical_cases(payload),
        stage="stage133_physical_triage",
        source="runs/oc20neb_tace_mace/rtece-stage133-teacher-relax-distill/stage133_interpretation.md",
        design_basis=(
            "Evaluate whether teacher PES relaxation trajectory distillation repairs force-tail and rattle-relax "
            "physical indicators for the fixed current-Pareto atomic rTECE architecture."
        ),
        job_name="rtece-phys133",
        rattle_start_config=58,
        rattle_limit_configs=8,
    )
    payload = dict(payload)
    payload["train_wrappers"] = [row["wrapper"] for row in sweep["rows"]]
    payload["physical_wrappers"] = [row["wrapper"] for row in physical["cases"]]
    payload["physical_cases"] = physical["cases"]
    return payload


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
    audit = audit_stage133_manifest(payload)
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
    parser.add_argument("--source-limit-configs", type=int, default=32)
    parser.add_argument("--copies-per-config", type=int, default=2)
    parser.add_argument("--rattle-std-a", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--relax-max-steps", type=int, default=4)
    parser.add_argument("--base-limit-configs", type=int, default=2048)
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
    payload = make_stage133_manifest(
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
        valid_limit_configs=args.valid_limit_configs,
        bench_limit_configs=args.bench_limit_configs,
        max_steps=args.max_steps,
        lr_warmup_steps=args.lr_warmup_steps,
        early_stopping_patience=args.early_stopping_patience,
    )
    if args.materialize:
        payload = materialize_stage133(payload)
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
