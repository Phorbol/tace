#!/usr/bin/env python3
"""Create Stage129 teacher-rattle distillation wrappers for current rTECE Pareto rows."""

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

CURRENT_PARETO_VARIANTS = (
    "l1_active_nrad12_species20_radial_species8_cross3_h64",
    "l1_active_nrad12_species24_radial_species8_cross3_h64",
)

DEFAULT_BASE_TRAIN = "runs/oc20neb_tace_mace/tece-distill-20260717/mixed_train_tw0.75.extxyz"
DEFAULT_TEACHER_VALID = "runs/oc20neb_tace_mace/tece-distill-20260717/teacher_valid.extxyz"
DEFAULT_DFT_VALID = "/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz"
DEFAULT_TEACHER_MODEL = "runs/oc20neb_tace_mace/675969_resume/checkpoints_epoch/TACE-0-100000-0.1329.ckpt"


def _std_token(value: float) -> str:
    return f"{float(value):g}".replace(".", "p").replace("-", "m")


def _command(parts: list[str | Path | int | float]) -> str:
    return " ".join(str(part) for part in parts if str(part) != "")


def _shell_export(name: str, value: str | Path | int | float) -> str:
    return f"export {name}={shlex.quote(str(value))}"


def _stage129_rows() -> list[dict[str, Any]]:
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import stage129_current_pareto_rows

    rows = stage129_current_pareto_rows()
    order = {name: idx for idx, name in enumerate(CURRENT_PARETO_VARIANTS)}
    return sorted(rows, key=lambda row: order[str(row["name"])])


def make_stage129_manifest(
    *,
    output_root: str | Path,
    base_train: str | Path = DEFAULT_BASE_TRAIN,
    source_configs: str | Path = DEFAULT_DFT_VALID,
    teacher_model: str | Path = DEFAULT_TEACHER_MODEL,
    train_valid_file: str | Path = DEFAULT_DFT_VALID,
    dft_valid_file: str | Path = DEFAULT_DFT_VALID,
    teacher_valid_file: str | Path = DEFAULT_TEACHER_VALID,
    source_start_config: int = 58,
    source_limit_configs: int = 8,
    copies_per_config: int = 16,
    rattle_std_a: float = 0.05,
    seed: int = 20260720,
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
    if float(rattle_std_a) < 0.0:
        raise ValueError("rattle_std_a must be non-negative")
    root = Path(output_root)
    rattle_name = (
        f"rattled_valid_start{int(source_start_config)}_limit{int(source_limit_configs)}_"
        f"copies{int(copies_per_config)}_std{_std_token(rattle_std_a)}.extxyz"
    )
    augmented_limit = int(base_limit_configs) + int(source_limit_configs) * int(copies_per_config)
    artifacts = {
        "rattled_configs": str(root / rattle_name),
        "rattled_summary": str(root / rattle_name.replace(".extxyz", "_summary.json")),
        "teacher_labeled_rattles": str(root / "teacher_labeled_stage128_window_rattles.extxyz"),
        "teacher_labeled_summary": str(root / "teacher_labeled_stage128_window_rattles_summary.json"),
        "augmented_train": str(root / "augmented_train_base2048_plus_teacher_rattle128.extxyz"),
        "augmented_train_summary": str(root / "augmented_train_base2048_plus_teacher_rattle128_summary.json"),
        "prep_wrapper": str(root / "rtece_stage129_prepare_no_export.sbatch"),
        "wrapper_root": str(root / "current_pareto_wrappers"),
        "run_root": str(root / "current_pareto_runs"),
        "physical_wrapper_root": str(root / "physical_triage_wrappers"),
        "physical_run_root": str(root / "physical_triage"),
    }
    py = "${TACE_PYTHON:-python}"
    steps = [
        {
            "id": "make_stage128_window_rattles",
            "command": _command([
                py,
                "benchmarks/oc20neb_tace_mace/make_rattle_distill_configs.py",
                "--input", source_configs,
                "--output", artifacts["rattled_configs"],
                "--summary", artifacts["rattled_summary"],
                "--start-config", int(source_start_config),
                "--limit-configs", int(source_limit_configs),
                "--copies-per-config", int(copies_per_config),
                "--rattle-std-a", float(rattle_std_a),
                "--seed", int(seed),
            ]),
        },
        {
            "id": "teacher_label_stage128_window_rattles",
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
                "--source-label", "base_mixed_train_tw0p75",
                "--input", artifacts["teacher_labeled_rattles"],
                "--source-label", "teacher_labeled_stage128_rattles",
                "--input-limit", int(base_limit_configs),
                "--input-limit", -1,
                "--output", artifacts["augmented_train"],
                "--summary", artifacts["augmented_train_summary"],
            ]),
        },
        {
            "id": "make_current_pareto_train_wrappers",
            "command": _command([
                py,
                "benchmarks/oc20neb_tace_mace/make_rtece_pareto_sweep.py",
                "--output-dir", artifacts["wrapper_root"],
                "--run-root", artifacts["run_root"],
                "--train-file", artifacts["augmented_train"],
                "--train-valid-file", train_valid_file,
                "--dft-valid-file", dft_valid_file,
                "--teacher-valid-file", teacher_valid_file,
                "--row-set", "stage129-current-pareto",
                "--limit-configs", augmented_limit,
                "--valid-limit-configs", int(valid_limit_configs),
                "--bench-limit-configs", int(bench_limit_configs),
                "--max-steps", int(max_steps),
                "--batch-size", 8,
                "--valid-batch-size", 16,
                "--lr-warmup-steps", int(lr_warmup_steps),
                "--early-stopping-patience", int(early_stopping_patience),
            ]),
        },
    ]
    return {
        "schema_version": "rtece_stage129_teacher_rattle_distill.v1",
        "stage": "stage129_teacher_rattle_distill_current_pareto",
        "distillation_semantics": "teacher_fake_labels_on_stage128_rattle_window",
        "comparison_question": (
            "With stage127/stage128 Pareto architectures fixed, does teacher-labeled rattle coverage "
            "reduce the stage128 rattle force-spike failure without C/N-specific architecture changes?"
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
        "base_limit_configs": int(base_limit_configs),
        "augmented_limit_configs": int(augmented_limit),
        "valid_limit_configs": int(valid_limit_configs),
        "bench_limit_configs": int(bench_limit_configs),
        "max_steps": int(max_steps),
        "lr_warmup_steps": int(lr_warmup_steps),
        "early_stopping_patience": int(early_stopping_patience),
        "row_set": "stage129-current-pareto",
        "rows": [{"variant": str(row["name"]), **row} for row in _stage129_rows()],
        "artifacts": artifacts,
        "steps": steps,
        "review_basis": [
            "TECE_design_space: deployment error separates projection error from distillation error.",
            "TECE_design_space stage D: teacher E/F distillation should precede architecture-specific hacks.",
            "rTECE_review: teacher cache/distillation closure is P1 and must be evaluated with E/F RMSE and physical tests.",
            "Stage128: dimer and C/N RMSD are not dominant; rattle force spikes are the current falsification target.",
        ],
    }


def audit_stage129_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    commands = "\n".join(str(step.get("command", "")) for step in payload.get("steps") or [])
    variants = [str(row.get("variant") or row.get("name")) for row in payload.get("rows") or []]
    checks = {
        "schema_version": payload.get("schema_version") == "rtece_stage129_teacher_rattle_distill.v1",
        "distillation_semantics": payload.get("distillation_semantics") == "teacher_fake_labels_on_stage128_rattle_window",
        "stage128_window": int(payload.get("source_start_config", -1)) == 58 and int(payload.get("source_limit_configs", 0)) == 8,
        "stage128_pareto_rows": variants == list(CURRENT_PARETO_VARIANTS),
        "augmented_limit_configs": int(payload.get("augmented_limit_configs", 0)) == int(payload.get("base_limit_configs", 0)) + int(payload.get("source_limit_configs", 0)) * int(payload.get("copies_per_config", 0)),
        "uses_teacher_fake_labels": "distill_tace_labels.py" in commands and "--reference-prefix source_" in commands,
        "uses_current_pareto_row_set": "--row-set stage129-current-pareto" in commands,
        "base_subset_before_rattles": f"--input-limit {int(payload.get('base_limit_configs', 0))} --input-limit -1" in commands,
        "training_contract": "--max-steps 20000" in commands and "--lr-warmup-steps 500" in commands and "--early-stopping-patience 400" in commands,
        "no_forbidden_sbatch_flags": all(flag not in commands for flag in ("--export", "--mem", "--cpus-per-task")),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "schema_version": "rtece_stage129_manifest_audit.v1",
        "contract_pass": not failed,
        "failed_checks": failed,
        "checks": checks,
        "checked": {
            "variants": variants,
            "augmented_limit_configs": payload.get("augmented_limit_configs"),
            "source_start_config": payload.get("source_start_config"),
            "source_limit_configs": payload.get("source_limit_configs"),
            "copies_per_config": payload.get("copies_per_config"),
            "row_set": payload.get("row_set"),
        },
    }


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage129 Teacher-Rattle Distillation Plan",
        "",
        f"- stage: `{payload['stage']}`",
        f"- row set: `{payload['row_set']}`",
        f"- semantics: `{payload['distillation_semantics']}`",
        f"- source window: `{payload['source_start_config']}:{payload['source_start_config'] + payload['source_limit_configs']}`",
        f"- augmented train limit: `{payload['augmented_limit_configs']}` configs",
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
        "# Stage129 Manifest Contract Audit",
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
        "#SBATCH --job-name=rtece-prep129",
        "#SBATCH --partition=16V100",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --gpus-per-node=1",
        "#SBATCH --qos=flood-1o2gpu",
        "#SBATCH --time=03:55:00",
        "#SBATCH --output=/home/gengjianrui/bin/logs/rtece-prep129-%j.out",
        "#SBATCH --error=/home/gengjianrui/bin/logs/rtece-prep129-%j.err",
        "",
        "set -euo pipefail",
        "set -x",
        _shell_export("PYTHONUNBUFFERED", 1),
        _shell_export("TACE_ROOT", "/home/gengjianrui/bin/tace"),
        _shell_export("ENV_DIR", env_dir),
        'export TACE_PYTHON="${ENV_DIR}/bin/python"',
        'export PATH="${ENV_DIR}/bin:${PATH}"',
        'export PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}"',
        _shell_export("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", 1),
        _shell_export("TORCH_CUDA_ARCH_LIST", "7.0"),
        "mkdir -p /home/gengjianrui/bin/logs",
        'cd "${TACE_ROOT}"',
        '"${TACE_PYTHON}" -V',
        "nvidia-smi -L",
        "",
    ]
    for step in payload["steps"][:3]:
        body.extend([f"# {step['id']}", step["command"], ""])
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("\n".join(body), encoding="utf-8")
    return wrapper


def materialize_stage129(payload: dict[str, Any]) -> dict[str, Any]:
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import write_pareto_sweep

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
        rows=_stage129_rows(),
        row_set="stage129-current-pareto",
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
    payload = dict(payload)
    payload["train_wrappers"] = [row["wrapper"] for row in sweep["rows"]]
    return payload


def write_manifest_files(payload: dict[str, Any], *, output_json: Path, output_md: Path, audit_json: Path | None, audit_md: Path | None) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(format_markdown(payload), encoding="utf-8")
    audit = audit_stage129_manifest(payload)
    if audit_json is not None:
        audit_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if audit_md is not None:
        audit_md.write_text(format_audit_markdown(audit), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--base-train", default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--source-configs", default=DEFAULT_DFT_VALID)
    parser.add_argument("--teacher-model", default=DEFAULT_TEACHER_MODEL)
    parser.add_argument("--train-valid-file", default=DEFAULT_DFT_VALID)
    parser.add_argument("--dft-valid-file", default=DEFAULT_DFT_VALID)
    parser.add_argument("--teacher-valid-file", default=DEFAULT_TEACHER_VALID)
    parser.add_argument("--source-start-config", type=int, default=58)
    parser.add_argument("--source-limit-configs", type=int, default=8)
    parser.add_argument("--copies-per-config", type=int, default=16)
    parser.add_argument("--rattle-std-a", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--base-limit-configs", type=int, default=2048)
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
    payload = make_stage129_manifest(
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
        base_limit_configs=args.base_limit_configs,
        max_steps=args.max_steps,
        lr_warmup_steps=args.lr_warmup_steps,
        early_stopping_patience=args.early_stopping_patience,
    )
    if args.materialize:
        payload = materialize_stage129(payload)
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
