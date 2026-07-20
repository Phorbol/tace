#!/usr/bin/env python3
"""Create Stage136 L2-vs-L1 rTECE projection diagnostic wrappers."""

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

from benchmarks.oc20neb_tace_mace.make_rtece_stage132_broad_teacher_distill import DEFAULT_DFT_VALID
from benchmarks.oc20neb_tace_mace.make_rtece_stage135_l2_atomic_projection import DEFAULT_TRAIN_FILE

REFERENCE_PATH_IDS = (
    "atomic.radial_density",
    "atomic.species_basis_density",
    "atomic.vector_norm",
    "atomic.vector_cross_radial_dot",
    "atomic.quadrupole_norm",
    "atomic.quadrupole_cross_radial_frobenius",
)
L1_PATH_IDS = (
    "atomic.radial_density",
    "atomic.species_basis_density",
    "atomic.vector_norm",
    "atomic.vector_cross_radial_dot",
)
CANDIDATE_SPECS = (
    ("full_l2_reference", REFERENCE_PATH_IDS),
    ("stage132_l1_atomic", L1_PATH_IDS),
    (
        "drop_quadrupole_cross",
        (
            "atomic.radial_density",
            "atomic.species_basis_density",
            "atomic.vector_norm",
            "atomic.vector_cross_radial_dot",
            "atomic.quadrupole_norm",
        ),
    ),
    (
        "drop_quadrupole_norm",
        (
            "atomic.radial_density",
            "atomic.species_basis_density",
            "atomic.vector_norm",
            "atomic.vector_cross_radial_dot",
            "atomic.quadrupole_cross_radial_frobenius",
        ),
    ),
)


def _shell_assign(name: str, value: str | Path | int | float) -> str:
    return f"{name}={shlex.quote(str(value))}"


def _join_paths(path_ids: tuple[str, ...] | list[str]) -> str:
    return ",".join(str(path_id) for path_id in path_ids)


def _candidate_payload() -> list[dict[str, Any]]:
    return [{"name": name, "path_ids": list(path_ids)} for name, path_ids in CANDIDATE_SPECS]


def make_stage136_manifest(
    *,
    output_root: str | Path,
    train_configs: str | Path = DEFAULT_TRAIN_FILE,
    valid_configs: str | Path = DEFAULT_DFT_VALID,
    limit_configs: int = 64,
    energy_eval_stride: int = 4,
    force_eval_stride: int = 4,
    energy_eval_offset: int = 0,
    force_eval_offset: int = 0,
    energy_target_key: str = "energy",
    force_target_key: str = "forces",
    num_radial: int = 12,
    species_basis_channels: int = 24,
    atomic_cross_radial_sketch_channels: int = 3,
) -> dict[str, Any]:
    if int(limit_configs) < 2:
        raise ValueError("limit_configs must be at least 2")
    if int(energy_eval_stride) < 2:
        raise ValueError("energy_eval_stride must be at least 2")
    if int(force_eval_stride) < 2:
        raise ValueError("force_eval_stride must be at least 2")
    root = Path(output_root)
    artifacts = {
        "wrapper": str(root / "rtece_stage136_projection_no_export.sbatch"),
        "output_root": str(root / "projection_outputs"),
        "train_projection_json": str(root / "projection_outputs" / f"stage136_train_mixed{int(limit_configs)}_l2_reference_projection.json"),
        "valid_projection_json": str(root / "projection_outputs" / f"stage136_valid_dft{int(limit_configs)}_l2_reference_projection.json"),
    }
    return {
        "schema_version": "rtece_stage136_l2_projection_diagnostic.v1",
        "stage": "stage136_l2_projection_diagnostic",
        "diagnostic_semantics": "stage135_l2_reference_vs_stage132_l1_projection_residual",
        "comparison_question": (
            "After Stage135 showed lower validation loss but worse throughput/F-RMSE/max, measure projection error and "
            "held-out E/F label residuals for the Stage132 L1 atomic subspace inside the Stage135 L_A=2 quadrupole "
            "reference. This separates true L2 projection value from optimization/distillation failure."
        ),
        "train_configs": str(train_configs),
        "valid_configs": str(valid_configs),
        "limit_configs": int(limit_configs),
        "energy_eval_stride": int(energy_eval_stride),
        "force_eval_stride": int(force_eval_stride),
        "energy_eval_offset": int(energy_eval_offset),
        "force_eval_offset": int(force_eval_offset),
        "energy_target_key": str(energy_target_key),
        "force_target_key": str(force_target_key),
        "num_radial": int(num_radial),
        "species_basis_channels": int(species_basis_channels),
        "atomic_cross_radial_sketch_channels": int(atomic_cross_radial_sketch_channels),
        "reference_path_ids": list(REFERENCE_PATH_IDS),
        "candidates": _candidate_payload(),
        "artifacts": artifacts,
        "review_basis": [
            "TECE_design_space: deleted TECE coordinates should be evaluated as projection/distillation error before choosing the next retained path set.",
            "rTECE_review: projection/importance diagnostics are higher priority than further kernel work when physical closure is still failing.",
            "Stage135: naive L_A=2 atomic quadrupole training lowered validation loss but worsened F RMSE/max and throughput, so L2 value must be diagnosed separately from optimization and label effects.",
            "Stage113: force-aware projection made L2 norm paths look force-relevant in a small k=2 smoke; Stage136 repeats the question on the current nrad12/species24/cross3 active row.",
        ],
    }


def audit_stage136_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = list(payload.get("candidates") or [])
    candidate_names = [str(candidate.get("name")) for candidate in candidates]
    reference = list(payload.get("reference_path_ids") or [])
    checks = {
        "schema_version": payload.get("schema_version") == "rtece_stage136_l2_projection_diagnostic.v1",
        "stage": payload.get("stage") == "stage136_l2_projection_diagnostic",
        "diagnostic_semantics": payload.get("diagnostic_semantics") == "stage135_l2_reference_vs_stage132_l1_projection_residual",
        "reference_is_l2_atomic": reference == list(REFERENCE_PATH_IDS),
        "candidate_set": candidate_names == [name for name, _paths in CANDIDATE_SPECS],
        "contains_stage132_l1_candidate": any(candidate.get("name") == "stage132_l1_atomic" and candidate.get("path_ids") == list(L1_PATH_IDS) for candidate in candidates),
        "current_active_row_ranks": int(payload.get("num_radial", 0)) == 12 and int(payload.get("species_basis_channels", 0)) == 24 and int(payload.get("atomic_cross_radial_sketch_channels", 0)) == 3,
        "ef_targets": payload.get("energy_target_key") == "energy" and payload.get("force_target_key") == "forces",
        "heldout_split": int(payload.get("energy_eval_stride", 0)) >= 2 and int(payload.get("force_eval_stride", 0)) >= 2,
        "artifacts": "wrapper" in (payload.get("artifacts") or {}) and "train_projection_json" in (payload.get("artifacts") or {}) and "valid_projection_json" in (payload.get("artifacts") or {}),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "schema_version": "rtece_stage136_manifest_audit.v1",
        "contract_pass": not failed,
        "failed_checks": failed,
        "checks": checks,
        "checked": {
            "candidate_names": candidate_names,
            "reference_path_ids": reference,
            "limit_configs": payload.get("limit_configs"),
        },
    }


def _analyzer_command(*, configs_var: str, output_var: str, payload: dict[str, Any]) -> str:
    ref = _join_paths(REFERENCE_PATH_IDS)
    parts = [
        '"${TACE_PYTHON}"',
        "benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py",
        "--configs", f'"${{{configs_var}}}"',
        "--output-json", f'"${{{output_var}}}"',
        "--reference-path-ids", f'"${{REFERENCE_PATH_IDS}}"',
    ]
    for name, path_ids in CANDIDATE_SPECS:
        parts.extend(["--candidate", f"{name}:{_join_paths(path_ids)}"])
    parts.extend([
        "--num-radial", '"${NUM_RADIAL}"',
        "--species-basis-channels", '"${SPECIES_BASIS_CHANNELS}"',
        "--atomic-cross-radial-sketch-channels", '"${ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS}"',
        "--limit-configs", '"${LIMIT_CONFIGS}"',
        "--default-dtype", "float64",
        "--neighborlist-backend", "matscipy",
        "--energy-target-key", str(payload["energy_target_key"]),
        "--energy-eval-stride", str(int(payload["energy_eval_stride"])),
        "--energy-eval-offset", str(int(payload["energy_eval_offset"])),
        "--force-target-key", str(payload["force_target_key"]),
        "--force-eval-stride", str(int(payload["force_eval_stride"])),
        "--force-eval-offset", str(int(payload["force_eval_offset"])),
    ])
    # Keep the literal reference variable visible in the wrapper for auditability.
    _ = ref
    return " ".join(parts)


def write_stage136_wrapper(path: str | Path, payload: dict[str, Any]) -> Path:
    wrapper = Path(path)
    env_dir = "/home/gengjianrui/bin/.venvs/tace-mace-cu126"
    artifacts = payload["artifacts"]
    body = [
        "#!/bin/bash",
        "#SBATCH --job-name=rtece-proj136",
        "#SBATCH --partition=16V100",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --gpus-per-node=1",
        "#SBATCH --qos=flood-1o2gpu",
        "#SBATCH --time=03:55:00",
        "#SBATCH --output=/home/gengjianrui/bin/logs/rtece-proj136-%j.out",
        "#SBATCH --error=/home/gengjianrui/bin/logs/rtece-proj136-%j.err",
        "",
        "set -euo pipefail",
        "set -x",
        "export PYTHONUNBUFFERED=1",
        _shell_assign("TACE_ROOT", "/home/gengjianrui/bin/tace"),
        _shell_assign("ENV_DIR", env_dir),
        'TACE_PYTHON="${ENV_DIR}/bin/python"',
        'PATH="${ENV_DIR}/bin:${PATH}"',
        'PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}"',
        _shell_assign("TRAIN_CONFIGS", payload["train_configs"]),
        _shell_assign("VALID_CONFIGS", payload["valid_configs"]),
        _shell_assign("OUT_ROOT", artifacts["output_root"]),
        _shell_assign("TRAIN_OUTPUT", artifacts["train_projection_json"]),
        _shell_assign("VALID_OUTPUT", artifacts["valid_projection_json"]),
        _shell_assign("LIMIT_CONFIGS", int(payload["limit_configs"])),
        _shell_assign("NUM_RADIAL", int(payload["num_radial"])),
        _shell_assign("SPECIES_BASIS_CHANNELS", int(payload["species_basis_channels"])),
        _shell_assign("ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS", int(payload["atomic_cross_radial_sketch_channels"])),
        _shell_assign("REFERENCE_PATH_IDS", _join_paths(REFERENCE_PATH_IDS)),
        "mkdir -p /home/gengjianrui/bin/logs \"${OUT_ROOT}\"",
        'cd "${TACE_ROOT}"',
        'export TACE_PYTHON PATH PYTHONPATH',
        '"${TACE_PYTHON}" -V',
        "nvidia-smi -L",
        "",
        "# Stage132 mixed DFT/teacher train-distribution E/F projection",
        _analyzer_command(configs_var="TRAIN_CONFIGS", output_var="TRAIN_OUTPUT", payload=payload),
        "",
        "# Pure DFT valid-distribution E/F projection",
        _analyzer_command(configs_var="VALID_CONFIGS", output_var="VALID_OUTPUT", payload=payload),
        "",
    ]
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("\n".join(body), encoding="utf-8")
    return wrapper


def materialize_stage136(payload: dict[str, Any]) -> dict[str, Any]:
    audit = audit_stage136_manifest(payload)
    if not audit["contract_pass"]:
        raise ValueError(f"Stage136 manifest audit failed: {audit['failed_checks']}")
    materialized = dict(payload)
    materialized["wrapper"] = str(write_stage136_wrapper(payload["artifacts"]["wrapper"], payload))
    return materialized


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage136 L2 Projection Diagnostic Plan",
        "",
        f"- stage: `{payload['stage']}`",
        f"- semantics: `{payload['diagnostic_semantics']}`",
        f"- train configs: `{payload['train_configs']}`",
        f"- valid configs: `{payload['valid_configs']}`",
        f"- limit configs: `{payload['limit_configs']}`",
        f"- active row ranks: nrad `{payload['num_radial']}`, species `{payload['species_basis_channels']}`, cross `{payload['atomic_cross_radial_sketch_channels']}`",
        "",
        "## Question",
        "",
        str(payload["comparison_question"]),
        "",
        "## Candidates",
        "",
        "| candidate | paths |",
        "|---|---|",
    ]
    for candidate in payload["candidates"]:
        lines.append(f"| {candidate['name']} | {', '.join(candidate['path_ids'])} |")
    lines.extend(["", "## Review Basis", ""])
    lines.extend(f"- {item}" for item in payload.get("review_basis", []))
    lines.append("")
    return "\n".join(lines)


def format_audit_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Stage136 Manifest Contract Audit",
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
    audit = audit_stage136_manifest(payload)
    if audit_json is not None:
        audit_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if audit_md is not None:
        audit_md.write_text(format_audit_markdown(audit), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-configs", default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument("--valid-configs", default=DEFAULT_DFT_VALID)
    parser.add_argument("--limit-configs", type=int, default=64)
    parser.add_argument("--energy-eval-stride", type=int, default=4)
    parser.add_argument("--force-eval-stride", type=int, default=4)
    parser.add_argument("--energy-eval-offset", type=int, default=0)
    parser.add_argument("--force-eval-offset", type=int, default=0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path)
    parser.add_argument("--audit-md", type=Path)
    parser.add_argument("--materialize", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = make_stage136_manifest(
        output_root=args.output_root,
        train_configs=args.train_configs,
        valid_configs=args.valid_configs,
        limit_configs=args.limit_configs,
        energy_eval_stride=args.energy_eval_stride,
        force_eval_stride=args.force_eval_stride,
        energy_eval_offset=args.energy_eval_offset,
        force_eval_offset=args.force_eval_offset,
    )
    if args.materialize:
        payload = materialize_stage136(payload)
    write_manifest_files(payload, output_json=args.output_json, output_md=args.output_md, audit_json=args.audit_json, audit_md=args.audit_md)
    print(json.dumps({"stage": payload["stage"], "output_json": str(args.output_json)}, sort_keys=True))


if __name__ == "__main__":
    main()
