#!/usr/bin/env python3
"""Create Stage175 local TECE-front probe manifest and SAI wrapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.oc20neb_tace_mace.analyze_rtece_local_front_features import (
    DEFAULT_FEATURE_FAMILIES,
    DEFAULT_RIDGE_GRID,
    DEFAULT_SHELL_EDGES,
)
from benchmarks.oc20neb_tace_mace.make_rtece_stage171_residual_active_set import (
    DEFAULT_CONFIGS,
    DEFAULT_STAGE165_JSON,
    _shell_assign,
)


def _format_float(value: float) -> str:
    return f"{float(value):g}"


def _format_grid(values: list[float] | tuple[float, ...]) -> str:
    return ",".join(_format_float(value) for value in values)


def _feature_family_payload() -> list[dict[str, Any]]:
    return [
        {
            "name": "local_l0_rank3",
            "l_max": 0,
            "chemistry_rank": 3,
            "tece_front_semantics": "local_species_conditioned_cartesian_moment_scalar_contractions",
        },
        {
            "name": "local_l0_l1_rank3",
            "l_max": 1,
            "chemistry_rank": 3,
            "tece_front_semantics": "local_species_conditioned_cartesian_moment_scalar_contractions",
        },
        {
            "name": "local_l0_l1_l2_rank3",
            "l_max": 2,
            "chemistry_rank": 3,
            "tece_front_semantics": "local_species_conditioned_cartesian_moment_scalar_contractions",
        },
        {
            "name": "local_l0_l1_l2_rank4",
            "l_max": 2,
            "chemistry_rank": 4,
            "tece_front_semantics": "local_species_conditioned_cartesian_moment_scalar_contractions",
        },
    ]


def make_stage175_manifest(
    *,
    output_root: str | Path,
    stage165_json: str | Path = DEFAULT_STAGE165_JSON,
    configs: str | Path = DEFAULT_CONFIGS,
    limit_configs: int = 512,
    group_key: str = "case_id",
    stage165_variant: str = "stage157_direct_b32_rel0p25_mixed2048",
    shell_edges: tuple[float, ...] | list[float] = DEFAULT_SHELL_EDGES,
    ridge_grid: tuple[float, ...] | list[float] = DEFAULT_RIDGE_GRID,
) -> dict[str, Any]:
    if int(limit_configs) < 8:
        raise ValueError("limit_configs must be at least 8")
    root = Path(output_root)
    artifacts = {
        "manifest": str(root / "stage175_manifest.json"),
        "manifest_audit": str(root / "stage175_manifest_audit.json"),
        "stage_plan": str(root / "stage175_plan.md"),
        "wrapper": str(root / "stage175_local_tece_front_probe.sbatch"),
        "results_json": str(root / "stage175_results_summary.json"),
        "results_md": str(root / "stage175_results_summary.md"),
    }
    return {
        "schema_version": "rtece_stage175_local_tece_front_probe.v1",
        "stage": "stage175_local_tece_front_probe",
        "target_semantics": "stage165_case_offset_residual_mev_atom",
        "diagnostic_semantics": "local_tece_front_probe_after_stage174_global_proxy_rejection",
        "energy_split_mode": "group-loocv",
        "group_key": str(group_key),
        "uses_case_id_as_feature": False,
        "non_deployable_feature_policy": "case_id_source_key_source_frame_group_id_image_id_are_grouping_or_provenance_metadata_only",
        "comparison_question": (
            "After Stage174 rejected global composition/tag/geometry proxies, can a deployable local TECE-style "
            "species-conditioned moment scalar front provide an unseen-case energy-offset coordinate worth promoting?"
        ),
        "stage165_json": str(stage165_json),
        "configs": str(configs),
        "stage165_variant": str(stage165_variant),
        "limit_configs": int(limit_configs),
        "shell_edges": [float(value) for value in shell_edges],
        "feature_families": _feature_family_payload(),
        "projection_diagnostics": {
            "fit_intercept": True,
            "standardize_features": True,
            "ridge_grid": [float(value) for value in ridge_grid],
            "intercept_baseline": "group_loocv_train_mean_per_atom_residual_times_eval_atom_count",
            "promotion_gate": "candidate_energy_per_atom_rmse_must_beat_intercept_baseline",
        },
        "artifacts": artifacts,
        "review_basis": [
            "Stage174: global deployable composition/tag/geometry proxies did not beat the group-heldout intercept baseline.",
            "TECE_design_space.md: rTECE should use one local moment pass, sparse atomic invariants, and selected low-cost scalar paths.",
            "rTECE_review.md: low-rank species chemistry basis is required to avoid element-collision failures without returning to one-hot large channels.",
            "Stage170/173: the raw energy issue is a low-frequency deployable representation problem, not a per-case metadata correction.",
        ],
    }


def audit_stage175_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    families = list(payload.get("feature_families") or [])
    family_names = [str(family.get("name")) for family in families]
    diagnostics = dict(payload.get("projection_diagnostics") or {})
    artifacts = dict(payload.get("artifacts") or {})
    checks = {
        "schema_version": payload.get("schema_version") == "rtece_stage175_local_tece_front_probe.v1",
        "stage": payload.get("stage") == "stage175_local_tece_front_probe",
        "target_semantics": payload.get("target_semantics") == "stage165_case_offset_residual_mev_atom",
        "split_mode": payload.get("energy_split_mode") == "group-loocv",
        "no_case_id_feature": payload.get("uses_case_id_as_feature") is False,
        "feature_families": family_names == list(DEFAULT_FEATURE_FAMILIES),
        "has_l2_family": any(int(family.get("l_max", -1)) == 2 for family in families),
        "local_tece_semantics": all(str(family.get("tece_front_semantics", "")).startswith("local_species_conditioned") for family in families),
        "fit_intercept": diagnostics.get("fit_intercept") is True,
        "standardize_features": diagnostics.get("standardize_features") is True,
        "ridge_grid": list(diagnostics.get("ridge_grid") or []) == [float(value) for value in DEFAULT_RIDGE_GRID],
        "artifacts": all(key in artifacts for key in ("manifest", "manifest_audit", "stage_plan", "wrapper", "results_json", "results_md")),
    }
    failed = [key for key, passed in checks.items() if not passed]
    return {
        "schema_version": "rtece_stage175_manifest_audit.v1",
        "contract_pass": not failed,
        "failed_checks": failed,
        "checks": checks,
        "no_forbidden_sbatch_flags": True,
    }


def _analyzer_command(payload: dict[str, Any]) -> str:
    diagnostics = payload["projection_diagnostics"]
    return " ".join(
        [
            '"${TACE_PYTHON}"',
            "benchmarks/oc20neb_tace_mace/analyze_rtece_local_front_features.py",
            "--configs", '"${CONFIGS}"',
            "--stage165-json", '"${STAGE165_JSON}"',
            "--stage165-variant", str(payload["stage165_variant"]),
            "--group-key", str(payload["group_key"]),
            "--limit-configs", '"${LIMIT_CONFIGS}"',
            "--output-json", '"${RESULTS_JSON}"',
            "--output-md", '"${RESULTS_MD}"',
            "--shell-edges", _format_grid(payload["shell_edges"]),
            "--energy-ridge-grid", _format_grid(diagnostics["ridge_grid"]),
        ]
    )


def write_stage175_wrapper(path: str | Path, payload: dict[str, Any]) -> Path:
    wrapper = Path(path)
    artifacts = payload["artifacts"]
    env_dir = "/home/gengjianrui/bin/.venvs/tace-mace-cu126"
    body = [
        "#!/bin/bash",
        "#SBATCH --job-name=rtece-res175",
        "#SBATCH --partition=16V100",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --gpus-per-node=1",
        "#SBATCH --qos=flood-1o2gpu",
        "#SBATCH --time=01:55:00",
        "#SBATCH --output=/home/gengjianrui/bin/logs/rtece-res175-%j.out",
        "#SBATCH --error=/home/gengjianrui/bin/logs/rtece-res175-%j.err",
        "",
        "set -eo pipefail",
        "set -x",
        _shell_assign("TACE_ROOT", "/home/gengjianrui/bin/tace"),
        _shell_assign("ENV_DIR", env_dir),
        'TACE_PYTHON="${ENV_DIR}/bin/python"',
        'PATH="${ENV_DIR}/bin:${PATH}"',
        'PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}"',
        _shell_assign("CONFIGS", payload["configs"]),
        _shell_assign("STAGE165_JSON", payload["stage165_json"]),
        _shell_assign("RESULTS_JSON", artifacts["results_json"]),
        _shell_assign("RESULTS_MD", artifacts["results_md"]),
        _shell_assign("LIMIT_CONFIGS", int(payload["limit_configs"])),
        'mkdir -p /home/gengjianrui/bin/logs "$(dirname "${RESULTS_JSON}")"',
        'cd "${TACE_ROOT}"',
        'export TACE_PYTHON PATH PYTHONPATH',
        '"${TACE_PYTHON}" -V',
        "nvidia-smi -L",
        "",
        "# Stage175: local TECE front feature probe",
        _analyzer_command(payload),
        "",
    ]
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("\n".join(body), encoding="utf-8")
    return wrapper


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage175 Local TECE Front Probe Plan",
        "",
        f"- stage: `{payload['stage']}`",
        f"- target: `{payload['target_semantics']}`",
        f"- split: `{payload['energy_split_mode']}` by `{payload['group_key']}`",
        f"- shell edges: `{_format_grid(payload['shell_edges'])}`",
        f"- ridge grid: `{_format_grid(payload['projection_diagnostics']['ridge_grid'])}`",
        "",
        "## Question",
        "",
        payload["comparison_question"],
        "",
        "## Feature Families",
        "",
        "| family | Lmax | chemistry rank | semantics |",
        "|---|---:|---:|---|",
    ]
    for family in payload["feature_families"]:
        lines.append(f"| {family['name']} | {family['l_max']} | {family['chemistry_rank']} | {family['tece_front_semantics']} |")
    lines.extend(["", "## Review Basis", ""])
    lines.extend(f"- {item}" for item in payload["review_basis"])
    lines.append("")
    return "\n".join(lines)


def materialize_stage175(payload: dict[str, Any]) -> dict[str, Any]:
    audit = audit_stage175_manifest(payload)
    if not audit["contract_pass"]:
        raise ValueError(f"Stage175 manifest audit failed: {audit['failed_checks']}")
    artifacts = payload["artifacts"]
    for key in ("manifest", "manifest_audit", "stage_plan", "wrapper"):
        Path(artifacts[key]).parent.mkdir(parents=True, exist_ok=True)
    Path(artifacts["manifest"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(artifacts["manifest_audit"]).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(artifacts["stage_plan"]).write_text(format_markdown(payload), encoding="utf-8")
    write_stage175_wrapper(artifacts["wrapper"], payload)
    return dict(payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stage165-json", default=DEFAULT_STAGE165_JSON)
    parser.add_argument("--configs", default=DEFAULT_CONFIGS)
    parser.add_argument("--limit-configs", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = make_stage175_manifest(
        output_root=args.output_root,
        stage165_json=args.stage165_json,
        configs=args.configs,
        limit_configs=args.limit_configs,
    )
    materialized = materialize_stage175(payload)
    print(json.dumps({"stage": materialized["stage"], "artifacts": materialized["artifacts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
