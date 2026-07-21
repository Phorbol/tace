#!/usr/bin/env python3
"""Create Stage174 low-frequency feature probe manifest and SAI wrapper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.oc20neb_tace_mace.analyze_rtece_lowfreq_features import (
    DEFAULT_FEATURE_FAMILIES,
    DEFAULT_RIDGE_GRID,
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
            "name": "composition_fraction",
            "requires_tags": False,
            "tece_semantics": "low_frequency_species_composition_scalar",
            "forbidden_feature_names": [],
        },
        {
            "name": "tag_composition",
            "requires_tags": True,
            "tece_semantics": "tag_conditioned_adsorbate_slab_species_scalar",
            "forbidden_feature_names": [],
        },
        {
            "name": "geometry_z_profile",
            "requires_tags": False,
            "tece_semantics": "coarse_cell_and_surface_normal_geometry_scalar",
            "forbidden_feature_names": [],
        },
        {
            "name": "composition_tag_geometry",
            "requires_tags": True,
            "tece_semantics": "tag_conditioned_low_frequency_site_front_scalar",
            "forbidden_feature_names": [],
        },
        {
            "name": "pair_histogram",
            "requires_tags": False,
            "tece_semantics": "coarse_local_pair_radial_histogram_scalar",
            "forbidden_feature_names": [],
        },
    ]


def make_stage174_manifest(
    *,
    output_root: str | Path,
    stage165_json: str | Path = DEFAULT_STAGE165_JSON,
    configs: str | Path = DEFAULT_CONFIGS,
    limit_configs: int = 512,
    group_key: str = "case_id",
    stage165_variant: str = "stage157_direct_b32_rel0p25_mixed2048",
    ridge_grid: tuple[float, ...] | list[float] = DEFAULT_RIDGE_GRID,
) -> dict[str, Any]:
    if int(limit_configs) < 8:
        raise ValueError("limit_configs must be at least 8")
    root = Path(output_root)
    artifacts = {
        "manifest": str(root / "stage174_manifest.json"),
        "manifest_audit": str(root / "stage174_manifest_audit.json"),
        "stage_plan": str(root / "stage174_plan.md"),
        "wrapper": str(root / "stage174_lowfreq_feature_probe.sbatch"),
        "results_json": str(root / "stage174_results_summary.json"),
        "results_md": str(root / "stage174_results_summary.md"),
    }
    return {
        "schema_version": "rtece_stage174_lowfreq_feature_probe.v1",
        "stage": "stage174_lowfreq_feature_probe",
        "target_semantics": "stage165_case_offset_residual_mev_atom",
        "diagnostic_semantics": "deployable_low_frequency_chemistry_site_front_probe_after_stage173_rejection",
        "energy_split_mode": "group-loocv",
        "group_key": str(group_key),
        "uses_case_id_as_feature": False,
        "non_deployable_feature_policy": "case_id_source_key_source_frame_group_id_image_id_are_grouping_or_provenance_metadata_only",
        "comparison_question": (
            "After Stage173 showed current rTECE path descriptors do not beat an intercept-only unseen-case offset baseline, "
            "can deployable low-frequency chemistry/site/front summaries provide a group-heldout coordinate worth promoting "
            "into the next trainable rTECE front module?"
        ),
        "stage165_json": str(stage165_json),
        "configs": str(configs),
        "stage165_variant": str(stage165_variant),
        "limit_configs": int(limit_configs),
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
            "Stage170: raw absolute E RMSE is dominated by low-frequency case/site/adsorbate offsets.",
            "Stage173: robust current rTECE semantic path descriptors do not beat the intercept-only group-heldout baseline.",
            "rTECE_review.md: low-rank species chemistry and per-element references are required, while case/source metadata must not become deployable features.",
            "TECE_design_space.md: model degradation should retain high-value low-cost scalar coordinates selected by heldout physical/error benefit.",
        ],
    }


def audit_stage174_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    families = list(payload.get("feature_families") or [])
    family_names = [str(family.get("name")) for family in families]
    diagnostics = dict(payload.get("projection_diagnostics") or {})
    artifacts = dict(payload.get("artifacts") or {})
    forbidden = ("case_id", "source_key", "source_frame", "group_id", "image_id")
    forbidden_hits = [
        name
        for family in families
        for name in family.get("forbidden_feature_names", [])
        if any(key in str(name) for key in forbidden)
    ]
    checks = {
        "schema_version": payload.get("schema_version") == "rtece_stage174_lowfreq_feature_probe.v1",
        "stage": payload.get("stage") == "stage174_lowfreq_feature_probe",
        "target_semantics": payload.get("target_semantics") == "stage165_case_offset_residual_mev_atom",
        "split_mode": payload.get("energy_split_mode") == "group-loocv",
        "no_case_id_feature": payload.get("uses_case_id_as_feature") is False,
        "feature_families": family_names == list(DEFAULT_FEATURE_FAMILIES),
        "has_tag_free_family": any(not bool(family.get("requires_tags")) for family in families),
        "has_tag_required_family": any(bool(family.get("requires_tags")) for family in families),
        "no_forbidden_feature_names": not forbidden_hits,
        "fit_intercept": diagnostics.get("fit_intercept") is True,
        "standardize_features": diagnostics.get("standardize_features") is True,
        "ridge_grid": list(diagnostics.get("ridge_grid") or []) == [float(value) for value in DEFAULT_RIDGE_GRID],
        "artifacts": all(key in artifacts for key in ("manifest", "manifest_audit", "stage_plan", "wrapper", "results_json", "results_md")),
    }
    failed = [key for key, passed in checks.items() if not passed]
    return {
        "schema_version": "rtece_stage174_manifest_audit.v1",
        "contract_pass": not failed,
        "failed_checks": failed,
        "checks": checks,
        "forbidden_feature_name_hits": forbidden_hits,
        "no_forbidden_sbatch_flags": True,
    }


def _analyzer_command(payload: dict[str, Any]) -> str:
    diagnostics = payload["projection_diagnostics"]
    return " ".join(
        [
            '"${TACE_PYTHON}"',
            "benchmarks/oc20neb_tace_mace/analyze_rtece_lowfreq_features.py",
            "--configs", '"${CONFIGS}"',
            "--stage165-json", '"${STAGE165_JSON}"',
            "--stage165-variant", str(payload["stage165_variant"]),
            "--group-key", str(payload["group_key"]),
            "--limit-configs", '"${LIMIT_CONFIGS}"',
            "--output-json", '"${RESULTS_JSON}"',
            "--output-md", '"${RESULTS_MD}"',
            "--energy-ridge-grid", _format_grid(diagnostics["ridge_grid"]),
        ]
    )


def write_stage174_wrapper(path: str | Path, payload: dict[str, Any]) -> Path:
    wrapper = Path(path)
    artifacts = payload["artifacts"]
    env_dir = "/home/gengjianrui/bin/.venvs/tace-mace-cu126"
    body = [
        "#!/bin/bash",
        "#SBATCH --job-name=rtece-res174",
        "#SBATCH --partition=16V100",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --gpus-per-node=1",
        "#SBATCH --qos=flood-1o2gpu",
        "#SBATCH --time=01:55:00",
        "#SBATCH --output=/home/gengjianrui/bin/logs/rtece-res174-%j.out",
        "#SBATCH --error=/home/gengjianrui/bin/logs/rtece-res174-%j.err",
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
        "# Stage174: deployable low-frequency chemistry/site/front feature probe",
        _analyzer_command(payload),
        "",
    ]
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("\n".join(body), encoding="utf-8")
    return wrapper


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage174 Low-Frequency Feature Probe Plan",
        "",
        f"- stage: `{payload['stage']}`",
        f"- target: `{payload['target_semantics']}`",
        f"- split: `{payload['energy_split_mode']}` by `{payload['group_key']}`",
        f"- configs: `{payload['configs']}`",
        f"- Stage165 source: `{payload['stage165_json']}`",
        f"- limit configs: `{payload['limit_configs']}`",
        f"- ridge grid: `{_format_grid(payload['projection_diagnostics']['ridge_grid'])}`",
        "",
        "## Question",
        "",
        payload["comparison_question"],
        "",
        "## Feature Families",
        "",
        "| family | requires tags | TECE semantics |",
        "|---|---|---|",
    ]
    for family in payload["feature_families"]:
        lines.append(f"| {family['name']} | {family['requires_tags']} | {family['tece_semantics']} |")
    lines.extend(["", "## Review Basis", ""])
    lines.extend(f"- {item}" for item in payload["review_basis"])
    lines.append("")
    return "\n".join(lines)


def materialize_stage174(payload: dict[str, Any]) -> dict[str, Any]:
    audit = audit_stage174_manifest(payload)
    if not audit["contract_pass"]:
        raise ValueError(f"Stage174 manifest audit failed: {audit['failed_checks']}")
    artifacts = payload["artifacts"]
    for key in ("manifest", "manifest_audit", "stage_plan", "wrapper"):
        Path(artifacts[key]).parent.mkdir(parents=True, exist_ok=True)
    Path(artifacts["manifest"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(artifacts["manifest_audit"]).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(artifacts["stage_plan"]).write_text(format_markdown(payload), encoding="utf-8")
    write_stage174_wrapper(artifacts["wrapper"], payload)
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
    payload = make_stage174_manifest(
        output_root=args.output_root,
        stage165_json=args.stage165_json,
        configs=args.configs,
        limit_configs=args.limit_configs,
    )
    materialized = materialize_stage174(payload)
    print(json.dumps({"stage": materialized["stage"], "artifacts": materialized["artifacts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
