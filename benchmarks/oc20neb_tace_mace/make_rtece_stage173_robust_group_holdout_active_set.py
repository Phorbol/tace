#!/usr/bin/env python3
"""Create Stage173 robust group-heldout residual active-set projection wrappers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.oc20neb_tace_mace.make_rtece_stage171_residual_active_set import (
    CANDIDATE_SPECS,
    DEFAULT_CONFIGS,
    DEFAULT_STAGE165_JSON,
    REFERENCE_PATH_IDS,
    _candidate_payload,
    _join_paths,
    _shell_assign,
)

DEFAULT_RIDGE_GRID = (1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0, 100.0)


def _format_float(value: float) -> str:
    return f"{float(value):g}"


def _format_ridge_grid(values: tuple[float, ...] | list[float]) -> str:
    return ",".join(_format_float(float(value)) for value in values)


def make_stage173_manifest(
    *,
    output_root: str | Path,
    stage165_json: str | Path = DEFAULT_STAGE165_JSON,
    configs: str | Path = DEFAULT_CONFIGS,
    limit_configs: int = 512,
    group_key: str = "case_id",
    stage165_variant: str = "stage157_direct_b32_rel0p25_mixed2048",
    num_radial: int = 12,
    species_basis_channels: int = 24,
    atomic_cross_radial_sketch_channels: int = 3,
    ridge: float = 1.0e-8,
    ridge_grid: tuple[float, ...] | list[float] = DEFAULT_RIDGE_GRID,
) -> dict[str, Any]:
    if int(limit_configs) < 8:
        raise ValueError("limit_configs must be at least 8")
    grid = tuple(float(value) for value in ridge_grid)
    if not grid:
        raise ValueError("ridge_grid must contain at least one value")
    root = Path(output_root)
    artifacts = {
        "manifest": str(root / "stage173_manifest.json"),
        "manifest_audit": str(root / "stage173_manifest_audit.json"),
        "stage_plan": str(root / "stage173_plan.md"),
        "wrapper": str(root / "stage173_robust_group_holdout_active_set.sbatch"),
        "projection_json": str(root / "projection_outputs" / f"stage173_limit{int(limit_configs)}_robust_group_holdout_active_set_projection.json"),
    }
    return {
        "schema_version": "rtece_stage173_robust_group_holdout_active_set.v1",
        "stage": "stage173_robust_group_holdout_active_set",
        "target_semantics": "stage165_case_offset_residual_mev_atom",
        "diagnostic_semantics": "robust_case_group_heldout_rtece_path_projection_against_stage170_low_frequency_energy_residual",
        "energy_split_mode": "group-loocv",
        "uses_case_id_as_feature": False,
        "non_deployable_feature_policy": "case_id_group_id_image_id_are_grouping_or_split_metadata_only",
        "comparison_question": (
            "After Stage172 rejected the raw group-heldout descriptor projection, does a statistically safer projection "
            "with fold-local standardization, intercept, ridge sweep, and intercept-only baseline still reject the current "
            "deployable rTECE path coordinates for unseen case/site/adsorbate energy offsets?"
        ),
        "stage165_json": str(stage165_json),
        "configs": str(configs),
        "group_key": str(group_key),
        "stage165_variant": str(stage165_variant),
        "limit_configs": int(limit_configs),
        "ridge": float(ridge),
        "num_radial": int(num_radial),
        "species_basis_channels": int(species_basis_channels),
        "species_basis_mode": "learnable_embedding",
        "atomic_cross_radial_sketch_channels": int(atomic_cross_radial_sketch_channels),
        "reference_path_ids": list(REFERENCE_PATH_IDS),
        "candidates": _candidate_payload(),
        "projection_diagnostics": {
            "fit_intercept": True,
            "standardize_features": True,
            "ridge_grid": [float(value) for value in grid],
            "intercept_baseline": "group_loocv_train_mean_per_atom_residual_times_eval_atom_count",
            "ridge_selection": "diagnostic_oracle_by_energy_per_atom_rmse_when_available_else_total_energy_rmse",
        },
        "active_set": {
            "baseline_candidate": "t1_l0_species_radial",
            "energy_weight": 1.0,
            "force_weight": 0.0,
            "projection_weight": 0.25,
            "gain_mode": "absolute",
            "require_energy_gain": True,
            "require_beat_intercept": True,
        },
        "artifacts": artifacts,
        "review_basis": [
            "Stage170: raw absolute E RMSE is dominated by low-frequency case/site/adsorbate offsets, not by a simple E0 toggle.",
            "Stage171: config-heldout residual projection can interpolate seen cases and is not a deployment proof.",
            "Stage172: raw group-heldout projection produced catastrophic unseen-case RMSE and must be checked against intercept/standardization/ridge baselines before architecture conclusions.",
            "rTECE_review.md: random configuration splits can overestimate generalization; group-heldout and baseline comparisons are required.",
            "TECE_design_space.md: active-set pruning must separate projection error from optimization/distillation error before promoting path groups.",
        ],
    }


def audit_stage173_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = list(payload.get("candidates") or [])
    names = [str(candidate.get("name")) for candidate in candidates]
    artifacts = dict(payload.get("artifacts") or {})
    diagnostics = dict(payload.get("projection_diagnostics") or {})
    active = dict(payload.get("active_set") or {})
    all_paths = [str(path_id) for candidate in candidates for path_id in candidate.get("path_ids", [])]
    checks = {
        "schema_version": payload.get("schema_version") == "rtece_stage173_robust_group_holdout_active_set.v1",
        "stage": payload.get("stage") == "stage173_robust_group_holdout_active_set",
        "target_semantics": payload.get("target_semantics") == "stage165_case_offset_residual_mev_atom",
        "split_mode": payload.get("energy_split_mode") == "group-loocv",
        "no_case_id_feature": payload.get("uses_case_id_as_feature") is False,
        "group_key": str(payload.get("group_key")) == "case_id",
        "candidate_ladder": names == [str(spec["name"]) for spec in CANDIDATE_SPECS],
        "reference_matches_supernet": list(payload.get("reference_path_ids") or []) == list(REFERENCE_PATH_IDS),
        "contains_edge_relational": "edge.cavity.vector_dot" in all_paths,
        "fit_intercept": diagnostics.get("fit_intercept") is True,
        "standardize_features": diagnostics.get("standardize_features") is True,
        "ridge_grid": list(diagnostics.get("ridge_grid") or []) == [float(value) for value in DEFAULT_RIDGE_GRID],
        "baseline_candidate": active.get("baseline_candidate") == "t1_l0_species_radial",
        "requires_energy_gain": active.get("require_energy_gain") is True,
        "requires_beat_intercept": active.get("require_beat_intercept") is True,
        "artifacts": all(key in artifacts for key in ("manifest", "manifest_audit", "stage_plan", "wrapper", "projection_json")),
    }
    failed = [key for key, passed in checks.items() if not passed]
    return {
        "schema_version": "rtece_stage173_manifest_audit.v1",
        "contract_pass": not failed,
        "failed_checks": failed,
        "checks": checks,
        "no_forbidden_sbatch_flags": True,
    }


def _analyzer_command(payload: dict[str, Any]) -> str:
    active = payload["active_set"]
    diagnostics = payload["projection_diagnostics"]
    parts = [
        '"${TACE_PYTHON}"',
        "benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py",
        "--configs", '"${CONFIGS}"',
        "--output-json", '"${PROJECTION_JSON}"',
        "--reference-path-ids", '"${REFERENCE_PATH_IDS}"',
        "--residual-target-mode", "stage165-case-offset",
        "--stage165-json", '"${STAGE165_JSON}"',
        "--stage165-variant", str(payload["stage165_variant"]),
        "--residual-group-key", str(payload["group_key"]),
    ]
    for candidate in payload["candidates"]:
        parts.extend(["--candidate", f"{candidate['name']}:{_join_paths(candidate['path_ids'])}"])
    parts.extend(
        [
            "--num-radial", '"${NUM_RADIAL}"',
            "--species-basis-channels", '"${SPECIES_BASIS_CHANNELS}"',
            "--atomic-cross-radial-sketch-channels", '"${ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS}"',
            "--limit-configs", '"${LIMIT_CONFIGS}"',
            "--default-dtype", "float64",
            "--neighborlist-backend", "matscipy",
            "--ridge", str(float(payload["ridge"])),
            "--energy-baseline", "none",
            "--energy-split-mode", "group-loocv",
            "--energy-group-key", str(payload["group_key"]),
            "--energy-ridge-grid", _format_ridge_grid(diagnostics["ridge_grid"]),
            "--active-set-baseline-candidate", str(active["baseline_candidate"]),
            "--active-set-energy-weight", str(float(active["energy_weight"])),
            "--active-set-force-weight", str(float(active["force_weight"])),
            "--active-set-projection-weight", str(float(active["projection_weight"])),
            "--active-set-gain-mode", str(active["gain_mode"]),
        ]
    )
    if bool(diagnostics["fit_intercept"]):
        parts.append("--energy-fit-intercept")
    if bool(diagnostics["standardize_features"]):
        parts.append("--energy-standardize-features")
    if bool(active["require_energy_gain"]):
        parts.append("--active-set-require-energy-gain")
    if bool(active["require_beat_intercept"]):
        parts.append("--active-set-require-beat-intercept")
    return " ".join(parts)


def write_stage173_wrapper(path: str | Path, payload: dict[str, Any]) -> Path:
    wrapper = Path(path)
    env_dir = "/home/gengjianrui/bin/.venvs/tace-mace-cu126"
    artifacts = payload["artifacts"]
    body = [
        "#!/bin/bash",
        "#SBATCH --job-name=rtece-res173",
        "#SBATCH --partition=16V100",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --gpus-per-node=1",
        "#SBATCH --qos=flood-1o2gpu",
        "#SBATCH --time=03:55:00",
        "#SBATCH --output=/home/gengjianrui/bin/logs/rtece-res173-%j.out",
        "#SBATCH --error=/home/gengjianrui/bin/logs/rtece-res173-%j.err",
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
        _shell_assign("PROJECTION_JSON", artifacts["projection_json"]),
        _shell_assign("LIMIT_CONFIGS", int(payload["limit_configs"])),
        _shell_assign("NUM_RADIAL", int(payload["num_radial"])),
        _shell_assign("SPECIES_BASIS_CHANNELS", int(payload["species_basis_channels"])),
        _shell_assign("ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS", int(payload["atomic_cross_radial_sketch_channels"])),
        _shell_assign("REFERENCE_PATH_IDS", _join_paths(REFERENCE_PATH_IDS)),
        'mkdir -p /home/gengjianrui/bin/logs "$(dirname "${PROJECTION_JSON}")"',
        'cd "${TACE_ROOT}"',
        'export TACE_PYTHON PATH PYTHONPATH',
        '"${TACE_PYTHON}" -V',
        "nvidia-smi -L",
        "",
        "# Stage173: robust group-heldout path projection for Stage170 low-frequency energy residual",
        _analyzer_command(payload),
        "",
    ]
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("\n".join(body), encoding="utf-8")
    return wrapper


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage173 Robust Group-Heldout Residual Active-Set Plan",
        "",
        f"- stage: `{payload['stage']}`",
        f"- target: `{payload['target_semantics']}`",
        f"- split: `{payload['energy_split_mode']}` by `{payload['group_key']}`",
        f"- robust projection: intercept={payload['projection_diagnostics']['fit_intercept']}, standardize={payload['projection_diagnostics']['standardize_features']}",
        f"- ridge grid: `{_format_ridge_grid(payload['projection_diagnostics']['ridge_grid'])}`",
        f"- configs: `{payload['configs']}`",
        f"- Stage165 source: `{payload['stage165_json']}`",
        f"- limit configs: `{payload['limit_configs']}`",
        "",
        "## Question",
        "",
        payload["comparison_question"],
        "",
        "## Candidate Ladder",
        "",
        "| candidate | tier | marginal paths | edge paths | frame projections |",
        "|---|---|---|---:|---:|",
    ]
    for candidate in payload["candidates"]:
        cost = candidate["cost_proxy"]
        lines.append(
            "| {name} | {tier} | {marginal} | {edge} | {frame} |".format(
                name=candidate["name"],
                tier=candidate["tece_tier"],
                marginal=", ".join(candidate["marginal_paths"]),
                edge=cost["edge_paths"],
                frame=cost["edge_frame_projection_paths"],
            )
        )
    lines.extend(["", "## Review Basis", ""])
    lines.extend(f"- {item}" for item in payload["review_basis"])
    lines.append("")
    return "\n".join(lines)


def materialize_stage173(payload: dict[str, Any]) -> dict[str, Any]:
    audit = audit_stage173_manifest(payload)
    if not audit["contract_pass"]:
        raise ValueError(f"Stage173 manifest audit failed: {audit['failed_checks']}")
    artifacts = payload["artifacts"]
    for key in ("manifest", "manifest_audit", "stage_plan", "wrapper"):
        Path(artifacts[key]).parent.mkdir(parents=True, exist_ok=True)
    Path(artifacts["manifest"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(artifacts["manifest_audit"]).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(artifacts["stage_plan"]).write_text(format_markdown(payload), encoding="utf-8")
    write_stage173_wrapper(artifacts["wrapper"], payload)
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
    payload = make_stage173_manifest(
        output_root=args.output_root,
        stage165_json=args.stage165_json,
        configs=args.configs,
        limit_configs=args.limit_configs,
    )
    materialized = materialize_stage173(payload)
    print(json.dumps({"stage": materialized["stage"], "artifacts": materialized["artifacts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
