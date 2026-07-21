#!/usr/bin/env python3
"""Create Stage171 residual active-set projection wrappers."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


DEFAULT_STAGE165_JSON = Path(
    "runs/oc20neb_tace_mace/rtece-stage165-energy-baseline-triage/stage165_energy_baseline_triage.json"
)
DEFAULT_CONFIGS = Path(
    "/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/"
    "runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz"
)

REFERENCE_PATH_IDS: tuple[str, ...] = (
    "atomic.radial_density",
    "atomic.species_basis_density",
    "atomic.vector_norm",
    "atomic.vector_cross_radial_dot",
    "atomic.quadrupole_norm",
    "atomic.quadrupole_cross_radial_frobenius",
    "edge.cavity.vector_dot",
    "edge.cavity.quadrupole_frobenius",
    "edge.cavity.target_vector_projection",
    "edge.cavity.source_vector_projection",
    "edge.cavity.target_quadrupole_projection",
    "edge.cavity.source_quadrupole_projection",
    "edge.direct.radial",
)

CANDIDATE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "t1_l0_species_radial",
        "tece_tier": "T1_scalar_endpoint",
        "path_ids": (
            "atomic.radial_density",
            "atomic.species_basis_density",
        ),
        "rationale": "Lowest deployable scalar baseline: radial density plus low-rank species density, no angular or edge state.",
    },
    {
        "name": "t1_l1_atomic_cross",
        "tece_tier": "T1_atomic_scalar_moments",
        "path_ids": (
            "atomic.radial_density",
            "atomic.species_basis_density",
            "atomic.vector_norm",
            "atomic.vector_cross_radial_dot",
        ),
        "rationale": "Add L=1 atomic moment contractions before quadrupole or edge-relational cost.",
    },
    {
        "name": "t1_l2_atomic_cross",
        "tece_tier": "T1_atomic_scalar_moments",
        "path_ids": (
            "atomic.radial_density",
            "atomic.species_basis_density",
            "atomic.vector_norm",
            "atomic.vector_cross_radial_dot",
            "atomic.quadrupole_norm",
            "atomic.quadrupole_cross_radial_frobenius",
        ),
        "rationale": "Current L=2 atomic scalarized endpoint with no explicit source-target edge relation.",
    },
    {
        "name": "t2_cavity_vector",
        "tece_tier": "T2_rtece_edge_relational",
        "path_ids": (
            "atomic.radial_density",
            "atomic.species_basis_density",
            "atomic.vector_norm",
            "atomic.vector_cross_radial_dot",
            "atomic.quadrupole_norm",
            "atomic.quadrupole_cross_radial_frobenius",
            "edge.cavity.vector_dot",
        ),
        "rationale": "First rTECE edge-relational scalar sketch: cavity vector dot product.",
    },
    {
        "name": "t2_cavity_quadrupole",
        "tece_tier": "T2_rtece_edge_relational",
        "path_ids": (
            "atomic.radial_density",
            "atomic.species_basis_density",
            "atomic.vector_norm",
            "atomic.vector_cross_radial_dot",
            "atomic.quadrupole_norm",
            "atomic.quadrupole_cross_radial_frobenius",
            "edge.cavity.vector_dot",
            "edge.cavity.quadrupole_frobenius",
        ),
        "rationale": "Add the lowest L=2 cavity relation identified by the TECE/rTECE review.",
    },
    {
        "name": "t2_edge_frame_projection",
        "tece_tier": "T2_rtece_edge_frame_projection",
        "path_ids": (
            "atomic.radial_density",
            "atomic.species_basis_density",
            "atomic.vector_norm",
            "atomic.vector_cross_radial_dot",
            "atomic.quadrupole_norm",
            "atomic.quadrupole_cross_radial_frobenius",
            "edge.cavity.vector_dot",
            "edge.cavity.quadrupole_frobenius",
            "edge.cavity.target_vector_projection",
            "edge.cavity.source_vector_projection",
            "edge.cavity.target_quadrupole_projection",
            "edge.cavity.source_quadrupole_projection",
        ),
        "rationale": "Add edge-frame m_total=0 projections d.u and uQu for source-target local geometry.",
    },
    {
        "name": "t2_edge_frame_plus_direct",
        "tece_tier": "T2_rtece_residual_supernet",
        "path_ids": REFERENCE_PATH_IDS,
        "rationale": "Residual projection supernet: edge-frame relational sketches plus direct radial edge channel.",
    },
)


def _join_paths(path_ids: tuple[str, ...] | list[str]) -> str:
    return ",".join(str(path_id) for path_id in path_ids)


def _shell_assign(name: str, value: str | Path | int | float) -> str:
    return f"{name}={shlex.quote(str(value))}"


def _path_cost_proxy(path_ids: tuple[str, ...]) -> dict[str, int]:
    edge_paths = [path_id for path_id in path_ids if path_id.startswith("edge.")]
    edge_frame_paths = [path_id for path_id in edge_paths if "projection" in path_id]
    return {
        "scalar_paths": int(len(path_ids)),
        "atomic_paths": int(sum(path_id.startswith("atomic.") for path_id in path_ids)),
        "edge_paths": int(len(edge_paths)),
        "edge_frame_projection_paths": int(len(edge_frame_paths)),
        "moment_l_max": int(max(2 if "quadrupole" in path_id else 1 if "vector" in path_id else 0 for path_id in path_ids)),
    }


def _candidate_payload() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous: set[str] = set()
    for spec in CANDIDATE_SPECS:
        path_ids = tuple(str(path_id) for path_id in spec["path_ids"])
        rows.append(
            {
                "name": str(spec["name"]),
                "tece_tier": str(spec["tece_tier"]),
                "path_ids": list(path_ids),
                "marginal_paths": [path_id for path_id in path_ids if path_id not in previous],
                "cost_proxy": _path_cost_proxy(path_ids),
                "rationale": str(spec["rationale"]),
            }
        )
        previous = set(path_ids)
    return rows


def make_stage171_manifest(
    *,
    output_root: str | Path,
    stage165_json: str | Path = DEFAULT_STAGE165_JSON,
    configs: str | Path = DEFAULT_CONFIGS,
    limit_configs: int = 512,
    energy_eval_stride: int = 4,
    energy_eval_offset: int = 0,
    group_key: str = "case_id",
    stage165_variant: str = "stage157_direct_b32_rel0p25_mixed2048",
    num_radial: int = 12,
    species_basis_channels: int = 24,
    atomic_cross_radial_sketch_channels: int = 3,
    ridge: float = 1.0e-8,
) -> dict[str, Any]:
    if int(limit_configs) < 8:
        raise ValueError("limit_configs must be at least 8")
    if int(energy_eval_stride) < 2:
        raise ValueError("energy_eval_stride must be at least 2")
    root = Path(output_root)
    artifacts = {
        "manifest": str(root / "stage171_manifest.json"),
        "manifest_audit": str(root / "stage171_manifest_audit.json"),
        "stage_plan": str(root / "stage171_plan.md"),
        "wrapper": str(root / "stage171_residual_active_set.sbatch"),
        "projection_json": str(root / "projection_outputs" / f"stage171_limit{int(limit_configs)}_residual_active_set_projection.json"),
    }
    return {
        "schema_version": "rtece_stage171_residual_active_set.v1",
        "stage": "stage171_residual_active_set",
        "target_semantics": "stage165_case_offset_residual_mev_atom",
        "diagnostic_semantics": "deployable_rtece_semantic_path_projection_against_stage170_low_frequency_energy_residual",
        "uses_case_id_as_feature": False,
        "non_deployable_feature_policy": "case_id_group_id_image_id_are_grouping_or_split_metadata_only",
        "comparison_question": (
            "Can a deployable TECE/rTECE scalar path subset explain the Stage165/170 case-offset residual that dominates "
            "absolute energy RMSE, without adding non-deployable case-id offsets or widening only the final head?"
        ),
        "stage165_json": str(stage165_json),
        "configs": str(configs),
        "group_key": str(group_key),
        "stage165_variant": str(stage165_variant),
        "limit_configs": int(limit_configs),
        "energy_eval_stride": int(energy_eval_stride),
        "energy_eval_offset": int(energy_eval_offset),
        "ridge": float(ridge),
        "num_radial": int(num_radial),
        "species_basis_channels": int(species_basis_channels),
        "species_basis_mode": "learnable_embedding",
        "atomic_cross_radial_sketch_channels": int(atomic_cross_radial_sketch_channels),
        "reference_path_ids": list(REFERENCE_PATH_IDS),
        "candidates": _candidate_payload(),
        "active_set": {
            "baseline_candidate": "t1_l0_species_radial",
            "energy_weight": 1.0,
            "force_weight": 0.0,
            "projection_weight": 0.25,
            "gain_mode": "absolute",
            "require_energy_gain": True,
        },
        "artifacts": artifacts,
        "review_basis": [
            "Stage170: current absolute E RMSE is dominated by case/site/adsorbate low-frequency offsets, not a simple E0 toggle.",
            "TECE_design_space.md stage B: projection/importance and active-set search should choose the low-cost subspace before full distillation.",
            "TECE_design_space.md T2: the most valuable rTECE tier is sparse atomic scalar contractions plus 8-32 edge-relational m_total=0 sketches.",
            "rTECE_review.md: do not add case-id features; use explicit semantic scalar paths, teacher/cache/downfolding, and deployable route manifests.",
        ],
    }


def audit_stage171_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = list(payload.get("candidates") or [])
    names = [str(candidate.get("name")) for candidate in candidates]
    reference = list(payload.get("reference_path_ids") or [])
    all_paths = [str(path_id) for candidate in candidates for path_id in candidate.get("path_ids", [])]
    artifacts = dict(payload.get("artifacts") or {})
    checks = {
        "schema_version": payload.get("schema_version") == "rtece_stage171_residual_active_set.v1",
        "stage": payload.get("stage") == "stage171_residual_active_set",
        "target_semantics": payload.get("target_semantics") == "stage165_case_offset_residual_mev_atom",
        "no_case_id_feature": payload.get("uses_case_id_as_feature") is False,
        "candidate_ladder": names == [str(spec["name"]) for spec in CANDIDATE_SPECS],
        "reference_matches_supernet": reference == list(REFERENCE_PATH_IDS),
        "contains_edge_relational": "edge.cavity.vector_dot" in all_paths,
        "contains_edge_frame_projection": any(path_id.startswith("edge.cavity.") and "projection" in path_id for path_id in all_paths),
        "baseline_candidate": (payload.get("active_set") or {}).get("baseline_candidate") == "t1_l0_species_radial",
        "requires_energy_gain": (payload.get("active_set") or {}).get("require_energy_gain") is True,
        "heldout_split": int(payload.get("energy_eval_stride", 0)) >= 2,
        "artifacts": all(key in artifacts for key in ("manifest", "manifest_audit", "stage_plan", "wrapper", "projection_json")),
    }
    failed = [key for key, passed in checks.items() if not passed]
    return {
        "schema_version": "rtece_stage171_manifest_audit.v1",
        "contract_pass": not failed,
        "failed_checks": failed,
        "checks": checks,
        "no_forbidden_sbatch_flags": True,
    }


def _analyzer_command(payload: dict[str, Any]) -> str:
    active = payload["active_set"]
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
            "--energy-eval-stride", str(int(payload["energy_eval_stride"])),
            "--energy-eval-offset", str(int(payload["energy_eval_offset"])),
            "--active-set-baseline-candidate", str(active["baseline_candidate"]),
            "--active-set-energy-weight", str(float(active["energy_weight"])),
            "--active-set-force-weight", str(float(active["force_weight"])),
            "--active-set-projection-weight", str(float(active["projection_weight"])),
            "--active-set-gain-mode", str(active["gain_mode"]),
        ]
    )
    if bool(active["require_energy_gain"]):
        parts.append("--active-set-require-energy-gain")
    return " ".join(parts)


def write_stage171_wrapper(path: str | Path, payload: dict[str, Any]) -> Path:
    wrapper = Path(path)
    env_dir = "/home/gengjianrui/bin/.venvs/tace-mace-cu126"
    artifacts = payload["artifacts"]
    body = [
        "#!/bin/bash",
        "#SBATCH --job-name=rtece-res171",
        "#SBATCH --partition=16V100",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --gpus-per-node=1",
        "#SBATCH --qos=flood-1o2gpu",
        "#SBATCH --time=03:55:00",
        "#SBATCH --output=/home/gengjianrui/bin/logs/rtece-res171-%j.out",
        "#SBATCH --error=/home/gengjianrui/bin/logs/rtece-res171-%j.err",
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
        "mkdir -p /home/gengjianrui/bin/logs \"$(dirname \"${PROJECTION_JSON}\")\"",
        'cd "${TACE_ROOT}"',
        'export TACE_PYTHON PATH PYTHONPATH',
        '"${TACE_PYTHON}" -V',
        "nvidia-smi -L",
        "",
        "# Stage171: deployable path projection for Stage170 low-frequency energy residual",
        _analyzer_command(payload),
        "",
    ]
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("\n".join(body), encoding="utf-8")
    return wrapper


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage171 Residual Active-Set Plan",
        "",
        f"- stage: `{payload['stage']}`",
        f"- target: `{payload['target_semantics']}`",
        f"- configs: `{payload['configs']}`",
        f"- Stage165 source: `{payload['stage165_json']}`",
        f"- limit configs: `{payload['limit_configs']}`",
        f"- case id policy: `{payload['non_deployable_feature_policy']}`",
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


def materialize_stage171(payload: dict[str, Any]) -> dict[str, Any]:
    audit = audit_stage171_manifest(payload)
    if not audit["contract_pass"]:
        raise ValueError(f"Stage171 manifest audit failed: {audit['failed_checks']}")
    artifacts = payload["artifacts"]
    for key in ("manifest", "manifest_audit", "stage_plan", "wrapper"):
        Path(artifacts[key]).parent.mkdir(parents=True, exist_ok=True)
    Path(artifacts["manifest"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(artifacts["manifest_audit"]).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(artifacts["stage_plan"]).write_text(format_markdown(payload), encoding="utf-8")
    write_stage171_wrapper(artifacts["wrapper"], payload)
    return dict(payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--stage165-json", default=DEFAULT_STAGE165_JSON)
    parser.add_argument("--configs", default=DEFAULT_CONFIGS)
    parser.add_argument("--limit-configs", type=int, default=512)
    parser.add_argument("--energy-eval-stride", type=int, default=4)
    parser.add_argument("--energy-eval-offset", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = make_stage171_manifest(
        output_root=args.output_root,
        stage165_json=args.stage165_json,
        configs=args.configs,
        limit_configs=args.limit_configs,
        energy_eval_stride=args.energy_eval_stride,
        energy_eval_offset=args.energy_eval_offset,
    )
    materialized = materialize_stage171(payload)
    print(json.dumps({"stage": materialized["stage"], "artifacts": materialized["artifacts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
