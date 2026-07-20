#!/usr/bin/env python3
"""Create Stage143 semantic path active-set projection wrappers."""

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

DEFAULT_STAGE142_ROOT = Path("runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage")
DEFAULT_TRAIN_CONFIGS = DEFAULT_STAGE142_ROOT / "weighted_train_base2048_plus_teacher_relax640_forceonly_eanchor.extxyz"
DEFAULT_STAGE140_RESULTS = Path("runs/oc20neb_tace_mace/rtece-stage140-source-balanced-distill/stage140_results_summary.json")
DEFAULT_STAGE141_RESULTS = Path("runs/oc20neb_tace_mace/rtece-stage141-loss-measure-frontier/stage141_results_summary.json")
DEFAULT_STAGE142_RESULTS = DEFAULT_STAGE142_ROOT / "stage142_results_summary.json"

REFERENCE_PATH_IDS = (
    "atomic.radial_density",
    "atomic.species_basis_density",
    "atomic.vector_norm",
    "atomic.vector_cross_radial_dot",
    "atomic.quadrupole_norm",
    "atomic.quadrupole_cross_radial_frobenius",
    "edge.cavity.vector_dot",
    "edge.cavity.quadrupole_frobenius",
    "edge.direct.radial",
)

CANDIDATE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "t2_l0_species_radial",
        "tece_tier": "T2_scalar_endpoint",
        "moment_l_max": 0,
        "path_ids": (
            "atomic.radial_density",
            "atomic.species_basis_density",
        ),
        "rationale": "NEP/DPA1-0-like scalar endpoint with low-rank species density and no angular or edge-relational paths.",
    },
    {
        "name": "t2_l1_atomic_cross",
        "tece_tier": "T2_atomic_moment_scalarized",
        "moment_l_max": 1,
        "path_ids": (
            "atomic.radial_density",
            "atomic.species_basis_density",
            "atomic.vector_norm",
            "atomic.vector_cross_radial_dot",
        ),
        "rationale": "Keep L=1 vector moment scalar contractions and cross-radial rank before paying quadrupole or edge cost.",
    },
    {
        "name": "t2_l2_atomic_cross",
        "tece_tier": "T2_atomic_moment_scalarized",
        "moment_l_max": 2,
        "path_ids": (
            "atomic.radial_density",
            "atomic.species_basis_density",
            "atomic.vector_norm",
            "atomic.vector_cross_radial_dot",
            "atomic.quadrupole_norm",
            "atomic.quadrupole_cross_radial_frobenius",
        ),
        "rationale": "Current Stage140-142 atomic L=2 incumbent without edge-relational paths.",
    },
    {
        "name": "t3_l2_cavity_vector",
        "tece_tier": "T3_rtece_edge_relational",
        "moment_l_max": 2,
        "path_ids": (
            "atomic.radial_density",
            "atomic.species_basis_density",
            "atomic.vector_norm",
            "atomic.vector_cross_radial_dot",
            "atomic.quadrupole_norm",
            "atomic.quadrupole_cross_radial_frobenius",
            "edge.cavity.vector_dot",
        ),
        "rationale": "First true rTECE edge-relational increment: one cavity vector total-m=0 scalar sketch.",
    },
    {
        "name": "t3_l2_cavity_vector_quadrupole_direct",
        "tece_tier": "T3_rtece_edge_relational",
        "moment_l_max": 2,
        "path_ids": REFERENCE_PATH_IDS,
        "rationale": "Semantic supernet for the next active-set decision: atomic L=2 plus cavity vector, cavity quadrupole, and direct radial edge paths.",
    },
)


def _join_paths(path_ids: tuple[str, ...] | list[str]) -> str:
    return ",".join(str(path_id) for path_id in path_ids)


def _shell_assign(name: str, value: str | Path | int | float) -> str:
    return f"{name}={shlex.quote(str(value))}"


def _path_cost_proxy(path_ids: tuple[str, ...]) -> dict[str, int]:
    atomic_paths = [path_id for path_id in path_ids if path_id.startswith("atomic.")]
    edge_paths = [path_id for path_id in path_ids if path_id.startswith("edge.")]
    max_l = max((2 if "quadrupole" in path_id else 1 if "vector" in path_id else 0) for path_id in path_ids)
    edge_scalar_dims = 0
    for path_id in edge_paths:
        if path_id == "edge.direct.radial":
            edge_scalar_dims += 2
        else:
            edge_scalar_dims += 1
    return {
        "atomic_scalar_paths": len(atomic_paths),
        "edge_scalar_paths": edge_scalar_dims,
        "has_edge_relational_paths": int(bool(edge_paths)),
        "moment_l_max": max_l,
    }


def _candidate_payload() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous: set[str] = set()
    for spec in CANDIDATE_SPECS:
        path_ids = tuple(spec["path_ids"])
        current = set(path_ids)
        rows.append(
            {
                "name": str(spec["name"]),
                "tece_tier": str(spec["tece_tier"]),
                "moment_l_max": int(spec["moment_l_max"]),
                "path_ids": list(path_ids),
                "marginal_paths": [path_id for path_id in path_ids if path_id not in previous],
                "cost_proxy": _path_cost_proxy(path_ids),
                "rationale": str(spec["rationale"]),
            }
        )
        previous = current
    return rows


def make_stage143_manifest(
    *,
    output_root: str | Path,
    train_configs: str | Path = DEFAULT_TRAIN_CONFIGS,
    valid_configs: str | Path = DEFAULT_DFT_VALID,
    limit_configs: int = 512,
    energy_eval_stride: int = 4,
    force_eval_stride: int = 0,
    energy_eval_offset: int = 0,
    force_eval_offset: int = 0,
    energy_target_key: str = "energy",
    force_target_key: str | None = None,
    num_radial: int = 12,
    species_basis_channels: int = 24,
    atomic_cross_radial_sketch_channels: int = 3,
) -> dict[str, Any]:
    if int(limit_configs) < 8:
        raise ValueError("limit_configs must be at least 8 for held-out active-set diagnostics")
    if int(energy_eval_stride) < 2:
        raise ValueError("energy_eval_stride must be at least 2")
    if force_target_key is not None and int(force_eval_stride) < 2:
        raise ValueError("force_eval_stride must be at least 2 when force_target_key is enabled")
    root = Path(output_root)
    artifacts = {
        "wrapper": str(root / "rtece_stage143_semantic_active_set_no_export.sbatch"),
        "output_root": str(root / "projection_outputs"),
        "stage_plan": str(root / "stage143_plan.md"),
        "manifest": str(root / "stage143_manifest.json"),
        "manifest_audit": str(root / "stage143_manifest_audit.json"),
        "train_projection_json": str(root / "projection_outputs" / f"stage143_train_limit{int(limit_configs)}_semantic_active_set_projection.json"),
        "valid_projection_json": str(root / "projection_outputs" / f"stage143_valid_limit{int(limit_configs)}_semantic_active_set_projection.json"),
        "stage140_results": str(DEFAULT_STAGE140_RESULTS),
        "stage141_results": str(DEFAULT_STAGE141_RESULTS),
        "stage142_results": str(DEFAULT_STAGE142_RESULTS),
    }
    return {
        "schema_version": "rtece_stage143_semantic_active_set.v1",
        "stage": "stage143_semantic_active_set",
        "diagnostic_semantics": "semantic_path_active_set_projection_after_stage142",
        "deployment_measure_source": "stage142_force_only_teacher_relax_distribution",
        "comparison_question": (
            "Stage142 showed that more same-window teacher-relax force-only coverage is not enough. "
            "Use a Schur-complement-style projection diagnostic over explicit TECE semantic path groups to decide which "
            "atomic and edge-relational paths deserve the next training budget; this is not another same-window teacher-relax expansion. "
            "The default uses 512 configs so the energy projection fit is not below the 300-334 descriptor dimensions."
        ),
        "train_configs": str(train_configs),
        "valid_configs": str(valid_configs),
        "limit_configs": int(limit_configs),
        "energy_eval_stride": int(energy_eval_stride),
        "force_eval_stride": int(force_eval_stride),
        "energy_eval_offset": int(energy_eval_offset),
        "force_eval_offset": int(force_eval_offset),
        "energy_target_key": str(energy_target_key),
        "force_target_key": None if force_target_key is None else str(force_target_key),
        "force_projection_status": "deferred_force_descriptor_jacobian_cost" if force_target_key is None else "enabled_expensive_force_descriptor_jacobian",
        "num_radial": int(num_radial),
        "species_basis_channels": int(species_basis_channels),
        "species_basis_mode": "learnable_embedding",
        "atomic_cross_radial_sketch_channels": int(atomic_cross_radial_sketch_channels),
        "atomic_cross_radial_projection": "learnable",
        "reference_path_ids": list(REFERENCE_PATH_IDS),
        "candidates": _candidate_payload(),
        "artifacts": artifacts,
        "review_basis": [
            "TECE_design_space: choose a low-cost subspace by projection error, Sobolev E/F sensitivity, and hardware cost, not by widening the final head.",
            "TECE_design_space: rTECE is specifically the missing middle tier with sparse edge-relational scalar ECE sketches after immediate scalarization.",
            "rTECE_review: after Stage142 physical closure remains false, priority moves to semantic path registry, projection diagnostics, and active-set selection.",
            "Stage142: same-window teacher-relax coverage did not create a new Pareto point; the next falsification should evaluate operator basis value.",
        ],
    }


def audit_stage143_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = list(payload.get("candidates") or [])
    candidate_names = [str(candidate.get("name")) for candidate in candidates]
    reference = list(payload.get("reference_path_ids") or [])
    final_paths = candidates[-1].get("path_ids") if candidates else []
    checks = {
        "schema_version": payload.get("schema_version") == "rtece_stage143_semantic_active_set.v1",
        "stage": payload.get("stage") == "stage143_semantic_active_set",
        "diagnostic_semantics": payload.get("diagnostic_semantics") == "semantic_path_active_set_projection_after_stage142",
        "stage142_measure": payload.get("deployment_measure_source") == "stage142_force_only_teacher_relax_distribution",
        "reference_is_t3_cavity_supernet": reference == list(REFERENCE_PATH_IDS),
        "candidate_lattice": candidate_names == [str(spec["name"]) for spec in CANDIDATE_SPECS],
        "final_candidate_matches_reference": final_paths == list(REFERENCE_PATH_IDS),
        "contains_edge_relational_step": any("edge.cavity.vector_dot" in candidate.get("path_ids", []) for candidate in candidates),
        "starts_without_edge_paths": bool(candidates) and candidates[0].get("cost_proxy", {}).get("edge_scalar_paths") == 0,
        "current_active_ranks": int(payload.get("num_radial", 0)) == 12 and int(payload.get("species_basis_channels", 0)) == 24 and int(payload.get("atomic_cross_radial_sketch_channels", 0)) == 3,
        "energy_target": payload.get("energy_target_key") == "energy",
        "force_projection_status": payload.get("force_projection_status") in {"deferred_force_descriptor_jacobian_cost", "enabled_expensive_force_descriptor_jacobian"},
        "heldout_split": int(payload.get("energy_eval_stride", 0)) >= 2,
        "artifacts": all(key in (payload.get("artifacts") or {}) for key in ("wrapper", "stage_plan", "train_projection_json", "valid_projection_json")),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "schema_version": "rtece_stage143_manifest_audit.v1",
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
    parts = [
        '"${TACE_PYTHON}"',
        "benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py",
        "--configs", f'"${{{configs_var}}}"',
        "--output-json", f'"${{{output_var}}}"',
        "--reference-path-ids", '"${REFERENCE_PATH_IDS}"',
    ]
    for candidate in payload["candidates"]:
        parts.extend(["--candidate", f"{candidate['name']}:{_join_paths(tuple(candidate['path_ids']))}"])
    parts.extend(
        [
            "--num-radial", '"${NUM_RADIAL}"',
            "--species-basis-channels", '"${SPECIES_BASIS_CHANNELS}"',
            "--atomic-cross-radial-sketch-channels", '"${ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS}"',
            "--limit-configs", '"${LIMIT_CONFIGS}"',
            "--default-dtype", "float64",
            "--neighborlist-backend", "matscipy",
            "--energy-target-key", str(payload["energy_target_key"]),
            "--energy-eval-stride", str(int(payload["energy_eval_stride"])),
            "--energy-eval-offset", str(int(payload["energy_eval_offset"])),
        ]
    )
    if payload.get("force_target_key") is not None:
        parts.extend(
            [
                "--force-target-key", str(payload["force_target_key"]),
                "--force-eval-stride", str(int(payload["force_eval_stride"])),
                "--force-eval-offset", str(int(payload["force_eval_offset"])),
            ]
        )
    return " ".join(parts)


def write_stage143_wrapper(path: str | Path, payload: dict[str, Any]) -> Path:
    wrapper = Path(path)
    env_dir = "/home/gengjianrui/bin/.venvs/tace-mace-cu126"
    artifacts = payload["artifacts"]
    body = [
        "#!/bin/bash",
        "#SBATCH --job-name=rtece-proj143",
        "#SBATCH --partition=16V100",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --gpus-per-node=1",
        "#SBATCH --qos=flood-1o2gpu",
        "#SBATCH --time=03:55:00",
        "#SBATCH --output=/home/gengjianrui/bin/logs/rtece-proj143-%j.out",
        "#SBATCH --error=/home/gengjianrui/bin/logs/rtece-proj143-%j.err",
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
        "# Stage142 force-only teacher-relax deployment-measure projection; force-Jacobian projection is deferred by default",
        _analyzer_command(configs_var="TRAIN_CONFIGS", output_var="TRAIN_OUTPUT", payload=payload),
        "",
        "# Pure DFT validation projection for held-out benchmark consistency",
        _analyzer_command(configs_var="VALID_CONFIGS", output_var="VALID_OUTPUT", payload=payload),
        "",
    ]
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("\n".join(body), encoding="utf-8")
    return wrapper


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage143 Semantic Active-Set Projection Plan",
        "",
        f"- stage: `{payload['stage']}`",
        f"- semantics: `{payload['diagnostic_semantics']}`",
        f"- deployment measure: `{payload['deployment_measure_source']}`",
        f"- train configs: `{payload['train_configs']}`",
        f"- valid configs: `{payload['valid_configs']}`",
        f"- limit configs: `{payload['limit_configs']}`",
        f"- force projection: `{payload['force_projection_status']}`",
        "",
        "## Question",
        "",
        payload["comparison_question"],
        "",
        "## Reference Supernet",
        "",
        ", ".join(payload["reference_path_ids"]),
        "",
        "## Candidates",
        "",
        "| candidate | tier | Lmax | marginal paths | edge scalar cost |",
        "|---|---|---:|---|---:|",
    ]
    for candidate in payload["candidates"]:
        lines.append(
            "| {name} | {tier} | {lmax} | {marginal} | {edge_cost} |".format(
                name=candidate["name"],
                tier=candidate["tece_tier"],
                lmax=candidate["moment_l_max"],
                marginal=", ".join(candidate["marginal_paths"]),
                edge_cost=candidate["cost_proxy"]["edge_scalar_paths"],
            )
        )
    lines.extend(["", "## Review Basis", ""])
    lines.extend(f"- {item}" for item in payload["review_basis"])
    lines.append("")
    return "\n".join(lines)


def materialize_stage143(payload: dict[str, Any]) -> dict[str, Any]:
    audit = audit_stage143_manifest(payload)
    if not audit["contract_pass"]:
        raise ValueError(f"Stage143 manifest audit failed: {audit['failed_checks']}")
    artifacts = payload["artifacts"]
    for key in ("manifest", "manifest_audit", "stage_plan"):
        Path(artifacts[key]).parent.mkdir(parents=True, exist_ok=True)
    Path(artifacts["manifest"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(artifacts["manifest_audit"]).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(artifacts["stage_plan"]).write_text(format_markdown(payload), encoding="utf-8")
    write_stage143_wrapper(artifacts["wrapper"], payload)
    return dict(payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-configs", default=DEFAULT_TRAIN_CONFIGS)
    parser.add_argument("--valid-configs", default=DEFAULT_DFT_VALID)
    parser.add_argument("--limit-configs", type=int, default=512)
    parser.add_argument("--energy-eval-stride", type=int, default=4)
    parser.add_argument("--force-eval-stride", type=int, default=0)
    parser.add_argument("--force-target-key", default=None)
    parser.add_argument("--energy-eval-offset", type=int, default=0)
    parser.add_argument("--force-eval-offset", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = make_stage143_manifest(
        output_root=args.output_root,
        train_configs=args.train_configs,
        valid_configs=args.valid_configs,
        limit_configs=args.limit_configs,
        energy_eval_stride=args.energy_eval_stride,
        force_eval_stride=args.force_eval_stride,
        energy_eval_offset=args.energy_eval_offset,
        force_eval_offset=args.force_eval_offset,
        force_target_key=args.force_target_key,
    )
    materialized = materialize_stage143(payload)
    print(json.dumps({"stage": materialized["stage"], "artifacts": materialized["artifacts"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
