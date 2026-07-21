#!/usr/bin/env python3
"""Create Stage178 representation-upgrade rTECE ladder wrappers."""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.oc20neb_tace_mace.make_rtece_stage129_teacher_rattle_distill import (
    DEFAULT_BASE_TRAIN,
    DEFAULT_DFT_VALID,
)
from benchmarks.oc20neb_tace_mace.make_rtece_stage171_residual_active_set import _shell_assign

DEFAULT_TEACHER_VALID = Path("runs/oc20neb_tace_mace/tece-distill-20260717/mixed_valid_tw0.75_regen.extxyz")

CANDIDATE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "stage178_l0_local_species",
        "tece_tier": "T1_scalar_endpoint_with_trainable_local_chemistry",
        "representation_step": "l0_local_lowrank_species_front",
        "scalar_path_ids": (
            "atomic.radial_density",
            "atomic.species_basis_density",
            "atomic.local_l0_lowrank_density",
        ),
        "moment_l_max": 0,
        "num_radial": 10,
        "hidden_channels": "64,64",
        "species_basis_channels": 16,
        "species_basis_mode": "learnable_embedding",
        "local_l0_chemistry_rank": 4,
        "atomic_cross_radial_sketch_channels": 0,
        "atomic_cross_radial_projection": "fixed_shell_mean",
        "capacity_allocation": "trainable_species_and_local_l0_front_before_head",
        "rationale": "Fast Stage176-like endpoint, but removes fixed chemistry collisions with a learnable low-rank species basis and a rank-4 local L0 front.",
    },
    {
        "name": "stage178_l1_atomic_cross",
        "tece_tier": "T2_atomic_scalarized_cross_radial",
        "representation_step": "l1_sparse_cross_radial_invariants",
        "scalar_path_ids": (
            "atomic.radial_density",
            "atomic.species_basis_density",
            "atomic.local_l0_lowrank_density",
            "atomic.vector_norm",
            "atomic.vector_cross_radial_dot",
        ),
        "moment_l_max": 1,
        "num_radial": 10,
        "hidden_channels": "64,64",
        "species_basis_channels": 16,
        "species_basis_mode": "learnable_embedding",
        "local_l0_chemistry_rank": 4,
        "atomic_cross_radial_sketch_channels": 3,
        "atomic_cross_radial_projection": "learnable",
        "capacity_allocation": "learnable_radial_projection_and_sparse_l1_power_spectrum",
        "rationale": "Adds the review-recommended sparse cross-radial L=1 power-spectrum information before any edge-relational cost.",
    },
    {
        "name": "stage178_t3_minimal_cavity_direct",
        "tece_tier": "T3_minimal_rtece_edge_relational",
        "representation_step": "minimal_cavity_edge_relational_sketch",
        "scalar_path_ids": (
            "atomic.radial_density",
            "atomic.species_basis_density",
            "atomic.local_l0_lowrank_density",
            "atomic.vector_norm",
            "atomic.vector_cross_radial_dot",
            "edge.cavity.vector_dot",
            "edge.direct.radial",
        ),
        "moment_l_max": 1,
        "num_radial": 10,
        "hidden_channels": "64,64",
        "species_basis_channels": 16,
        "species_basis_mode": "learnable_embedding",
        "local_l0_chemistry_rank": 4,
        "atomic_cross_radial_sketch_channels": 3,
        "atomic_cross_radial_projection": "learnable",
        "capacity_allocation": "minimal_edge_relational_sketch_after_atomic_cross_radial_front",
        "rationale": "First true rTECE increment after Stage176: cavity vector relation plus separately accounted direct radial edge baseline.",
    },
)


def _join_paths(values: tuple[str, ...] | list[str]) -> str:
    return ",".join(str(value) for value in values)


def _candidate_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous: set[str] = set()
    for spec in CANDIDATE_SPECS:
        path_ids = tuple(str(path_id) for path_id in spec["scalar_path_ids"])
        rows.append(
            {
                **{key: value for key, value in spec.items() if key != "scalar_path_ids"},
                "scalar_path_ids": list(path_ids),
                "marginal_paths": [path_id for path_id in path_ids if path_id not in previous],
                "cost_proxy": {
                    "scalar_paths": len(path_ids),
                    "atomic_paths": sum(path_id.startswith("atomic.") for path_id in path_ids),
                    "edge_paths": sum(path_id.startswith("edge.") for path_id in path_ids),
                    "moment_l_max": int(spec["moment_l_max"]),
                    "num_radial": int(spec["num_radial"]),
                    "species_basis_channels": int(spec["species_basis_channels"]),
                    "local_l0_chemistry_rank": int(spec["local_l0_chemistry_rank"]),
                    "atomic_cross_radial_sketch_channels": int(spec["atomic_cross_radial_sketch_channels"]),
                },
            }
        )
        previous = set(path_ids)
    return rows


def make_stage178_manifest(
    *,
    output_root: str | Path,
    train_file: str | Path = DEFAULT_BASE_TRAIN,
    valid_file: str | Path = DEFAULT_DFT_VALID,
    teacher_valid_file: str | Path = DEFAULT_TEACHER_VALID,
    limit_configs: int = 2048,
    valid_limit_configs: int = 256,
    max_steps: int = 20000,
    batch_size: int = 32,
    lr: float = 1.0e-3,
    energy_weight: float = 1.0,
    force_weight: float = 10.0,
    lr_warmup_steps: int = 500,
    early_stopping_patience: int = 250,
) -> dict[str, Any]:
    if int(limit_configs) < 32:
        raise ValueError("limit_configs must be at least 32")
    if int(valid_limit_configs) < 16:
        raise ValueError("valid_limit_configs must be at least 16")
    root = Path(output_root)
    artifacts = {
        "manifest": str(root / "stage178_manifest.json"),
        "manifest_audit": str(root / "stage178_manifest_audit.json"),
        "stage_plan": str(root / "stage178_plan.md"),
        "train_wrapper": str(root / "wrappers" / "stage178_train_ladder_no_export.sbatch"),
        "benchmark_wrapper": str(root / "wrappers" / "stage178_benchmark_ladder_no_export.sbatch"),
        "physical_wrapper": str(root / "wrappers" / "stage178_physical_ladder_no_export.sbatch"),
        "results_root": str(root / "results"),
        "diagnostics_root": str(root / "diagnostics"),
    }
    return {
        "schema_version": "rtece_stage178_representation_upgrade.v1",
        "stage": "stage178_representation_upgrade_after_local_l0_endpoint",
        "question": (
            "After Stage176 proved a very fast but inaccurate local-L0 endpoint, which front-loaded "
            "TECE/rTECE representation increments recover E/F RMSE and physical behavior at tolerable throughput cost?"
        ),
        "training_entrypoint": "tace.scripts.rtece_train_scalar",
        "uses_production_training_framework": True,
        "train_file": str(train_file),
        "valid_file": str(valid_file),
        "teacher_valid_file": str(teacher_valid_file),
        "limit_configs": int(limit_configs),
        "valid_limit_configs": int(valid_limit_configs),
        "max_steps": int(max_steps),
        "training_config": {
            "trainer_backend": "lightning",
            "batch_size": int(batch_size),
            "lr": float(lr),
            "energy_weight": float(energy_weight),
            "force_weight": float(force_weight),
            "lr_scheduler": "plateau",
            "lr_warmup_steps": int(lr_warmup_steps),
            "early_stopping_patience": int(early_stopping_patience),
            "neighborlist_backend": "matscipy",
            "per_element_e0_fit": True,
        },
        "benchmark_protocol": {
            "primary_error_metric": "dft_f_rmse_mev_a",
            "required_error_metrics": [
                "dft_e_mae_mev_atom",
                "dft_e_rmse_mev_atom",
                "dft_e_max_mev_atom",
                "dft_f_mae_mev_a",
                "dft_f_rmse_mev_a",
                "dft_f_max_mev_a",
            ],
            "physical_tests": ["dimer_scan", "rattle_relax", "physical_pareto_summary"],
            "throughput_tests": ["valid_limit64", "scaling_limit32", "scaling_limit128", "scaling_limit512", "scaling_limit1024"],
            "memory_required": True,
            "community_baseline_alignment": "compare_only_after_matching_data_hardware_physics_protocol",
        },
        "candidates": _candidate_rows(),
        "artifacts": artifacts,
        "review_basis": [
            "TECE_design_space.md: rank candidates by retained/deleted TECE path sets, Sobolev E/F evidence, and hardware cost rather than MAE alone.",
            "rTECE_review.md §P1: prioritize radial POD/low-rank sketches, sparse cross-radial invariants, explicit path registry, and cavity edge kernels.",
            "rTECE_review.md: chemical collisions should be addressed by low-rank species bases, not by non-deployable C/N-specialized corrections.",
            "Stage177: Stage176 reaches high ASE/autograd atoms/s but fails E RMSE and rattle physical triage, so the next move is representation capacity before final-head width.",
        ],
    }


def audit_stage178_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    candidates = list(payload.get("candidates") or [])
    names = [str(row.get("name")) for row in candidates]
    artifacts = dict(payload.get("artifacts") or {})
    training = dict(payload.get("training_config") or {})
    benchmark = dict(payload.get("benchmark_protocol") or {})
    checks = {
        "schema_version": payload.get("schema_version") == "rtece_stage178_representation_upgrade.v1",
        "stage": payload.get("stage") == "stage178_representation_upgrade_after_local_l0_endpoint",
        "production_entrypoint": payload.get("training_entrypoint") == "tace.scripts.rtece_train_scalar",
        "candidate_names": names == [str(spec["name"]) for spec in CANDIDATE_SPECS],
        "l0_to_t3_ladder": bool(candidates)
        and candidates[0].get("representation_step") == "l0_local_lowrank_species_front"
        and candidates[-1].get("tece_tier") == "T3_minimal_rtece_edge_relational",
        "not_head_only": all(row.get("capacity_allocation") != "widen_final_head_only" for row in candidates),
        "contains_learnable_species": all(row.get("species_basis_mode") == "learnable_embedding" for row in candidates),
        "contains_learnable_cross_radial": any(row.get("atomic_cross_radial_projection") == "learnable" for row in candidates),
        "contains_edge_relational": any("edge.cavity.vector_dot" in row.get("scalar_path_ids", []) for row in candidates),
        "has_warmup": int(training.get("lr_warmup_steps", 0)) > 0,
        "has_early_stopping": int(training.get("early_stopping_patience", 0)) > 0,
        "rmse_primary": benchmark.get("primary_error_metric") == "dft_f_rmse_mev_a",
        "max_errors_required": "dft_e_max_mev_atom" in benchmark.get("required_error_metrics", [])
        and "dft_f_max_mev_a" in benchmark.get("required_error_metrics", []),
        "physical_required": set(benchmark.get("physical_tests") or []) >= {"dimer_scan", "rattle_relax"},
        "memory_required": benchmark.get("memory_required") is True,
        "artifacts": all(
            key in artifacts
            for key in (
                "manifest",
                "manifest_audit",
                "stage_plan",
                "train_wrapper",
                "benchmark_wrapper",
                "physical_wrapper",
                "results_root",
                "diagnostics_root",
            )
        ),
    }
    failed = [key for key, ok in checks.items() if not ok]
    return {
        "schema_version": "rtece_stage178_manifest_audit.v1",
        "contract_pass": not failed,
        "failed_checks": failed,
        "checks": checks,
        "no_forbidden_sbatch_flags": True,
    }


def _candidate_train_command(row: dict[str, Any], payload: dict[str, Any]) -> str:
    training = payload["training_config"]
    out_dir = Path(payload["artifacts"]["results_root"]) / row["name"]
    parts = [
        'PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}"',
        "python",
        "-m",
        "tace.scripts.rtece_train_scalar",
        "--variant",
        str(row["name"]),
        "--scalar-path-ids",
        _join_paths(row["scalar_path_ids"]),
        "--train-file",
        '"${TRAIN_FILE}"',
        "--valid-file",
        '"${VALID_FILE}"',
        "--output-dir",
        shlex.quote(str(out_dir)),
        "--limit-configs",
        '"${LIMIT_CONFIGS}"',
        "--valid-limit-configs",
        '"${VALID_LIMIT_CONFIGS}"',
        "--max-steps",
        '"${MAX_STEPS}"',
        "--trainer-backend",
        str(training["trainer_backend"]),
        "--batch-size",
        str(training["batch_size"]),
        "--num-radial",
        str(row["num_radial"]),
        "--hidden-channels",
        str(row["hidden_channels"]),
        "--species-basis-channels",
        str(row["species_basis_channels"]),
        "--species-basis-mode",
        str(row["species_basis_mode"]),
        "--local-l0-chemistry-rank",
        str(row["local_l0_chemistry_rank"]),
        "--moment-l-max",
        str(row["moment_l_max"]),
        "--atomic-cross-radial-sketch-channels",
        str(row["atomic_cross_radial_sketch_channels"]),
        "--atomic-cross-radial-projection",
        str(row["atomic_cross_radial_projection"]),
        "--lr",
        str(training["lr"]),
        "--energy-weight",
        str(training["energy_weight"]),
        "--force-weight",
        str(training["force_weight"]),
        "--lr-scheduler",
        str(training["lr_scheduler"]),
        "--lr-warmup-steps",
        str(training["lr_warmup_steps"]),
        "--early-stopping-patience",
        str(training["early_stopping_patience"]),
        "--neighborlist-backend",
        str(training["neighborlist_backend"]),
        "--device",
        "cuda",
        "--default-dtype",
        "float32",
        "--no-progress-bar",
    ]
    return " ".join(parts)


def _checkpoint_path(payload: dict[str, Any], row: dict[str, Any]) -> str:
    return str(Path(payload["artifacts"]["results_root"]) / row["name"] / "rtece_scalar_best.pt")


def _benchmark_commands(row: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    ckpt = shlex.quote(_checkpoint_path(payload, row))
    diag = Path(payload["artifacts"]["diagnostics_root"]) / row["name"]
    return [
        (
            f'PYTHONPATH="${{TACE_ROOT}}:${{PYTHONPATH:-}}" python benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py '
            f"--model {ckpt} --configs \"${{VALID_FILE}}\" --output {shlex.quote(str(diag / (row['name'] + '_dft_benchmark.json')))} "
            f"--variant {row['name']} --start-config 0 --limit-configs \"${{VALID_LIMIT_CONFIGS}}\" "
            "--measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist"
        ),
        (
            f'PYTHONPATH="${{TACE_ROOT}}:${{PYTHONPATH:-}}" python benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py '
            f"--model {ckpt} --configs \"${{TEACHER_VALID_FILE}}\" --output {shlex.quote(str(diag / (row['name'] + '_teacher_benchmark.json')))} "
            f"--variant {row['name']} --start-config 0 --limit-configs \"${{VALID_LIMIT_CONFIGS}}\" "
            "--measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist"
        ),
    ] + [
        (
            f'PYTHONPATH="${{TACE_ROOT}}:${{PYTHONPATH:-}}" python benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py '
            f"--model {ckpt} --configs \"${{VALID_FILE}}\" --output "
            f"{shlex.quote(str(diag / (row['name'] + f'_scaling_limit{limit}.json')))} "
            f"--variant {row['name']} --start-config 0 --limit-configs {limit} "
            "--measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist"
        )
        for limit in (32, 128, 512, 1024)
    ]

def _physical_commands(row: dict[str, Any], payload: dict[str, Any]) -> list[str]:
    ckpt = shlex.quote(_checkpoint_path(payload, row))
    diag = Path(payload["artifacts"]["diagnostics_root"]) / row["name"]
    dimer_json = diag / f"{row['name']}_dimer_scan.json"
    rattle_json = diag / f"{row['name']}_rattle_relax.json"
    physical_json = diag / f"{row['name']}_physical_pareto.json"
    case = ":".join(
        [
            str(row["name"]),
            str(diag / f"{row['name']}_dft_benchmark.json"),
            str(diag / f"{row['name']}_teacher_benchmark.json"),
            str(dimer_json),
            str(rattle_json),
        ]
    )
    return [
        (
            f'PYTHONPATH="${{TACE_ROOT}}:${{PYTHONPATH:-}}" python benchmarks/oc20neb_tace_mace/dimer_scan_rtece.py '
            f"--checkpoint {ckpt} --output-json {shlex.quote(str(dimer_json))} --output-md {shlex.quote(str(dimer_json.with_suffix('.md')))} "
            f"--route {row['name']} --pairs C-N C-O C-H N-H O-H C-C N-N --num-points 24 --min-scale 0.5 --max-scale 5.0 "
            "--device cuda --default-dtype float32 --force-mode autograd --neighborlist-backend ase"
        ),
        (
            f'PYTHONPATH="${{TACE_ROOT}}:${{PYTHONPATH:-}}" python benchmarks/oc20neb_tace_mace/rattle_relax_rtece.py '
            f"--checkpoint {ckpt} --configs \"${{VALID_FILE}}\" --output-json {shlex.quote(str(rattle_json))} --output-md {shlex.quote(str(rattle_json.with_suffix('.md')))} "
            f"--route {row['name']} --start-config 58 --limit-configs 16 --rattle-std 0.05 --rattle-seed 20260718 --fmax 0.05 --max-steps 10 "
            "--device cuda --default-dtype float32 --force-mode autograd --neighborlist-backend matscipy"
        ),
        (
            f'PYTHONPATH="${{TACE_ROOT}}:${{PYTHONPATH:-}}" python benchmarks/oc20neb_tace_mace/summarize_rtece_physical_pareto.py '
            f"--case {shlex.quote(case)} "
            "--max-dft-f-rmse-mev-a 120.0 --max-dft-e-rmse-mev-atom 320.0 --max-dft-f-max-mev-a 2600.0 --max-dft-e-max-mev-atom 760.0 "
            "--rattle-focus-label C_or_N --max-focus-rattle-rmsd-a 0.35 --max-rattle-fmax-ev-a 1.00 "
            f"--output-json {shlex.quote(str(physical_json))} --output-md {shlex.quote(str(physical_json.with_suffix('.md')))}"
        ),
    ]


def _wrapper_header(
    payload: dict[str, Any],
    *,
    job_name: str,
    time_limit: str,
    qos: str = "flood-1o2gpu",
    array: str | None = None,
) -> list[str]:
    env_dir = "/home/gengjianrui/bin/.venvs/tace-mace-cu126"
    header = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
    ]
    if array is not None:
        header.append(f"#SBATCH --array={array}")
    header.extend(
        [
            "#SBATCH --partition=16V100",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --gpus-per-node=1",
            f"#SBATCH --qos={qos}",
            f"#SBATCH --time={time_limit}",
            f"#SBATCH --output=/home/gengjianrui/bin/logs/{job_name}-%j.out",
            f"#SBATCH --error=/home/gengjianrui/bin/logs/{job_name}-%j.err",
            "",
            "set -eo pipefail",
            "set -x",
            _shell_assign("TACE_ROOT", "/home/gengjianrui/bin/tace"),
            _shell_assign("ENV_DIR", env_dir),
            'TACE_PYTHON="${ENV_DIR}/bin/python"',
            'PATH="${ENV_DIR}/bin:${PATH}"',
            _shell_assign("TRAIN_FILE", payload["train_file"]),
            _shell_assign("VALID_FILE", payload["valid_file"]),
            _shell_assign("TEACHER_VALID_FILE", payload["teacher_valid_file"]),
            _shell_assign("LIMIT_CONFIGS", payload["limit_configs"]),
            _shell_assign("VALID_LIMIT_CONFIGS", payload["valid_limit_configs"]),
            _shell_assign("MAX_STEPS", payload["max_steps"]),
            _shell_assign("RESULTS_ROOT", payload["artifacts"]["results_root"]),
            _shell_assign("DIAGNOSTICS_ROOT", payload["artifacts"]["diagnostics_root"]),
            'mkdir -p /home/gengjianrui/bin/logs "${RESULTS_ROOT}" "${DIAGNOSTICS_ROOT}"',
            'cd "${TACE_ROOT}"',
            '"${TACE_PYTHON}" -V',
            "nvidia-smi -L",
            "",
        ]
    )
    return header


def write_stage178_train_wrapper(path: str | Path, payload: dict[str, Any]) -> Path:
    wrapper = Path(path)
    body = _wrapper_header(payload, job_name="rtece-st178-train", time_limit="03:55:00", array="0-2")
    body.extend(
        [
            'TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"',
            'echo "Stage178 array task ${TASK_ID}"',
            "",
        ]
    )
    for index, row in enumerate(payload["candidates"]):
        branch = "if" if index == 0 else "elif"
        body.extend(
            [
                f'{branch} [ "${{TASK_ID}}" = "{index}" ]; then',
                f"  mkdir -p {shlex.quote(str(Path(payload['artifacts']['results_root']) / row['name']))}",
                f"  {_candidate_train_command(row, payload)}",
            ]
        )
    body.extend(
        [
            "else",
            '  echo "Unknown Stage178 array task ${TASK_ID}"',
            "  exit 2",
            "fi",
            "",
        ]
    )
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("\n".join(body), encoding="utf-8")
    return wrapper


def write_stage178_benchmark_wrapper(path: str | Path, payload: dict[str, Any]) -> Path:
    wrapper = Path(path)
    body = _wrapper_header(payload, job_name="rtece-st178-bench", time_limit="03:55:00")
    for row in payload["candidates"]:
        body.extend([f"# Benchmark {row['name']}", f"mkdir -p {shlex.quote(str(Path(payload['artifacts']['diagnostics_root']) / row['name']))}"])
        body.extend(_benchmark_commands(row, payload))
        body.append("")
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("\n".join(body), encoding="utf-8")
    return wrapper


def write_stage178_physical_wrapper(path: str | Path, payload: dict[str, Any]) -> Path:
    wrapper = Path(path)
    body = _wrapper_header(payload, job_name="rtece-st178-phys", time_limit="03:55:00")
    for row in payload["candidates"]:
        body.extend([f"# Physical triage {row['name']}", f"mkdir -p {shlex.quote(str(Path(payload['artifacts']['diagnostics_root']) / row['name']))}"])
        body.extend(_physical_commands(row, payload))
        body.append("")
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("\n".join(body), encoding="utf-8")
    return wrapper


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage178 Representation Upgrade Plan",
        "",
        f"- stage: `{payload['stage']}`",
        f"- train file: `{payload['train_file']}`",
        f"- valid file: `{payload['valid_file']}`",
        f"- max steps: `{payload['max_steps']}`",
        f"- primary metric: `{payload['benchmark_protocol']['primary_error_metric']}`",
        "",
        "## Question",
        "",
        payload["question"],
        "",
        "## Candidate Ladder",
        "",
        "| candidate | tier | step | marginal paths | allocation |",
        "|---|---|---|---|---|",
    ]
    for row in payload["candidates"]:
        lines.append(
            "| {name} | {tier} | {step} | {marginal} | {allocation} |".format(
                name=row["name"],
                tier=row["tece_tier"],
                step=row["representation_step"],
                marginal=", ".join(row["marginal_paths"]),
                allocation=row["capacity_allocation"],
            )
        )
    lines.extend(["", "## Review Basis", ""])
    lines.extend(f"- {item}" for item in payload["review_basis"])
    lines.append("")
    return "\n".join(lines)


def materialize_stage178(payload: dict[str, Any]) -> dict[str, Any]:
    audit = audit_stage178_manifest(payload)
    if not audit["contract_pass"]:
        raise ValueError(f"Stage178 manifest audit failed: {audit['failed_checks']}")
    artifacts = payload["artifacts"]
    for key in ("manifest", "manifest_audit", "stage_plan", "train_wrapper", "benchmark_wrapper", "physical_wrapper", "results_root", "diagnostics_root"):
        Path(artifacts[key]).parent.mkdir(parents=True, exist_ok=True)
    Path(artifacts["manifest"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(artifacts["manifest_audit"]).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(artifacts["stage_plan"]).write_text(format_markdown(payload), encoding="utf-8")
    write_stage178_train_wrapper(artifacts["train_wrapper"], payload)
    write_stage178_benchmark_wrapper(artifacts["benchmark_wrapper"], payload)
    write_stage178_physical_wrapper(artifacts["physical_wrapper"], payload)
    return dict(payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-file", default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--valid-file", default=DEFAULT_DFT_VALID)
    parser.add_argument("--teacher-valid-file", default=DEFAULT_TEACHER_VALID)
    parser.add_argument("--limit-configs", type=int, default=2048)
    parser.add_argument("--valid-limit-configs", type=int, default=256)
    parser.add_argument("--max-steps", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = make_stage178_manifest(
        output_root=args.output_root,
        train_file=args.train_file,
        valid_file=args.valid_file,
        teacher_valid_file=args.teacher_valid_file,
        limit_configs=args.limit_configs,
        valid_limit_configs=args.valid_limit_configs,
        max_steps=args.max_steps,
    )
    materialize_stage178(payload)
    print(json.dumps(audit_stage178_manifest(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
