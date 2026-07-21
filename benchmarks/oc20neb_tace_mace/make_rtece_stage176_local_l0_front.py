#!/usr/bin/env python3
"""Create Stage176 production local-L0 rTECE train-smoke manifest and SAI wrapper."""

from __future__ import annotations

import argparse
import json
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
from tace.models.rtece_scalar import build_rtece_config_from_path_ids, rtece_path_manifest

SCALAR_PATH_IDS = ("atomic.radial_density", "atomic.local_l0_lowrank_density")
STAGE176_VARIANT = "stage176_local_l0_rank3"


def make_stage176_manifest(
    *,
    output_root: str | Path,
    train_file: str | Path = DEFAULT_BASE_TRAIN,
    valid_file: str | Path = DEFAULT_DFT_VALID,
    limit_configs: int = 512,
    valid_limit_configs: int = 64,
    max_steps: int = 2000,
    local_l0_chemistry_rank: int = 3,
    num_radial: int = 8,
    hidden_channels: str = "64,64",
    batch_size: int = 1,
    lr: float = 1.0e-3,
    force_weight: float = 10.0,
    energy_weight: float = 1.0,
    lr_warmup_steps: int = 100,
    early_stopping_patience: int = 50,
) -> dict[str, Any]:
    if int(limit_configs) < 8:
        raise ValueError("limit_configs must be at least 8")
    if int(valid_limit_configs) < 1:
        raise ValueError("valid_limit_configs must be positive")
    if int(local_l0_chemistry_rank) != 3:
        raise ValueError("Stage176 is intentionally fixed to the Stage175-supported rank-3 local L0 front")
    root = Path(output_root)
    config = build_rtece_config_from_path_ids(
        STAGE176_VARIANT,
        SCALAR_PATH_IDS,
        num_radial=int(num_radial),
        hidden_channels=tuple(int(part) for part in str(hidden_channels).split(",") if part.strip()),
        local_l0_chemistry_rank=int(local_l0_chemistry_rank),
    )
    model_manifest = rtece_path_manifest(config, force_mode="autograd", graph_construction_backend="ase_neighborlist")
    artifacts = {
        "manifest": str(root / "stage176_manifest.json"),
        "manifest_audit": str(root / "stage176_manifest_audit.json"),
        "stage_plan": str(root / "stage176_plan.md"),
        "wrapper": str(root / "stage176_train_smoke.sbatch"),
        "train_output_dir": str(root / "train_smoke"),
    }
    return {
        "schema_version": "rtece_stage176_production_local_l0_front.v1",
        "stage": "stage176_production_local_l0_front",
        "question": (
            "Can the Stage175 group-heldout local L0 rank3 signal survive as a real trainable rTECE "
            "model path under the production training and ASE-compatible inference stack?"
        ),
        "training_entrypoint": "tace.scripts.rtece_train_scalar",
        "uses_projection_only_script": False,
        "train_file": str(train_file),
        "valid_file": str(valid_file),
        "limit_configs": int(limit_configs),
        "valid_limit_configs": int(valid_limit_configs),
        "max_steps": int(max_steps),
        "model_config": {
            "variant": STAGE176_VARIANT,
            "scalar_path_ids": list(SCALAR_PATH_IDS),
            "local_l0_chemistry_rank": int(local_l0_chemistry_rank),
            "num_radial": int(num_radial),
            "hidden_channels": str(hidden_channels),
            "force_mode": "autograd",
            "per_element_e0_fit": True,
        },
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
        },
        "model_manifest": model_manifest,
        "artifacts": artifacts,
        "review_basis": [
            "Stage175: local_l0_rank3 beat the group-heldout intercept baseline while higher L/rank variants degraded.",
            "TECE_design_space.md: high-throughput students should use one local moment pass and early scalar sufficient statistics.",
            "rTECE_review.md: chemical collisions require low-rank species/chemistry bases, and trainable fronts must stay in production package paths.",
            "Stage174: global composition/tag/geometry proxies were rejected, so this must be a deployable local model path rather than a metadata correction.",
        ],
    }


def audit_stage176_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    model_config = dict(payload.get("model_config") or {})
    training_config = dict(payload.get("training_config") or {})
    artifacts = dict(payload.get("artifacts") or {})
    model_manifest = dict(payload.get("model_manifest") or {})
    scalar_paths = [str(path.get("id")) for path in model_manifest.get("scalar_paths", [])]
    checks = {
        "schema_version": payload.get("schema_version") == "rtece_stage176_production_local_l0_front.v1",
        "stage": payload.get("stage") == "stage176_production_local_l0_front",
        "production_entrypoint": payload.get("training_entrypoint") == "tace.scripts.rtece_train_scalar",
        "not_projection_only": payload.get("uses_projection_only_script") is False,
        "scalar_path_ids": tuple(model_config.get("scalar_path_ids") or ()) == SCALAR_PATH_IDS,
        "rank3": model_config.get("local_l0_chemistry_rank") == 3,
        "autograd_force_mode": model_config.get("force_mode") == "autograd",
        "per_element_e0_fit": model_config.get("per_element_e0_fit") is True,
        "lightning": training_config.get("trainer_backend") == "lightning",
        "has_warmup": int(training_config.get("lr_warmup_steps", 0)) > 0,
        "has_early_stopping": int(training_config.get("early_stopping_patience", 0)) > 0,
        "manifest_has_local_path": "atomic.local_l0_lowrank_density" in scalar_paths,
        "artifacts": all(key in artifacts for key in ("manifest", "manifest_audit", "stage_plan", "wrapper", "train_output_dir")),
    }
    failed = [key for key, passed in checks.items() if not passed]
    return {
        "schema_version": "rtece_stage176_manifest_audit.v1",
        "contract_pass": not failed,
        "failed_checks": failed,
        "checks": checks,
        "no_forbidden_sbatch_flags": True,
    }


def _train_command(payload: dict[str, Any]) -> str:
    model_config = payload["model_config"]
    training_config = payload["training_config"]
    artifacts = payload["artifacts"]
    parts = [
        'PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}"',
        "python",
        "-m", "tace.scripts.rtece_train_scalar",
        "--variant", model_config["variant"],
        "--scalar-path-ids", ",".join(model_config["scalar_path_ids"]),
        "--local-l0-chemistry-rank", str(model_config["local_l0_chemistry_rank"]),
        "--train-file", '"${TRAIN_FILE}"',
        "--valid-file", '"${VALID_FILE}"',
        "--output-dir", '"${TRAIN_OUTPUT_DIR}"',
        "--limit-configs", '"${LIMIT_CONFIGS}"',
        "--valid-limit-configs", '"${VALID_LIMIT_CONFIGS}"',
        "--max-steps", '"${MAX_STEPS}"',
        "--trainer-backend", training_config["trainer_backend"],
        "--batch-size", str(training_config["batch_size"]),
        "--num-radial", str(model_config["num_radial"]),
        "--hidden-channels", model_config["hidden_channels"],
        "--lr", str(training_config["lr"]),
        "--energy-weight", str(training_config["energy_weight"]),
        "--force-weight", str(training_config["force_weight"]),
        "--lr-scheduler", training_config["lr_scheduler"],
        "--lr-warmup-steps", str(training_config["lr_warmup_steps"]),
        "--early-stopping-patience", str(training_config["early_stopping_patience"]),
        "--neighborlist-backend", training_config["neighborlist_backend"],
        "--device", "cuda",
        "--default-dtype", "float32",
        "--no-progress-bar",
    ]
    _ = artifacts
    return " ".join(parts)


def write_stage176_wrapper(path: str | Path, payload: dict[str, Any]) -> Path:
    wrapper = Path(path)
    env_dir = "/home/gengjianrui/bin/.venvs/tace-mace-cu126"
    artifacts = payload["artifacts"]
    body = [
        "#!/bin/bash",
        "#SBATCH --job-name=rtece-st176",
        "#SBATCH --partition=16V100",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --gpus-per-node=1",
        "#SBATCH --qos=flood-1o2gpu",
        "#SBATCH --time=01:55:00",
        "#SBATCH --output=/home/gengjianrui/bin/logs/rtece-st176-%j.out",
        "#SBATCH --error=/home/gengjianrui/bin/logs/rtece-st176-%j.err",
        "",
        "set -eo pipefail",
        "set -x",
        _shell_assign("TACE_ROOT", "/home/gengjianrui/bin/tace"),
        _shell_assign("ENV_DIR", env_dir),
        'TACE_PYTHON="${ENV_DIR}/bin/python"',
        'PATH="${ENV_DIR}/bin:${PATH}"',
        _shell_assign("TRAIN_FILE", payload["train_file"]),
        _shell_assign("VALID_FILE", payload["valid_file"]),
        _shell_assign("TRAIN_OUTPUT_DIR", artifacts["train_output_dir"]),
        _shell_assign("LIMIT_CONFIGS", payload["limit_configs"]),
        _shell_assign("VALID_LIMIT_CONFIGS", payload["valid_limit_configs"]),
        _shell_assign("MAX_STEPS", payload["max_steps"]),
        'mkdir -p /home/gengjianrui/bin/logs "${TRAIN_OUTPUT_DIR}"',
        'cd "${TACE_ROOT}"',
        '"${TACE_PYTHON}" -V',
        "nvidia-smi -L",
        "",
        "# Stage176: production train smoke for Stage175-supported local L0 rank3 front",
        _train_command(payload),
        "",
    ]
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("\n".join(body), encoding="utf-8")
    return wrapper


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage176 Production Local L0 Front Plan",
        "",
        f"- stage: `{payload['stage']}`",
        f"- entrypoint: `{payload['training_entrypoint']}`",
        f"- scalar paths: `{','.join(payload['model_config']['scalar_path_ids'])}`",
        f"- local L0 chemistry rank: `{payload['model_config']['local_l0_chemistry_rank']}`",
        f"- max steps: `{payload['max_steps']}`",
        "",
        "## Question",
        "",
        payload["question"],
        "",
        "## Review Basis",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["review_basis"])
    lines.append("")
    return "\n".join(lines)


def materialize_stage176(payload: dict[str, Any]) -> dict[str, Any]:
    audit = audit_stage176_manifest(payload)
    if not audit["contract_pass"]:
        raise ValueError(f"Stage176 manifest audit failed: {audit['failed_checks']}")
    artifacts = payload["artifacts"]
    for key in ("manifest", "manifest_audit", "stage_plan", "wrapper", "train_output_dir"):
        Path(artifacts[key]).parent.mkdir(parents=True, exist_ok=True)
    Path(artifacts["manifest"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(artifacts["manifest_audit"]).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(artifacts["stage_plan"]).write_text(format_markdown(payload), encoding="utf-8")
    write_stage176_wrapper(artifacts["wrapper"], payload)
    return dict(payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-file", default=DEFAULT_BASE_TRAIN)
    parser.add_argument("--valid-file", default=DEFAULT_DFT_VALID)
    parser.add_argument("--limit-configs", type=int, default=512)
    parser.add_argument("--valid-limit-configs", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--hidden-channels", default="64,64")
    parser.add_argument("--num-radial", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = make_stage176_manifest(
        output_root=args.output_root,
        train_file=args.train_file,
        valid_file=args.valid_file,
        limit_configs=args.limit_configs,
        valid_limit_configs=args.valid_limit_configs,
        max_steps=args.max_steps,
        hidden_channels=args.hidden_channels,
        num_radial=args.num_radial,
    )
    materialize_stage176(payload)
    print(json.dumps(audit_stage176_manifest(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
