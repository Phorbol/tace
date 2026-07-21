#!/usr/bin/env python3
"""Materialize Stage169 rTECE descriptor-cache diagnostics.

Stage169 turns the Stage168 semantic descriptor proxy into an offline, cacheable
job so full rTECE path-family diagnostics can feed the TECE path-selection loop
without recomputing heavy descriptors in every analysis run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_ROOT = Path("runs/oc20neb_tace_mace/rtece-stage169-descriptor-cache")
DEFAULT_BENCHMARK = (
    "runs/oc20neb_tace_mace/rtece-stage157-relative-neb-loss/relative_loss_runs/"
    "stage157_direct_b32_rel0p25_mixed2048/stage157_direct_b32_rel0p25_mixed2048_dft_benchmark.json"
)
DEFAULT_CONFIGS = (
    "/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/"
    "Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz"
)
PYTHON = "/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python"
FORBIDDEN_SBATCH_FLAGS = ("--export", "--mem", "--cpus-per-task")


def make_stage169_manifest(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    benchmark: str = DEFAULT_BENCHMARK,
    configs: str = DEFAULT_CONFIGS,
    max_configs_per_case: int = 2,
    num_radial: int = 8,
    ridge: float = 1.0e-3,
    neighborlist_backend: str = "matscipy",
) -> dict[str, Any]:
    root = Path(output_root)
    return {
        "schema_version": "rtece_stage169_descriptor_cache.v1",
        "stage": "stage169_rtece_descriptor_cache",
        "row_set": "stage169-rtece-descriptor-cache",
        "purpose": "cache rTECE semantic descriptor features for Stage168 energy-offset proxy diagnostics",
        "benchmark": str(benchmark),
        "configs": str(configs),
        "group_key": "case_id",
        "num_radial": int(num_radial),
        "max_configs_per_case": int(max_configs_per_case),
        "ridge": float(ridge),
        "neighborlist_backend": str(neighborlist_backend),
        "artifacts": {
            "manifest": str(root / "stage169_manifest.json"),
            "audit": str(root / "stage169_manifest_audit.json"),
            "wrapper": str(root / "stage169_descriptor_cache.sbatch"),
            "feature_cache": str(root / "stage169_rtece_descriptor_features.json"),
            "summary_json": str(root / "stage169_rtece_descriptor_proxy.json"),
            "summary_md": str(root / "stage169_rtece_descriptor_proxy.md"),
            "plan": str(root / "stage169_plan.md"),
        },
    }


def _stage169_command(payload: dict[str, Any]) -> str:
    artifacts = payload["artifacts"]
    return " ".join(
        [
            PYTHON,
            "benchmarks/oc20neb_tace_mace/summarize_rtece_stage168_rtece_descriptor_proxy.py",
            "--benchmark",
            f'"{payload["benchmark"]}"',
            "--configs",
            f'"{payload["configs"]}"',
            "--num-radial",
            str(int(payload["num_radial"])),
            "--max-configs-per-case",
            str(int(payload["max_configs_per_case"])),
            "--ridge",
            str(float(payload["ridge"])),
            "--neighborlist-backend",
            f'"{payload["neighborlist_backend"]}"',
            "--feature-cache-json",
            f'"{artifacts["feature_cache"]}"',
            "--write-feature-cache-json",
            f'"{artifacts["feature_cache"]}"',
            "--output-json",
            f'"{artifacts["summary_json"]}"',
            "--output-md",
            f'"{artifacts["summary_md"]}"',
        ]
    )


def _wrapper_text(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "#SBATCH --job-name=rtece-desc169",
            "#SBATCH --partition=16V100",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --gpus-per-node=1",
            "#SBATCH --qos=flood-1o2gpu",
            "#SBATCH --time=03:55:00",
            "#SBATCH --output=logs/rtece-desc169-%j.out",
            "#SBATCH --error=logs/rtece-desc169-%j.err",
            "",
            "set -eo pipefail",
            "cd /home/gengjianrui/bin/tace",
            "mkdir -p logs",
            f"mkdir -p \"{Path(payload['artifacts']['feature_cache']).parent}\"",
            _stage169_command(payload),
            "",
        ]
    )


def _plan_text(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Stage169 rTECE Descriptor Cache Plan",
            "",
            "This materializes a SAI-safe sbatch wrapper for Stage168 descriptor proxy diagnostics.",
            "The feature cache is a reusable data product for TECE semantic-path selection, not a model feature that uses case_id.",
            "",
            f"- benchmark: `{payload['benchmark']}`",
            f"- configs: `{payload['configs']}`",
            f"- max configs per case: `{payload['max_configs_per_case']}`",
            f"- feature cache: `{payload['artifacts']['feature_cache']}`",
            f"- summary: `{payload['artifacts']['summary_md']}`",
            "",
        ]
    )


def audit_stage169_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    wrapper = _wrapper_text(payload)
    command = _stage169_command(payload)
    combined = wrapper + "\n" + command
    forbidden = [flag for flag in FORBIDDEN_SBATCH_FLAGS if flag in combined]
    checks = {
        "schema_version": payload.get("schema_version") == "rtece_stage169_descriptor_cache.v1",
        "stage": payload.get("stage") == "stage169_rtece_descriptor_cache",
        "has_feature_cache": bool(payload.get("artifacts", {}).get("feature_cache")),
        "has_wrapper": bool(payload.get("artifacts", {}).get("wrapper")),
        "uses_stage168_summary": "summarize_rtece_stage168_rtece_descriptor_proxy.py" in command,
        "uses_feature_cache_flags": "--feature-cache-json" in command and "--write-feature-cache-json" in command,
        "no_forbidden_sbatch_flags": not forbidden,
        "no_set_u": "set -u" not in wrapper,
        "uses_pipefail": "set -eo pipefail" in wrapper,
        "max_configs_per_case_positive": int(payload.get("max_configs_per_case", 0)) > 0,
    }
    return {
        "schema_version": "rtece_stage169_descriptor_cache_audit.v1",
        **checks,
        "checks": checks,
        "passed": all(checks.values()),
        "forbidden_sbatch_flags": forbidden,
        "command": command,
    }


def materialize_stage169(payload: dict[str, Any]) -> dict[str, Any]:
    audit = audit_stage169_manifest(payload)
    if not audit["passed"]:
        raise ValueError(f"Stage169 manifest audit failed: {audit}")
    artifacts = {key: Path(value) for key, value in payload["artifacts"].items()}
    artifacts["manifest"].parent.mkdir(parents=True, exist_ok=True)
    artifacts["manifest"].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts["audit"].write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifacts["wrapper"].write_text(_wrapper_text(payload), encoding="utf-8")
    artifacts["plan"].write_text(_plan_text(payload), encoding="utf-8")
    return {
        "schema_version": "rtece_stage169_descriptor_cache_materialized.v1",
        "artifacts": {key: str(value) for key, value in artifacts.items()},
        "audit": audit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--configs", default=DEFAULT_CONFIGS)
    parser.add_argument("--max-configs-per-case", type=int, default=2)
    parser.add_argument("--num-radial", type=int, default=8)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--neighborlist-backend", default="matscipy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = make_stage169_manifest(
        output_root=args.output_root,
        benchmark=args.benchmark,
        configs=args.configs,
        max_configs_per_case=args.max_configs_per_case,
        num_radial=args.num_radial,
        ridge=args.ridge,
        neighborlist_backend=args.neighborlist_backend,
    )
    print(json.dumps(materialize_stage169(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
