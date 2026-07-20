#!/usr/bin/env python3
"""Create Stage145 NEP and DeepMD community baseline wrappers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_TRAIN_FILE = Path(
    "runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/"
    "weighted_train_base2048_plus_teacher_relax640_forceonly_eanchor.extxyz"
)
DEFAULT_OUTPUT_ROOT = Path("runs/oc20neb_tace_mace/community-baselines-stage145")


def make_stage145_manifest(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    train_file: str | Path = DEFAULT_TRAIN_FILE,
    limit_configs: int = 2688,
    valid_limit_configs: int = 256,
    bench_limit_configs: int = 1024,
    nep_generations: int = 20000,
    deepmd_stop_batch: int = 20000,
) -> dict[str, Any]:
    root = Path(output_root)
    if int(limit_configs) < 1:
        raise ValueError("limit_configs must be positive")
    if int(valid_limit_configs) < 1:
        raise ValueError("valid_limit_configs must be positive")
    if int(bench_limit_configs) < 1:
        raise ValueError("bench_limit_configs must be positive")
    rows = [
        {
            "name": "nep4_mixed_smoke",
            "engine": "nep",
            "module": "gpumd/4.8-cuda12.4",
            "descriptor_label": "nep4",
            "train_dir": str(root / "nep4_mixed_smoke"),
            "generation": int(nep_generations),
            "wrapper": str(root / "wrappers" / "nep4_mixed_smoke_no_export.sbatch"),
        },
        {
            "name": "deepmd_dpa_like_mixed_smoke",
            "engine": "deepmd",
            "module": "deepmd-kit/3.1.2",
            "descriptor_label": "dpa1_zero_attention",
            "train_dir": str(root / "deepmd_dpa_like_mixed_smoke"),
            "stop_batch": int(deepmd_stop_batch),
            "wrapper": str(root / "wrappers" / "deepmd_dpa_like_mixed_smoke_no_export.sbatch"),
        },
    ]
    return {
        "schema_version": "community_baselines_stage145.v1",
        "stage": "community_baselines_stage145",
        "row_set": "community-baselines-stage145",
        "output_root": str(root),
        "train_contract": {
            "train_file": str(train_file),
            "label_target": "mixed_energy_forces",
            "energy_key": "energy",
            "forces_key": "forces",
            "dft_energy_key": "dft_energy",
            "dft_forces_key": "dft_forces",
            "teacher_energy_key": "teacher_energy",
            "teacher_forces_key": "teacher_forces",
            "energy_weight_key": "energy_weight",
            "forces_weight_key": "forces_weight",
            "limit_configs": int(limit_configs),
            "valid_limit_configs": int(valid_limit_configs),
            "bench_limit_configs": int(bench_limit_configs),
        },
        "metrics_contract": [
            "mae_e_mev_atom",
            "rmse_e_mev_atom",
            "max_abs_e_mev_atom",
            "bias_e_mev_atom",
            "mae_f_mev_a",
            "rmse_f_mev_a",
            "max_abs_f_mev_a",
            "atoms_per_second",
            "dimer_scan",
            "rattle_relax",
        ],
        "comparison_anchors": [
            "stage140_forceonly_ew2",
            "stage142_teacher_relax_coverage",
            "stage144_t3_cavity_vector",
        ],
        "rows": rows,
        "artifacts": {
            "manifest": str(root / "stage145_manifest.json"),
            "audit": str(root / "stage145_manifest_audit.json"),
            "stage_plan": str(root / "stage145_plan.md"),
            "wrapper_root": str(root / "wrappers"),
        },
    }


def audit_stage145_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    rows = list(payload.get("rows") or [])
    row_names = {str(row.get("name")) for row in rows}
    wrappers = [str(row.get("wrapper", "")) for row in rows]
    checks = {
        "schema_version": payload.get("schema_version") == "community_baselines_stage145.v1",
        "stage": payload.get("stage") == "community_baselines_stage145",
        "row_set": payload.get("row_set") == "community-baselines-stage145",
        "has_nep_and_deepmd": row_names == {"nep4_mixed_smoke", "deepmd_dpa_like_mixed_smoke"},
        "mixed_label_contract": (payload.get("train_contract") or {}).get("label_target") == "mixed_energy_forces",
        "rmse_metrics_present": "rmse_e_mev_atom" in payload.get("metrics_contract", [])
        and "rmse_f_mev_a" in payload.get("metrics_contract", []),
        "no_export_wrapper_names": all(path.endswith("_no_export.sbatch") for path in wrappers),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "schema_version": "community_baselines_stage145_audit.v1",
        "contract_pass": not failed,
        "failed_checks": failed,
        "checks": checks,
    }


def _write_wrapper(row: dict[str, Any], payload: dict[str, Any]) -> str:
    wrapper = Path(row["wrapper"])
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    train_file = payload["train_contract"]["train_file"]
    limit = payload["train_contract"]["limit_configs"]
    train_dir = row["train_dir"]
    if row["engine"] == "nep":
        command = f"""module load {row['module']}
export STAGE145_TRAIN_FILE="{train_file}"
export STAGE145_OUTPUT_DIR="{train_dir}"
export STAGE145_LIMIT_CONFIGS="{limit}"
python benchmarks/oc20neb_tace_mace/convert_stage145_nep.py --manifest {payload['artifacts']['manifest']} --row {row['name']}
cd "{train_dir}"
nep
"""
    else:
        command = f"""module load {row['module']}
export STAGE145_TRAIN_FILE="{train_file}"
export STAGE145_OUTPUT_DIR="{train_dir}"
export STAGE145_LIMIT_CONFIGS="{limit}"
python benchmarks/oc20neb_tace_mace/convert_stage145_deepmd.py --manifest {payload['artifacts']['manifest']} --row {row['name']}
cd "{train_dir}"
dp --pt train input.json
dp --pt freeze -o frozen_model.pth
"""
    text = f"""#!/bin/bash
#SBATCH --job-name={row['name']}
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=rush-1o2gpu
#SBATCH --output=logs/{row['name']}-%j.out
#SBATCH --error=logs/{row['name']}-%j.err

set -eo pipefail
export STAGE145_ROOT_LABEL="community-baselines-stage145"
cd "$(pwd)"
{command}
"""
    wrapper.write_text(text, encoding="utf-8")
    return str(wrapper)


def _write_stage_plan(payload: dict[str, Any]) -> str:
    path = Path(payload["artifacts"]["stage_plan"])
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage145 Community Baselines Plan",
        "",
        f"- train file: `{payload['train_contract']['train_file']}`",
        "- label target: mixed `energy` and `forces`",
        "- rows: `nep4_mixed_smoke`, `deepmd_dpa_like_mixed_smoke`",
        "- report DFT, teacher, mixed RMSE/MAE/max/bias and atoms/s.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def materialize_stage145(payload: dict[str, Any]) -> dict[str, Any]:
    audit = audit_stage145_manifest(payload)
    if not audit["contract_pass"]:
        raise ValueError(f"Stage145 manifest audit failed: {audit['failed_checks']}")
    manifest_path = Path(payload["artifacts"]["manifest"])
    audit_path = Path(payload["artifacts"]["audit"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    wrappers = {row["name"]: _write_wrapper(row, payload) for row in payload["rows"]}
    stage_plan = _write_stage_plan(payload)
    return {"manifest": str(manifest_path), "audit": str(audit_path), "stage_plan": stage_plan, "wrappers": wrappers}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--limit-configs", type=int, default=2688)
    parser.add_argument("--valid-limit-configs", type=int, default=256)
    parser.add_argument("--bench-limit-configs", type=int, default=1024)
    parser.add_argument("--nep-generations", type=int, default=20000)
    parser.add_argument("--deepmd-stop-batch", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = make_stage145_manifest(
        output_root=args.output_root,
        train_file=args.train_file,
        limit_configs=args.limit_configs,
        valid_limit_configs=args.valid_limit_configs,
        bench_limit_configs=args.bench_limit_configs,
        nep_generations=args.nep_generations,
        deepmd_stop_batch=args.deepmd_stop_batch,
    )
    print(json.dumps(materialize_stage145(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
