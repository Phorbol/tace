#!/usr/bin/env python3
"""Create Stage181 conventional molecular-MD rTECE validation scaffold."""

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

from benchmarks.oc20neb_tace_mace.prepare_stage181_conventional_datasets import (
    make_3bpa_dataset_manifest,
    make_rmd17_download_manifest,
)

DEFAULT_OUTPUT_ROOT = Path("runs/oc20neb_tace_mace/rtece-stage181-conventional-md")
DEFAULT_3BPA_ROOT = Path("datasets/3BPA/dataset_3BPA")
DEFAULT_RMD17_ROOT = Path("datasets/rMD17")

STUDENT_PATH_IDS = [
    "atomic.radial_density",
    "atomic.species_basis_density",
    "atomic.local_l0_lowrank_density",
    "atomic.vector_norm",
    "atomic.vector_cross_radial_dot",
]

BENCHMARK_SPLITS = ["test_300K", "test_600K", "test_1200K", "test_dih"]


def _shell_assign(name: str, value: str | Path | int | float) -> str:
    return f"{name}={shlex.quote(str(value))}"


def _path_csv(paths: list[str]) -> str:
    return ",".join(paths)


def _artifacts(root: Path) -> dict[str, Any]:
    return {
        "manifest": str(root / "stage181_manifest.json"),
        "manifest_audit": str(root / "stage181_manifest_audit.json"),
        "stage_plan": str(root / "stage181_plan.md"),
        "dataset_manifests": {
            "3BPA": str(root / "datasets" / "3bpa_manifest.json"),
            "rMD17": str(root / "datasets" / "rmd17_manifest.json"),
        },
        "results_root": str(root / "results"),
        "diagnostics_root": str(root / "diagnostics"),
        "wrappers": {
            "train_3bpa_300k": str(root / "wrappers" / "stage181_train_3bpa_300k_no_export.sbatch"),
            "benchmark_3bpa": str(root / "wrappers" / "stage181_benchmark_3bpa_no_export.sbatch"),
            "train_3bpa_mixedT": str(root / "wrappers" / "stage181_train_3bpa_mixedT_no_export.sbatch"),
            "benchmark_3bpa_mixedT": str(root / "wrappers" / "stage181_benchmark_3bpa_mixedT_no_export.sbatch"),
            "train_rmd17_ethanol": str(root / "wrappers" / "stage181_train_rmd17_ethanol_no_export.sbatch"),
            "benchmark_rmd17_ethanol": str(root / "wrappers" / "stage181_benchmark_rmd17_ethanol_no_export.sbatch"),
        },
    }


def make_stage181_manifest(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    dataset_root: str | Path = DEFAULT_3BPA_ROOT,
    rmd17_root: str | Path = DEFAULT_RMD17_ROOT,
    max_steps: int = 20000,
) -> dict[str, Any]:
    root = Path(output_root)
    return {
        "schema_version": "rtece_stage181_conventional_md.v1",
        "stage": "stage181_conventional_md",
        "scientific_question": (
            "Can the same TECE-derived scalar-sketched student be evaluated on clean molecular MD splits "
            "where temperature and dihedral OOD generalization are explicit?"
        ),
        "dataset_matrix": {
            "primary": "3BPA",
            "secondary": "rMD17",
            "rationale": (
                "3BPA is small, eV-based, and has ID/temperature/dihedral OOD splits; rMD17 adds larger "
                "single-molecule trajectories but must be converted from kcal/mol to eV before training."
            ),
        },
        "dataset_root": str(dataset_root),
        "rmd17_root": str(rmd17_root),
        "rmd17_smoke": {
            "molecule": "ethanol",
            "train_configs_recommended_max": 1000,
            "train_file": str(Path(rmd17_root) / "converted_extxyz" / "rmd17_ethanol_train.extxyz"),
            "valid_file": str(Path(rmd17_root) / "converted_extxyz" / "rmd17_ethanol_valid.extxyz"),
            "test_file": str(Path(rmd17_root) / "converted_extxyz" / "rmd17_ethanol_test.extxyz"),
            "source_units": {"energy": "kcal/mol", "forces": "kcal/mol/A", "distance": "A"},
            "target_units": {"energy": "eV", "forces": "eV/A", "distance": "A"},
        },
        "distribution_coverage_comparison": {
            "baseline_train_split": "train_300K",
            "coverage_train_split": "train_mixedT",
            "id_split": "test_300K",
            "ood_splits": ["test_600K", "test_1200K", "test_dih"],
            "controlled_variables": [
                "same rTECE student path set",
                "same optimizer schedule",
                "same benchmark splits",
                "same energy/force units",
            ],
            "interpretation": (
                "If train_mixedT reduces high-temperature/dihedral F RMSE at similar throughput, the present "
                "failure mode is at least partly deployment-distribution coverage rather than only student capacity."
            ),
        },
        "train_split": "train_300K",
        "valid_split": "test_300K",
        "benchmark_splits": list(BENCHMARK_SPLITS),
        "labels": {"energy_key": "energy", "forces_key": "forces"},
        "units": {"energy": "eV", "forces": "eV/A", "distance": "A"},
        "student_path_ids": list(STUDENT_PATH_IDS),
        "student_config": {
            "variant": "stage181_3bpa_l1_local_l0_student",
            "num_radial": 10,
            "hidden_channels": "64,64",
            "species_basis_channels": 16,
            "species_basis_mode": "learnable_embedding",
            "local_l0_chemistry_rank": 4,
            "moment_l_max": 1,
            "atomic_cross_radial_sketch_channels": 3,
            "atomic_cross_radial_projection": "learnable",
            "short_range_repulsion_potential": "none",
        },
        "training_config": {
            "entrypoint": "tace.scripts.rtece_train_scalar",
            "trainer_backend": "lightning",
            "batch_size": 32,
            "lr": 1.0e-3,
            "energy_weight": 1.0,
            "force_weight": 10.0,
            "lr_scheduler": "plateau",
            "lr_warmup_steps": 500,
            "early_stopping_patience": 250,
            "neighborlist_backend": "matscipy",
            "per_element_e0_fit": True,
            "max_steps": int(max_steps),
        },
        "metrics_required": [
            "E MAE/RMSE/max per split",
            "F MAE/RMSE/max per split",
            "test_600K/test_1200K temperature OOD degradation",
            "test_dih dihedral PES error",
            "atoms/s and peak memory per split",
        ],
        "artifacts": _artifacts(root),
        "review_basis": [
            "TECE_design_space.md: deployment distribution mu and physical metric must define the low-cost projection target.",
            "rTECE_review.md: use held-out deployment splits, not random config splits, to decide whether degradation is safe.",
            "User Stage181 suggestion: 3BPA is eV/A extxyz; rMD17 must be converted from kcal/mol and kcal/mol/A.",
        ],
    }


def _wrapper_header(payload: dict[str, Any], *, job_name: str, time_limit: str) -> list[str]:
    env_dir = "/home/gengjianrui/bin/.venvs/tace-mace-cu126"
    return [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        "#SBATCH --partition=16V100",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --gpus-per-node=1",
        "#SBATCH --qos=flood-1o2gpu",
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
        _shell_assign("DATASET_ROOT", payload["dataset_root"]),
        _shell_assign("RMD17_ROOT", payload["rmd17_root"]),
        _shell_assign("RESULTS_ROOT", payload["artifacts"]["results_root"]),
        _shell_assign("DIAGNOSTICS_ROOT", payload["artifacts"]["diagnostics_root"]),
        'mkdir -p /home/gengjianrui/bin/logs "${RESULTS_ROOT}" "${DIAGNOSTICS_ROOT}"',
        'cd "${TACE_ROOT}"',
        '"${TACE_PYTHON}" -V',
        "nvidia-smi -L",
        "",
    ]


def _split_path(dataset_root: str | Path, split: str) -> str:
    filename = "iso_atoms.xyz" if split == "iso_atoms" else f"{split}.xyz"
    return str(Path(dataset_root) / filename)


def _student_train_command(
    payload: dict[str, Any],
    *,
    output_dir: str,
    train_split: str,
    variant: str,
) -> str:
    cfg = payload["student_config"]
    train = payload["training_config"]
    parts = [
        'PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}"',
        '"${TACE_PYTHON}"',
        "-m",
        "tace.scripts.rtece_train_scalar",
        "--variant",
        variant,
        "--scalar-path-ids",
        _path_csv(payload["student_path_ids"]),
        "--train-file",
        f'"${{DATASET_ROOT}}/{train_split}.xyz"',
        "--valid-file",
        '"${DATASET_ROOT}/test_300K.xyz"',
        "--output-dir",
        shlex.quote(output_dir),
        "--max-steps",
        str(train["max_steps"]),
        "--trainer-backend",
        train["trainer_backend"],
        "--batch-size",
        str(train["batch_size"]),
        "--num-radial",
        str(cfg["num_radial"]),
        "--hidden-channels",
        cfg["hidden_channels"],
        "--species-basis-channels",
        str(cfg["species_basis_channels"]),
        "--species-basis-mode",
        cfg["species_basis_mode"],
        "--local-l0-chemistry-rank",
        str(cfg["local_l0_chemistry_rank"]),
        "--moment-l-max",
        str(cfg["moment_l_max"]),
        "--atomic-cross-radial-sketch-channels",
        str(cfg["atomic_cross_radial_sketch_channels"]),
        "--atomic-cross-radial-projection",
        cfg["atomic_cross_radial_projection"],
        "--lr",
        str(train["lr"]),
        "--energy-weight",
        str(train["energy_weight"]),
        "--force-weight",
        str(train["force_weight"]),
        "--lr-scheduler",
        train["lr_scheduler"],
        "--lr-warmup-steps",
        str(train["lr_warmup_steps"]),
        "--early-stopping-patience",
        str(train["early_stopping_patience"]),
        "--neighborlist-backend",
        train["neighborlist_backend"],
        "--device",
        "cuda",
        "--default-dtype",
        "float32",
        "--no-progress-bar",
    ]
    return " ".join(parts)


def _write_3bpa_train_wrapper(
    path: str | Path,
    payload: dict[str, Any],
    *,
    train_split: str,
    output_name: str,
    variant: str,
    job_name: str,
) -> Path:
    wrapper = Path(path)
    body = _wrapper_header(payload, job_name=job_name, time_limit="03:55:00")
    out = str(Path(payload["artifacts"]["results_root"]) / output_name)
    body.extend([
        f"mkdir -p {shlex.quote(out)}",
        _student_train_command(payload, output_dir=out, train_split=train_split, variant=variant),
        "",
    ])
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("\n".join(body), encoding="utf-8")
    return wrapper


def write_train_wrapper(path: str | Path, payload: dict[str, Any]) -> Path:
    return _write_3bpa_train_wrapper(
        path,
        payload,
        train_split="train_300K",
        output_name="stage181_3bpa_train300K",
        variant=payload["student_config"]["variant"],
        job_name="rtece-st181-3bpa-train",
    )


def write_mixedt_train_wrapper(path: str | Path, payload: dict[str, Any]) -> Path:
    return _write_3bpa_train_wrapper(
        path,
        payload,
        train_split="train_mixedT",
        output_name="stage181_3bpa_trainMixedT",
        variant="stage181_3bpa_mixedT_l1_local_l0_student",
        job_name="rtece-st181-3bpa-mix-tr",
    )


def _write_3bpa_benchmark_wrapper(
    path: str | Path,
    payload: dict[str, Any],
    *,
    trained_output_name: str,
    variant_prefix: str,
    job_name: str,
) -> Path:
    wrapper = Path(path)
    body = _wrapper_header(payload, job_name=job_name, time_limit="02:00:00")
    checkpoint = str(Path(payload["artifacts"]["results_root"]) / trained_output_name / "rtece_scalar_best.pt")
    out_root = Path(payload["artifacts"]["diagnostics_root"]) / trained_output_name
    body.append(f"mkdir -p {shlex.quote(str(out_root))}")
    for split in payload["benchmark_splits"]:
        output = out_root / f"{split}_benchmark.json"
        body.append(
            'PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" "${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py '
            f"--model {shlex.quote(checkpoint)} --configs \"${{DATASET_ROOT}}/{split}.xyz\" "
            f"--output {shlex.quote(str(output))} --variant {variant_prefix}_{split} "
            "--start-config 0 --limit-configs 512 --measure-passes 5 --device cuda --default-dtype float32 "
            "--force-mode autograd --graph-construction-backend matscipy_neighborlist"
        )
    body.append("")
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("\n".join(body), encoding="utf-8")
    return wrapper


def write_benchmark_wrapper(path: str | Path, payload: dict[str, Any]) -> Path:
    return _write_3bpa_benchmark_wrapper(
        path,
        payload,
        trained_output_name="stage181_3bpa_train300K",
        variant_prefix="stage181_3bpa_train300K",
        job_name="rtece-st181-3bpa-bench",
    )


def write_mixedt_benchmark_wrapper(path: str | Path, payload: dict[str, Any]) -> Path:
    return _write_3bpa_benchmark_wrapper(
        path,
        payload,
        trained_output_name="stage181_3bpa_trainMixedT",
        variant_prefix="stage181_3bpa_trainMixedT",
        job_name="rtece-st181-3bpa-mix-bn",
    )


def _rmd17_train_command(payload: dict[str, Any], *, output_dir: str) -> str:
    cfg = payload["student_config"]
    train = payload["training_config"]
    parts = [
        'PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}"',
        '"${TACE_PYTHON}"',
        "-m",
        "tace.scripts.rtece_train_scalar",
        "--variant",
        "stage181_rmd17_ethanol_l1_local_l0_student",
        "--scalar-path-ids",
        _path_csv(payload["student_path_ids"]),
        "--train-file",
        '"${RMD17_ROOT}/converted_extxyz/rmd17_ethanol_train.extxyz"',
        "--valid-file",
        '"${RMD17_ROOT}/converted_extxyz/rmd17_ethanol_valid.extxyz"',
        "--output-dir",
        shlex.quote(output_dir),
        "--max-steps",
        str(train["max_steps"]),
        "--trainer-backend",
        train["trainer_backend"],
        "--batch-size",
        str(train["batch_size"]),
        "--num-radial",
        str(cfg["num_radial"]),
        "--hidden-channels",
        cfg["hidden_channels"],
        "--species-basis-channels",
        str(cfg["species_basis_channels"]),
        "--species-basis-mode",
        cfg["species_basis_mode"],
        "--local-l0-chemistry-rank",
        str(cfg["local_l0_chemistry_rank"]),
        "--moment-l-max",
        str(cfg["moment_l_max"]),
        "--atomic-cross-radial-sketch-channels",
        str(cfg["atomic_cross_radial_sketch_channels"]),
        "--atomic-cross-radial-projection",
        cfg["atomic_cross_radial_projection"],
        "--lr",
        str(train["lr"]),
        "--energy-weight",
        str(train["energy_weight"]),
        "--force-weight",
        str(train["force_weight"]),
        "--lr-scheduler",
        train["lr_scheduler"],
        "--lr-warmup-steps",
        str(train["lr_warmup_steps"]),
        "--early-stopping-patience",
        str(train["early_stopping_patience"]),
        "--neighborlist-backend",
        train["neighborlist_backend"],
        "--device",
        "cuda",
        "--default-dtype",
        "float32",
        "--no-progress-bar",
    ]
    return " ".join(parts)


def write_rmd17_train_wrapper(path: str | Path, payload: dict[str, Any]) -> Path:
    wrapper = Path(path)
    body = _wrapper_header(payload, job_name="rtece-st181-rmd17-train", time_limit="03:55:00")
    out = str(Path(payload["artifacts"]["results_root"]) / "stage181_rmd17_ethanol")
    body.extend([f"mkdir -p {shlex.quote(out)}", _rmd17_train_command(payload, output_dir=out), ""])
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("\n".join(body), encoding="utf-8")
    return wrapper


def write_rmd17_benchmark_wrapper(path: str | Path, payload: dict[str, Any]) -> Path:
    wrapper = Path(path)
    body = _wrapper_header(payload, job_name="rtece-st181-rmd17-bench", time_limit="01:00:00")
    checkpoint = str(Path(payload["artifacts"]["results_root"]) / "stage181_rmd17_ethanol" / "rtece_scalar_best.pt")
    output = str(Path(payload["artifacts"]["diagnostics_root"]) / "stage181_rmd17_ethanol_test_benchmark.json")
    body.append(f"mkdir -p {shlex.quote(str(Path(output).parent))}")
    body.append(
        'PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" "${TACE_PYTHON}" benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py '
        f"--model {shlex.quote(checkpoint)} --configs \"${{RMD17_ROOT}}/converted_extxyz/rmd17_ethanol_test.extxyz\" "
        f"--output {shlex.quote(output)} --variant stage181_rmd17_ethanol_test "
        "--start-config 0 --limit-configs 512 --measure-passes 5 --device cuda --default-dtype float32 "
        "--force-mode autograd --graph-construction-backend matscipy_neighborlist"
    )
    body.append("")
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("\n".join(body), encoding="utf-8")
    return wrapper

def render_stage_plan(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage181 Conventional Molecular MD Validation",
        "",
        "This stage adds clean molecular MD deployment splits before continuing renormalized initialization comparisons.",
        "",
        "## Datasets",
        "",
        "- primary: 3BPA, eV/eV-A extxyz with 300K ID, 600K/1200K OOD, and dihedral PES splits",
        "- secondary: rMD17, npz source requiring kcal/mol to eV conversion before training",
        "- rMD17 smoke wrappers expect converted ethanol train/valid/test extxyz files under `converted_extxyz/`",
        "- 3BPA distribution coverage check compares train_300K against train_mixedT with the same student",
        "",
        "## Student",
        "",
        f"- paths: `{_path_csv(payload['student_path_ids'])}`",
        f"- local L0 chemistry rank: `{payload['student_config']['local_l0_chemistry_rank']}`",
        "",
        "## Required Outputs",
        "",
    ]
    for item in payload["metrics_required"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def audit_stage181_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    wrappers = payload.get("artifacts", {}).get("wrappers", {})
    wrapper_text = ""
    for path in wrappers.values():
        p = Path(path)
        if p.exists():
            wrapper_text += p.read_text(encoding="utf-8")
    forbidden = ("--export", "#SBATCH --mem", "--mem=", "#SBATCH --cpus-per-task", "--cpus-per-task", "set -u")
    checks = {
        "schema": payload.get("schema_version") == "rtece_stage181_conventional_md.v1",
        "stage": payload.get("stage") == "stage181_conventional_md",
        "primary_dataset": payload.get("dataset_matrix", {}).get("primary") == "3BPA",
        "secondary_dataset": payload.get("dataset_matrix", {}).get("secondary") == "rMD17",
        "units": payload.get("units") == {"energy": "eV", "forces": "eV/A", "distance": "A"},
        "student_paths": payload.get("student_path_ids") == STUDENT_PATH_IDS,
        "benchmark_splits": payload.get("benchmark_splits") == BENCHMARK_SPLITS,
        "wrappers": set(wrappers) == {
            "train_3bpa_300k",
            "benchmark_3bpa",
            "train_3bpa_mixedT",
            "benchmark_3bpa_mixedT",
            "train_rmd17_ethanol",
            "benchmark_rmd17_ethanol",
        },
        "distribution_coverage_comparison": payload.get("distribution_coverage_comparison", {}).get(
            "coverage_train_split"
        ) == "train_mixedT",
        "rmd17_smoke_units": payload.get("rmd17_smoke", {}).get("target_units")
        == {"energy": "eV", "forces": "eV/A", "distance": "A"},
        "no_forbidden_sbatch_flags": not any(token in wrapper_text for token in forbidden),
    }
    failed = [key for key, ok in checks.items() if not ok]
    return {
        "schema_version": "rtece_stage181_manifest_audit.v1",
        "contract_pass": not failed,
        "failed_checks": failed,
        "checks": checks,
        "no_forbidden_sbatch_flags": checks["no_forbidden_sbatch_flags"],
    }


def materialize_stage181(payload: dict[str, Any]) -> dict[str, Any]:
    artifacts = payload["artifacts"]
    Path(artifacts["manifest"]).parent.mkdir(parents=True, exist_ok=True)
    dataset_manifest_3bpa = make_3bpa_dataset_manifest(payload["dataset_root"])
    dataset_manifest_rmd17 = make_rmd17_download_manifest(payload["rmd17_root"])
    Path(artifacts["dataset_manifests"]["3BPA"]).parent.mkdir(parents=True, exist_ok=True)
    Path(artifacts["dataset_manifests"]["3BPA"]).write_text(
        json.dumps(dataset_manifest_3bpa, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    Path(artifacts["dataset_manifests"]["rMD17"]).write_text(
        json.dumps(dataset_manifest_rmd17, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_train_wrapper(artifacts["wrappers"]["train_3bpa_300k"], payload)
    write_benchmark_wrapper(artifacts["wrappers"]["benchmark_3bpa"], payload)
    write_mixedt_train_wrapper(artifacts["wrappers"]["train_3bpa_mixedT"], payload)
    write_mixedt_benchmark_wrapper(artifacts["wrappers"]["benchmark_3bpa_mixedT"], payload)
    write_rmd17_train_wrapper(artifacts["wrappers"]["train_rmd17_ethanol"], payload)
    write_rmd17_benchmark_wrapper(artifacts["wrappers"]["benchmark_rmd17_ethanol"], payload)
    audit = audit_stage181_manifest(payload)
    Path(artifacts["manifest"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(artifacts["manifest_audit"]).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(artifacts["stage_plan"]).write_text(render_stage_plan(payload), encoding="utf-8")
    return {"payload": payload, "audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_3BPA_ROOT)
    parser.add_argument("--rmd17-root", type=Path, default=DEFAULT_RMD17_ROOT)
    args = parser.parse_args()
    payload = make_stage181_manifest(
        output_root=args.output_root,
        dataset_root=args.dataset_root,
        rmd17_root=args.rmd17_root,
    )
    result = materialize_stage181(payload)
    print(json.dumps(result["audit"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
