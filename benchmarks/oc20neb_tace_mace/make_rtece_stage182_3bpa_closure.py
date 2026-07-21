#!/usr/bin/env python3
"""Create Stage182 3BPA conventional-MD closure scaffold.

Stage182 turns the Stage181 3BPA observation into a controlled comparison:
the same train_300K deployment distribution is evaluated with rTECE, NEP4,
and a DeepMD DPA1 zero-attention-style baseline on the same ID/OOD splits.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_ROOT = Path("runs/oc20neb_tace_mace/rtece-stage182-3bpa-closure")
DEFAULT_DATASET_ROOT = Path("datasets/3BPA/dataset_3BPA")
DEFAULT_PYTHON = Path("/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python")

BENCHMARK_SPLITS = ["test_300K", "test_600K", "test_1200K", "test_dih"]
FORBIDDEN_SBATCH_TOKENS = (
    "--export",
    "#SBATCH --mem",
    "--mem=",
    "#SBATCH --cpus-per-task",
    "--cpus-per-task",
    "set -u",
    "export ",
)

RTECE_PATH_IDS = [
    "atomic.radial_density",
    "atomic.species_basis_density",
    "atomic.local_l0_lowrank_density",
    "atomic.vector_norm",
    "atomic.vector_cross_radial_dot",
]


def _artifacts(root: Path) -> dict[str, Any]:
    return {
        "manifest": str(root / "stage182_manifest.json"),
        "audit": str(root / "stage182_manifest_audit.json"),
        "summary_json": str(root / "stage182_summary.json"),
        "summary_md": str(root / "stage182_summary.md"),
        "stage_plan": str(root / "stage182_plan.md"),
        "wrapper_root": str(root / "wrappers"),
        "benchmark_wrapper_root": str(root / "benchmark_wrappers"),
        "results_root": str(root / "results"),
        "diagnostics_root": str(root / "diagnostics"),
    }


def _row_outputs(root: Path, row_name: str) -> dict[str, str]:
    return {
        split: str(root / "diagnostics" / row_name / f"{split}_benchmark.json")
        for split in BENCHMARK_SPLITS
    }


def make_stage182_manifest(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    train_split: str = "train_300K",
    train_limit_configs: int = 500,
    bench_limit_configs: int = 512,
    max_steps: int = 20000,
    nep_generations: int = 20000,
    deepmd_stop_batch: int = 20000,
) -> dict[str, Any]:
    root = Path(output_root)
    if int(train_limit_configs) < 1:
        raise ValueError("train_limit_configs must be positive")
    if int(bench_limit_configs) < 1:
        raise ValueError("bench_limit_configs must be positive")
    artifacts = _artifacts(root)
    rows = [
        {
            "name": "rtece_l1_local_l0_train300k",
            "engine": "rtece",
            "descriptor_label": "tece_scalar_l1_local_l0_rank4",
            "train_dir": str(root / "results" / "rtece_l1_local_l0_train300k"),
            "wrapper": str(root / "wrappers" / "rtece_l1_local_l0_train300k_no_export.sbatch"),
            "benchmark_wrapper": str(root / "benchmark_wrappers" / "rtece_l1_local_l0_train300k_benchmark_no_export.sbatch"),
            "model_artifact": str(root / "results" / "rtece_l1_local_l0_train300k" / "rtece_scalar_best.pt"),
            "benchmark_outputs": _row_outputs(root, "rtece_l1_local_l0_train300k"),
            "student_config": {
                "scalar_path_ids": list(RTECE_PATH_IDS),
                "num_radial": 10,
                "hidden_channels": "64,64",
                "species_basis_channels": 16,
                "species_basis_mode": "learnable_embedding",
                "local_l0_chemistry_rank": 4,
                "moment_l_max": 1,
                "atomic_cross_radial_sketch_channels": 3,
                "atomic_cross_radial_projection": "learnable",
            },
        },
        {
            "name": "nep4_train300k",
            "engine": "nep",
            "module": "gpumd/4.8-cuda12.4",
            "descriptor_label": "nep4",
            "train_dir": str(root / "results" / "nep4_train300k"),
            "wrapper": str(root / "wrappers" / "nep4_train300k_no_export.sbatch"),
            "benchmark_wrapper": str(root / "benchmark_wrappers" / "nep4_train300k_benchmark_no_export.sbatch"),
            "model_artifact": str(root / "results" / "nep4_train300k" / "nep.txt"),
            "benchmark_outputs": _row_outputs(root, "nep4_train300k"),
            "generation": int(nep_generations),
        },
        {
            "name": "deepmd_dpa1_zero_train300k",
            "engine": "deepmd",
            "module": "deepmd-kit/3.1.2",
            "descriptor_label": "dpa1_zero_attention",
            "train_dir": str(root / "results" / "deepmd_dpa1_zero_train300k"),
            "wrapper": str(root / "wrappers" / "deepmd_dpa1_zero_train300k_no_export.sbatch"),
            "benchmark_wrapper": str(root / "benchmark_wrappers" / "deepmd_dpa1_zero_train300k_benchmark_no_export.sbatch"),
            "model_artifact": str(root / "results" / "deepmd_dpa1_zero_train300k" / "frozen_model.pth"),
            "benchmark_outputs": _row_outputs(root, "deepmd_dpa1_zero_train300k"),
            "stop_batch": int(deepmd_stop_batch),
        },
    ]
    return {
        "schema_version": "rtece_stage182_3bpa_closure.v1",
        "stage": "stage182_3bpa_conventional_closure",
        "scientific_question": (
            "On a clean molecular-MD deployment distribution, does the TECE-derived scalar student occupy a "
            "useful accuracy/throughput region relative to NEP4 and DPA1 zero-attention-style baselines?"
        ),
        "theory_alignment": [
            "TECE_design_space.md: define the projection target by deployment distribution mu, physical error metric, and hardware cost.",
            "rTECE_review.md: compare against community high-throughput baselines before spending effort on low-level kernels.",
            "Stage181 result: mixed-temperature coverage helped OOD force RMSE but did not close energy/tail errors, so train_300K is the controlled first row.",
        ],
        "dataset": {
            "name": "3BPA",
            "root": str(dataset_root),
            "train_split": str(train_split),
            "benchmark_splits": list(BENCHMARK_SPLITS),
            "units": {"energy": "eV", "forces": "eV/A", "distance": "A"},
            "labels": {"energy_key": "energy", "forces_key": "forces"},
            "ood_axes": ["temperature_600K", "temperature_1200K", "dihedral_pes"],
        },
        "train_contract": {
            "train_file": str(Path(dataset_root) / f"{train_split}.xyz"),
            "energy_key": "energy",
            "forces_key": "forces",
            "limit_configs": int(train_limit_configs),
            "bench_limit_configs": int(bench_limit_configs),
            "max_steps": int(max_steps),
            "rtece_entrypoint": "tace.scripts.rtece_train_scalar",
            "nep_converter": "benchmarks/oc20neb_tace_mace/convert_stage145_nep.py",
            "deepmd_converter": "benchmarks/oc20neb_tace_mace/convert_stage145_deepmd.py",
        },
        "comparison_contract": {
            "primary_ranking_metric": "rmse_f_mev_a",
            "secondary_ranking_metrics": ["rmse_e_mev_atom", "atoms_per_second", "max_abs_f_mev_a"],
            "required_metrics": [
                "mae_e_mev_atom",
                "rmse_e_mev_atom",
                "max_abs_e_mev_atom",
                "mae_f_mev_a",
                "rmse_f_mev_a",
                "max_abs_f_mev_a",
                "atoms_per_second",
                "peak_memory_mb",
            ],
            "physical_followups_after_numeric_closure": ["dimer_scan", "rattle_relax_rmsd"],
        },
        "rows": rows,
        "artifacts": artifacts,
    }


def _header(job_name: str, *, time_limit: str = "03:55:00") -> list[str]:
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
        "TACE_ROOT=/home/gengjianrui/bin/tace",
        f"TACE_PYTHON={shlex.quote(str(DEFAULT_PYTHON))}",
        "PATH=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin:${PATH}",
        "mkdir -p /home/gengjianrui/bin/logs",
        "cd ${TACE_ROOT}",
        "${TACE_PYTHON} -V",
        "",
    ]


def _write(path: str | Path, lines: list[str]) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(lines).rstrip() + "\n"
    p.write_text(text, encoding="utf-8")
    return str(p)


def _dataset_split(payload: dict[str, Any], split: str) -> str:
    return str(Path(payload["dataset"]["root"]) / f"{split}.xyz")


def _write_rtece_train_wrapper(row: dict[str, Any], payload: dict[str, Any]) -> str:
    cfg = row["student_config"]
    train = payload["train_contract"]
    lines = _header("rtece-st182-rtece-train")
    lines.extend([
        f"mkdir -p {shlex.quote(row['train_dir'])}",
        "PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} -m tace.scripts.rtece_train_scalar "
        f"--variant {shlex.quote(row['name'])} "
        f"--scalar-path-ids {shlex.quote(','.join(cfg['scalar_path_ids']))} "
        f"--train-file {shlex.quote(train['train_file'])} "
        f"--valid-file {shlex.quote(_dataset_split(payload, 'test_300K'))} "
        f"--output-dir {shlex.quote(row['train_dir'])} "
        f"--max-steps {int(train['max_steps'])} "
        "--trainer-backend lightning --batch-size 32 "
        f"--num-radial {int(cfg['num_radial'])} "
        f"--hidden-channels {shlex.quote(cfg['hidden_channels'])} "
        f"--species-basis-channels {int(cfg['species_basis_channels'])} "
        f"--species-basis-mode {shlex.quote(cfg['species_basis_mode'])} "
        f"--local-l0-chemistry-rank {int(cfg['local_l0_chemistry_rank'])} "
        f"--moment-l-max {int(cfg['moment_l_max'])} "
        f"--atomic-cross-radial-sketch-channels {int(cfg['atomic_cross_radial_sketch_channels'])} "
        f"--atomic-cross-radial-projection {shlex.quote(cfg['atomic_cross_radial_projection'])} "
        f"--energy-key {shlex.quote(train['energy_key'])} --forces-key {shlex.quote(train['forces_key'])} "
        "--energy-weight 1.0 --force-weight 10.0 --lr 0.001 --lr-scheduler plateau "
        "--lr-warmup-steps 500 --early-stopping-patience 250 "
        "--neighborlist-backend matscipy --device cuda --default-dtype float32 --no-progress-bar",
    ])
    return _write(row["wrapper"], lines)


def _write_community_train_wrapper(row: dict[str, Any], payload: dict[str, Any]) -> str:
    lines = _header(f"rtece-st182-{row['engine']}-train")
    lines.append(f"module load {row['module']}")
    converter = payload["train_contract"]["nep_converter" if row["engine"] == "nep" else "deepmd_converter"]
    lines.append(f"PYTHONPATH=${{TACE_ROOT}}:${{PYTHONPATH:-}} ${{TACE_PYTHON}} {converter} --manifest {shlex.quote(payload['artifacts']['manifest'])} --row {shlex.quote(row['name'])}")
    lines.append(f"cd {shlex.quote(row['train_dir'])}")
    if row["engine"] == "nep":
        lines.append("nep")
    else:
        lines.append("dp --pt train input.json")
        lines.append("dp --pt freeze -o frozen_model.pth")
    return _write(row["wrapper"], lines)


def _write_train_wrapper(row: dict[str, Any], payload: dict[str, Any]) -> str:
    if row["engine"] == "rtece":
        return _write_rtece_train_wrapper(row, payload)
    return _write_community_train_wrapper(row, payload)


def _write_benchmark_wrapper(row: dict[str, Any], payload: dict[str, Any]) -> str:
    lines = _header(f"rtece-st182-{row['engine']}-bench", time_limit="01:40:00")
    if row["engine"] in {"nep", "deepmd"}:
        lines.append(f"module load {row['module']}")
    lines.append(f"mkdir -p {shlex.quote(str(Path(payload['artifacts']['diagnostics_root']) / row['name']))}")
    train = payload["train_contract"]
    for split in BENCHMARK_SPLITS:
        configs = _dataset_split(payload, split)
        output = row["benchmark_outputs"][split]
        if row["engine"] == "rtece":
            lines.append(
                "PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py "
                f"--model {shlex.quote(row['model_artifact'])} --configs {shlex.quote(configs)} "
                f"--output {shlex.quote(output)} --variant {shlex.quote(row['name'] + '_' + split)} "
                f"--energy-key {shlex.quote(train['energy_key'])} --forces-key {shlex.quote(train['forces_key'])} "
                f"--start-config 0 --limit-configs {int(train['bench_limit_configs'])} --measure-passes 5 "
                "--device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist"
            )
        else:
            lines.append(
                "PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/benchmark_stage145_community.py "
                f"--engine {shlex.quote(row['engine'])} --model-artifact {shlex.quote(row['model_artifact'])} "
                f"--configs {shlex.quote(configs)} --output {shlex.quote(output)} --row-name {shlex.quote(row['name'] + '_' + split)} "
                f"--energy-key {shlex.quote(train['energy_key'])} --forces-key {shlex.quote(train['forces_key'])} "
                f"--start-config 0 --limit-configs {int(train['bench_limit_configs'])} --measure-passes 5 --device cuda"
            )
    return _write(row["benchmark_wrapper"], lines)


def _write_stage_plan(payload: dict[str, Any]) -> str:
    path = Path(payload["artifacts"]["stage_plan"])
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage182 3BPA Conventional Closure",
        "",
        "Goal: compare the TECE-derived scalar student against NEP4 and DPA1 zero-attention-style baselines on one clean molecular-MD deployment distribution.",
        "",
        "## Controlled Variables",
        "",
        f"- train split: `{payload['dataset']['train_split']}`",
        f"- train configs: `{payload['train_contract']['limit_configs']}`",
        f"- benchmark configs per split: `{payload['train_contract']['bench_limit_configs']}`",
        "- labels: `energy` and `forces` in eV/eV-A",
        "",
        "## Splits",
        "",
    ]
    lines.extend(f"- `{split}`" for split in BENCHMARK_SPLITS)
    lines.extend([
        "",
        "## Interpretation",
        "",
        "- If rTECE is much slower than NEP/DPA1 at similar error, the front-end descriptor path is still too expensive.",
        "- If rTECE is much less accurate at similar train coverage, the current scalarized TECE projection lacks needed representation rank.",
        "- If rTECE has competitive force RMSE but poor energy RMSE/max, prioritize energy-gauge/E0 and residual distillation before kernel work.",
    ])
    return _write(path, lines)


def audit_stage182_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    rows = list(payload.get("rows") or [])
    row_names = {str(row.get("name")) for row in rows}
    wrapper_paths = [str(row.get("wrapper", "")) for row in rows] + [str(row.get("benchmark_wrapper", "")) for row in rows]
    wrapper_text = ""
    for path in wrapper_paths:
        p = Path(path)
        if p.exists():
            wrapper_text += p.read_text(encoding="utf-8") + "\n"
    checks = {
        "schema_version": payload.get("schema_version") == "rtece_stage182_3bpa_closure.v1",
        "stage": payload.get("stage") == "stage182_3bpa_conventional_closure",
        "dataset": (payload.get("dataset") or {}).get("name") == "3BPA",
        "units": (payload.get("dataset") or {}).get("units") == {"energy": "eV", "forces": "eV/A", "distance": "A"},
        "benchmark_splits": (payload.get("dataset") or {}).get("benchmark_splits") == BENCHMARK_SPLITS,
        "rows": row_names == {"rtece_l1_local_l0_train300k", "nep4_train300k", "deepmd_dpa1_zero_train300k"},
        "rmse_primary": (payload.get("comparison_contract") or {}).get("primary_ranking_metric") == "rmse_f_mev_a",
        "required_metrics": all(
            metric in (payload.get("comparison_contract") or {}).get("required_metrics", [])
            for metric in ["rmse_e_mev_atom", "rmse_f_mev_a", "max_abs_e_mev_atom", "max_abs_f_mev_a", "atoms_per_second", "peak_memory_mb"]
        ),
        "no_forbidden_sbatch_tokens": not any(token in wrapper_text for token in FORBIDDEN_SBATCH_TOKENS),
        "wrapper_names": all(path.endswith("_no_export.sbatch") for path in wrapper_paths if path),
        "split_complete": all(
            split in (row.get("benchmark_outputs") or {}) for row in rows for split in BENCHMARK_SPLITS
        ),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "schema_version": "rtece_stage182_3bpa_closure_audit.v1",
        "contract_pass": not failed,
        "failed_checks": failed,
        "checks": checks,
        "no_forbidden_sbatch_tokens": checks["no_forbidden_sbatch_tokens"],
    }


def materialize_stage182(payload: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(payload["artifacts"]["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    wrappers = {row["name"]: _write_train_wrapper(row, payload) for row in payload["rows"]}
    benchmark_wrappers = {row["name"]: _write_benchmark_wrapper(row, payload) for row in payload["rows"]}
    stage_plan = _write_stage_plan(payload)
    audit = audit_stage182_manifest(payload)
    if not audit["contract_pass"]:
        raise ValueError(f"Stage182 manifest audit failed: {audit['failed_checks']}")
    audit_path = Path(payload["artifacts"]["audit"])
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "manifest": str(manifest_path),
        "audit": audit,
        "audit_path": str(audit_path),
        "stage_plan": stage_plan,
        "wrappers": wrappers,
        "benchmark_wrappers": benchmark_wrappers,
    }


def _read_benchmark(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _summary_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    output = row["benchmark_outputs"][split]
    payload = _read_benchmark(output)
    if payload is None:
        return {
            "row_name": row["name"],
            "engine": row["engine"],
            "split": split,
            "status": "missing",
            "benchmark_output": output,
        }
    status = str(payload.get("status", "completed"))
    result = {
        "row_name": row["name"],
        "engine": row["engine"],
        "split": split,
        "status": status,
        "benchmark_output": output,
    }
    for key in [
        "mae_e_mev_atom",
        "rmse_e_mev_atom",
        "max_abs_e_mev_atom",
        "mae_f_mev_a",
        "rmse_f_mev_a",
        "max_abs_f_mev_a",
        "atoms_per_second",
        "peak_memory_mb",
        "peak_reserved_mb",
        "peak_allocated_mb",
    ]:
        if key in payload:
            result[key] = payload[key]
    if "peak_memory_mb" not in result:
        if result.get("peak_reserved_mb") is not None:
            result["peak_memory_mb"] = result["peak_reserved_mb"]
        elif result.get("peak_allocated_mb") is not None:
            result["peak_memory_mb"] = result["peak_allocated_mb"]
    return result


def _sort_key(row: dict[str, Any]) -> tuple[int, str, float, str]:
    missing = 0 if row.get("status") == "completed" else 1
    split_rank = BENCHMARK_SPLITS.index(row["split"]) if row.get("split") in BENCHMARK_SPLITS else len(BENCHMARK_SPLITS)
    rmse = row.get("rmse_f_mev_a")
    return (missing, str(split_rank), float(rmse) if rmse is not None else float("inf"), str(row.get("row_name")))


def _render_summary_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage182 3BPA Conventional Closure Summary",
        "",
        "| row | engine | split | status | rmse_f_mev_a | rmse_e_mev_atom | max_abs_f_mev_a | atoms_per_second | peak_memory_mb |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        def fmt(key: str) -> str:
            value = row.get(key)
            if value is None:
                return ""
            if isinstance(value, (int, float)):
                return f"{float(value):.3f}"
            return str(value)
        lines.append(
            f"| {row['row_name']} | {row['engine']} | {row['split']} | {row['status']} | "
            f"{fmt('rmse_f_mev_a')} | {fmt('rmse_e_mev_atom')} | {fmt('max_abs_f_mev_a')} | {fmt('atoms_per_second')} | {fmt('peak_memory_mb')} |"
        )
    lines.append("")
    return "\n".join(lines)


def summarize_stage182_results(payload: dict[str, Any], *, write: bool = False) -> dict[str, Any]:
    rows = [_summary_row(row, split) for row in payload["rows"] for split in BENCHMARK_SPLITS]
    rows.sort(key=_sort_key)
    markdown = _render_summary_markdown(rows)
    summary = {
        "schema_version": "rtece_stage182_3bpa_closure_summary.v1",
        "stage": payload["stage"],
        "primary_ranking_metric": payload["comparison_contract"]["primary_ranking_metric"],
        "rows": rows,
        "markdown": markdown,
    }
    if write:
        json_path = Path(payload["artifacts"]["summary_json"])
        md_path = Path(payload["artifacts"]["summary_md"])
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        md_path.write_text(markdown, encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--train-limit-configs", type=int, default=500)
    parser.add_argument("--bench-limit-configs", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--nep-generations", type=int, default=20000)
    parser.add_argument("--deepmd-stop-batch", type=int, default=20000)
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = make_stage182_manifest(
        output_root=args.output_root,
        dataset_root=args.dataset_root,
        train_limit_configs=args.train_limit_configs,
        bench_limit_configs=args.bench_limit_configs,
        max_steps=args.max_steps,
        nep_generations=args.nep_generations,
        deepmd_stop_batch=args.deepmd_stop_batch,
    )
    if args.summarize_only:
        print(json.dumps(summarize_stage182_results(payload, write=True), indent=2, sort_keys=True))
    else:
        result = materialize_stage182(payload)
        summarize_stage182_results(payload, write=True)
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
