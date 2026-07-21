#!/usr/bin/env python3
"""Create Stage185 rMD17 rTECE representation-ladder scaffold.

Stage185 turns rMD17 from a Stage181 smoke dataset into an independent
conventional MD closure.  The goal is not to invent a new student here; it is
to run the same TECE semantic L0/L1/L2/T3 degradation ladder used on 3BPA under
a clean rMD17 unit-conversion and time-ordered split contract.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.oc20neb_tace_mace.prepare_stage181_conventional_datasets import KCAL_MOL_TO_EV
from benchmarks.oc20neb_tace_mace.make_rtece_stage183_3bpa_representation_ladder import CANDIDATE_SPECS

DEFAULT_OUTPUT_ROOT = Path("runs/oc20neb_tace_mace/rtece-stage185-rmd17-representation-ladder")
DEFAULT_RMD17_ROOT = Path("datasets/rMD17")
DEFAULT_PYTHON = Path("/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python")
RMD17_ARCHIVE_URL = "https://figshare.com/ndownloader/articles/12672038/versions/4"
DEFAULT_MOLECULE = "ethanol"
RMD17_SPLITS = ("train", "valid", "test")

FORBIDDEN_SBATCH_TOKENS = (
    "--export",
    "#SBATCH --mem",
    "--mem=",
    "#SBATCH --cpus-per-task",
    "--cpus-per-task",
    "set -u",
    "export ",
)


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _artifacts(root: Path) -> dict[str, str]:
    return {
        "manifest": str(root / "stage185_manifest.json"),
        "audit": str(root / "stage185_manifest_audit.json"),
        "stage_plan": str(root / "stage185_plan.md"),
        "summary_json": str(root / "stage185_summary.json"),
        "summary_md": str(root / "stage185_summary.md"),
        "prep_wrapper": str(root / "wrappers" / "stage185_rmd17_ethanol_prep_no_export.sbatch"),
        "wrapper_root": str(root / "wrappers"),
        "benchmark_wrapper_root": str(root / "benchmark_wrappers"),
        "results_root": str(root / "results"),
        "diagnostics_root": str(root / "diagnostics"),
    }


def _rmd17_split_paths(rmd17_root: Path, molecule: str) -> dict[str, str]:
    return {
        split: str(rmd17_root / "converted_extxyz" / f"rmd17_{molecule}_{split}.extxyz")
        for split in RMD17_SPLITS
    }


def _row_outputs(root: Path, row_name: str) -> dict[str, str]:
    return {split: str(root / "diagnostics" / row_name / f"{split}_benchmark.json") for split in RMD17_SPLITS}


def _stage185_row_name(stage183_name: str, molecule: str) -> str:
    suffix = stage183_name.removeprefix("stage183_")
    return f"stage185_rmd17_{suffix}" if molecule == DEFAULT_MOLECULE else f"stage185_rmd17_{molecule}_{suffix}"


def _candidate_rows(root: Path, molecule: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    previous: set[str] = set()
    for spec in CANDIDATE_SPECS:
        paths = tuple(str(path_id) for path_id in spec["scalar_path_ids"])
        cfg = {
            "scalar_path_ids": list(paths),
            "num_radial": 10,
            "hidden_channels": "64,64",
            "species_basis_channels": 16,
            "species_basis_mode": "learnable_embedding",
            "local_l0_chemistry_rank": 4,
            "moment_l_max": int(spec["moment_l_max"]),
            "atomic_cross_radial_sketch_channels": int(spec["atomic_cross_radial_sketch_channels"]),
            "atomic_cross_radial_projection": str(spec["atomic_cross_radial_projection"]),
        }
        name = _stage185_row_name(str(spec["name"]), molecule)
        rows.append({
            "name": name,
            "engine": "rtece",
            "dataset_role": "single_molecule_rmd17_temporal_generalization",
            "source_stage183_row": str(spec["name"]),
            "tece_tier": str(spec["tece_tier"]),
            "representation_step": str(spec["representation_step"]),
            "capacity_allocation": str(spec["capacity_allocation"]),
            "rationale": str(spec["rationale"]),
            "marginal_paths": [path_id for path_id in paths if path_id not in previous],
            "cost_proxy": {
                "scalar_paths": len(paths),
                "atomic_paths": sum(path_id.startswith("atomic.") for path_id in paths),
                "edge_paths": sum(path_id.startswith("edge.") for path_id in paths),
                "moment_l_max": int(spec["moment_l_max"]),
                "num_radial": cfg["num_radial"],
                "species_basis_channels": cfg["species_basis_channels"],
                "local_l0_chemistry_rank": cfg["local_l0_chemistry_rank"],
                "atomic_cross_radial_sketch_channels": cfg["atomic_cross_radial_sketch_channels"],
            },
            "student_config": cfg,
            "train_dir": str(root / "results" / name),
            "wrapper": str(root / "wrappers" / f"{name}_no_export.sbatch"),
            "benchmark_wrapper": str(root / "benchmark_wrappers" / f"{name}_benchmark_no_export.sbatch"),
            "model_artifact": str(root / "results" / name / "rtece_scalar_best.pt"),
            "benchmark_outputs": _row_outputs(root, name),
        })
        previous = set(paths)
    return rows


def make_stage185_manifest(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    rmd17_root: str | Path = DEFAULT_RMD17_ROOT,
    molecule: str = DEFAULT_MOLECULE,
    train_count: int = 1000,
    valid_count: int = 1000,
    test_count: int = 1000,
    max_steps: int = 20000,
    bench_limit_configs: int = 1000,
) -> dict[str, Any]:
    counts = {"train_count": int(train_count), "valid_count": int(valid_count), "test_count": int(test_count)}
    if any(value < 1 for value in counts.values()):
        raise ValueError("rMD17 split counts must be positive")
    if counts["train_count"] > 1000:
        raise ValueError("rMD17 train_count should not exceed the PyG-recommended 1000 correlated-sample ceiling")
    root = Path(output_root)
    rmd17 = Path(rmd17_root)
    split_paths = _rmd17_split_paths(rmd17, molecule)
    artifacts = _artifacts(root)
    return {
        "schema_version": "rtece_stage185_rmd17_representation_ladder.v1",
        "stage": "stage185_rmd17_representation_ladder",
        "scientific_question": (
            "Does the 3BPA-derived TECE semantic representation ladder behave consistently on a conventional "
            "single-molecule rMD17 trajectory once units and temporal splits are controlled?"
        ),
        "theory_alignment": [
            "TECE_design_space.md §11/§12: compare a tiered student family by physical error and hardware cost across deployment distributions.",
            "TECE_design_space.md §5: keep one low-order moment pass and scalarize sparse high-value paths before expensive persistent equivariant state.",
            "rTECE_review.md P1: path IDs, held-out deployment splits, and comparable supervised/distilled students are prerequisites for real renormalization.",
            "Stage184 result: L2 is the current 3BPA rTECE Pareto candidate; rMD17 checks whether that conclusion transfers to conventional molecular MD.",
            "PyG MD17 documentation: original/revised MD17 energy and force labels are kcal/mol and kcal/mol/A, and highly correlated trajectories should use no more than about 1000 training samples.",
        ],
        "dataset": {
            "name": "rMD17",
            "molecule": str(molecule),
            "root": str(rmd17),
            "source": {
                "figshare_article": "12672038",
                "version": 4,
                "archive_url": RMD17_ARCHIVE_URL,
                "source_npz": str(rmd17 / f"rmd17_{molecule}.npz"),
            },
            "source_units": {"energy": "kcal/mol", "forces": "kcal/mol/A", "distance": "A"},
            "target_units": {"energy": "eV", "forces": "eV/A", "distance": "A"},
            "conversion_factor_energy": float(KCAL_MOL_TO_EV),
            "conversion_factor_forces": float(KCAL_MOL_TO_EV),
            "labels": {"energy_key": "energy", "forces_key": "forces"},
            "converted_splits": split_paths,
        },
        "split_policy": {
            **counts,
            "indexing": "contiguous_time_ordered_blocks",
            "rationale": (
                "Use explicit contiguous blocks for the first rMD17 closure so train/valid/test membership is reproducible "
                "and does not silently mix kcal-source frames after conversion.  The train block respects the documented "
                "1000-sample ceiling for highly correlated rMD17 trajectories."
            ),
        },
        "train_contract": {
            "train_file": split_paths["train"],
            "valid_file": split_paths["valid"],
            "energy_key": "energy",
            "forces_key": "forces",
            "limit_configs": int(train_count),
            "valid_limit_configs": int(valid_count),
            "bench_limit_configs": int(bench_limit_configs),
            "max_steps": int(max_steps),
            "rtece_entrypoint": "tace.scripts.rtece_train_scalar",
            "trainer_backend": "lightning",
            "batch_size": 32,
            "lr": 0.001,
            "lr_scheduler": "plateau",
            "lr_warmup_steps": 500,
            "early_stopping_patience": 250,
            "energy_weight": 1.0,
            "force_weight": 10.0,
            "neighborlist_backend": "matscipy",
            "per_element_e0_fit": True,
        },
        "comparison_contract": {
            "primary_ranking_metric": "rmse_f_mev_a",
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
            "benchmark_splits": list(RMD17_SPLITS),
            "interpretation": (
                "If the L2-vs-T3 ordering matches 3BPA, the representation-ladder conclusion is dataset-robust; "
                "if rMD17 prefers a lower tier, 3BPA dihedral behavior is a stronger angular/relational stress test than standard molecular MD."
            ),
            "head_capacity_control": "hidden_channels fixed at 64,64 across rows",
        },
        "rows": _candidate_rows(root, str(molecule)),
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
    p.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return str(p)


def _path_csv(paths: list[str]) -> str:
    return ",".join(paths)


def _write_prep_wrapper(payload: dict[str, Any]) -> str:
    dataset = payload["dataset"]
    split = payload["split_policy"]
    rmd17_root = dataset["root"]
    molecule = dataset["molecule"]
    source_npz = dataset["source"]["source_npz"]
    archive = str(Path(rmd17_root) / "rMD17_v4.zip")
    lines = _header("rtece-st185-rmd17-prep", time_limit="03:55:00")
    lines.extend([
        f"RMD17_ROOT={shlex.quote(rmd17_root)}",
        f"RMD17_ARCHIVE_URL={shlex.quote(dataset['source']['archive_url'])}",
        f"RMD17_ARCHIVE={shlex.quote(archive)}",
        f"RMD17_NPZ={shlex.quote(source_npz)}",
        "mkdir -p ${RMD17_ROOT} ${RMD17_ROOT}/converted_extxyz",
        "if [ ! -f ${RMD17_NPZ} ]; then",
        "  if [ ! -f ${RMD17_ARCHIVE} ]; then",
        "    curl -L ${RMD17_ARCHIVE_URL} -o ${RMD17_ARCHIVE}",
        "  fi",
        f"  unzip -j -o ${{RMD17_ARCHIVE}} rmd17_{shlex.quote(molecule)}.npz -d ${{RMD17_ROOT}}",
        "fi",
        "PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/prepare_stage181_conventional_datasets.py rmd17-npz-to-splits "
        f"--source-npz {shlex.quote(source_npz)} --output-dir {shlex.quote(str(Path(rmd17_root) / 'converted_extxyz'))} "
        f"--molecule {shlex.quote(molecule)} --train-count {int(split['train_count'])} --valid-count {int(split['valid_count'])} --test-count {int(split['test_count'])} "
        f"--summary-json {shlex.quote(str(Path(payload['artifacts']['diagnostics_root']) / 'rmd17_split_conversion_summary.json'))}",
    ])
    return _write(payload["artifacts"]["prep_wrapper"], lines)


def _common_train_args(row: dict[str, Any], payload: dict[str, Any]) -> str:
    cfg = row["student_config"]
    train = payload["train_contract"]
    return (
        f"--variant {shlex.quote(row['name'])} "
        f"--scalar-path-ids {shlex.quote(_path_csv(cfg['scalar_path_ids']))} "
        f"--train-file {shlex.quote(train['train_file'])} "
        f"--valid-file {shlex.quote(train['valid_file'])} "
        f"--output-dir {shlex.quote(row['train_dir'])} "
        f"--max-steps {int(train['max_steps'])} "
        f"--limit-configs {int(train['limit_configs'])} --valid-limit-configs {int(train['valid_limit_configs'])} "
        f"--trainer-backend {shlex.quote(train['trainer_backend'])} --batch-size {int(train['batch_size'])} "
        f"--num-radial {int(cfg['num_radial'])} "
        f"--hidden-channels {shlex.quote(cfg['hidden_channels'])} "
        f"--species-basis-channels {int(cfg['species_basis_channels'])} "
        f"--species-basis-mode {shlex.quote(cfg['species_basis_mode'])} "
        f"--local-l0-chemistry-rank {int(cfg['local_l0_chemistry_rank'])} "
        f"--moment-l-max {int(cfg['moment_l_max'])} "
        f"--atomic-cross-radial-sketch-channels {int(cfg['atomic_cross_radial_sketch_channels'])} "
        f"--atomic-cross-radial-projection {shlex.quote(cfg['atomic_cross_radial_projection'])} "
        f"--energy-key {shlex.quote(train['energy_key'])} --forces-key {shlex.quote(train['forces_key'])} "
        f"--energy-weight {float(train['energy_weight'])} --force-weight {float(train['force_weight'])} "
        f"--lr {float(train['lr'])} --lr-scheduler {shlex.quote(train['lr_scheduler'])} "
        f"--lr-warmup-steps {int(train['lr_warmup_steps'])} --early-stopping-patience {int(train['early_stopping_patience'])} "
        f"--neighborlist-backend {shlex.quote(train['neighborlist_backend'])} --device cuda --default-dtype float32 --no-progress-bar"
    )


def _write_train_wrapper(row: dict[str, Any], payload: dict[str, Any]) -> str:
    lines = _header(f"rtece-st185-{row['representation_step']}", time_limit="04:55:00")
    lines.extend([
        f"mkdir -p {shlex.quote(row['train_dir'])}",
        "PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} -m tace.scripts.rtece_train_scalar " + _common_train_args(row, payload),
    ])
    return _write(row["wrapper"], lines)


def _write_benchmark_wrapper(row: dict[str, Any], payload: dict[str, Any]) -> str:
    train = payload["train_contract"]
    split_paths = payload["dataset"]["converted_splits"]
    out_root = Path(payload["artifacts"]["diagnostics_root"]) / row["name"]
    lines = _header(f"rtece-st185-bench-{row['representation_step']}", time_limit="01:40:00")
    lines.append(f"mkdir -p {shlex.quote(str(out_root))}")
    for split in RMD17_SPLITS:
        lines.append(
            "PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py "
            f"--model {shlex.quote(row['model_artifact'])} --configs {shlex.quote(split_paths[split])} "
            f"--output {shlex.quote(row['benchmark_outputs'][split])} --variant {shlex.quote(row['name'] + '_' + split)} "
            f"--energy-key {shlex.quote(train['energy_key'])} --forces-key {shlex.quote(train['forces_key'])} "
            f"--start-config 0 --limit-configs {int(train['bench_limit_configs'])} --measure-passes 5 "
            "--device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist"
        )
    return _write(row["benchmark_wrapper"], lines)


def _write_stage_plan(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage185 rMD17 Representation Ladder",
        "",
        "Goal: promote rMD17 from smoke data prep to an independent conventional MD closure for the Stage183 TECE semantic ladder.",
        "",
        "## Dataset Contract",
        "",
        f"- molecule: `{payload['dataset']['molecule']}`",
        "- source labels are kcal/mol and kcal/mol/A; converted extxyz labels are eV and eV/A.",
        "- train/valid/test are contiguous time-ordered blocks, with train count capped at 1000 for the first closure.",
        "",
        "## Document Alignment",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["theory_alignment"])
    lines.extend(["", "## Ladder", ""])
    for row in payload["rows"]:
        lines.append(f"- `{row['name']}`: {row['representation_step']}; marginal paths `{', '.join(row['marginal_paths'])}`")
    lines.extend([
        "",
        "## Decision Rule",
        "",
        "- If L2 remains the best accuracy/throughput tradeoff, the Stage184 conclusion transfers beyond 3BPA.",
        "- If T3 wins rMD17 but not 3BPA, edge-relational sketches are dataset/task dependent rather than uniformly Pareto-optimal.",
        "- If all tiers are close on rMD17, rMD17 is a weaker representation stress test and should mainly validate data/engine correctness.",
        "",
    ])
    return _write(payload["artifacts"]["stage_plan"], lines)


def audit_stage185_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    rows = list(payload.get("rows") or [])
    wrapper_paths = [payload.get("artifacts", {}).get("prep_wrapper", "")]
    wrapper_paths += [str(row.get("wrapper", "")) for row in rows]
    wrapper_paths += [str(row.get("benchmark_wrapper", "")) for row in rows]
    wrapper_text = ""
    for wrapper in wrapper_paths:
        p = Path(wrapper)
        if p.exists():
            wrapper_text += p.read_text(encoding="utf-8") + "\n"
    row_names = [row.get("name") for row in rows]
    expected_names = [
        "stage185_rmd17_l0_local_species",
        "stage185_rmd17_l1_cross",
        "stage185_rmd17_l2_atomic_quadrupole",
        "stage185_rmd17_t3_cavity_vecq",
    ]
    nested = True
    previous: set[str] = set()
    for row in rows:
        paths = set((row.get("student_config") or {}).get("scalar_path_ids") or [])
        nested = nested and previous <= paths
        previous = paths
    checks = {
        "schema_version": payload.get("schema_version") == "rtece_stage185_rmd17_representation_ladder.v1",
        "stage": payload.get("stage") == "stage185_rmd17_representation_ladder",
        "dataset": (payload.get("dataset") or {}).get("name") == "rMD17",
        "molecule": (payload.get("dataset") or {}).get("molecule") == DEFAULT_MOLECULE,
        "source_units": (payload.get("dataset") or {}).get("source_units") == {"energy": "kcal/mol", "forces": "kcal/mol/A", "distance": "A"},
        "target_units": (payload.get("dataset") or {}).get("target_units") == {"energy": "eV", "forces": "eV/A", "distance": "A"},
        "train_count_ceiling": (payload.get("split_policy") or {}).get("train_count", 10**9) <= 1000,
        "contiguous_splits": (payload.get("split_policy") or {}).get("indexing") == "contiguous_time_ordered_blocks",
        "row_names": row_names == expected_names,
        "nested_paths": nested,
        "fixed_head": all((row.get("student_config") or {}).get("hidden_channels") == "64,64" for row in rows),
        "no_forbidden_sbatch_tokens": not any(token in wrapper_text for token in FORBIDDEN_SBATCH_TOKENS),
        "prep_command": "rmd17-npz-to-splits" in wrapper_text and RMD17_ARCHIVE_URL in wrapper_text,
        "train_and_benchmark": "tace.scripts.rtece_train_scalar" in wrapper_text and "benchmark_rtece_scalar.py" in wrapper_text,
        "metric_contract": "rmse_f_mev_a" == (payload.get("comparison_contract") or {}).get("primary_ranking_metric"),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "schema_version": "rtece_stage185_rmd17_representation_ladder_audit.v1",
        "contract_pass": not failed,
        "failed_checks": failed,
        "checks": checks,
        "no_forbidden_sbatch_tokens": checks["no_forbidden_sbatch_tokens"],
    }


def materialize_stage185(payload: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(payload["artifacts"]["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    prep_wrapper = _write_prep_wrapper(payload)
    wrappers = {row["name"]: _write_train_wrapper(row, payload) for row in payload["rows"]}
    benchmark_wrappers = {row["name"]: _write_benchmark_wrapper(row, payload) for row in payload["rows"]}
    stage_plan = _write_stage_plan(payload)
    audit = audit_stage185_manifest(payload)
    if not audit["contract_pass"]:
        raise ValueError(f"Stage185 manifest audit failed: {audit['failed_checks']}")
    audit_path = Path(payload["artifacts"]["audit"])
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summarize_stage185_results(payload, write=True)
    return {
        "manifest": str(manifest_path),
        "audit": audit,
        "audit_path": str(audit_path),
        "stage_plan": stage_plan,
        "prep_wrapper": prep_wrapper,
        "wrappers": wrappers,
        "benchmark_wrappers": benchmark_wrappers,
    }


def _read_json(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _benchmark_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    output = row["benchmark_outputs"][split]
    payload = _read_json(output)
    result = {
        "row_name": row["name"],
        "split": split,
        "engine": row["engine"],
        "status": "missing",
        "benchmark_output": output,
        "tece_tier": row["tece_tier"],
        "scalar_paths": row["cost_proxy"]["scalar_paths"],
        "edge_paths": row["cost_proxy"]["edge_paths"],
        "moment_l_max": row["cost_proxy"]["moment_l_max"],
    }
    if payload is None:
        return result
    result["status"] = str(payload.get("status", "completed"))
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
        result[key] = _finite_float(payload.get(key))
    if result.get("peak_memory_mb") is None:
        result["peak_memory_mb"] = result.get("peak_reserved_mb") or result.get("peak_allocated_mb")
    return result


def _sort_key(row: dict[str, Any]) -> tuple[int, str, float, float, str]:
    missing = 0 if row.get("status") == "completed" else 1
    split_order = {"test": "0", "valid": "1", "train": "2"}.get(str(row.get("split")), "9")
    f = row.get("rmse_f_mev_a")
    speed = row.get("atoms_per_second")
    return (missing, split_order, float(f) if f is not None else float("inf"), -(float(speed) if speed is not None else 0.0), str(row.get("row_name")))


def _render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage185 rMD17 Representation Ladder Summary",
        "",
        "| row | split | status | rmse_f_mev_a | rmse_e_mev_atom | max_abs_f_mev_a | max_abs_e_mev_atom | atoms/s | peak_memory_mb |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
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
            f"| {row['row_name']} | {row['split']} | {row['status']} | {fmt('rmse_f_mev_a')} | {fmt('rmse_e_mev_atom')} | "
            f"{fmt('max_abs_f_mev_a')} | {fmt('max_abs_e_mev_atom')} | {fmt('atoms_per_second')} | {fmt('peak_memory_mb')} |"
        )
    lines.extend(["", "Primary ranking is force RMSE; energy RMSE/max and force max remain required diagnostics.", ""])
    return "\n".join(lines)


def summarize_stage185_results(payload: dict[str, Any], *, write: bool = False) -> dict[str, Any]:
    rows = [_benchmark_row(row, split) for row in payload["rows"] for split in RMD17_SPLITS]
    rows.sort(key=_sort_key)
    markdown = _render_markdown(rows)
    summary = {
        "schema_version": "rtece_stage185_rmd17_representation_ladder_summary.v1",
        "stage": payload["stage"],
        "dataset": payload["dataset"],
        "split_policy": payload["split_policy"],
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
    parser.add_argument("--rmd17-root", type=Path, default=DEFAULT_RMD17_ROOT)
    parser.add_argument("--molecule", default=DEFAULT_MOLECULE)
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--valid-count", type=int, default=1000)
    parser.add_argument("--test-count", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--bench-limit-configs", type=int, default=1000)
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = make_stage185_manifest(
        output_root=args.output_root,
        rmd17_root=args.rmd17_root,
        molecule=args.molecule,
        train_count=args.train_count,
        valid_count=args.valid_count,
        test_count=args.test_count,
        max_steps=args.max_steps,
        bench_limit_configs=args.bench_limit_configs,
    )
    if args.summarize_only:
        print(json.dumps(summarize_stage185_results(payload, write=True), indent=2, sort_keys=True))
        return
    result = materialize_stage185(payload)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
