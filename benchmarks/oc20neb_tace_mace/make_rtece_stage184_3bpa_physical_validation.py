#!/usr/bin/env python3
"""Create Stage184 3BPA physical-validation scaffold.

Stage184 follows the Stage183 representation ladder: L2 atomic quadrupoles are
now the current rTECE Pareto candidate, while T3 edge-relational paths are more
expensive and not clearly better on the dihedral split.  This stage adds the
physical diagnostics requested in the TECE/rTECE plan: dimer scans and
rattle-relax RMSD, treated as continuous external metrics rather than hard
binary gates.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_ROOT = Path("runs/oc20neb_tace_mace/rtece-stage184-3bpa-physical-validation")
DEFAULT_DATASET_ROOT = Path("datasets/3BPA/dataset_3BPA")
DEFAULT_STAGE183_ROOT = Path("runs/oc20neb_tace_mace/rtece-stage183-3bpa-representation-ladder")
DEFAULT_STAGE182_ROOT = Path("runs/oc20neb_tace_mace/rtece-stage182-3bpa-closure")
DEFAULT_PYTHON = Path("/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python")

FORBIDDEN_SBATCH_TOKENS = (
    "--export",
    "#SBATCH --mem",
    "--mem=",
    "#SBATCH --cpus-per-task",
    "--cpus-per-task",
    "set -u",
    "export ",
)
DIMER_PAIRS = ("C-N", "C-O", "C-H", "N-H", "O-H", "C-C", "N-N")
RTECE_ROWS = (
    "stage183_l0_local_species",
    "stage183_l1_cross",
    "stage183_l2_atomic_quadrupole",
    "stage183_t3_cavity_vecq",
)


def _artifacts(root: Path) -> dict[str, str]:
    return {
        "manifest": str(root / "stage184_manifest.json"),
        "audit": str(root / "stage184_manifest_audit.json"),
        "stage_plan": str(root / "stage184_plan.md"),
        "summary_json": str(root / "stage184_summary.json"),
        "summary_md": str(root / "stage184_summary.md"),
        "wrapper_root": str(root / "wrappers"),
        "diagnostics_root": str(root / "diagnostics"),
    }


def _read_json(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _stage183_decision(stage183_root: str | Path = DEFAULT_STAGE183_ROOT) -> dict[str, Any]:
    summary_path = Path(stage183_root) / "stage183_summary.json"
    summary = _read_json(summary_path)
    if summary is None:
        return {
            "source_summary": str(summary_path),
            "status": "missing_stage183_summary",
            "current_pareto_candidate": "stage183_l2_atomic_quadrupole",
            "reason": "Use the document-aligned L2 candidate as the physical-validation focus until summary is available.",
        }
    rows = [row for row in summary.get("rows", []) if row.get("status") == "completed"]
    by_row_split = {(row.get("row_name"), row.get("split")): row for row in rows}
    l2_dih = by_row_split.get(("stage183_l2_atomic_quadrupole", "test_dih"), {})
    t3_dih = by_row_split.get(("stage183_t3_cavity_vecq", "test_dih"), {})
    l2_speed = _finite_float(l2_dih.get("atoms_per_second"))
    t3_speed = _finite_float(t3_dih.get("atoms_per_second"))
    return {
        "source_summary": str(summary_path),
        "status": "completed",
        "current_pareto_candidate": "stage183_l2_atomic_quadrupole",
        "reason": (
            "L2 gives the best Stage183 dihedral force RMSE among rTECE rows while keeping much higher throughput than T3; "
            "T3 remains an expensive edge-relational diagnostic row."
        ),
        "stage183_l2_test_dih_rmse_f_mev_a": l2_dih.get("rmse_f_mev_a"),
        "stage183_t3_test_dih_rmse_f_mev_a": t3_dih.get("rmse_f_mev_a"),
        "stage183_l2_test_dih_atoms_per_second": l2_speed,
        "stage183_t3_test_dih_atoms_per_second": t3_speed,
    }


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _rtece_row(root: Path, stage183_root: Path, name: str) -> dict[str, Any]:
    diag = root / "diagnostics" / name
    return {
        "name": name,
        "engine": "rtece",
        "model_artifact": str(stage183_root / "results" / name / "rtece_scalar_best.pt"),
        "benchmark_outputs": {
            "test_300K": str(stage183_root / "diagnostics" / name / "test_300K_benchmark.json"),
            "test_dih": str(stage183_root / "diagnostics" / name / "test_dih_benchmark.json"),
        },
        "wrapper": str(root / "wrappers" / f"{name}_physical_no_export.sbatch"),
        "run_dir": str(diag),
        "dimer_json": str(diag / f"{name}_dimer.json"),
        "dimer_md": str(diag / f"{name}_dimer.md"),
        "rattle_json": str(diag / f"{name}_rattle.json"),
        "rattle_md": str(diag / f"{name}_rattle.md"),
        "selection_basis": "Stage183 representation-ladder row for physical external validation.",
    }


def _nep_row(root: Path, stage182_root: Path) -> dict[str, Any]:
    name = "nep4_train300k"
    diag = root / "diagnostics" / name
    return {
        "name": name,
        "engine": "nep",
        "module": "gpumd/4.8-cuda12.4",
        "model_artifact": str(stage182_root / "results" / name / "nep.txt"),
        "benchmark_outputs": {
            "test_300K": str(stage182_root / "diagnostics" / name / "test_300K_benchmark.json"),
            "test_dih": str(stage182_root / "diagnostics" / name / "test_dih_benchmark.json"),
        },
        "wrapper": str(root / "wrappers" / f"{name}_physical_no_export.sbatch"),
        "run_dir": str(diag / "nep_prediction_work"),
        "community_physical_json": str(diag / f"{name}_community_physical.json"),
        "selection_basis": "Stage182 matched NEP4 high-throughput community baseline for physical external comparison.",
    }


def make_stage184_manifest(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    stage183_root: str | Path = DEFAULT_STAGE183_ROOT,
    stage182_root: str | Path = DEFAULT_STAGE182_ROOT,
    dimer_num_points: int = 16,
    rattle_start_config: int = 0,
    rattle_limit_configs: int = 8,
    rattle_max_steps: int = 10,
) -> dict[str, Any]:
    if int(dimer_num_points) < 2:
        raise ValueError("dimer_num_points must be at least 2")
    if int(rattle_limit_configs) < 1:
        raise ValueError("rattle_limit_configs must be positive")
    root = Path(output_root)
    stage183 = Path(stage183_root)
    stage182 = Path(stage182_root)
    rows = [_rtece_row(root, stage183, name) for name in RTECE_ROWS]
    rows.append(_nep_row(root, stage182))
    artifacts = _artifacts(root)
    return {
        "schema_version": "rtece_stage184_3bpa_physical_validation.v1",
        "stage": "stage184_3bpa_physical_validation",
        "scientific_question": (
            "Does the Stage183 L2 rTECE Pareto candidate retain physical smoothness and rattle-relax stability, "
            "and how does it compare with L0/L1/T3 and NEP4 on continuous physical diagnostics?"
        ),
        "theory_alignment": [
            "TECE_design_space.md: deployment distribution, physical error metrics, and hardware cost jointly define the student projection target.",
            "TECE_design_space.md §11: L_A=2/3 scalarized students should be evaluated by physical error, not only split RMSE.",
            "rTECE_review.md P0/P1: physical diagnostics and package/API correctness are higher priority than kernel-only speedups.",
            "User requirement: dimer scans from 0.5 to 5.0 covalent-radius scale and rattle-relax RMSD are external physical tests, with rattle treated as continuous RMSD rather than a binary gate.",
        ],
        "dataset": {
            "name": "3BPA",
            "root": str(dataset_root),
            "units": {"energy": "eV", "forces": "eV/A", "distance": "A"},
            "rattle_configs": str(Path(dataset_root) / "test_300K.xyz"),
        },
        "stage183_decision": _stage183_decision(stage183),
        "physical_contract": {
            "dimer_pairs": list(DIMER_PAIRS),
            "dimer_num_points": int(dimer_num_points),
            "dimer_distance_scales": {"min": 0.5, "max": 5.0},
            "rattle_policy": "continuous_rmsd_not_binary_gate",
            "rattle_configs": "test_300K.xyz",
            "rattle_start_config": int(rattle_start_config),
            "rattle_limit_configs": int(rattle_limit_configs),
            "rattle_std_a": 0.05,
            "rattle_max_steps": int(rattle_max_steps),
            "rattle_fmax_ev_a": 0.05,
            "primary_numeric_anchors": ["test_300K", "test_dih"],
        },
        "rows": rows,
        "artifacts": artifacts,
    }


def _header(job_name: str, *, time_limit: str = "02:30:00") -> list[str]:
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
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1",
        "TORCH_CUDA_ARCH_LIST=7.0",
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


def _pairs_arg(payload: dict[str, Any]) -> str:
    return " ".join(shlex.quote(pair) for pair in payload["physical_contract"]["dimer_pairs"])


def _write_rtece_wrapper(row: dict[str, Any], payload: dict[str, Any]) -> str:
    phys = payload["physical_contract"]
    configs = payload["dataset"]["rattle_configs"]
    lines = _header(f"rtece-st184-{row['name']}")
    lines.append(f"mkdir -p {shlex.quote(row['run_dir'])}")
    lines.append(
        "PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} ${TACE_PYTHON} "
        "benchmarks/oc20neb_tace_mace/dimer_scan_rtece.py "
        f"--checkpoint {shlex.quote(row['model_artifact'])} --output-json {shlex.quote(row['dimer_json'])} --output-md {shlex.quote(row['dimer_md'])} "
        f"--route {shlex.quote(row['name'])} --pairs {_pairs_arg(payload)} --num-points {int(phys['dimer_num_points'])} "
        "--min-scale 0.5 --max-scale 5.0 --device cuda --default-dtype float32 --force-mode autograd --neighborlist-backend ase"
    )
    lines.append(
        "PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=${TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD} TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} ${TACE_PYTHON} "
        "benchmarks/oc20neb_tace_mace/rattle_relax_rtece.py "
        f"--checkpoint {shlex.quote(row['model_artifact'])} --configs {shlex.quote(configs)} "
        f"--output-json {shlex.quote(row['rattle_json'])} --output-md {shlex.quote(row['rattle_md'])} --route {shlex.quote(row['name'])} "
        f"--start-config {int(phys['rattle_start_config'])} --limit-configs {int(phys['rattle_limit_configs'])} "
        f"--rattle-std {float(phys['rattle_std_a'])} --rattle-seed 20260721 --fmax {float(phys['rattle_fmax_ev_a'])} "
        f"--max-steps {int(phys['rattle_max_steps'])} --device cuda --default-dtype float32 --force-mode autograd --neighborlist-backend matscipy"
    )
    return _write(row["wrapper"], lines)


def _write_nep_wrapper(row: dict[str, Any], payload: dict[str, Any]) -> str:
    phys = payload["physical_contract"]
    configs = payload["dataset"]["rattle_configs"]
    lines = _header("rtece-st184-nep4-phys", time_limit="03:55:00")
    lines.append("module load gpumd/4.8-cuda12.4")
    lines.append(f"mkdir -p {shlex.quote(str(Path(row['community_physical_json']).parent))} {shlex.quote(row['run_dir'])}")
    lines.append(
        "PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/physical_stage145_community.py "
        f"--engine nep --model-artifact {shlex.quote(row['model_artifact'])} --configs {shlex.quote(configs)} "
        f"--output-json {shlex.quote(row['community_physical_json'])} --row-name {shlex.quote(row['name'])} --run-dir {shlex.quote(row['run_dir'])} "
        f"--pairs {_pairs_arg(payload)} --dimer-points {int(phys['dimer_num_points'])} --min-scale 0.5 --max-scale 5.0 "
        f"--start-config {int(phys['rattle_start_config'])} --limit-configs {int(phys['rattle_limit_configs'])} "
        f"--rattle-std {float(phys['rattle_std_a'])} --rattle-seed 20260721 --fmax {float(phys['rattle_fmax_ev_a'])} "
        f"--max-steps {int(phys['rattle_max_steps'])} --device cuda"
    )
    return _write(row["wrapper"], lines)


def _write_wrapper(row: dict[str, Any], payload: dict[str, Any]) -> str:
    if row["engine"] == "rtece":
        return _write_rtece_wrapper(row, payload)
    if row["engine"] == "nep":
        return _write_nep_wrapper(row, payload)
    raise ValueError(f"unsupported Stage184 engine {row['engine']!r}")


def _write_stage_plan(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage184 3BPA Physical Validation",
        "",
        "Goal: test whether the Stage183 numeric Pareto result survives external physical diagnostics.",
        "",
        "## Policy",
        "",
        "- Rattle-relax is read as continuous RMSD and force-tail evidence, not as a long-term binary gate.",
        "- Dimer scan covers 0.5 to 5.0 covalent-radius scale for the same CHNO element pairs across rTECE and NEP.",
        "- L2 is the current rTECE Pareto candidate; T3 remains a cost/edge-relational diagnostic row.",
        "",
        "## Rows",
        "",
    ]
    for row in payload["rows"]:
        lines.append(f"- `{row['name']}` ({row['engine']}): `{row['selection_basis']}`")
    lines.append("")
    return _write(payload["artifacts"]["stage_plan"], lines)


def _dimer_metrics(pair_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(pair_summaries)
    if total == 0:
        return {
            "dimer_num_pairs": 0,
            "dimer_short_repulsive_fraction": None,
            "dimer_nonfinite_pair_count": None,
            "dimer_min_short_energy_lift_eV": None,
        }
    repulsive = 0
    nonfinite = 0
    lifts: list[float] = []
    for item in pair_summaries:
        summary = item.get("summary") or {}
        if bool(summary.get("short_force_repulsive")):
            repulsive += 1
        if bool(summary.get("has_nonfinite")):
            nonfinite += 1
        lift = _finite_float(summary.get("short_minus_long_energy_eV"))
        if lift is not None:
            lifts.append(lift)
    return {
        "dimer_num_pairs": int(total),
        "dimer_short_repulsive_fraction": float(repulsive / total),
        "dimer_nonfinite_pair_count": int(nonfinite),
        "dimer_min_short_energy_lift_eV": min(lifts) if lifts else None,
    }


def _rattle_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "rattle_num_configs": summary.get("num_configs"),
        "rattle_converged_fraction": _finite_float(summary.get("converged_fraction")),
        "rattle_mean_final_rmsd_a": _finite_float(summary.get("mean_final_rmsd_a")),
        "rattle_max_final_rmsd_a": _finite_float(summary.get("max_final_rmsd_a")),
        "rattle_max_fmax_ev_a": _finite_float(summary.get("max_fmax_ev_a")),
    }


def _benchmark_metrics(row: dict[str, Any], split: str) -> dict[str, Any]:
    data = _read_json((row.get("benchmark_outputs") or {}).get(split, "")) or {}
    return {
        f"{split}_rmse_f_mev_a": _finite_float(data.get("rmse_f_mev_a")),
        f"{split}_rmse_e_mev_atom": _finite_float(data.get("rmse_e_mev_atom")),
        f"{split}_max_abs_f_mev_a": _finite_float(data.get("max_abs_f_mev_a")),
        f"{split}_max_abs_e_mev_atom": _finite_float(data.get("max_abs_e_mev_atom")),
        f"{split}_atoms_per_second": _finite_float(data.get("atoms_per_second")),
        f"{split}_peak_memory_mb": _finite_float(data.get("peak_memory_mb") or data.get("peak_reserved_mb") or data.get("peak_allocated_mb")),
    }


def _summary_row(row: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "row_name": row["name"],
        "engine": row["engine"],
        "status": "missing",
    }
    result.update(_benchmark_metrics(row, "test_300K"))
    result.update(_benchmark_metrics(row, "test_dih"))
    if row["engine"] == "rtece":
        dimer = _read_json(row["dimer_json"])
        rattle = _read_json(row["rattle_json"])
        if dimer is None or rattle is None:
            return result
        result["status"] = "completed"
        result.update(_dimer_metrics(list(dimer.get("pair_summaries") or [])))
        result.update(_rattle_metrics(dict(rattle.get("summary") or {})))
        result["dimer_json"] = row["dimer_json"]
        result["rattle_json"] = row["rattle_json"]
        return result
    if row["engine"] == "nep":
        payload = _read_json(row["community_physical_json"])
        if payload is None:
            return result
        result["status"] = str(payload.get("status", "completed"))
        result.update(_dimer_metrics(list(((payload.get("dimer_scan") or {}).get("pair_summaries")) or [])))
        result.update(_rattle_metrics(dict(((payload.get("rattle_relax") or {}).get("summary")) or {})))
        result["community_physical_json"] = row["community_physical_json"]
        return result
    return result


def _sort_key(row: dict[str, Any]) -> tuple[int, float, float, str]:
    missing = 0 if row.get("status") == "completed" else 1
    rmsd = row.get("rattle_mean_final_rmsd_a")
    force = row.get("test_dih_rmse_f_mev_a")
    return (missing, float(rmsd) if rmsd is not None else float("inf"), float(force) if force is not None else float("inf"), str(row.get("row_name")))


def _render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage184 3BPA Physical Validation Summary",
        "",
        "| row | engine | status | test_dih_rmse_f_mev_a | test_300K_rmse_f_mev_a | rattle_mean_final_rmsd_a | rattle_max_fmax_ev_a | dimer_short_repulsive_fraction | atoms/s test_dih |",
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
            f"| {row['row_name']} | {row['engine']} | {row['status']} | {fmt('test_dih_rmse_f_mev_a')} | "
            f"{fmt('test_300K_rmse_f_mev_a')} | {fmt('rattle_mean_final_rmsd_a')} | {fmt('rattle_max_fmax_ev_a')} | "
            f"{fmt('dimer_short_repulsive_fraction')} | {fmt('test_dih_atoms_per_second')} |"
        )
    lines.extend([
        "",
        "Rattle-relax is interpreted as continuous RMSD and force-tail evidence, not as a binary long-term gate.",
        "",
    ])
    return "\n".join(lines)


def summarize_stage184_results(payload: dict[str, Any], *, write: bool = False) -> dict[str, Any]:
    rows = [_summary_row(row) for row in payload["rows"]]
    rows.sort(key=_sort_key)
    markdown = _render_markdown(rows)
    summary = {
        "schema_version": "rtece_stage184_physical_validation_summary.v1",
        "stage": payload["stage"],
        "rattle_policy": payload["physical_contract"]["rattle_policy"],
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


def audit_stage184_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    rows = list(payload.get("rows") or [])
    wrapper_paths = [str(row.get("wrapper", "")) for row in rows]
    wrapper_text = ""
    for wrapper in wrapper_paths:
        p = Path(wrapper)
        if p.exists():
            wrapper_text += p.read_text(encoding="utf-8") + "\n"
    row_names = {str(row.get("name")) for row in rows}
    checks = {
        "schema_version": payload.get("schema_version") == "rtece_stage184_3bpa_physical_validation.v1",
        "stage": payload.get("stage") == "stage184_3bpa_physical_validation",
        "l2_priority": (payload.get("stage183_decision") or {}).get("current_pareto_candidate") == "stage183_l2_atomic_quadrupole",
        "continuous_rattle": (payload.get("physical_contract") or {}).get("rattle_policy") == "continuous_rmsd_not_binary_gate",
        "dimer_scale": (payload.get("physical_contract") or {}).get("dimer_distance_scales") == {"min": 0.5, "max": 5.0},
        "row_set": row_names == {"stage183_l0_local_species", "stage183_l1_cross", "stage183_l2_atomic_quadrupole", "stage183_t3_cavity_vecq", "nep4_train300k"},
        "has_nep": any(row.get("engine") == "nep" for row in rows),
        "has_rtece": sum(row.get("engine") == "rtece" for row in rows) == 4,
        "no_forbidden_sbatch_tokens": not any(token in wrapper_text for token in FORBIDDEN_SBATCH_TOKENS),
        "reuses_existing_tools": all(tool in wrapper_text for tool in ["dimer_scan_rtece.py", "rattle_relax_rtece.py", "physical_stage145_community.py"]),
        "wrapper_names": all(path.endswith("_no_export.sbatch") for path in wrapper_paths if path),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "schema_version": "rtece_stage184_3bpa_physical_validation_audit.v1",
        "contract_pass": not failed,
        "failed_checks": failed,
        "checks": checks,
        "no_forbidden_sbatch_tokens": checks["no_forbidden_sbatch_tokens"],
    }


def materialize_stage184(payload: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(payload["artifacts"]["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    wrappers = {row["name"]: _write_wrapper(row, payload) for row in payload["rows"]}
    stage_plan = _write_stage_plan(payload)
    audit = audit_stage184_manifest(payload)
    if not audit["contract_pass"]:
        raise ValueError(f"Stage184 manifest audit failed: {audit['failed_checks']}")
    audit_path = Path(payload["artifacts"]["audit"])
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "manifest": str(manifest_path),
        "audit": audit,
        "audit_path": str(audit_path),
        "stage_plan": stage_plan,
        "wrappers": wrappers,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--stage183-root", type=Path, default=DEFAULT_STAGE183_ROOT)
    parser.add_argument("--stage182-root", type=Path, default=DEFAULT_STAGE182_ROOT)
    parser.add_argument("--dimer-num-points", type=int, default=16)
    parser.add_argument("--rattle-start-config", type=int, default=0)
    parser.add_argument("--rattle-limit-configs", type=int, default=8)
    parser.add_argument("--rattle-max-steps", type=int, default=10)
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = make_stage184_manifest(
        output_root=args.output_root,
        dataset_root=args.dataset_root,
        stage183_root=args.stage183_root,
        stage182_root=args.stage182_root,
        dimer_num_points=args.dimer_num_points,
        rattle_start_config=args.rattle_start_config,
        rattle_limit_configs=args.rattle_limit_configs,
        rattle_max_steps=args.rattle_max_steps,
    )
    if args.summarize_only:
        print(json.dumps(summarize_stage184_results(payload, write=True), indent=2, sort_keys=True))
        return
    result = materialize_stage184(payload)
    summarize_stage184_results(payload, write=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
