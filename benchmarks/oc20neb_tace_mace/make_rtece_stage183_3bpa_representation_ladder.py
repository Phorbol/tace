#!/usr/bin/env python3
"""Create Stage183 3BPA rTECE representation-ladder scaffold.

Stage183 follows Stage182's conventional 3BPA closure result: the current
L1 local-L0 rTECE student is close to NEP on ID/temperature force RMSE but
has a much larger dihedral PES energy/force gap.  The next experiment is a
nested TECE semantic path ladder, not a wider final head.
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
from pathlib import Path
from typing import Any

DEFAULT_OUTPUT_ROOT = Path("runs/oc20neb_tace_mace/rtece-stage183-3bpa-representation-ladder")
DEFAULT_DATASET_ROOT = Path("datasets/3BPA/dataset_3BPA")
DEFAULT_STAGE182_SUMMARY = Path("runs/oc20neb_tace_mace/rtece-stage182-3bpa-closure/stage182_summary.json")
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

BASE_PATHS = (
    "atomic.radial_density",
    "atomic.species_basis_density",
    "atomic.local_l0_lowrank_density",
)
L1_PATHS = BASE_PATHS + (
    "atomic.vector_norm",
    "atomic.vector_cross_radial_dot",
)
L2_PATHS = L1_PATHS + (
    "atomic.quadrupole_norm",
    "atomic.quadrupole_cross_radial_frobenius",
)
T3_CAVITY_VECQ_PATHS = L2_PATHS + (
    "edge.cavity.vector_dot",
    "edge.cavity.quadrupole_frobenius",
)

CANDIDATE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "stage183_l0_local_species",
        "tece_tier": "T1_scalar_endpoint_local_chemistry",
        "representation_step": "l0_local_species_basis",
        "scalar_path_ids": BASE_PATHS,
        "moment_l_max": 0,
        "atomic_cross_radial_sketch_channels": 0,
        "atomic_cross_radial_projection": "fixed_shell_mean",
        "capacity_allocation": "learnable_species_basis_and_local_l0_before_head",
        "rationale": "NEP/DPA1-0-like scalar endpoint: one moment pass, learnable species basis, no angular or edge-relational retention.",
    },
    {
        "name": "stage183_l1_cross",
        "tece_tier": "T2_atomic_l1_cross_radial",
        "representation_step": "l1_sparse_cross_radial_power_spectrum",
        "scalar_path_ids": L1_PATHS,
        "moment_l_max": 1,
        "atomic_cross_radial_sketch_channels": 3,
        "atomic_cross_radial_projection": "learnable",
        "capacity_allocation": "learnable_l1_cross_radial_front_before_head",
        "rationale": "Stage182 rTECE row: keeps sparse L1 atomic moment information while still scalarizing immediately.",
    },
    {
        "name": "stage183_l2_atomic_quadrupole",
        "tece_tier": "T2_l2_atomic_scalarized_quadrupole",
        "representation_step": "l2_atomic_quadrupole_without_edge_cost",
        "scalar_path_ids": L2_PATHS,
        "moment_l_max": 2,
        "atomic_cross_radial_sketch_channels": 3,
        "atomic_cross_radial_projection": "learnable",
        "capacity_allocation": "learnable_l2_quadrupole_front_before_head",
        "rationale": "Tests whether angular/dihedral PES error is mainly missing L2 local moment information before adding edge-relational cost.",
    },
    {
        "name": "stage183_t3_cavity_vecq",
        "tece_tier": "T3_sparse_cavity_edge_relational",
        "representation_step": "cavity_vector_quadrupole_edge_relational_sketch",
        "scalar_path_ids": T3_CAVITY_VECQ_PATHS,
        "moment_l_max": 2,
        "atomic_cross_radial_sketch_channels": 3,
        "atomic_cross_radial_projection": "learnable",
        "capacity_allocation": "sparse_edge_relational_paths_after_l2_atomic_front",
        "rationale": "True rTECE increment: source-target cavity vector/quadrupole relations with no persistent high-l edge state.",
    },
)


def _artifacts(root: Path) -> dict[str, str]:
    return {
        "manifest": str(root / "stage183_manifest.json"),
        "audit": str(root / "stage183_manifest_audit.json"),
        "stage_plan": str(root / "stage183_plan.md"),
        "stage182_analysis_json": str(root / "stage182_to_stage183_analysis.json"),
        "stage182_analysis_md": str(root / "stage182_to_stage183_analysis.md"),
        "summary_json": str(root / "stage183_summary.json"),
        "summary_md": str(root / "stage183_summary.md"),
        "wrapper_root": str(root / "wrappers"),
        "benchmark_wrapper_root": str(root / "benchmark_wrappers"),
        "results_root": str(root / "results"),
        "diagnostics_root": str(root / "diagnostics"),
    }


def _row_outputs(root: Path, row_name: str) -> dict[str, str]:
    return {split: str(root / "diagnostics" / row_name / f"{split}_benchmark.json") for split in BENCHMARK_SPLITS}


def _ratio(numerator: float | int | None, denominator: float | int | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    value = float(numerator) / float(denominator)
    if not math.isfinite(value):
        return None
    return value


def analyze_stage182_for_stage183(summary_path: str | Path = DEFAULT_STAGE182_SUMMARY) -> dict[str, Any]:
    path = Path(summary_path)
    if not path.exists():
        return {
            "schema_version": "rtece_stage183_stage182_analysis.v1",
            "source_summary": str(path),
            "status": "missing_stage182_summary",
            "primary_failure_mode": "dihedral_pes_gap",
            "force_rmse_ratios": {},
            "energy_rmse_ratios": {},
            "priority_axes": ["l2_atomic_quadrupole", "t3_cavity_edge_relational"],
            "next_decision": "representation_ladder_before_more_kernel_work",
            "rationale": "Stage182 summary missing locally; keep the document-derived priority until matched rows are available.",
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in payload.get("rows", []) if row.get("status") == "completed"]
    by_engine_split: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        by_engine_split[(str(row.get("engine")), str(row.get("split")))] = row
    force_ratios: dict[str, float] = {}
    energy_ratios: dict[str, float] = {}
    for split in BENCHMARK_SPLITS:
        rtece = by_engine_split.get(("rtece", split))
        nep = by_engine_split.get(("nep", split))
        if not rtece or not nep:
            continue
        f_ratio = _ratio(rtece.get("rmse_f_mev_a"), nep.get("rmse_f_mev_a"))
        e_ratio = _ratio(rtece.get("rmse_e_mev_atom"), nep.get("rmse_e_mev_atom"))
        if f_ratio is not None:
            force_ratios[split] = f_ratio
        if e_ratio is not None:
            energy_ratios[split] = e_ratio
    id_force = force_ratios.get("test_300K")
    dih_force = force_ratios.get("test_dih")
    id_energy = energy_ratios.get("test_300K")
    dih_energy = energy_ratios.get("test_dih")
    dihedral_gap = any(
        ratio is not None and base is not None and ratio > max(1.25 * base, base + 0.25)
        for ratio, base in [(dih_force, id_force), (dih_energy, id_energy)]
    )
    return {
        "schema_version": "rtece_stage183_stage182_analysis.v1",
        "source_summary": str(path),
        "status": "completed" if force_ratios or energy_ratios else "no_matched_rows",
        "primary_failure_mode": "dihedral_pes_gap" if dihedral_gap or "test_dih" in energy_ratios else "general_accuracy_gap",
        "force_rmse_ratios": force_ratios,
        "energy_rmse_ratios": energy_ratios,
        "priority_axes": ["l2_atomic_quadrupole", "t3_cavity_edge_relational"],
        "next_decision": "representation_ladder_before_more_kernel_work",
        "rationale": (
            "Stage182 shows the L1/local-L0 student is roughly NEP-close on ID/temperature force RMSE but much worse on the dihedral PES split; "
            "the clean next test is L2 atomic quadrupoles and then cavity edge-relational sketches before kernel work or head widening."
        ),
    }


def _candidate_rows(root: Path) -> list[dict[str, Any]]:
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
        name = str(spec["name"])
        rows.append({
            "name": name,
            "engine": "rtece",
            "descriptor_label": str(spec["tece_tier"]),
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


def make_stage183_manifest(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    stage182_summary: str | Path = DEFAULT_STAGE182_SUMMARY,
    train_split: str = "train_300K",
    train_limit_configs: int = 500,
    bench_limit_configs: int = 512,
    max_steps: int = 20000,
) -> dict[str, Any]:
    if int(train_limit_configs) < 1:
        raise ValueError("train_limit_configs must be positive")
    if int(bench_limit_configs) < 1:
        raise ValueError("bench_limit_configs must be positive")
    root = Path(output_root)
    artifacts = _artifacts(root)
    return {
        "schema_version": "rtece_stage183_3bpa_representation_ladder.v1",
        "stage": "stage183_3bpa_representation_ladder",
        "scientific_question": (
            "On 3BPA, can a TECE semantic path ladder recover the Stage182 dihedral PES gap by adding L2 atomic "
            "quadrupoles and sparse cavity edge-relational sketches, while holding final-head capacity fixed?"
        ),
        "theory_alignment": [
            "TECE_design_space.md §5: rTECE should keep one fused L_A=2/3 moment pass, sparse atomic scalar paths, and sparse edge-relational m_total=0 sketches.",
            "TECE_design_space.md §11: compare a tiered T1/T2/T3 path ladder by physical error and hardware cost, not parameter count alone.",
            "rTECE_review.md §4: replace variant switches with semantic path IDs and fix edge sketch duplication by explicit path registry.",
            "rTECE_review.md: increase representation capacity in front-loaded chemistry/angular/edge paths before widening the final MLP head.",
            "Stage182: 3BPA gives matched NEP/DPA/rTECE numeric closure and exposes a dihedral PES gap in the current L1/local-L0 student.",
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
        "stage182_analysis": analyze_stage182_for_stage183(stage182_summary),
        "train_contract": {
            "train_file": str(Path(dataset_root) / f"{train_split}.xyz"),
            "valid_file": str(Path(dataset_root) / "test_300K.xyz"),
            "energy_key": "energy",
            "forces_key": "forces",
            "limit_configs": int(train_limit_configs),
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
            "secondary_ranking_metrics": ["rmse_e_mev_atom", "max_abs_f_mev_a", "max_abs_e_mev_atom", "atoms_per_second", "peak_memory_mb"],
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
            "physical_followups_after_numeric_ladder": ["dimer_scan", "rattle_relax_rmsd"],
            "head_capacity_control": "hidden_channels fixed at 64,64 across rows",
        },
        "rows": _candidate_rows(root),
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


def _dataset_split(payload: dict[str, Any], split: str) -> str:
    return str(Path(payload["dataset"]["root"]) / f"{split}.xyz")


def _common_train_args(row: dict[str, Any], payload: dict[str, Any]) -> str:
    cfg = row["student_config"]
    train = payload["train_contract"]
    return (
        f"--variant {shlex.quote(row['name'])} "
        f"--scalar-path-ids {shlex.quote(','.join(cfg['scalar_path_ids']))} "
        f"--train-file {shlex.quote(train['train_file'])} "
        f"--valid-file {shlex.quote(train['valid_file'])} "
        f"--output-dir {shlex.quote(row['train_dir'])} "
        f"--max-steps {int(train['max_steps'])} "
        f"--limit-configs {int(train['limit_configs'])} "
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
    lines = _header(f"rtece-st183-{row['representation_step']}")
    lines.extend([
        f"mkdir -p {shlex.quote(row['train_dir'])}",
        "PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} -m tace.scripts.rtece_train_scalar " + _common_train_args(row, payload),
    ])
    return _write(row["wrapper"], lines)


def _write_benchmark_wrapper(row: dict[str, Any], payload: dict[str, Any]) -> str:
    train = payload["train_contract"]
    lines = _header(f"rtece-st183-bench-{row['representation_step']}", time_limit="01:40:00")
    lines.append(f"mkdir -p {shlex.quote(str(Path(payload['artifacts']['diagnostics_root']) / row['name']))}")
    for split in BENCHMARK_SPLITS:
        lines.append(
            "PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py "
            f"--model {shlex.quote(row['model_artifact'])} --configs {shlex.quote(_dataset_split(payload, split))} "
            f"--output {shlex.quote(row['benchmark_outputs'][split])} --variant {shlex.quote(row['name'] + '_' + split)} "
            f"--energy-key {shlex.quote(train['energy_key'])} --forces-key {shlex.quote(train['forces_key'])} "
            f"--start-config 0 --limit-configs {int(train['bench_limit_configs'])} --measure-passes 5 "
            "--device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist"
        )
    return _write(row["benchmark_wrapper"], lines)


def _write_stage_plan(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage183 3BPA Representation Ladder",
        "",
        "Goal: test whether TECE semantic representation increments close the Stage182 dihedral PES gap before kernel work or head widening.",
        "",
        "## Document Alignment",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["theory_alignment"])
    lines.extend([
        "",
        "## Ladder",
        "",
    ])
    for row in payload["rows"]:
        lines.append(f"- `{row['name']}`: {row['representation_step']}; marginal paths `{', '.join(row['marginal_paths'])}`")
    lines.extend([
        "",
        "## Metrics",
        "",
        "- Primary: force RMSE in meV/A across test_300K, test_600K, test_1200K, and test_dih.",
        "- Required: energy RMSE/max, force max, atoms/s, and peak memory.",
        "- Physical followups after numeric ladder: dimer scan and rattle-relax RMSD, treated as continuous diagnostics.",
        "",
        "## Decision Rule",
        "",
        "- If L2 improves test_dih energy/force RMSE without a large throughput penalty, keep L_A=2 in the active rTECE family.",
        "- If T3 improves test_dih beyond L2, edge-relational sketches are a real rTECE advantage over pure NEP-like scalar endpoints.",
        "- If neither improves test_dih, prioritize teacher trajectory/distillation coverage rather than adding more hand-designed paths.",
    ])
    return _write(payload["artifacts"]["stage_plan"], lines)


def _write_analysis(payload: dict[str, Any]) -> tuple[str, str]:
    analysis = payload["stage182_analysis"]
    json_path = Path(payload["artifacts"]["stage182_analysis_json"])
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Stage182 to Stage183 Analysis",
        "",
        f"- source: `{analysis['source_summary']}`",
        f"- status: `{analysis['status']}`",
        f"- primary failure mode: `{analysis['primary_failure_mode']}`",
        f"- next decision: `{analysis['next_decision']}`",
        "",
        "## Force RMSE Ratios rTECE / NEP",
        "",
    ]
    if analysis["force_rmse_ratios"]:
        lines.extend(f"- `{split}`: {value:.3f}" for split, value in analysis["force_rmse_ratios"].items())
    else:
        lines.append("- no matched Stage182 rows available")
    lines.extend(["", "## Energy RMSE Ratios rTECE / NEP", ""])
    if analysis["energy_rmse_ratios"]:
        lines.extend(f"- `{split}`: {value:.3f}" for split, value in analysis["energy_rmse_ratios"].items())
    else:
        lines.append("- no matched Stage182 rows available")
    lines.extend(["", "## Priority", ""])
    lines.extend(f"- `{axis}`" for axis in analysis["priority_axes"])
    lines.extend(["", analysis["rationale"], ""])
    md_path = Path(payload["artifacts"]["stage182_analysis_md"])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return str(json_path), str(md_path)


def audit_stage183_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    rows = list(payload.get("rows") or [])
    names = [str(row.get("name")) for row in rows]
    wrapper_paths = [str(row.get("wrapper", "")) for row in rows] + [str(row.get("benchmark_wrapper", "")) for row in rows]
    wrapper_text = ""
    for wrapper_path in wrapper_paths:
        p = Path(wrapper_path)
        if p.exists():
            wrapper_text += p.read_text(encoding="utf-8") + "\n"
    nested = True
    previous: set[str] = set()
    for row in rows:
        current = set((row.get("student_config") or {}).get("scalar_path_ids") or [])
        nested = nested and previous <= current
        previous = current
    required_metrics = (payload.get("comparison_contract") or {}).get("required_metrics", [])
    all_paths = {path_id for row in rows for path_id in (row.get("student_config") or {}).get("scalar_path_ids", [])}
    checks = {
        "schema_version": payload.get("schema_version") == "rtece_stage183_3bpa_representation_ladder.v1",
        "stage": payload.get("stage") == "stage183_3bpa_representation_ladder",
        "dataset": (payload.get("dataset") or {}).get("name") == "3BPA",
        "splits": (payload.get("dataset") or {}).get("benchmark_splits") == BENCHMARK_SPLITS,
        "candidate_names": names == [str(spec["name"]) for spec in CANDIDATE_SPECS],
        "nested_paths": nested,
        "fixed_head": len({(row.get("student_config") or {}).get("hidden_channels") for row in rows}) == 1,
        "not_head_only": all(row.get("capacity_allocation") != "widen_final_head_only" for row in rows),
        "contains_l2": {"atomic.quadrupole_norm", "atomic.quadrupole_cross_radial_frobenius"} <= all_paths,
        "contains_t3_edge": {"edge.cavity.vector_dot", "edge.cavity.quadrupole_frobenius"} <= all_paths,
        "stage182_dihedral_priority": (payload.get("stage182_analysis") or {}).get("primary_failure_mode") == "dihedral_pes_gap",
        "production_entrypoint": (payload.get("train_contract") or {}).get("rtece_entrypoint") == "tace.scripts.rtece_train_scalar",
        "label_keys": (payload.get("dataset") or {}).get("labels") == {"energy_key": "energy", "forces_key": "forces"},
        "rmse_primary": (payload.get("comparison_contract") or {}).get("primary_ranking_metric") == "rmse_f_mev_a",
        "required_metrics": all(metric in required_metrics for metric in ["rmse_e_mev_atom", "rmse_f_mev_a", "max_abs_e_mev_atom", "max_abs_f_mev_a", "atoms_per_second", "peak_memory_mb"]),
        "physical_followups": set((payload.get("comparison_contract") or {}).get("physical_followups_after_numeric_ladder") or []) >= {"dimer_scan", "rattle_relax_rmsd"},
        "no_forbidden_sbatch_tokens": not any(token in wrapper_text for token in FORBIDDEN_SBATCH_TOKENS),
        "wrapper_names": all(path.endswith("_no_export.sbatch") for path in wrapper_paths if path),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "schema_version": "rtece_stage183_3bpa_representation_ladder_audit.v1",
        "contract_pass": not failed,
        "failed_checks": failed,
        "checks": checks,
        "no_forbidden_sbatch_tokens": checks["no_forbidden_sbatch_tokens"],
    }


def materialize_stage183(payload: dict[str, Any]) -> dict[str, Any]:
    manifest_path = Path(payload["artifacts"]["manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    wrappers = {row["name"]: _write_train_wrapper(row, payload) for row in payload["rows"]}
    benchmark_wrappers = {row["name"]: _write_benchmark_wrapper(row, payload) for row in payload["rows"]}
    stage_plan = _write_stage_plan(payload)
    analysis_json, analysis_md = _write_analysis(payload)
    audit = audit_stage183_manifest(payload)
    if not audit["contract_pass"]:
        raise ValueError(f"Stage183 manifest audit failed: {audit['failed_checks']}")
    audit_path = Path(payload["artifacts"]["audit"])
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "manifest": str(manifest_path),
        "audit": audit,
        "audit_path": str(audit_path),
        "stage_plan": stage_plan,
        "stage182_analysis_json": analysis_json,
        "stage182_analysis_md": analysis_md,
        "wrappers": wrappers,
        "benchmark_wrappers": benchmark_wrappers,
    }


def _read_json(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _train_summary_for_row(row: dict[str, Any]) -> dict[str, Any]:
    return _read_json(Path(row["train_dir"]) / "train_summary.json") or {}


def _summary_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    train_summary = _train_summary_for_row(row)
    bench = _read_json(row["benchmark_outputs"][split])
    result: dict[str, Any] = {
        "row_name": row["name"],
        "engine": row["engine"],
        "split": split,
        "tece_tier": row.get("tece_tier"),
        "representation_step": row.get("representation_step"),
        "train_best_valid_loss": train_summary.get("best_valid_loss"),
        "train_best_step": train_summary.get("best_step"),
        "train_steps": train_summary.get("steps"),
        "status": "missing",
        "benchmark_output": row["benchmark_outputs"][split],
    }
    if bench is None:
        return result
    result["status"] = str(bench.get("status", "completed"))
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
        if key in bench:
            result[key] = bench[key]
    if "peak_memory_mb" not in result:
        if result.get("peak_reserved_mb") is not None:
            result["peak_memory_mb"] = result["peak_reserved_mb"]
        elif result.get("peak_allocated_mb") is not None:
            result["peak_memory_mb"] = result["peak_allocated_mb"]
    return result


def _summary_sort_key(row: dict[str, Any]) -> tuple[int, int, float, str]:
    missing = 0 if row.get("status") == "completed" else 1
    split_rank = BENCHMARK_SPLITS.index(row["split"]) if row.get("split") in BENCHMARK_SPLITS else len(BENCHMARK_SPLITS)
    rmse = row.get("rmse_f_mev_a")
    return (missing, split_rank, float(rmse) if rmse is not None else float("inf"), str(row.get("row_name")))


def _render_summary_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Stage183 3BPA Representation Ladder Summary",
        "",
        "| row | tier | split | status | train_best_valid_loss | rmse_f_mev_a | rmse_e_mev_atom | max_abs_f_mev_a | max_abs_e_mev_atom | atoms_per_second | peak_memory_mb |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
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
            f"| {row['row_name']} | {fmt('tece_tier')} | {row['split']} | {row['status']} | "
            f"{fmt('train_best_valid_loss')} | {fmt('rmse_f_mev_a')} | {fmt('rmse_e_mev_atom')} | "
            f"{fmt('max_abs_f_mev_a')} | {fmt('max_abs_e_mev_atom')} | {fmt('atoms_per_second')} | {fmt('peak_memory_mb')} |"
        )
    lines.append("")
    return "\n".join(lines)


def summarize_stage183_results(payload: dict[str, Any], *, write: bool = False) -> dict[str, Any]:
    rows = [_summary_row(row, split) for row in payload["rows"] for split in BENCHMARK_SPLITS]
    rows.sort(key=_summary_sort_key)
    markdown = _render_summary_markdown(rows)
    summary = {
        "schema_version": "rtece_stage183_representation_ladder_summary.v1",
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
    parser.add_argument("--stage182-summary", type=Path, default=DEFAULT_STAGE182_SUMMARY)
    parser.add_argument("--train-limit-configs", type=int, default=500)
    parser.add_argument("--bench-limit-configs", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = make_stage183_manifest(
        output_root=args.output_root,
        dataset_root=args.dataset_root,
        stage182_summary=args.stage182_summary,
        train_limit_configs=args.train_limit_configs,
        bench_limit_configs=args.bench_limit_configs,
        max_steps=args.max_steps,
    )
    if args.analyze_only:
        print(json.dumps(payload["stage182_analysis"], indent=2, sort_keys=True))
        return
    if args.summarize_only:
        print(json.dumps(summarize_stage183_results(payload, write=True), indent=2, sort_keys=True))
        return
    result = materialize_stage183(payload)
    summarize_stage183_results(payload, write=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
