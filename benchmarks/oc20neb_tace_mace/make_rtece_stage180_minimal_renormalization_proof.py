#!/usr/bin/env python3
"""Create Stage180 minimal renormalization-proof scaffold."""

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

DEFAULT_OUTPUT_ROOT = Path("runs/oc20neb_tace_mace/rtece-stage180-minimal-renormalization-proof")
DEFAULT_IMPLEMENTATION_PLAN = Path("docs/superpowers/plans/2026-07-21-stage180-minimal-renormalization-proof.md")
DEFAULT_TEACHER_VALID = Path("runs/oc20neb_tace_mace/tece-distill-20260717/mixed_valid_tw0.75_regen.extxyz")

FIXED_STUDENT_PATH_IDS = [
    "atomic.radial_density",
    "atomic.species_basis_density",
    "atomic.local_l0_lowrank_density",
    "atomic.vector_norm",
    "atomic.vector_cross_radial_dot",
]

REFERENCE_PATH_IDS = FIXED_STUDENT_PATH_IDS + [
    "edge.cavity.vector_dot",
    "edge.direct.radial",
]


def _path_csv(paths: list[str]) -> str:
    return ",".join(paths)


def _artifacts(root: Path) -> dict[str, Any]:
    implementation_plan = DEFAULT_IMPLEMENTATION_PLAN if Path(root) == DEFAULT_OUTPUT_ROOT else root / "stage180_implementation_plan.md"
    return {
        "manifest": str(root / "stage180_manifest.json"),
        "manifest_audit": str(root / "stage180_manifest_audit.json"),
        "stage_plan": str(root / "stage180_plan.md"),
        "implementation_plan": str(implementation_plan),
        "results_root": str(root / "results"),
        "diagnostics_root": str(root / "diagnostics"),
        "wrappers": {
            "projection_diagnostic": str(root / "wrappers" / "stage180_projection_diagnostic_no_export.sbatch"),
            "scratch_train": str(root / "wrappers" / "stage180_scratch_train_no_export.sbatch"),
            "benchmark_physical_after_training": str(root / "wrappers" / "stage180_benchmark_physical_no_export.sbatch"),
        },
    }


def make_stage180_manifest(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    train_file: str | Path = DEFAULT_BASE_TRAIN,
    valid_file: str | Path = DEFAULT_DFT_VALID,
    teacher_valid_file: str | Path = DEFAULT_TEACHER_VALID,
    limit_configs: int = 2048,
    valid_limit_configs: int = 256,
    projection_limit_configs: int = 256,
    max_steps: int = 20000,
) -> dict[str, Any]:
    root = Path(output_root)
    return {
        "schema_version": "rtece_stage180_minimal_renormalization_proof.v1",
        "stage": "stage180_minimal_renormalization_proof",
        "scientific_question": (
            "Does TECE-style renormalized initialization or downfolded distillation improve a fixed rTECE "
            "student path set over the same architecture trained from scratch?"
        ),
        "fixed_student_path_ids": list(FIXED_STUDENT_PATH_IDS),
        "reference_teacher_proxy_path_ids": list(REFERENCE_PATH_IDS),
        "train_file": str(train_file),
        "valid_file": str(valid_file),
        "teacher_valid_file": str(teacher_valid_file),
        "limit_configs": int(limit_configs),
        "valid_limit_configs": int(valid_limit_configs),
        "projection_limit_configs": int(projection_limit_configs),
        "max_steps": int(max_steps),
        "student_config": {
            "variant": "stage180_fixed_student_l1_atomic_cross",
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
            "relative_energy_weight": 0.25,
            "lr_scheduler": "plateau",
            "lr_warmup_steps": 500,
            "early_stopping_patience": 250,
            "neighborlist_backend": "matscipy",
            "per_element_e0_fit": True,
        },
        "comparison_arms": [
            {
                "id": "scratch_same_student",
                "purpose": "Control arm: train the fixed student path set from random initialization.",
                "runnable_now": True,
                "blocking_tooling": [],
                "output_dir": str(root / "results" / "scratch_same_student"),
            },
            {
                "id": "linear_projection_diagnostic",
                "purpose": "Quantify projection/downfolding gap from reference proxy paths onto the fixed student path set.",
                "runnable_now": True,
                "blocking_tooling": [],
                "output_json": str(root / "diagnostics" / "stage180_projection_diagnostic.json"),
            },
            {
                "id": "renorm_initialized_same_student",
                "purpose": "Train the same fixed student path set from a projection/GN initialized checkpoint.",
                "runnable_now": False,
                "blocking_tooling": [
                    "checkpoint_initialization_from_projection_coefficients",
                    "train_entrypoint_init_checkpoint_or_init_state",
                ],
                "required_before_running": "Implement a deterministic initializer that maps projection/GN coefficients into the fixed student model state and lets the train entrypoint load it before optimizer construction.",
            },
            {
                "id": "renorm_initialized_teacher_residual_distill",
                "purpose": "Use the same renormalized initialization plus teacher E/F residual distillation labels.",
                "runnable_now": False,
                "blocking_tooling": [
                    "checkpoint_initialization_from_projection_coefficients",
                    "train_entrypoint_init_checkpoint_or_init_state",
                    "teacher_residual_cache_or_extxyz_labels",
                    "distillation_loss_mixing_real_and_teacher_labels",
                ],
                "required_before_running": "Create a versioned teacher cache or extxyz residual-label product with E/F schema, teacher hash, units, and PBC convention.",
            },
        ],
        "metrics_required": [
            "E MAE/RMSE/max",
            "F MAE/RMSE/max",
            "relative image RMSE",
            "barrier RMSE",
            "dimer scan smoothness",
            "rattle-relax final RMSD and force tail",
            "atoms/s scaling",
            "peak allocated/reserved memory",
        ],
        "success_criteria": {
            "primary_claim_requires": [
                "renorm_initialized_same_student beats scratch_same_student under matched E/F/relative/physical metrics",
                "or the result is recorded as falsifying the current renormalization recipe",
            ],
            "not_sufficient": [
                "linear projection diagnostic alone",
                "improved MAE without E/F max and physical diagnostics",
                "unmatched NEP/DPA smoke baseline comparison",
            ],
        },
        "artifacts": _artifacts(root),
        "review_basis": [
            "TECE_design_space.md: deleted paths should be downfolded into retained paths by Gram/Schur or local GN correction, not merely set to zero.",
            "rTECE_review.md: only renormalization+distillation outperforming from-scratch proves the branch is more than scalar student architecture search.",
            "Stage179: operator_space_renormalization_beats_from_scratch is the central missing hypothesis.",
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
        _shell_assign("TRAIN_FILE", payload["train_file"]),
        _shell_assign("VALID_FILE", payload["valid_file"]),
        _shell_assign("TEACHER_VALID_FILE", payload["teacher_valid_file"]),
        _shell_assign("LIMIT_CONFIGS", payload["limit_configs"]),
        _shell_assign("VALID_LIMIT_CONFIGS", payload["valid_limit_configs"]),
        _shell_assign("PROJECTION_LIMIT_CONFIGS", payload["projection_limit_configs"]),
        _shell_assign("MAX_STEPS", payload["max_steps"]),
        _shell_assign("RESULTS_ROOT", payload["artifacts"]["results_root"]),
        _shell_assign("DIAGNOSTICS_ROOT", payload["artifacts"]["diagnostics_root"]),
        'mkdir -p /home/gengjianrui/bin/logs "${RESULTS_ROOT}" "${DIAGNOSTICS_ROOT}"',
        'cd "${TACE_ROOT}"',
        '"${TACE_PYTHON}" -V',
        "nvidia-smi -L",
        "",
    ]


def _student_train_command(payload: dict[str, Any], *, output_dir: str) -> str:
    cfg = payload["student_config"]
    train = payload["training_config"]
    parts = [
        'PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}"',
        "python",
        "-m",
        "tace.scripts.rtece_train_scalar",
        "--variant",
        cfg["variant"],
        "--scalar-path-ids",
        _path_csv(payload["fixed_student_path_ids"]),
        "--train-file",
        '"${TRAIN_FILE}"',
        "--valid-file",
        '"${VALID_FILE}"',
        "--output-dir",
        shlex.quote(output_dir),
        "--limit-configs",
        '"${LIMIT_CONFIGS}"',
        "--valid-limit-configs",
        '"${VALID_LIMIT_CONFIGS}"',
        "--max-steps",
        '"${MAX_STEPS}"',
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
        "--relative-energy-weight",
        str(train["relative_energy_weight"]),
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


def write_projection_wrapper(path: str | Path, payload: dict[str, Any]) -> Path:
    wrapper = Path(path)
    body = _wrapper_header(payload, job_name="rtece-st180-proj", time_limit="01:00:00")
    out = Path(payload["artifacts"]["diagnostics_root"]) / "stage180_projection_diagnostic.json"
    body.extend(
        [
            f"mkdir -p {shlex.quote(str(out.parent))}",
            (
                'PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" python benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py '
                '--configs "${TRAIN_FILE}" '
                f"--output-json {shlex.quote(str(out))} "
                f"--reference-path-ids {_path_csv(payload['reference_teacher_proxy_path_ids'])} "
                f"--candidate scratch_same_student:{_path_csv(payload['fixed_student_path_ids'])} "
                "--num-radial 10 --species-basis-channels 16 --species-basis-mode learnable_embedding "
                "--local-l0-chemistry-rank 4 "
                "--reference-atomic-cross-radial-sketch-channels 3 --candidate-atomic-cross-radial-sketch-channels 3 "
                "--limit-configs \"${PROJECTION_LIMIT_CONFIGS}\" --default-dtype float64 --neighborlist-backend matscipy "
                "--energy-target-key energy --energy-baseline element_counts --energy-fit-intercept --energy-standardize-features "
                "--energy-ridge-grid 1e-12,1e-10,1e-8,1e-6 --energy-split-mode group-loocv --energy-group-key case_id "
                "--force-target-key forces --force-component-sample-count 6000 --force-eval-stride 4"
            ),
            "",
        ]
    )
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("\n".join(body), encoding="utf-8")
    return wrapper


def write_scratch_train_wrapper(path: str | Path, payload: dict[str, Any]) -> Path:
    wrapper = Path(path)
    body = _wrapper_header(payload, job_name="rtece-st180-scratch", time_limit="03:55:00")
    arm = next(row for row in payload["comparison_arms"] if row["id"] == "scratch_same_student")
    body.extend(
        [
            f"mkdir -p {shlex.quote(arm['output_dir'])}",
            _student_train_command(payload, output_dir=str(arm["output_dir"])),
            "",
        ]
    )
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("\n".join(body), encoding="utf-8")
    return wrapper


def write_benchmark_physical_wrapper(path: str | Path, payload: dict[str, Any]) -> Path:
    wrapper = Path(path)
    body = _wrapper_header(payload, job_name="rtece-st180-diag", time_limit="03:55:00")
    arm = next(row for row in payload["comparison_arms"] if row["id"] == "scratch_same_student")
    variant = "scratch_same_student"
    checkpoint = str(Path(arm["output_dir"]) / "rtece_scalar_best.pt")
    diag = Path(payload["artifacts"]["diagnostics_root"]) / variant
    dimer_json = diag / f"{variant}_dimer_scan.json"
    rattle_json = diag / f"{variant}_rattle_relax.json"
    physical_json = diag / f"{variant}_physical_pareto.json"
    dft_benchmark = diag / f"{variant}_dft_benchmark.json"
    teacher_benchmark = diag / f"{variant}_teacher_benchmark.json"
    case = ":".join([variant, str(dft_benchmark), str(teacher_benchmark), str(dimer_json), str(rattle_json)])
    body.extend(
        [
            f"mkdir -p {shlex.quote(str(diag))}",
            (
                'PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" python benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py '
                f"--model {shlex.quote(checkpoint)} --configs \"${{VALID_FILE}}\" --output {shlex.quote(str(dft_benchmark))} "
                f"--variant {variant} --start-config 0 --limit-configs \"${{VALID_LIMIT_CONFIGS}}\" "
                "--measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist"
            ),
            (
                'PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" python benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py '
                f"--model {shlex.quote(checkpoint)} --configs \"${{TEACHER_VALID_FILE}}\" --output {shlex.quote(str(teacher_benchmark))} "
                f"--variant {variant} --start-config 0 --limit-configs \"${{VALID_LIMIT_CONFIGS}}\" "
                "--measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist"
            ),
        ]
    )
    for limit in (32, 128, 512, 1024):
        body.append(
            'PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" python benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py '
            f"--model {shlex.quote(checkpoint)} --configs \"${{VALID_FILE}}\" --output {shlex.quote(str(diag / f'{variant}_scaling_limit{limit}.json'))} "
            f"--variant {variant} --start-config 0 --limit-configs {limit} "
            "--measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist"
        )
    body.extend(
        [
            (
                'PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" python benchmarks/oc20neb_tace_mace/dimer_scan_rtece.py '
                f"--checkpoint {shlex.quote(checkpoint)} --output-json {shlex.quote(str(dimer_json))} --output-md {shlex.quote(str(dimer_json.with_suffix('.md')))} "
                f"--route {variant} --pairs C-N C-O C-H N-H O-H C-C N-N --num-points 24 --min-scale 0.5 --max-scale 5.0 "
                "--device cuda --default-dtype float32 --force-mode autograd --neighborlist-backend ase"
            ),
            (
                'PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" python benchmarks/oc20neb_tace_mace/rattle_relax_rtece.py '
                f"--checkpoint {shlex.quote(checkpoint)} --configs \"${{VALID_FILE}}\" --output-json {shlex.quote(str(rattle_json))} --output-md {shlex.quote(str(rattle_json.with_suffix('.md')))} "
                f"--route {variant} --start-config 58 --limit-configs 16 --rattle-std 0.05 --rattle-seed 20260718 --fmax 0.05 --max-steps 10 "
                "--device cuda --default-dtype float32 --force-mode autograd --neighborlist-backend matscipy"
            ),
            (
                'PYTHONPATH="${TACE_ROOT}:${PYTHONPATH:-}" python benchmarks/oc20neb_tace_mace/summarize_rtece_physical_pareto.py '
                f"--case {shlex.quote(case)} "
                "--max-dft-f-rmse-mev-a 120.0 --max-dft-e-rmse-mev-atom 320.0 --max-dft-f-max-mev-a 2600.0 --max-dft-e-max-mev-atom 760.0 "
                "--rattle-focus-label C_or_N --max-focus-rattle-rmsd-a 0.35 --max-rattle-fmax-ev-a 1.00 "
                f"--output-json {shlex.quote(str(physical_json))} --output-md {shlex.quote(str(physical_json.with_suffix('.md')))}"
            ),
            "",
        ]
    )
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text("\n".join(body), encoding="utf-8")
    return wrapper


def render_stage_plan(payload: dict[str, Any]) -> str:
    lines = [
        "# Stage180 Minimal Renormalization Proof",
        "",
        "This stage is a proof scaffold, not a final renormalization result.",
        "",
        "## Fixed Student",
        "",
        f"- fixed student paths: `{_path_csv(payload['fixed_student_path_ids'])}`",
        f"- reference proxy paths: `{_path_csv(payload['reference_teacher_proxy_path_ids'])}`",
        "",
        "## Comparison Arms",
        "",
        "| arm | runnable now | blocking tooling | purpose |",
        "|---|---:|---|---|",
    ]
    for arm in payload["comparison_arms"]:
        blockers = ", ".join(arm.get("blocking_tooling") or [])
        lines.append(f"| `{arm['id']}` | {arm['runnable_now']} | {blockers or 'none'} | {arm['purpose']} |")
    lines.extend(
        [
            "",
            "## Success Criteria",
            "",
        ]
    )
    for item in payload["success_criteria"]["primary_claim_requires"]:
        lines.append(f"- {item}")
    lines.extend(["", "## Metrics Required", ""])
    for item in payload["metrics_required"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def render_implementation_plan(payload: dict[str, Any]) -> str:
    return f"""# Stage180 Minimal Renormalization Proof Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove or falsify whether TECE-style projection/GN renormalized initialization improves the fixed rTECE student `{payload['student_config']['variant']}` over from-scratch training.

**Architecture:** Keep one frozen student path manifest and vary only initialization/distillation treatment. First run the currently available scratch and linear projection diagnostic arms; then add explicit initializer and teacher-residual tooling before running the two renormalized arms.

**Tech Stack:** Python, PyTorch, Lightning, ASE/matscipy neighbor lists, Slurm on SAI, existing `tace.scripts.rtece_train_scalar`, `analyze_rtece_projection_error.py`, `benchmark_rtece_scalar.py`, dimer/rattle diagnostics.

## Global Constraints

- Do not use `sbatch --export`, `#SBATCH --mem`, `#SBATCH --cpus-per-task`, or `set -u` in SAI wrappers.
- Rattle-relax is a continuous RMSD/force-tail metric, not a binary scientific gate.
- NEP/DPA comparisons remain non-rankable until data, hardware scope, metrics, physics tests, and throughput taxonomy are matched.
- The primary Stage180 claim requires the same student architecture across all arms.

---

### Task 1: Run Available Baselines

**Files:**
- Use: `{payload['artifacts']['wrappers']['projection_diagnostic']}`
- Use: `{payload['artifacts']['wrappers']['scratch_train']}`
- Use: `{payload['artifacts']['wrappers']['benchmark_physical_after_training']}`

**Interfaces:**
- Consumes: Stage180 manifest at `{payload['artifacts']['manifest']}`
- Produces: projection JSON, scratch checkpoint, E/F benchmark JSONs, dimer/rattle/physical JSONs

- [ ] **Step 1: Validate wrappers with Slurm**

```bash
sbatch --test-only {payload['artifacts']['wrappers']['projection_diagnostic']}
sbatch --test-only {payload['artifacts']['wrappers']['scratch_train']}
sbatch --test-only {payload['artifacts']['wrappers']['benchmark_physical_after_training']}
```

Expected: all three commands accepted by Slurm; no `--export`, mem, or cpus-per-task flags.

- [ ] **Step 2: Submit projection diagnostic**

```bash
sbatch {payload['artifacts']['wrappers']['projection_diagnostic']}
```

Expected: `{payload['artifacts']['diagnostics_root']}/stage180_projection_diagnostic.json` exists and reports descriptor, energy, and sampled force projection rows.

- [ ] **Step 3: Submit scratch training**

```bash
sbatch {payload['artifacts']['wrappers']['scratch_train']}
```

Expected: `{payload['artifacts']['results_root']}/scratch_same_student/rtece_scalar_best.pt` and `train_summary.json` exist.

- [ ] **Step 4: Submit scratch benchmark and physical diagnostics**

```bash
sbatch {payload['artifacts']['wrappers']['benchmark_physical_after_training']}
```

Expected: `scratch_same_student_dft_benchmark.json`, `scratch_same_student_teacher_benchmark.json`, scaling JSONs, dimer JSON, rattle JSON, and physical summary JSON exist under `{payload['artifacts']['diagnostics_root']}/scratch_same_student`.

### Task 2: Add Projection/GN Initializer Support

**Files:**
- Modify: `tace/scripts/rtece_train_scalar.py`
- Modify: `tace/lightning/rtece.py`
- Create: `benchmarks/oc20neb_tace_mace/initialize_rtece_from_projection.py`
- Test: `test/test_stage180_minimal_renormalization_proof.py`

**Interfaces:**
- Produces: `--init-checkpoint PATH` or `--init-state PATH` support before optimizer construction.
- Produces: projection-initialized checkpoint with the exact same `RTECEScalarConfig` as the fixed student.

- [ ] **Step 1: Write failing test**

```python
def test_stage180_requires_projection_initializer_before_renorm_arms_are_runnable():
    payload = make_stage180_manifest()
    arms = {{row["id"]: row for row in payload["comparison_arms"]}}
    assert arms["renorm_initialized_same_student"]["runnable_now"] is False
```

Expected before implementation: renorm arms remain blocked.

- [ ] **Step 2: Implement initializer**

Add a deterministic script that reads a projection coefficient artifact, creates the fixed student config, initializes compatible scalar-head parameters, saves `rtece_scalar_init.pt`, and records projection source metadata.

- [ ] **Step 3: Add train-entrypoint load hook**

Load the initialized state before Lightning module/optimizer construction. Reject config mismatch with a clear error.

- [ ] **Step 4: Update Stage180 manifest generator**

Mark `renorm_initialized_same_student` runnable only after wrapper generation includes a concrete initializer command and train command using the init state.

### Task 3: Add Teacher Residual Distillation Arm

**Files:**
- Create: `benchmarks/oc20neb_tace_mace/make_rtece_teacher_residual_cache.py`
- Modify: `tace/lightning/rtece.py`
- Modify: `tace/scripts/rtece_train_scalar.py`
- Test: `test/test_stage180_minimal_renormalization_proof.py`

**Interfaces:**
- Produces: versioned teacher residual label file with teacher hash, units, cutoff, PBC convention, energy residual, force residual, and source structure hash.
- Consumes: residual label file in training with explicit real-label/teacher-label loss weights.

- [ ] **Step 1: Write failing schema test**

```python
def test_teacher_residual_cache_requires_teacher_hash_units_and_force_residuals():
    record = make_teacher_residual_cache_record(...)
    assert record["teacher_model_hash"]
    assert record["units"] == {{"energy": "eV", "forces": "eV/A"}}
    assert "force_residual" in record
```

Expected before implementation: cache builder is missing.

- [ ] **Step 2: Implement cache builder**

Read DFT configs and teacher-labeled configs with matching structure order, compute residual E/F labels, and write a schema-versioned extxyz or JSONL cache.

- [ ] **Step 3: Add mixed real/teacher loss**

Extend training to consume the residual cache with explicit weights while preserving original DFT E/F loss.

- [ ] **Step 4: Run four-arm Stage180 comparison**

Run scratch, projection diagnostic, renorm init, and renorm+teacher residual distill under the same fixed student path set and report the required metrics.
"""


def audit_stage180_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    arms = {row.get("id"): row for row in payload.get("comparison_arms") or []}
    wrappers = payload.get("artifacts", {}).get("wrappers", {})
    wrapper_text = ""
    for path in wrappers.values():
        p = Path(path)
        if p.exists():
            wrapper_text += p.read_text(encoding="utf-8")
    forbidden = ("--export", "#SBATCH --mem", "--mem=", "#SBATCH --cpus-per-task", "--cpus-per-task", "set -u")
    checks = {
        "schema": payload.get("schema_version") == "rtece_stage180_minimal_renormalization_proof.v1",
        "stage": payload.get("stage") == "stage180_minimal_renormalization_proof",
        "fixed_path_set": payload.get("fixed_student_path_ids") == FIXED_STUDENT_PATH_IDS,
        "four_arms": set(arms) == {
            "scratch_same_student",
            "linear_projection_diagnostic",
            "renorm_initialized_same_student",
            "renorm_initialized_teacher_residual_distill",
        },
        "runnable_truth": arms.get("scratch_same_student", {}).get("runnable_now") is True
        and arms.get("linear_projection_diagnostic", {}).get("runnable_now") is True
        and arms.get("renorm_initialized_same_student", {}).get("runnable_now") is False
        and arms.get("renorm_initialized_teacher_residual_distill", {}).get("runnable_now") is False,
        "renorm_blockers": arms.get("renorm_initialized_same_student", {}).get("blocking_tooling")
        == [
            "checkpoint_initialization_from_projection_coefficients",
            "train_entrypoint_init_checkpoint_or_init_state",
        ],
        "success_claim": payload.get("success_criteria", {}).get("primary_claim_requires")
        == [
            "renorm_initialized_same_student beats scratch_same_student under matched E/F/relative/physical metrics",
            "or the result is recorded as falsifying the current renormalization recipe",
        ],
        "wrappers": set(wrappers) == {
            "projection_diagnostic",
            "scratch_train",
            "benchmark_physical_after_training",
        },
        "no_forbidden_sbatch_flags": not any(token in wrapper_text for token in forbidden),
    }
    failed = [key for key, ok in checks.items() if not ok]
    return {
        "schema_version": "rtece_stage180_manifest_audit.v1",
        "contract_pass": not failed,
        "failed_checks": failed,
        "checks": checks,
        "no_forbidden_sbatch_flags": checks["no_forbidden_sbatch_flags"],
    }


def materialize_stage180(payload: dict[str, Any]) -> dict[str, Any]:
    artifacts = payload["artifacts"]
    Path(artifacts["manifest"]).parent.mkdir(parents=True, exist_ok=True)
    write_projection_wrapper(artifacts["wrappers"]["projection_diagnostic"], payload)
    write_scratch_train_wrapper(artifacts["wrappers"]["scratch_train"], payload)
    write_benchmark_physical_wrapper(artifacts["wrappers"]["benchmark_physical_after_training"], payload)
    audit = audit_stage180_manifest(payload)
    Path(artifacts["manifest"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(artifacts["manifest_audit"]).write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    Path(artifacts["stage_plan"]).write_text(render_stage_plan(payload), encoding="utf-8")
    plan_path = Path(artifacts["implementation_plan"])
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(render_implementation_plan(payload), encoding="utf-8")
    return {"payload": payload, "audit": audit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    payload = make_stage180_manifest(output_root=args.output_root)
    result = materialize_stage180(payload)
    print(json.dumps(result["audit"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
