# Stage180 Minimal Renormalization Proof Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove or falsify whether TECE-style projection/GN renormalized initialization improves the fixed rTECE student `stage180_fixed_student_l1_atomic_cross` over from-scratch training.

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
- Use: `/tmp/pytest-of-gengjianrui/pytest-1094/test_stage180_generator_runs_a0/stage180/wrappers/stage180_projection_diagnostic_no_export.sbatch`
- Use: `/tmp/pytest-of-gengjianrui/pytest-1094/test_stage180_generator_runs_a0/stage180/wrappers/stage180_scratch_train_no_export.sbatch`
- Use: `/tmp/pytest-of-gengjianrui/pytest-1094/test_stage180_generator_runs_a0/stage180/wrappers/stage180_benchmark_physical_no_export.sbatch`

**Interfaces:**
- Consumes: Stage180 manifest at `/tmp/pytest-of-gengjianrui/pytest-1094/test_stage180_generator_runs_a0/stage180/stage180_manifest.json`
- Produces: projection JSON, scratch checkpoint, E/F benchmark JSONs, dimer/rattle/physical JSONs

- [ ] **Step 1: Validate wrappers with Slurm**

```bash
sbatch --test-only /tmp/pytest-of-gengjianrui/pytest-1094/test_stage180_generator_runs_a0/stage180/wrappers/stage180_projection_diagnostic_no_export.sbatch
sbatch --test-only /tmp/pytest-of-gengjianrui/pytest-1094/test_stage180_generator_runs_a0/stage180/wrappers/stage180_scratch_train_no_export.sbatch
sbatch --test-only /tmp/pytest-of-gengjianrui/pytest-1094/test_stage180_generator_runs_a0/stage180/wrappers/stage180_benchmark_physical_no_export.sbatch
```

Expected: all three commands accepted by Slurm; no `--export`, mem, or cpus-per-task flags.

- [ ] **Step 2: Submit projection diagnostic**

```bash
sbatch /tmp/pytest-of-gengjianrui/pytest-1094/test_stage180_generator_runs_a0/stage180/wrappers/stage180_projection_diagnostic_no_export.sbatch
```

Expected: `/tmp/pytest-of-gengjianrui/pytest-1094/test_stage180_generator_runs_a0/stage180/diagnostics/stage180_projection_diagnostic.json` exists and reports descriptor, energy, and sampled force projection rows.

- [ ] **Step 3: Submit scratch training**

```bash
sbatch /tmp/pytest-of-gengjianrui/pytest-1094/test_stage180_generator_runs_a0/stage180/wrappers/stage180_scratch_train_no_export.sbatch
```

Expected: `/tmp/pytest-of-gengjianrui/pytest-1094/test_stage180_generator_runs_a0/stage180/results/scratch_same_student/rtece_scalar_best.pt` and `train_summary.json` exist.

- [ ] **Step 4: Submit scratch benchmark and physical diagnostics**

```bash
sbatch /tmp/pytest-of-gengjianrui/pytest-1094/test_stage180_generator_runs_a0/stage180/wrappers/stage180_benchmark_physical_no_export.sbatch
```

Expected: `scratch_same_student_dft_benchmark.json`, `scratch_same_student_teacher_benchmark.json`, scaling JSONs, dimer JSON, rattle JSON, and physical summary JSON exist under `/tmp/pytest-of-gengjianrui/pytest-1094/test_stage180_generator_runs_a0/stage180/diagnostics/scratch_same_student`.

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
    arms = {row["id"]: row for row in payload["comparison_arms"]}
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
    assert record["units"] == {"energy": "eV", "forces": "eV/A"}
    assert "force_residual" in record
```

Expected before implementation: cache builder is missing.

- [ ] **Step 2: Implement cache builder**

Read DFT configs and teacher-labeled configs with matching structure order, compute residual E/F labels, and write a schema-versioned extxyz or JSONL cache.

- [ ] **Step 3: Add mixed real/teacher loss**

Extend training to consume the residual cache with explicit weights while preserving original DFT E/F loss.

- [ ] **Step 4: Run four-arm Stage180 comparison**

Run scratch, projection diagnostic, renorm init, and renorm+teacher residual distill under the same fixed student path set and report the required metrics.
