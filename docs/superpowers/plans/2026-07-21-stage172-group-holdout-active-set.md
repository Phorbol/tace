# Stage172 Group-Heldout Active-Set Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a case/group-heldout residual active-set projection diagnostic that tests whether Stage171 deployable rTECE paths generalize the low-frequency energy offset residual to unseen cases.

**Architecture:** Reuse the Stage171 TECE/rTECE candidate ladder and Stage165 residual target, but replace deterministic config-heldout evaluation with group-LOOCV keyed by extxyz metadata. The group key is split metadata only and is never passed as a descriptor, baseline feature, or model input.

**Tech Stack:** Python, PyTorch linear projection diagnostics, ASE extxyz metadata, existing rTECE path manifest/wrapper machinery, SAI Slurm sbatch.

## Global Constraints

- Keep `case_id` non-deployable: split/group metadata only.
- Preserve TECE_design_space.md Stage B active-set semantics and rTECE_review.md group-heldout validation requirement.
- Use E RMSE/MAE/max and per-atom metrics for active-set ranking; do not rely only on MAE.
- Sbatch wrappers must not use `--export`, `--mem`, `--cpus-per-task`, or `set -u`.

---

### Task 1: Add Group-LOOCV Energy Projection Metrics

**Files:**
- Modify: `benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py`
- Test: `test/test_rtece_scalar.py`

- [x] Write a failing unit test for `energy_group_loocv_projection_metrics`.
- [x] Implement LOOCV over unique group labels with the group labels excluded from features.
- [x] Verify the unit test passes.

### Task 2: Add Analyzer CLI Group Split

**Files:**
- Modify: `benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py`
- Test: `test/test_rtece_scalar.py`

- [x] Write a failing CLI test for `--energy-split-mode group-loocv --energy-group-key case_id`.
- [x] Add group-label loading from extxyz info.
- [x] Route energy projection rows through group-LOOCV metrics.
- [x] Emit split metadata in projection JSON.

### Task 3: Add Stage172 Wrapper

**Files:**
- Create: `benchmarks/oc20neb_tace_mace/make_rtece_stage172_group_holdout_active_set.py`
- Create: `runs/oc20neb_tace_mace/rtece-stage172-group-holdout-active-set/stage172_*`
- Test: `test/test_rtece_scalar.py`

- [x] Write a failing Stage172 manifest/wrapper test.
- [x] Reuse Stage171 candidate ladder and switch to group-LOOCV split.
- [x] Materialize manifest, audit, plan, and sbatch wrapper.
- [ ] Run focused tests, py_compile, and wrapper audit.
- [ ] Commit and push small source/manifest artifacts.
- [ ] Submit the Stage172 Slurm job when verification is clean.
