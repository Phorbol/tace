# TECE Renormalized Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a closed-loop OC20NEB experiment workflow for teacher-labeled reduced TACE students.

**Architecture:** Keep core TACE model code untouched for the first pass. Add benchmark-local scripts that generate teacher labels, generate reduced student configs from the current baseline config, run Slurm jobs, and summarize the results against TECE source-document priorities.

**Tech Stack:** Python 3.9+, ASE extxyz, PyTorch/TACE runtime, PyYAML, pytest, Slurm sbatch, existing `benchmarks/oc20neb_tace_mace` benchmark scripts.

## Global Constraints

- Do not modify core model internals unless the first closed-loop experiment proves the benchmark/config approach is insufficient.
- Preserve existing DFT labels when writing teacher-labeled extxyz files.
- Measure throughput and memory with `benchmark_models.py`; do not use parameter count as the decision metric.
- Prefer `CgtpInteraction` and OEq-compatible paths in the first round because TACE docs state that `SO2Interaction` currently lacks operator-fusion support.
- Keep existing uncommitted user changes untouched.

---

### Task 1: Teacher Label Utility

**Files:**
- Create: `test/test_oc20neb_distill_tools.py`
- Create: `benchmarks/oc20neb_tace_mace/distill_tace_labels.py`

**Interfaces:**
- Produces: `copy_atoms_with_teacher_labels(atoms_list, energies, forces, target_energy_key, target_forces_key, reference_prefix) -> list`
- Produces CLI: `distill_tace_labels.py --model TEACHER.ckpt --input in.extxyz --output out.extxyz`

- [ ] **Step 1: Write failing tests for label preservation and shape validation**

- [ ] **Step 2: Run the focused test and verify it fails because the script does not exist**

- [ ] **Step 3: Implement the pure label-copy helper and TACE inference CLI**

- [ ] **Step 4: Run the focused test and verify it passes**

### Task 2: Student Config Matrix Utility

**Files:**
- Modify: `test/test_oc20neb_distill_tools.py`
- Create: `benchmarks/oc20neb_tace_mace/make_tece_student_configs.py`

**Interfaces:**
- Produces: `build_variant_config(base_cfg, variant_name, train_file, valid_file) -> dict`
- Produces CLI: `make_tece_student_configs.py --base-config tace_oc20neb.yaml --train-file distill_train.extxyz --valid-file distill_valid.extxyz --output-dir configs`

- [ ] **Step 1: Write failing tests for the three named variants**

- [ ] **Step 2: Run the focused test and verify it fails because config generation does not exist**

- [ ] **Step 3: Implement deep-copy updates for `scalar_fast`, `edge_min`, and `path_scalar`**

- [ ] **Step 4: Run the focused test and verify it passes**

### Task 3: Slurm Runner and Review Summary

**Files:**
- Create: `benchmarks/oc20neb_tace_mace/tece_distill_matrix.sbatch`
- Create: `benchmarks/oc20neb_tace_mace/summarize_tece_distill.py`
- Modify: `benchmarks/oc20neb_tace_mace/README.md`

**Interfaces:**
- Runner consumes generated configs and existing `train_tace_oc20neb.sbatch` / `benchmark_oc20neb.sbatch` patterns.
- Summary consumes benchmark JSON files and emits a Markdown table with TECE source-document review prompts.

- [ ] **Step 1: Write a lightweight test for summary table ordering**

- [ ] **Step 2: Implement Slurm runner and summary script**

- [ ] **Step 3: Document the exact closed-loop commands**

- [ ] **Step 4: Run focused tests and Python syntax checks**
