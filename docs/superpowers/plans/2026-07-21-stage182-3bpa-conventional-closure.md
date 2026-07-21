# Stage182 3BPA Conventional Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a clean 3BPA conventional-MD closure scaffold that compares the current TECE-derived scalar rTECE student against NEP4 and a DeepMD DPA1 zero-attention-style baseline on the same train/test splits.

**Architecture:** Stage182 reuses the existing rTECE production CLI plus Stage145 NEP/DeepMD conversion and benchmark tools. It only adds a manifest/wrapper/result-summary layer so the comparison is controlled by dataset split, label keys, train budget, and metric contract.

**Tech Stack:** Python, ASE extxyz, rTECE Lightning CLI, GPUMD/NEP wrappers, DeepMD-kit wrappers, Slurm sbatch on SAI.

## Global Constraints

- Do not use `--export`, shell `export`, `#SBATCH --mem`, `--mem=`, `#SBATCH --cpus-per-task`, `--cpus-per-task`, or `set -u` in generated sbatch wrappers.
- Use 3BPA labels `energy` and `forces` in eV/eV-A without unit conversion.
- Primary ranking metric is `rmse_f_mev_a`; report `rmse_e_mev_atom`, E/F max errors, throughput, and peak memory when present.
- Do not judge rattle-relax as binary pass/fail; it is a later continuous RMSD/force-tail physical probe.

---

### Task 1: Stage182 Manifest And Wrapper Scaffold

**Files:**
- Create: `benchmarks/oc20neb_tace_mace/make_rtece_stage182_3bpa_closure.py`
- Test: `test/test_stage181_conventional_datasets.py`

**Interfaces:**
- Consumes: `tace.scripts.rtece_train_scalar`, `benchmarks/oc20neb_tace_mace/convert_stage145_nep.py`, `benchmarks/oc20neb_tace_mace/convert_stage145_deepmd.py`, `benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py`, `benchmarks/oc20neb_tace_mace/benchmark_stage145_community.py`.
- Produces: `make_stage182_manifest`, `materialize_stage182`, `audit_stage182_manifest`, `summarize_stage182_results`.

- [x] **Step 1: Write failing tests**

Tests require rTECE, NEP4, and DeepMD DPA1-zero rows; all four 3BPA benchmark splits; RMSE/max/throughput/peak-memory metrics; and SAI-safe wrappers.

- [x] **Step 2: Verify tests fail before implementation**

Run: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_stage181_conventional_datasets.py::test_stage182_3bpa_closure_manifest_compares_rtece_nep_and_dpalike test/test_stage181_conventional_datasets.py::test_stage182_3bpa_closure_wrappers_are_sai_safe_and_split_complete test/test_stage181_conventional_datasets.py::test_stage182_summary_marks_missing_results_and_orders_by_force_rmse -q`
Expected before implementation: `ModuleNotFoundError: No module named 'benchmarks.oc20neb_tace_mace.make_rtece_stage182_3bpa_closure'`.

- [x] **Step 3: Implement scaffold**

Add manifest construction, wrapper generation, audit, and summary collection in the new Stage182 module.

- [x] **Step 4: Verify tests pass**

Run the same command.
Expected after implementation: `3 passed`.

### Task 2: Materialize And Run Stage182

**Files:**
- Generated: `runs/oc20neb_tace_mace/rtece-stage182-3bpa-closure/stage182_manifest.json`
- Generated: `runs/oc20neb_tace_mace/rtece-stage182-3bpa-closure/wrappers/*.sbatch`
- Generated: `runs/oc20neb_tace_mace/rtece-stage182-3bpa-closure/benchmark_wrappers/*.sbatch`

**Interfaces:**
- Consumes: local `datasets/3BPA/dataset_3BPA/*.xyz`.
- Produces: submitted train jobs for rTECE, NEP4, and DeepMD DPA1-zero; then submitted benchmark jobs for completed models.

- [ ] **Step 1: Materialize wrappers**

Run: `/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python benchmarks/oc20neb_tace_mace/make_rtece_stage182_3bpa_closure.py`
Expected: JSON listing manifest, audit, wrappers, and benchmark_wrappers.

- [ ] **Step 2: Submit training jobs if dataset is present**

Run: `sbatch runs/oc20neb_tace_mace/rtece-stage182-3bpa-closure/wrappers/rtece_l1_local_l0_train300k_no_export.sbatch`, `sbatch runs/oc20neb_tace_mace/rtece-stage182-3bpa-closure/wrappers/nep4_train300k_no_export.sbatch`, and `sbatch runs/oc20neb_tace_mace/rtece-stage182-3bpa-closure/wrappers/deepmd_dpa1_zero_train300k_no_export.sbatch`.
Expected: three Slurm job ids or explicit module/data availability errors recorded in logs.

- [ ] **Step 3: Submit benchmark jobs after model artifacts exist**

Run each generated benchmark wrapper with `sbatch`.
Expected: one JSON per row per split under `diagnostics/<row>/<split>_benchmark.json`.

- [ ] **Step 4: Summarize**

Run: `/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python benchmarks/oc20neb_tace_mace/make_rtece_stage182_3bpa_closure.py --summarize-only`
Expected: `stage182_summary.json` and `stage182_summary.md` with completed and missing rows.
