# Stage173 Robust Group-Holdout Active Set Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Stage172 group-heldout residual projection statistically robust before using it to accept or reject rTECE path groups.

**Architecture:** Extend the existing projection analyzer with fold-local feature standardization, optional intercept, ridge-grid diagnostics, and an intercept-only per-atom baseline. Add an active-set gate that refuses candidate promotion unless the selected energy metric beats the intercept baseline, then materialize a Stage173 wrapper using the same TECE path ladder as Stage172.

**Tech Stack:** Python, PyTorch CPU linear algebra, pytest, existing rTECE projection analyzer and SAI-safe sbatch wrapper style.

## Global Constraints

- Keep `case_id` and related group IDs as split metadata only; never pass them as model descriptors.
- Do not use forbidden sbatch flags: `--export`, `--mem`, `--cpus-per-task`, or `set -u`.
- Preserve Stage172 raw result interpretation; Stage173 only tests whether baseline/standardization/ridge changes the conclusion.
- Commit only code, tests, plans, manifests, wrappers, and compact summaries; exclude raw projection JSON/checkpoints/logs.

---

### Task 1: Robust Projection Metrics

**Files:**
- Modify: `benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py`
- Test: `test/test_rtece_scalar.py`

**Interfaces:**
- Consumes: `energy_group_loocv_projection_metrics(...)`
- Produces: `constant_group_loocv_energy_baseline_metrics(...)`
- Produces new keyword args on `energy_group_loocv_projection_metrics(...)`: `include_intercept`, `standardize_features`, `ridge_values`

- [ ] **Step 1: Write failing tests**

Add tests that call `energy_group_loocv_projection_metrics(..., include_intercept=True, standardize_features=True, ridge_values=(0.0, 1.0))` and assert that the payload reports `energy_selected_ridge`, `energy_ridge_grid`, `energy_intercept_baseline_per_atom_rmse`, and `energy_beats_intercept_baseline`.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py -k 'robust_group_loocv or beat_intercept' -q
```

Expected: fail because the new keyword args and gate do not exist.

- [ ] **Step 3: Implement robust fold fitting**

Add fold-local standardization from fit rows only, append intercept after standardization, evaluate each ridge value, and record the best diagnostic row by per-atom RMSE when atom counts exist.

- [ ] **Step 4: Run tests and verify pass**

Run the focused pytest command above. Expected: pass.

---

### Task 2: Active-Set Intercept Gate

**Files:**
- Modify: `benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py`
- Test: `test/test_rtece_scalar.py`

**Interfaces:**
- Consumes: `rank_active_set_candidate_rows(...)`
- Produces new keyword arg: `require_beat_intercept`

- [ ] **Step 1: Write failing test**

Add a test where a candidate improves over a weak path baseline but remains worse than `energy_intercept_baseline_per_atom_rmse`, and assert rejection reason `intercept_baseline_not_beaten`.

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py -k beat_intercept -q
```

Expected: fail because `require_beat_intercept` is not implemented.

- [ ] **Step 3: Implement gate and CLI flag**

Add `--active-set-require-beat-intercept` and include the value in the payload active-set metadata.

- [ ] **Step 4: Run focused tests**

Run the focused pytest command. Expected: pass.

---

### Task 3: Stage173 Wrapper

**Files:**
- Create: `benchmarks/oc20neb_tace_mace/make_rtece_stage173_robust_group_holdout_active_set.py`
- Create: `runs/oc20neb_tace_mace/rtece-stage173-robust-group-holdout-active-set/stage173_manifest.json`
- Create: `runs/oc20neb_tace_mace/rtece-stage173-robust-group-holdout-active-set/stage173_manifest_audit.json`
- Create: `runs/oc20neb_tace_mace/rtece-stage173-robust-group-holdout-active-set/stage173_plan.md`
- Create: `runs/oc20neb_tace_mace/rtece-stage173-robust-group-holdout-active-set/stage173_robust_group_holdout_active_set.sbatch`
- Test: `test/test_rtece_scalar.py`

**Interfaces:**
- Consumes: Stage172 candidate ladder from `make_rtece_stage171_residual_active_set.py`
- Produces: `make_stage173_manifest`, `audit_stage173_manifest`, `materialize_stage173`

- [ ] **Step 1: Write failing manifest/wrapper test**

Assert schema, group-LOOCV split, robust projection flags, require-beat-intercept gate, and absence of forbidden sbatch flags.

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py -k stage173 -q
```

Expected: fail because the Stage173 generator does not exist.

- [ ] **Step 3: Implement generator**

Create the Stage173 generator by reusing Stage172 semantics plus robust analyzer options: `--energy-fit-intercept`, `--energy-standardize-features`, `--energy-ridge-grid 1e-8,1e-6,1e-4,1e-2,1e0,1e2`, and `--active-set-require-beat-intercept`.

- [ ] **Step 4: Materialize, verify, submit**

Run py_compile, focused pytest, manifest audit, forbidden sbatch flag search, then submit sbatch if checks pass.
