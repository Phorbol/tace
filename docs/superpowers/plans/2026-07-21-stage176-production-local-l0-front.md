# Stage176 Production Local L0 Front Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the Stage175 local L0 rank3 diagnostic into the production rTECE model/training/ASE path.

**Architecture:** Add a manifest-addressable atomic scalar path that keeps the TECE degradation clean: one neighbor pass, early scalarization, low-rank center-neighbor chemistry gates, no persistent equivariant state, and no extra edge relation lifetime. The first supported realization is L0 only, rank-controlled, and trainable where Stage175 indicated fixed low-rank local density has group-heldout signal.

**Tech Stack:** PyTorch, `tace.models.rtece_scalar`, `tace.lightning.rtece`, `tace.scripts.rtece_train_scalar`, existing rTECE path manifest and ASE calculator.

## Global Constraints

- Stay aligned with `TECE_design_space.md`: model changes must be expressible as a retained/deleted TECE path set and a hardware-cost axis.
- Stay aligned with `rTECE_review.md`: use production package entry points, preserve PBC edge-shift semantics, retain per-element E0 fitting, and mark analytic/Triton force backends as unsupported for trainable new front until derivatives are implemented.
- Do not use Slurm `--export`, `--mem`, `--cpus-per-task`, or `set -u` in generated sbatch wrappers.
- Do not commit raw checkpoints, large logs, caches, or full run directories.

---

### Task 1: Model-Path Contract And Descriptor Test

**Files:**
- Modify: `test/test_rtece_scalar.py`
- Modify: `tace/models/rtece_scalar.py`

**Interfaces:**
- Consumes: `build_rtece_config_from_path_ids`, `atomic_scalar_descriptors`, `rtece_path_manifest`, `rtece_route_contract`, `RTECEScalarModel`.
- Produces: `atomic.local_l0_lowrank_density` scalar path and `local_l0_chemistry_rank` config field.

- [ ] **Step 1: Write failing tests**

Add tests that require:
- `build_rtece_config_from_path_ids(..., ("atomic.radial_density", "atomic.local_l0_lowrank_density"), local_l0_chemistry_rank=3)` to create a 32-feature descriptor for `num_radial=8`: 8 base density + 8*3 local rank features.
- descriptors to be rotation invariant and chemistry sensitive.
- manifest to expose `moment.l0.local_lowrank_density`, `atomic.local_l0_lowrank_density`, `trainable_local_l0_chemistry_front`, and `early_scalarized_local_l0_density`.
- analytic/Triton inference backend contract to reject the trainable local front.

- [ ] **Step 2: Verify RED**

Run:
`/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py -k 'stage176 or local_l0_lowrank' -q`

Expected: fail because the path id and config field are not implemented.

- [ ] **Step 3: Implement production model support**

Add `local_l0_chemistry_rank: int = 0` to `RTECEScalarConfig`; normalize it as non-negative. Implement a small trainable `LocalL0ChemistryFront` initialized to Stage175's fixed basis `[1, z, z^2, sqrt(z)]` truncated to rank, with center and neighbor embeddings plus a zero-initialized projection/gate so the new path starts as a conservative fixed low-rank density front.

- [ ] **Step 4: Verify GREEN**

Run the same focused pytest command and expect all selected tests to pass.

### Task 2: Training CLI And Lightning Plumbing

**Files:**
- Modify: `tace/scripts/rtece_train_scalar.py`
- Modify: `tace/lightning/rtece.py`
- Modify: `test/test_rtece_scalar.py`

**Interfaces:**
- Consumes: Task 1 config field.
- Produces: `--local-l0-chemistry-rank` in both step-loop and Lightning training paths.

- [ ] **Step 1: Write failing tests**

Add tests that `tace.lightning.rtece.build_training_config` and CLI-style `tace.scripts.rtece_train_scalar.build_training_config` pass `local_l0_chemistry_rank=3` through path-id configs.

- [ ] **Step 2: Verify RED**

Run:
`/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py -k 'stage176 or local_l0_lowrank' -q`

Expected: fail on missing keyword/attribute.

- [ ] **Step 3: Implement plumbing**

Add the CLI option and pass it through `build_training_config`, `fit_rtece_lightning`, summaries, and manifest payload/checkpoint config loading.

- [ ] **Step 4: Verify GREEN**

Run the focused pytest command and expect all selected tests to pass.

### Task 3: Stage176 Smoke Wrapper And Manifest

**Files:**
- Create: `benchmarks/oc20neb_tace_mace/make_rtece_stage176_local_l0_front.py`
- Create: `runs/oc20neb_tace_mace/rtece-stage176-production-local-l0-front/stage176_manifest.json`
- Create: `runs/oc20neb_tace_mace/rtece-stage176-production-local-l0-front/stage176_manifest_audit.json`
- Create: `runs/oc20neb_tace_mace/rtece-stage176-production-local-l0-front/stage176_train_smoke.sbatch`

**Interfaces:**
- Consumes: production CLI path.
- Produces: safe Slurm smoke wrapper for a small true train/inference path, not a projection-only script.

- [ ] **Step 1: Write failing generator test**

Add a test that the generated wrapper contains `--scalar-path-ids atomic.radial_density,atomic.local_l0_lowrank_density`, `--local-l0-chemistry-rank 3`, and none of the forbidden Slurm flags.

- [ ] **Step 2: Verify RED**

Run the focused pytest command and expect failure due missing generator.

- [ ] **Step 3: Implement generator**

Generate a minimal Stage176 train smoke using the existing full training CLI, `max_steps` small, per-element E0 enabled, Lightning backend, warmup/scheduler available, and no forbidden Slurm flags.

- [ ] **Step 4: Verify and optionally submit**

Run pytest, `py_compile`, wrapper forbidden-flag scan, manifest audit, and submit a short sbatch smoke if the wrapper materializes successfully.
