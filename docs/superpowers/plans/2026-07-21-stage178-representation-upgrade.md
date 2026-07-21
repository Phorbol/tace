# Stage178 Representation Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Materialize the next rTECE experiment ladder after Stage176/177: add document-grounded front-loaded representation capacity and benchmark it with E/F RMSE, max error, physical probes, throughput, and memory fields.

**Architecture:** Stage178 does not add a parallel model implementation. It reuses the production `tace.scripts.rtece_train_scalar` entrypoint and existing rTECE scalar paths to compare three controlled representation increments: L0 local/species front, L1 sparse atomic cross-radial invariants, and minimal T3 cavity/direct edge sketches. The wrappers emit train, benchmark, and physical triage commands against the same checkpoint naming and benchmark parser contracts used by the repo.

**Tech Stack:** PyTorch, Lightning training backend, `tace.models.rtece_scalar`, `benchmark_rtece_scalar.py`, `dimer_scan_rtece.py`, `rattle_relax_rtece.py`, `summarize_rtece_physical_pareto.py`, Slurm no-export wrappers.

## Global Constraints

- Stay aligned with `TECE_design_space.md`: compare retained/deleted TECE path sets and hardware cost, not only parameter count or MAE.
- Stay aligned with `rTECE_review.md §P1`: prioritize radial low-rank sketches, sparse cross-radial invariants, explicit path registry, and cavity edge kernels.
- Do not widen only the final MLP head to create capacity; allocate capacity in the feature extractor/front representation.
- Report E/F MAE, RMSE, max error, dimer scan, rattle relax, atom-count throughput scaling, and memory fields before claiming a Pareto position.
- Do not use Slurm `--export`, `--mem`, `--cpus-per-task`, `set -u`, or shell `export` in generated wrappers.
- Do not commit checkpoints, full logs, caches, or large run directories.

---

### Task 1: Stage178 Manifest Contract

**Files:**
- Modify: `test/test_rtece_scalar.py`
- Create: `benchmarks/oc20neb_tace_mace/make_rtece_stage178_representation_upgrade.py`

**Interfaces:**
- Produces: `make_stage178_manifest(...) -> dict[str, Any]`
- Produces: `audit_stage178_manifest(payload: dict[str, Any]) -> dict[str, Any]`
- Produces: `materialize_stage178(payload: dict[str, Any]) -> dict[str, Any]`

- [x] **Step 1: Write failing test**

Add `test_stage178_representation_upgrade_manifest_builds_doc_grounded_ladder_and_safe_wrappers`, requiring:
- schema `rtece_stage178_representation_upgrade.v1`;
- three candidates: `stage178_l0_local_species`, `stage178_l1_atomic_cross`, `stage178_t3_minimal_cavity_direct`;
- RMSE-first benchmark protocol plus E/F MAE/RMSE/max fields;
- no head-only capacity allocation;
- no forbidden Slurm/shell patterns.

- [x] **Step 2: Verify RED**

Run:
`/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py::test_stage178_representation_upgrade_manifest_builds_doc_grounded_ladder_and_safe_wrappers -q`

Expected before implementation: `ModuleNotFoundError` for `make_rtece_stage178_representation_upgrade`.

- [x] **Step 3: Implement manifest generator**

Create `CANDIDATE_SPECS` with:
- L0 endpoint: radial density + learnable species basis + rank-4 local L0 chemistry front;
- L1 atomic cross: add vector norm and learnable atomic cross-radial projection with sketch rank 3;
- T3 minimal edge: add `edge.cavity.vector_dot` and `edge.direct.radial`.

- [x] **Step 4: Verify GREEN**

Run focused pytest and `py_compile`.

### Task 2: Production Wrapper Contract

**Files:**
- Modify: `benchmarks/oc20neb_tace_mace/make_rtece_stage178_representation_upgrade.py`
- Create by command: `runs/oc20neb_tace_mace/rtece-stage178-representation-upgrade/stage178_manifest.json`
- Create by command: `runs/oc20neb_tace_mace/rtece-stage178-representation-upgrade/wrappers/stage178_train_ladder_no_export.sbatch`
- Create by command: `runs/oc20neb_tace_mace/rtece-stage178-representation-upgrade/wrappers/stage178_benchmark_ladder_no_export.sbatch`
- Create by command: `runs/oc20neb_tace_mace/rtece-stage178-representation-upgrade/wrappers/stage178_physical_ladder_no_export.sbatch`

**Interfaces:**
- Train wrapper calls `python -m tace.scripts.rtece_train_scalar`.
- Benchmark wrapper calls `benchmark_rtece_scalar.py --model .../rtece_scalar_best.pt --output ... --variant ... --graph-construction-backend matscipy_neighborlist`.
- Physical wrapper calls `dimer_scan_rtece.py`, `rattle_relax_rtece.py`, and `summarize_rtece_physical_pareto.py --case variant:dft:teacher:dimer:rattle`.

- [x] **Step 1: Materialize artifacts**

Run:
`/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python benchmarks/oc20neb_tace_mace/make_rtece_stage178_representation_upgrade.py --output-root runs/oc20neb_tace_mace/rtece-stage178-representation-upgrade`

Expected: manifest audit `contract_pass: true`.

- [x] **Step 2: Scan forbidden wrapper patterns**

Run:
`rg -n -- '--export|#SBATCH --mem|#SBATCH --cpus-per-task|set -u|export ' runs/oc20neb_tace_mace/rtece-stage178-representation-upgrade/wrappers benchmarks/oc20neb_tace_mace/make_rtece_stage178_representation_upgrade.py`

Expected: no output, exit code 1.

- [x] **Step 3: Submit train ladder smoke**

Run after code commit or when ready for cluster time:
`sbatch runs/oc20neb_tace_mace/rtece-stage178-representation-upgrade/wrappers/stage178_train_ladder_no_export.sbatch`

Actual: submitted Slurm array job `688517` after `sbatch --test-only` accepted the `flood-1o2gpu` array wrapper; tasks `688517_0`, `688517_1`, and `688517_2` entered RUNNING. The earlier non-array `rush-1o2gpu` attempt `688511` was cancelled because `16V100` does not permit that QOS. Task `688517_2` exposed a production model bug where `local_l0_chemistry_front` was not forwarded into edge-relational moment recomputation; the plumbing fix is covered by `test_stage178_local_l0_front_can_be_combined_with_edge_relational_paths`, and T3 task 2 was resubmitted as `688528_2`.

Expected: three production rTECE checkpoints under `runs/oc20neb_tace_mace/rtece-stage178-representation-upgrade/results/*/rtece_scalar_best.pt`.

- [ ] **Step 4: Run benchmark and physical wrappers after training**

Run:
`sbatch runs/oc20neb_tace_mace/rtece-stage178-representation-upgrade/wrappers/stage178_benchmark_ladder_no_export.sbatch`

Then:
`sbatch runs/oc20neb_tace_mace/rtece-stage178-representation-upgrade/wrappers/stage178_physical_ladder_no_export.sbatch`

Expected: DFT/teacher benchmark JSON, scaling JSON for 32/128/512/1024, dimer/rattle JSON, and physical Pareto summaries per candidate.

### Task 3: Stage178 Analysis Handoff

**Files:**
- Future modify: `benchmarks/oc20neb_tace_mace/make_rtece_stage177_unified_pareto_audit.py` or successor Stage179 analyzer.

**Interfaces:**
- Consumes Stage178 train summaries, DFT/teacher benchmark JSON, scaling JSON, and physical Pareto JSON.
- Produces a comparable row set against Stage176, Stage157, NEP, and DPA-like baselines.

- [ ] **Step 1: Normalize Stage178 rows**

After jobs complete, add rows with E/F MAE/RMSE/max, relative NEB metrics, atoms/s, memory, dimer/rattle gates, and training controls.

- [ ] **Step 2: Interpret against NEP/DPA**

Only claim superiority if data/hardware/physical scopes match. Otherwise report exactly which columns are not protocol-aligned.

- [ ] **Step 3: Decide next algorithm move**

If L1 improves RMSE/physical behavior at modest throughput cost, promote atomic cross-radial front. If only T3 improves physical behavior, prioritize fused cavity-edge implementation. If neither improves energy RMSE, return to teacher residual projection/renormalized initialization rather than further widening the head.
