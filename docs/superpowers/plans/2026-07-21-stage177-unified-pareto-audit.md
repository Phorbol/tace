# Stage177 Unified Pareto Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a lightweight evidence collector that compares current rTECE candidates, Stage176, NEP, and DPA-like baselines under one accuracy/physical/throughput schema.

**Architecture:** Stage177 does not train a new model. It reads existing committed or local lightweight summaries, normalizes them into one Pareto-audit schema, marks missing evidence explicitly, and emits the next benchmark actions needed to make Stage176 comparable. The audit separates raw DFT E/F metrics, relative NEB energy metrics, physical extrapolation probes, training controls, throughput, memory, and TECE semantic path metadata.

**Tech Stack:** Python 3 standard library, existing JSON/Markdown artifacts under `runs/oc20neb_tace_mace`, pytest.

## Global Constraints

- Stay aligned with `TECE_design_space.md`: rank candidates by deployment Sobolev-style evidence, not by MAE alone or parameter count alone.
- Stay aligned with `rTECE_review.md`: report E/F RMSE, MAE, max error, physical dimer/rattle status, throughput, memory, E0/reference semantics, and missing output contracts.
- Treat Stage176 as `benchmark_missing` until a real DFT/teacher benchmark and physical triage have been run from its production checkpoint.
- Do not claim NEP/DPA/rTECE superiority unless all compared rows have matching benchmark scope and physical evidence.
- Do not commit raw checkpoints, large logs, converted datasets, or full run directories.

---

### Task 1: Stage177 Schema And Existing-Evidence Collector

**Files:**
- Modify: `test/test_rtece_scalar.py`
- Create: `benchmarks/oc20neb_tace_mace/make_rtece_stage177_unified_pareto_audit.py`
- Create by command: `runs/oc20neb_tace_mace/rtece-stage177-unified-pareto-audit/stage177_pareto_audit.json`
- Create by command: `runs/oc20neb_tace_mace/rtece-stage177-unified-pareto-audit/stage177_pareto_audit.md`

**Interfaces:**
- Produces: `make_stage177_audit(output_root: str | Path = DEFAULT_OUTPUT_ROOT) -> dict[str, Any]`
- Produces: `audit_stage177_payload(payload: dict[str, Any]) -> dict[str, Any]`
- Produces: `materialize_stage177(payload: dict[str, Any]) -> dict[str, str]`
- Produces rows with `candidate`, `family`, `benchmark_status`, `physics_status`, `throughput_status`, `rankable_now`, and normalized DFT metric keys.

- [ ] **Step 1: Write failing tests**

Add tests requiring:
- rows for `stage176_local_l0_rank3`, `stage157_direct_b32_rel0p25_mixed2048`, `nep4_mixed_smoke`, and `deepmd_dpa_like_mixed_smoke`;
- Stage176 to be present but not rankable because DFT benchmark, physical triage, and throughput scaling are missing;
- Stage157 and Stage145 rows to carry RMSE/max/throughput metrics when source summaries exist;
- comparison caveats to include non-matching throughput protocols and missing Stage176 full benchmark.

- [ ] **Step 2: Verify RED**

Run:
`/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py -k stage177 -q`

Expected: fail because the Stage177 module does not exist.

- [ ] **Step 3: Implement collector**

Create the Stage177 script with small JSON readers for:
- `runs/oc20neb_tace_mace/rtece-stage176-production-local-l0-front/stage176_manifest.json`
- `runs/oc20neb_tace_mace/rtece-stage176-production-local-l0-front/train_smoke/train_summary.json`
- `runs/oc20neb_tace_mace/rtece-stage157-relative-neb-loss/stage157_metrics.json`
- `runs/oc20neb_tace_mace/community-baselines-stage145/stage145_results_summary.json`

Normalize values into one schema and never infer missing values as zero.

- [ ] **Step 4: Verify GREEN**

Run the Stage177 pytest selection, `py_compile`, `git diff --check`, and inspect the generated JSON/Markdown.

### Task 2: Next-Benchmark Action Manifest

**Files:**
- Modify: `benchmarks/oc20neb_tace_mace/make_rtece_stage177_unified_pareto_audit.py`
- Modify: `test/test_rtece_scalar.py`

**Interfaces:**
- Produces `next_required_actions` with concrete Stage176 benchmark, physical triage, and throughput scaling actions.

- [ ] **Step 1: Write failing tests**

Require the payload to include actions:
- `stage176_dft_teacher_benchmark`
- `stage176_dimer_scan`
- `stage176_rattle_relax`
- `stage176_atom_count_throughput_scaling`
- `community_baseline_protocol_alignment`

- [ ] **Step 2: Verify RED**

Run the focused Stage177 tests and confirm failure on missing actions.

- [ ] **Step 3: Implement actions**

Add explicit commands or command templates where the existing repo entry points are known, otherwise add `status: "needs_wrapper"` rather than an invented command.

- [ ] **Step 4: Verify GREEN**

Run focused tests and materialize the Stage177 artifacts.
