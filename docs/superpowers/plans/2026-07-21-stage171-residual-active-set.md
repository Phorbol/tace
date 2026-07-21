# Stage171 Residual Active Set Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Stage171 manifest/wrapper for TECE-aligned active-set projection against the Stage170 case-offset energy residual.

**Architecture:** Add a focused wrapper generator under `benchmarks/oc20neb_tace_mace/` that reuses the existing rTECE path manifest and projection analyzer. The new stage targets deployable descriptor projection of the low-frequency case/site/adsorbate residual instead of raw energy or non-deployable case-id offsets.

**Tech Stack:** Python stdlib, existing `analyze_rtece_projection_error.py`, rTECE scalar path ids, pytest, Slurm sbatch wrapper.

## Global Constraints

- Keep alignment with `/home/gengjianrui/bin/TECE_design_space.md`: projection/importance/active-set before uncontrolled architecture widening.
- Keep alignment with `/home/gengjianrui/bin/rTECE_review.md`: no case-id feature, use deployable semantic scalar/edge paths, and keep teacher/downfolding route explicit.
- SAI sbatch wrapper must not use `--export`, `--mem`, `--cpus-per-task`, or `set -u`.
- Do not commit large feature caches, checkpoints, raw train directories, or logs.

---

### Task 1: Stage171 Manifest and Wrapper

**Files:**
- Create: `benchmarks/oc20neb_tace_mace/make_rtece_stage171_residual_active_set.py`
- Test: `test/test_rtece_scalar.py`
- Create: `runs/oc20neb_tace_mace/rtece-stage171-residual-active-set/stage171_plan.md`
- Create: `runs/oc20neb_tace_mace/rtece-stage171-residual-active-set/stage171_manifest.json`
- Create: `runs/oc20neb_tace_mace/rtece-stage171-residual-active-set/stage171_manifest_audit.json`
- Create: `runs/oc20neb_tace_mace/rtece-stage171-residual-active-set/stage171_residual_active_set.sbatch`

**Interfaces:**
- Consumes: Stage165/170 residual diagnosis files and existing rTECE scalar path ids.
- Produces: `make_stage171_manifest(output_root, ...) -> dict[str, Any]`, `audit_stage171_manifest(payload) -> dict[str, Any]`, `materialize_stage171(payload) -> dict[str, Any]`.

- [ ] **Step 1: Write failing pytest**

Add `test_stage171_residual_active_set_manifest_is_tece_aligned_and_sbatch_safe` to `test/test_rtece_scalar.py`. It imports `make_stage171_manifest`, `audit_stage171_manifest`, and `materialize_stage171`, builds a tmp manifest, asserts schema, residual target semantics, no case-id feature, candidate ladder contains edge-relational paths, and wrapper excludes forbidden sbatch patterns.

- [ ] **Step 2: Verify RED**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py -k stage171_residual_active_set -q
```

Expected: FAIL because `benchmarks.oc20neb_tace_mace.make_rtece_stage171_residual_active_set` does not exist.

- [ ] **Step 3: Implement minimal generator**

Create the Stage171 generator with explicit candidate path ladder and SAI-safe sbatch text. The wrapper should call an existing or subsequent analyzer with a residual-target mode; Stage171 is a formal plan/wrapper artifact, not the heavy projection result.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py -k stage171_residual_active_set -q
```

Expected: PASS.

- [ ] **Step 5: Materialize artifacts and static checks**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python benchmarks/oc20neb_tace_mace/make_rtece_stage171_residual_active_set.py --output-root runs/oc20neb_tace_mace/rtece-stage171-residual-active-set
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m py_compile benchmarks/oc20neb_tace_mace/make_rtece_stage171_residual_active_set.py
jq -e '.contract_pass == true' runs/oc20neb_tace_mace/rtece-stage171-residual-active-set/stage171_manifest_audit.json
```

Expected: all commands exit 0.

- [ ] **Step 6: Commit and push**

Stage only the generator, test, plan, and small Stage171 manifest/wrapper artifacts. Exclude caches and logs.
