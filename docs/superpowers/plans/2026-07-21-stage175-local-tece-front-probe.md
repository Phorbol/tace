# Stage175 Local TECE Front Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether a deployable local TECE-style atomic-moment scalar front explains the Stage165/170 raw-energy residual better than Stage174 global proxies under robust group-heldout validation.

**Architecture:** Add a diagnostic feature builder that computes structure-level summaries from local low-rank species-conditioned Cartesian moments: `ell=0` shell density, `ell=1` vector norm squared, and `ell=2` quadrupole Frobenius squared. Evaluate the resulting feature family with the same Stage173/174 group-LOOCV intercept/standardization/ridge gate before any model architecture promotion.

**Tech Stack:** Python, ASE, NumPy, PyTorch, Stage174/Stage173 projection helpers, Slurm wrapper pattern used by recent rTECE stages.

## Global Constraints

- Do not use `case_id`, `source_key`, `source_frame`, `group_id`, or image identifiers as deployable features.
- Keep the local front exactly rotation-invariant by using scalar contractions of Cartesian moments.
- Use group-LOOCV by `case_id`, fold-local standardization, intercept, and ridge grid `[1e-8, 1e-6, 1e-4, 1e-2, 1, 100]`.
- A feature family is only architecture-promotable if it beats the intercept-only group-heldout per-atom residual baseline.
- SAI wrappers must not use `--export`, `--mem`, `--cpus-per-task`, or `set -u`.
- Commit only code, tests, plans, wrappers, manifests, and small summaries.

---

### Task 1: Local Moment Feature Builder

**Files:**
- Create: `benchmarks/oc20neb_tace_mace/analyze_rtece_local_front_features.py`
- Modify: `test/test_rtece_scalar.py`

**Interfaces:**
- Produces: `local_tece_front_feature_row_from_atoms(atoms, *, shell_edges: tuple[float, ...], chemistry_rank: int, l_max: int, center_condition: bool) -> tuple[dict[str, float], dict[str, object]]`
- Produces: `local_tece_front_feature_matrix(atoms_list, *, feature_family: str) -> tuple[torch.Tensor, list[str], dict[str, object]]`

- [ ] **Step 1: Write failing tests**

Add tests for forbidden metadata rejection and exact rotation invariance on a small molecule/slab-like structure.

- [ ] **Step 2: Run red test**

Run: `/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py -k 'stage175_local_front_feature' -q`

Expected: FAIL because the Stage175 module is missing.

- [ ] **Step 3: Implement local moment features**

Use fixed low-rank chemistry basis `[1, z/zmax, (z/zmax)^2, sqrt(z/zmax)]` truncated by `chemistry_rank`. For each center atom and shell, compute neighbor moment sums for `ell=0`, `ell=1`, and `ell=2`; contract to scalars and aggregate over centers with optional center chemistry conditioning.

- [ ] **Step 4: Run green test**

Run the same pytest command. Expected: PASS.

### Task 2: Stage175 Group-Heldout Probe

**Files:**
- Modify: `benchmarks/oc20neb_tace_mace/analyze_rtece_local_front_features.py`
- Modify: `test/test_rtece_scalar.py`

**Interfaces:**
- Produces: `evaluate_local_front_feature_family(...) -> dict[str, object]`
- Produces: `run_stage175_local_front_probe(configs: Path, stage165_json: Path, ...) -> dict[str, object]`

- [ ] **Step 1: Write failing synthetic probe test**

Create synthetic groups where a local shell moment feature predicts the heldout residual better than intercept-only. Assert the group-heldout metric beats the intercept baseline.

- [ ] **Step 2: Run red test**

Run: `/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py -k 'stage175_local_front_probe' -q`

Expected: FAIL because the evaluator is missing.

- [ ] **Step 3: Implement probe**

Load Stage165 residual targets, build local front matrices, call `energy_group_loocv_projection_metrics`, and emit JSON/Markdown summary rows with E RMSE/MAE/max, selected ridge, feature count, and intercept-beating flag.

- [ ] **Step 4: Run green test**

Run the same pytest command. Expected: PASS.

### Task 3: Stage175 Manifest And Wrapper

**Files:**
- Create: `benchmarks/oc20neb_tace_mace/make_rtece_stage175_local_tece_front_probe.py`
- Modify: `test/test_rtece_scalar.py`

**Interfaces:**
- Produces: `make_stage175_manifest(output_root: str | Path, ...) -> dict[str, object]`
- Produces: `audit_stage175_manifest(payload: dict[str, object]) -> dict[str, object]`
- Produces: `materialize_stage175(payload: dict[str, object]) -> dict[str, object]`

- [ ] **Step 1: Write failing manifest test**

Assert the manifest names Stage174, uses local TECE moment semantics, and materializes a wrapper with no forbidden Slurm flags.

- [ ] **Step 2: Run red test**

Run: `/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py -k 'stage175_manifest' -q`

Expected: FAIL because the generator is missing.

- [ ] **Step 3: Implement generator**

Follow the Stage174 generator pattern with output root `runs/oc20neb_tace_mace/rtece-stage175-local-tece-front-probe/`.

- [ ] **Step 4: Run green test**

Run the same pytest command. Expected: PASS.

### Task 4: Real Stage175 Diagnostic

**Files:**
- Create lightweight artifacts under `runs/oc20neb_tace_mace/rtece-stage175-local-tece-front-probe/`

- [ ] **Step 1: Materialize wrapper**

Run: `/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python benchmarks/oc20neb_tace_mace/make_rtece_stage175_local_tece_front_probe.py --output-root runs/oc20neb_tace_mace/rtece-stage175-local-tece-front-probe --limit-configs 512`

- [ ] **Step 2: Audit wrapper**

Run: `rg -- '--export|--mem|--cpus-per-task|set -u' runs/oc20neb_tace_mace/rtece-stage175-local-tece-front-probe/stage175_local_tece_front_probe.sbatch`

Expected: exit code 1, no matches.

- [ ] **Step 3: Submit and monitor Slurm**

Run: `sbatch runs/oc20neb_tace_mace/rtece-stage175-local-tece-front-probe/stage175_local_tece_front_probe.sbatch`, then check `sacct`.

- [ ] **Step 4: Interpret**

If local front features beat intercept, promote the winning local moment family to the next trainable rTECE architecture branch. If not, record that the current fixed local front is still insufficient and move toward teacher-projected/learnable path selection.

### Task 5: Verification And Commit

- [ ] **Step 1: Focused pytest**

Run: `/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py -k 'stage175 or stage174 or energy_group_loocv' -q`

- [ ] **Step 2: py_compile**

Run: `/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m py_compile benchmarks/oc20neb_tace_mace/analyze_rtece_local_front_features.py benchmarks/oc20neb_tace_mace/make_rtece_stage175_local_tece_front_probe.py`

- [ ] **Step 3: Diff check**

Run: `git diff --check && git diff --cached --check`

- [ ] **Step 4: Commit and push**

Commit Stage175 code/tests/plan/results and push to `phorbol/tece-renorm-distill`.
