# Stage174 Low-Frequency Feature Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether deployable low-frequency chemistry/site/front features can explain the Stage165/170 raw-energy case-offset residual under robust group-heldout validation before promoting any new rTECE architecture.

**Architecture:** Add a focused diagnostic script that builds feature matrices from ASE structures only, rejects non-deployable metadata as model features, and evaluates each feature family with the Stage173 robust group-LOOCV energy projection gate. Add a generator that materializes a SAI-safe wrapper, manifest, audit, and markdown plan for the diagnostic.

**Tech Stack:** Python, ASE, NumPy, PyTorch projection helpers from `benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py`, existing Stage171/173 wrapper patterns, pytest.

## Global Constraints

- Do not use `case_id`, `source_key`, `source_frame`, `group_id`, or image identifiers as deployable features; `case_id` may only define group-heldout folds and align Stage165 residual targets.
- If `tags` are used, mark the feature family with `requires_tags=true`; also provide tag-free feature families.
- Use group-LOOCV by `case_id`, fold-local standardization, intercept, and ridge grid `[1e-8, 1e-6, 1e-4, 1e-2, 1, 100]`.
- A candidate is promotion-supported only if it beats the intercept-only group-heldout per-atom residual baseline.
- Keep SAI wrappers free of `--export`, `--mem`, `--cpus-per-task`, and `set -u`.
- Commit only lightweight manifests/summaries/wrappers; leave raw projection outputs local.

---

### Task 1: Low-Frequency Feature Builder Tests

**Files:**
- Modify: `test/test_rtece_scalar.py`
- Create: `benchmarks/oc20neb_tace_mace/analyze_rtece_lowfreq_features.py`

**Interfaces:**
- Produces: `lowfreq_feature_row_from_atoms(atoms, *, include_tags: bool, include_geometry: bool, include_pair_histogram: bool = False, pair_bins: tuple[float, ...] = (1.5, 2.5, 3.5, 5.0)) -> tuple[dict[str, float], dict[str, object]]`
- Produces: `lowfreq_feature_matrix(atoms_list, *, feature_family: str) -> tuple[torch.Tensor, list[str], dict[str, object]]`

- [ ] **Step 1: Write failing tests**

Add tests that create small ASE `Atoms` objects with forbidden metadata and tags. Assert that tag features exist only for tag-aware families, forbidden metadata never appears in feature names, and metadata reports rejected fields.

- [ ] **Step 2: Run tests to verify failure**

Run: `/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py -k 'stage174_lowfreq_feature_builder' -q`

Expected: FAIL because `benchmarks.oc20neb_tace_mace.analyze_rtece_lowfreq_features` does not exist.

- [ ] **Step 3: Implement feature builder**

Implement deterministic feature rows:
- `composition_fraction`: `natoms`, `frac_z{Z}`, `count_z{Z}`.
- `tag_composition`: `tag{t}_frac_z{Z}`, `tag{t}_count_z{Z}`, `tag{t}_frac_atoms`, marked `requires_tags=true`.
- `geometry_z_profile`: cell lengths, volume/atom, z mean/std/span, and tag-wise z stats when tags are enabled.
- `pair_histogram`: coarse deployable element-pair radial counts/fractions with small fixed bins.

- [ ] **Step 4: Run tests to verify pass**

Run the same pytest command. Expected: PASS.

### Task 2: Stage174 Group-Heldout Probe

**Files:**
- Modify: `benchmarks/oc20neb_tace_mace/analyze_rtece_lowfreq_features.py`
- Modify: `test/test_rtece_scalar.py`

**Interfaces:**
- Produces: `run_stage174_lowfreq_probe(configs: Path, stage165_json: Path, *, limit_configs: int, group_key: str, stage165_variant: str, output_json: Path | None = None, output_md: Path | None = None) -> dict[str, object]`
- Consumes: `load_stage165_case_offset_residual_targets`, `load_config_group_labels`, and `energy_group_loocv_projection_metrics`.

- [ ] **Step 1: Write failing synthetic probe test**

Create a four-group synthetic fixture where tag composition predicts heldout offsets better than intercept-only. Assert the tag candidate beats the intercept baseline while metadata candidates are excluded.

- [ ] **Step 2: Run tests to verify failure**

Run: `/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py -k 'stage174_lowfreq_probe' -q`

Expected: FAIL because the probe runner is missing.

- [ ] **Step 3: Implement group-heldout probe**

For each feature family, load configs, build a feature matrix, call `energy_group_loocv_projection_metrics` with intercept, standardization, and ridge grid, then compare to the intercept baseline. Output JSON and markdown with RMSE/MAE/max, selected ridge, `beats_intercept`, feature count, and `requires_tags`.

- [ ] **Step 4: Run focused tests**

Run the same pytest command. Expected: PASS.

### Task 3: Stage174 Manifest And SAI Wrapper

**Files:**
- Create: `benchmarks/oc20neb_tace_mace/make_rtece_stage174_lowfreq_feature_probe.py`
- Modify: `test/test_rtece_scalar.py`

**Interfaces:**
- Produces: `make_stage174_manifest(output_root: str | Path, ...) -> dict[str, object]`
- Produces: `audit_stage174_manifest(payload: dict[str, object]) -> dict[str, object]`
- Produces: `materialize_stage174(payload: dict[str, object]) -> dict[str, object]`

- [ ] **Step 1: Write failing manifest test**

Assert the manifest names the Stage173 negative result, has no forbidden metadata features, includes both tag-free and tag-required feature families, uses the robust ridge grid, and materializes a wrapper without forbidden Slurm flags.

- [ ] **Step 2: Run tests to verify failure**

Run: `/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py -k 'stage174_manifest' -q`

Expected: FAIL because the generator is missing.

- [ ] **Step 3: Implement generator**

Follow the Stage173 wrapper pattern but call `analyze_rtece_lowfreq_features.py`. Use `set -eo pipefail`, no forbidden Slurm directives, and write manifest/audit/plan/wrapper paths under `runs/oc20neb_tace_mace/rtece-stage174-lowfreq-feature-probe/`.

- [ ] **Step 4: Run manifest tests**

Run the same pytest command. Expected: PASS.

### Task 4: Materialize, Run, Summarize

**Files:**
- Create lightweight files under `runs/oc20neb_tace_mace/rtece-stage174-lowfreq-feature-probe/`

**Interfaces:**
- Consumes: Stage174 generator CLI.
- Produces: `stage174_results_summary.json`, `stage174_results_summary.md`.

- [ ] **Step 1: Materialize wrapper**

Run: `/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python benchmarks/oc20neb_tace_mace/make_rtece_stage174_lowfreq_feature_probe.py --output-root runs/oc20neb_tace_mace/rtece-stage174-lowfreq-feature-probe --limit-configs 512`

- [ ] **Step 2: Audit wrapper**

Run: `rg -- '--export|--mem|--cpus-per-task|set -u' runs/oc20neb_tace_mace/rtece-stage174-lowfreq-feature-probe/stage174_lowfreq_feature_probe.sbatch`

Expected: exit code 1, no matches.

- [ ] **Step 3: Submit and monitor sbatch**

Run: `sbatch runs/oc20neb_tace_mace/rtece-stage174-lowfreq-feature-probe/stage174_lowfreq_feature_probe.sbatch`, then check `sacct` until completion.

- [ ] **Step 4: Summarize result**

If no feature family beats intercept, record that the next architecture must add a richer deployable local TECE front or change dataset/coverage. If a feature family beats intercept, record it as the first candidate front module for the next trainable rTECE stage.

### Task 5: Verification And Commit

**Files:**
- Modify/create all Stage174 files above.

**Interfaces:**
- Consumes: focused tests and generated summaries.
- Produces: committed lightweight Stage174 artifacts.

- [ ] **Step 1: Run focused pytest**

Run: `/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py -k 'stage174 or energy_group_loocv' -q`

- [ ] **Step 2: Run py_compile**

Run: `/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m py_compile benchmarks/oc20neb_tace_mace/analyze_rtece_lowfreq_features.py benchmarks/oc20neb_tace_mace/make_rtece_stage174_lowfreq_feature_probe.py`

- [ ] **Step 3: Git diff check**

Run: `git diff --check`

- [ ] **Step 4: Commit and push**

Commit only code, tests, plan, wrapper, manifest/audit, and small summaries. Push to `phorbol/tece-renorm-distill`.
