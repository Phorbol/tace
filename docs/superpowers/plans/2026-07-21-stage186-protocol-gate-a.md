# Stage186 Protocol Foundations And Gate A Exact Downfolding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the immutable Stage186 protocol layer and prove exact linear semantic Schur downfolding end to end before changing nonlinear rTECE training.

**Architecture:** Add canonical Stage186 hashes and manifests without changing legacy `rtece_path_manifest.v1`, materialize a leakage-free 3BPA split, enforce optional Stage186 checkpoint compatibility, and strengthen SAI wrapper auditing. Implement Gate A in a separate `tace.renormalization` package with an explicit coefficient-per-state linear checkpoint; do not route it through the nonlinear rTECE MLP or the historical Stage180 prefit initializer.

**Tech Stack:** Python 3.11, NumPy, PyTorch checkpoint serialization, ASE/extxyz, pytest, JSON/NPZ schemas, existing rTECE path manifests, Slurm `sbatch --test-only`.

## Global Constraints

- Source of truth: `docs/superpowers/specs/2026-07-21-tece-renorm-core-proof-design.md`, `/home/gengjianrui/bin/TECE_design_space.md`, and `/home/gengjianrui/bin/rTECE_review.md`.
- Stage180 remains historical ordinary E/F prefit evidence; no new code may relabel it as Schur or GN renormalization.
- Preserve `rtece_path_manifest.v1` and its 16-character legacy `manifest_hash`; Stage186 adds full SHA256 hashes alongside it.
- Gate A uses an explicit linear semantic model. Do not map rTECE MLP head weights to semantic path coefficients.
- Never form an explicit matrix inverse. Use rank-revealing least squares, pseudoinverse application, or stable linear solves.
- Unregularized Schur and anchored damped results are separate schema fields and separate claims.
- Build training/validation only from 3BPA `train_300K.xyz`; all named test files remain untouched.
- Fit per-element E0 and E/F normalization scales from the materialized training split only.
- SAI wrappers must not contain `--export`, shell `export`, `#SBATCH --mem`, `--mem=`, `#SBATCH --cpus-per-task`, `--cpus-per-task`, or `set -u`.
- No Stage186 GPU training or cross-engine Pareto ranking belongs to this plan.
- Every implementation and review subagent uses `gpt-5.6-terra` with `reasoning_effort=xhigh`.

---

### Task 1: Canonical Stage186 Protocol Hashes

**Files:**
- Create: `tace/models/rtece_protocol.py`
- Test: `test/test_stage186_protocol_foundations.py`

**Interfaces:**
- Produces: `canonical_json_bytes(value) -> bytes`
- Produces: `sha256_json(value) -> str`
- Produces: `sha256_file(path) -> str`
- Produces: `build_compatibility_tuple(...) -> dict[str, str]`
- Produces: `validate_compatibility_tuple(payload, expected=None) -> dict[str, str]`
- Produces: `write_canonical_json(path, payload) -> str`

- [ ] **Step 1: Write failing protocol tests**

```python
from pathlib import Path

import pytest

from tace.models.rtece_protocol import (
    build_compatibility_tuple,
    canonical_json_bytes,
    sha256_json,
    validate_compatibility_tuple,
)


def test_stage186_canonical_json_and_hash_ignore_mapping_order():
    left = {"b": [2, 1], "a": {"x": 1.0}}
    right = {"a": {"x": 1.0}, "b": [2, 1]}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert sha256_json(left) == sha256_json(right)
    assert sha256_json(left).startswith("sha256:")
    assert len(sha256_json(left)) == 71


def test_stage186_compatibility_tuple_rejects_missing_or_tampered_fields():
    payload = build_compatibility_tuple(
        operator_manifest_hash="sha256:" + "1" * 64,
        feature_schema_hash="sha256:" + "2" * 64,
        model_config_hash="sha256:" + "3" * 64,
        implementation_revision="git:abc123",
        teacher_checkpoint_hash="sha256:" + "4" * 64,
        teacher_config_hash="sha256:" + "5" * 64,
        data_manifest_hash="sha256:" + "6" * 64,
    )
    assert validate_compatibility_tuple(payload) == payload
    with pytest.raises(ValueError, match="data_manifest_hash"):
        validate_compatibility_tuple({k: v for k, v in payload.items() if k != "data_manifest_hash"})
    with pytest.raises(ValueError, match="compatibility mismatch"):
        validate_compatibility_tuple(payload, {**payload, "teacher_config_hash": "sha256:" + "9" * 64})
```

- [ ] **Step 2: Run the focused test and confirm import failure**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest -q test/test_stage186_protocol_foundations.py
```

Expected: collection fails because `tace.models.rtece_protocol` does not exist.

- [ ] **Step 3: Implement the canonical protocol module**

Implement these exact constants and signatures:

```python
STAGE186_COMPATIBILITY_SCHEMA = "rtece_stage186_compatibility.v1"
COMPATIBILITY_FIELDS = (
    "operator_manifest_hash",
    "feature_schema_hash",
    "model_config_hash",
    "implementation_revision",
    "teacher_checkpoint_hash",
    "teacher_config_hash",
    "data_manifest_hash",
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_json(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
```

`build_compatibility_tuple` must require every field in `COMPATIBILITY_FIELDS`, reject blank values, add `schema_version`, and add `compatibility_hash=sha256_json(payload_without_hash)`. `validate_compatibility_tuple` must recompute the hash, reject missing/extra semantic fields, and compare every field when `expected` is provided. `write_canonical_json` writes indented sorted JSON with a final newline and returns the payload SHA256.

- [ ] **Step 4: Run focused tests**

Run the Task 1 command. Expected: all tests in `test_stage186_protocol_foundations.py` pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add tace/models/rtece_protocol.py test/test_stage186_protocol_foundations.py
git commit -m "Add Stage186 canonical protocol hashes"
```

### Task 2: Bind Active-Set Projection To An Immutable Operator Manifest

**Files:**
- Modify: `tace/models/rtece_protocol.py`
- Create: `benchmarks/oc20neb_tace_mace/make_rtece_stage186_protocol.py`
- Modify: `benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py`
- Modify: `test/test_stage186_protocol_foundations.py`

**Interfaces:**
- Consumes: existing Stage171 ordered path IDs and `rtece_path_manifest(...)`
- Produces dataclasses: `OperatorEntry`, `OperatorManifest`
- Produces: `build_stage186_operator_manifest(config, *, retained_groups, implementation_revision) -> OperatorManifest`
- Produces: `write_operator_manifest(path, manifest) -> str`
- Produces: `load_operator_manifest(path) -> OperatorManifest`
- Produces: `validate_projection_operator_binding(payload, operator_manifest) -> None`
- Adds CLI: `analyze_rtece_projection_error.py --operator-manifest PATH`

- [ ] **Step 1: Add failing operator-binding tests**

```python
def test_stage186_projection_binds_exact_operator_order_and_hash(tmp_path):
    manifest = build_stage186_operator_manifest(
        build_rtece_config("rtece_atomic_moments"),
        retained_groups={"L0": ["atomic.radial_density"]},
        implementation_revision="git:test",
    )
    assert manifest.schema_version == \"rtece_stage186_operator_manifest.v1\"
    assert manifest.operator_manifest_hash.startswith(\"sha256:\")
    payload = {
        \"operator_manifest_schema\": manifest.schema_version,
        \"operator_manifest_hash\": manifest.operator_manifest_hash,
        \"operator_path_ids\": list(manifest.operator_path_ids),
    }
    validate_projection_operator_binding(payload, manifest)
    payload[\"operator_path_ids\"] = list(reversed(payload[\"operator_path_ids\"]))
    with pytest.raises(ValueError, match=\"operator path order\"):
        validate_projection_operator_binding(payload, manifest)
```

Add a CLI-level test that passes `--operator-manifest`, reads the emitted JSON, and asserts the full hash/order fields. Keep the existing test expectation for `rtece_projection_diagnostic.v1`; Stage186 binding is additive.

- [ ] **Step 2: Run the new binding tests and confirm failure**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest -q test/test_stage186_protocol_foundations.py -k operator
```

Expected: import or missing-argument failure.

- [ ] **Step 3: Implement manifest construction and validation**

The operator manifest must contain:

```python
{
    "schema_version": "rtece_stage186_operator_manifest.v1",
    "legacy_path_manifest_schema": legacy["schema_version"],
    "legacy_path_manifest_hash": legacy["manifest_hash"],
    "operator_path_ids": [row["id"] for row in legacy["scalar_paths"]],
    "retained_groups": {name: list(path_ids) for name, path_ids in retained_groups.items()},
    "feature_schema": {"units": {"distance": "A"}, "dtype": "float64"},
    "implementation_revision": implementation_revision,
}
```

Compute `operator_manifest_hash` over the payload before adding that field. Reject duplicate path IDs, unknown retained IDs, non-nested declared groups, or a manifest whose recomputed hash differs.

When `--operator-manifest` is supplied, the analyzer must validate candidate paths against this exact order and add `operator_manifest_schema`, `operator_manifest_hash`, and `operator_path_ids` to the top-level projection JSON and each active-set row. Without the argument, preserve legacy behavior byte-for-byte except formatting.

- [ ] **Step 4: Run binding and legacy projection tests**

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest -q \
  test/test_stage186_protocol_foundations.py \
  test/test_rtece_scalar.py -k 'projection_cli_emits_active_set_candidate_rows or stage186_projection or active_set'
```

Expected: all selected tests pass; the former schema-drift assertion includes `require_beat_intercept=False` consistently rather than removing the implementation field.

- [ ] **Step 5: Commit Task 2**

```bash
git add tace/models/rtece_protocol.py \
  benchmarks/oc20neb_tace_mace/make_rtece_stage186_protocol.py \
  benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py \
  test/test_stage186_protocol_foundations.py test/test_rtece_scalar.py
git commit -m "Bind Stage186 projections to operator manifests"
```

### Task 3: Materialize The Leakage-Free 3BPA Split And Training-Only E0

**Files:**
- Create: `benchmarks/oc20neb_tace_mace/prepare_rtece_stage186_data.py`
- Modify: `test/test_stage186_protocol_foundations.py`

**Interfaces:**
- Produces: `choose_validation_window(source_hash, *, split_seed=186) -> int`
- Produces: `split_stage186_indices(num_frames, source_hash, *, split_seed=186, window_size=100, embargo=5) -> dict[str, list[int] | int]`
- Produces: `materialize_stage186_3bpa_split(dataset_root, output_root, *, split_seed=186) -> dict[str, Any]`
- Artifacts: `train.extxyz`, `valid.extxyz`, `data_manifest.json`, `e0.json`

- [ ] **Step 1: Write deterministic split tests**

```python
def test_stage186_split_uses_one_window_and_nonwrapping_embargo():
    split = split_stage186_indices(500, "sha256:" + "a" * 64)
    valid = split["validation_indices"]
    train = split["training_indices"]
    embargo = split["embargo_indices"]
    assert len(valid) == 100
    assert valid == list(range(valid[0], valid[0] + 100))
    assert not set(train) & set(valid)
    assert not set(train) & set(embargo)
    assert not set(valid) & set(embargo)
    assert min(train + valid + embargo) >= 0
    assert max(train + valid + embargo) < 500


def test_stage186_data_manifest_never_uses_named_test_files(tmp_path):
    dataset = make_synthetic_3bpa_tree(tmp_path / "3bpa", num_train=500)
    manifest = materialize_stage186_3bpa_split(dataset, tmp_path / "out")
    assert manifest["source_split"] == "train_300K.xyz"
    assert manifest["test_files_touched"] == []
    assert manifest["e0_fit"]["source"] == "training_indices_only"
    assert manifest["normalization"]["energy_scale_mev_atom"] > 0
    assert manifest["normalization"]["force_scale_mev_a"] > 0
```

The test helper writes 500 small ASE `Atoms` frames with valid `energy` and `forces` labels and four sentinel test files whose hashes are checked before/after materialization.

- [ ] **Step 2: Run split tests and confirm failure**

Run the Task 1 test command with `-k split`. Expected: import failure.

- [ ] **Step 3: Implement the exact split algorithm**

Use:

```python
window = int(hashlib.sha256(f"186|{source_hash}".encode("ascii")).hexdigest(), 16) % 5
valid_start = window * 100
valid_stop = valid_start + 100
left_embargo = range(max(0, valid_start - 5), valid_start)
right_embargo = range(valid_stop, min(500, valid_stop + 5))
```

Training indices are all remaining source indices. Reject any source count other than 500 for the Stage186 3BPA protocol. Read only `train_300K.xyz`; calculate and record hashes of named test files without opening them through ASE.

Fit per-element E0 by the same weighted least-squares convention as `tace.lightning.rtece.fit_atomic_energies`, using training frames only. Store integer atomic-number keys, solver rank, residual RMSE, and source indices in `e0.json`. Define scales exactly as the spec: standard deviation of E0-subtracted energy per atom with a 1 meV/atom floor, and RMS of all training force components with a 1 meV/A floor.

- [ ] **Step 4: Run focused data tests**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest -q \
  test/test_stage186_protocol_foundations.py -k 'split or data_manifest or e0'
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 3**

```bash
git add benchmarks/oc20neb_tace_mace/prepare_rtece_stage186_data.py \
  test/test_stage186_protocol_foundations.py
git commit -m "Materialize leakage-free Stage186 3BPA data"
```

### Task 4: Enforce Optional Stage186 Checkpoint Compatibility

**Files:**
- Modify: `tace/models/rtece_workflow.py`
- Modify: `tace/lightning/rtece.py`
- Modify: `test/test_stage186_protocol_foundations.py`
- Modify: `test/test_rtece_scalar.py`

**Interfaces:**
- Extends: `save_checkpoint(..., compatibility: Mapping[str, str] | None = None)`
- Extends: `load_checkpoint(..., expected_compatibility: Mapping[str, str] | None = None)`
- Extends: `load_rtece_init_state(..., expected_compatibility: Mapping[str, str] | None = None)`
- Preserves: loading historical checkpoints when no expected Stage186 tuple is requested

- [ ] **Step 1: Add tamper and legacy tests**

```python
def test_stage186_checkpoint_recomputes_and_rejects_compatibility_tamper(tmp_path):
    compatibility = make_test_compatibility_tuple()
    path = tmp_path / "model.pt"
    save_checkpoint(path, model, config, compatibility=compatibility)
    _, _, metadata = load_checkpoint(path, expected_compatibility=compatibility)
    assert metadata["stage186_compatibility"] == compatibility
    payload = torch.load(path, weights_only=False)
    payload["stage186_compatibility"]["data_manifest_hash"] = "sha256:" + "0" * 64
    torch.save(payload, path)
    with pytest.raises(ValueError, match="compatibility"):
        load_checkpoint(path, expected_compatibility=compatibility)


def test_legacy_checkpoint_load_remains_supported_without_stage186_expectation(tmp_path):
    path = tmp_path / "legacy.pt"
    save_checkpoint(path, model, config)
    load_checkpoint(path)
    with pytest.raises(ValueError, match="missing Stage186 compatibility"):
        load_checkpoint(path, expected_compatibility=make_test_compatibility_tuple())
```

- [ ] **Step 2: Run compatibility tests and confirm signature failure**

Run focused tests with `-k compatibility`. Expected: unexpected keyword argument failures.

- [ ] **Step 3: Implement additive checkpoint enforcement**

When saving with `compatibility`, call `validate_compatibility_tuple` and store the validated copy under `stage186_compatibility`. On every load of a checkpoint containing the field, validate its internal hash even if the caller supplies no expectation. When an expectation is supplied, reject missing compatibility or any field mismatch before `model.load_state_dict`.

`load_rtece_init_state` must apply the same rule before copying state. It must keep the existing config equality check and return the validated tuple in initializer metadata. Do not change legacy path-manifest hashes.

- [ ] **Step 4: Run checkpoint regression tests**

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest -q \
  test/test_stage186_protocol_foundations.py \
  test/test_rtece_scalar.py -k 'checkpoint or init_state or compatibility'
```

Expected: selected Stage186 and legacy tests pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add tace/models/rtece_workflow.py tace/lightning/rtece.py \
  test/test_stage186_protocol_foundations.py test/test_rtece_scalar.py
git commit -m "Enforce Stage186 checkpoint compatibility"
```

### Task 5: Audit Complete SAI Wrapper Bodies And Test-Only Evidence

**Files:**
- Modify: `benchmarks/oc20neb_tace_mace/audit_rtece_wrapper_contract.py`
- Modify: `test/test_stage186_protocol_foundations.py`
- Modify: `test/test_rtece_scalar.py`

**Interfaces:**
- Produces: `forbidden_wrapper_tokens(text) -> list[str]`
- Produces: `record_sbatch_test_only(wrapper, *, runner=subprocess.run) -> dict[str, Any]`
- Extends: `audit_wrapper_file(..., require_test_only=False, test_only_result=None)`

- [ ] **Step 1: Write full-body audit tests**

```python
@pytest.mark.parametrize(
    "line, token",
    [
        ("export FOO=bar", "shell export"),
        ("set -u", "set -u"),
        ("#SBATCH --export=ALL", "--export"),
        ("#SBATCH --mem=8G", "--mem"),
        ("#SBATCH --cpus-per-task=4", "--cpus-per-task"),
        ("srun --mem=8G command", "--mem="),
    ],
)
def test_stage186_wrapper_audit_rejects_forbidden_full_body_tokens(line, token):
    assert token in forbidden_wrapper_tokens("#!/bin/bash\n" + line + "\n")


def test_stage186_wrapper_audit_requires_successful_test_only_record(tmp_path):
    wrapper = tmp_path / "ok.sbatch"
    wrapper.write_text("#!/bin/bash\n#SBATCH --partition=16V100\npython -m x\n")
    result = {"command": ["sbatch", "--test-only", str(wrapper)], "returncode": 0, "stdout": "ok", "stderr": ""}
    row = audit_wrapper_file(wrapper, expected_exports={}, require_test_only=True, test_only_result=result)
    assert row["contract_pass"] is True
```

- [ ] **Step 2: Run wrapper tests and confirm failure**

Run focused tests with `-k wrapper_audit`. Expected: missing-function/signature failure.

- [ ] **Step 3: Implement the audit**

Scan stripped non-comment shell lines for `export ` and exact shell option `set -u`; scan every line for all forbidden option forms. Avoid matching explanatory comments unless the line is a `#SBATCH` directive. `record_sbatch_test_only` runs exactly `sbatch --test-only <wrapper>`, captures text output, and returns a JSON-safe record. It must not submit a job.

Keep legacy export expectation support for historical audits, but Stage186 calls `expected_exports={}` and rejects every shell export. Include `test_only` evidence and `test_only_pass` in rows and `wrapper_audit.json`.

- [ ] **Step 4: Run wrapper regressions**

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest -q \
  test/test_stage186_protocol_foundations.py \
  test/test_rtece_scalar.py -k 'wrapper_contract or wrapper_audit'
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 5**

```bash
git add benchmarks/oc20neb_tace_mace/audit_rtece_wrapper_contract.py \
  test/test_stage186_protocol_foundations.py test/test_rtece_scalar.py
git commit -m "Audit complete Stage186 Slurm wrappers"
```

### Task 6: Define Gate A Linear Problem And Operator-State Schemas

**Files:**
- Create: `tace/renormalization/__init__.py`
- Create: `tace/renormalization/linear_semantic.py`
- Create: `test/test_stage186_gate_a_linear_semantic.py`

**Interfaces:**
- Consumes: `OperatorManifest` from `tace.models.rtece_protocol`
- Produces dataclass: `LinearSemanticProblem`
- Produces: `write_linear_problem(path, problem) -> None`
- Produces: `load_linear_problem(path) -> LinearSemanticProblem`

- [ ] **Step 1: Write schema round-trip and rejection tests**

```python
def test_gate_a_linear_problem_roundtrip_preserves_rows_and_partitions(tmp_path):
    problem, manifest = make_known_gate_a_problem()
    write_operator_manifest(tmp_path / "operator_manifest.json", manifest)
    write_linear_problem(tmp_path / "linear_problem.npz", problem)
    loaded = load_linear_problem(tmp_path / "linear_problem.npz")
    np.testing.assert_array_equal(loaded.retained_indices, problem.retained_indices)
    np.testing.assert_array_equal(loaded.deleted_indices, problem.deleted_indices)
    np.testing.assert_allclose(loaded.A_E, problem.A_E)
    assert loaded.operator_manifest_hash == manifest.operator_manifest_hash


def test_gate_a_manifest_requires_one_coefficient_state_key_per_operator():
    _, manifest = make_known_gate_a_problem()
    broken = replace(manifest, operators=manifest.operators[:-1])
    with pytest.raises(ValueError, match="coefficient mapping"):
        broken.validate()
```

- [ ] **Step 2: Run Gate A tests and confirm module failure**

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest -q \
  test/test_stage186_gate_a_linear_semantic.py
```

Expected: collection fails because `tace.renormalization` does not exist.

- [ ] **Step 3: Implement immutable schema dataclasses**

`LinearSemanticProblem` contains float64 `A_E`, `A_F`, `A_V`, diagonal `weights`, `theta`, integer retained/deleted indices, fit/held-out row indices, units, dtype, gauge, operator-manifest hash, and metadata. Reuse `OperatorManifest` serialization and validation from `tace.models.rtece_protocol`; do not define a second operator schema in `tace.renormalization`. Problem validation requires:

```python
num_rows = A_E.shape[0] + A_F.shape[0] + A_V.shape[0]
assert weights.shape == (num_rows,)
assert theta.shape == (A_E.shape[1],)
assert sorted(np.concatenate([retained_indices, deleted_indices]).tolist()) == list(range(theta.size))
assert not set(retained_indices.tolist()) & set(deleted_indices.tolist())
assert sorted(np.concatenate([fit_rows, heldout_rows]).tolist()) == list(range(num_rows))
assert np.all(np.isfinite(A)) and np.all(weights >= 0)
```

The NPZ stores arrays without pickles and one UTF-8 JSON metadata scalar. Loading uses `allow_pickle=False`, validates schema/hash/shape/dtype, and rejects unknown schema versions.

- [ ] **Step 4: Run schema tests**

Run the Task 6 command. Expected: schema tests pass.

- [ ] **Step 5: Commit Task 6**

```bash
git add tace/renormalization/__init__.py tace/renormalization/linear_semantic.py \
  test/test_stage186_gate_a_linear_semantic.py
git commit -m "Define Gate A linear semantic schemas"
```

### Task 7: Solve Exact Schur And Anchored Damped Downfolding

**Files:**
- Modify: `tace/renormalization/linear_semantic.py`
- Modify: `test/test_stage186_gate_a_linear_semantic.py`

**Interfaces:**
- Produces dataclasses: `ResidualMetrics`, `DirectFit`, `ProjectionFit`
- Produces: `solve_direct_retained_wls(problem, *, rcond=1e-12) -> DirectFit`
- Produces: `solve_exact_downfold(problem, *, damping=0.0, rcond=1e-12) -> ProjectionFit`
- Produces: `projection_fit_payload(fit) -> dict[str, Any]`

- [ ] **Step 1: Write analytic equality and separation tests**

```python
def test_gate_a_exact_schur_matches_direct_weighted_least_squares():
    problem, _ = make_known_gate_a_problem()
    direct = solve_direct_retained_wls(problem)
    fit = solve_exact_downfold(problem, damping=0.0)
    np.testing.assert_allclose(fit.retained_coefficients, direct.coefficients, rtol=1e-10, atol=1e-12)
    assert fit.exact_schur is not None
    assert fit.anchored_damped is None
    assert fit.fit_residual.weighted_rmse == pytest.approx(direct.fit_residual.weighted_rmse, rel=1e-10)


def test_gate_a_damped_fit_never_relabels_augmented_objective_as_schur():
    problem, _ = make_known_gate_a_problem()
    fit = solve_exact_downfold(problem, damping=1e-4)
    payload = projection_fit_payload(fit)
    assert payload["exact_schur"] is None
    assert payload["anchored_damped"]["lambda"] == pytest.approx(1e-4)
    assert "augmented_objective" in payload["anchored_damped"]
    assert "heldout_observable_residual" in payload
```

Add tests for rank-deficient `G_RR`, zero-weight virial rows, a relevant deleted group increasing residual, and batching/row-order invariance.

- [ ] **Step 2: Run solver tests and confirm missing-function failure**

Run the Task 6 command. Expected: imports fail for solver functions.

- [ ] **Step 3: Implement stable weighted solves**

Stack `A=[A_E; A_F; A_V]`, select fit rows, and whiten with `sqrt(weights)`. The independent direct retained-space solve uses the declared anchored gauge: solve `Phi_R delta ~= Phi_D theta_D` with `np.linalg.lstsq(..., rcond=rcond)` and set `beta=theta_R+delta`. This keeps rank-deficient null-space components anchored to `theta_R`.

For exact Schur, solve `G_RR delta = G_RD theta_D` by rank-revealing least squares and set `beta=theta_R+delta`. For damping, solve the augmented system:

```python
aug_A = np.concatenate([Phi_R, np.sqrt(damping) * np.eye(num_retained)], axis=0)
aug_b = np.concatenate([Phi_R @ theta_R + Phi_D @ theta_D, np.sqrt(damping) * theta_R])
beta, _, rank, singular_values = np.linalg.lstsq(aug_A, aug_b, rcond=rcond)
```

Report fit and held-out observable residuals from the unaugmented `A/W`; report the anchored penalty and augmented objective only under `anchored_damped`. Record rank, singular values, condition estimate, `rcond`, and solver status. Never call `np.linalg.inv`.

- [ ] **Step 4: Run all Gate A solver tests**

Run the Task 6 command. Expected: all solver/schema tests pass.

- [ ] **Step 5: Commit Task 7**

```bash
git add tace/renormalization/linear_semantic.py \
  test/test_stage186_gate_a_linear_semantic.py
git commit -m "Implement exact Gate A downfolding"
```

### Task 8: Save And Replay The Retained Linear Checkpoint

**Files:**
- Create: `tace/renormalization/linear_checkpoint.py`
- Modify: `tace/renormalization/__init__.py`
- Modify: `test/test_stage186_gate_a_linear_semantic.py`

**Interfaces:**
- Produces: `RetainedLinearModel(torch.nn.Module)`
- Produces: `save_retained_linear_checkpoint(path, model, manifest, compatibility, metadata) -> None`
- Produces: `load_retained_linear_checkpoint(path, *, expected_compatibility) -> tuple[RetainedLinearModel, OperatorManifest, dict[str, Any]]`

- [ ] **Step 1: Write checkpoint replay/tamper tests**

```python
def test_gate_a_checkpoint_writes_each_retained_coefficient_and_replays(tmp_path):
    problem, manifest = make_known_gate_a_problem()
    fit = solve_exact_downfold(problem)
    model = RetainedLinearModel.from_manifest(manifest, fit.retained_coefficients)
    path = tmp_path / "retained.pt"
    save_retained_linear_checkpoint(path, model, manifest, make_test_compatibility_tuple(), {})
    loaded, loaded_manifest, _ = load_retained_linear_checkpoint(
        path, expected_compatibility=make_test_compatibility_tuple()
    )
    np.testing.assert_allclose(
        loaded(problem.stacked_A[:, problem.retained_indices]).detach().numpy(),
        problem.stacked_A[:, problem.retained_indices] @ fit.retained_coefficients,
    )
    assert set(loaded.state_dict()) == {entry.state_dict_key for entry in loaded_manifest.retained_operators}
```

- [ ] **Step 2: Run checkpoint tests and confirm failure**

Run Gate A tests with `-k checkpoint`. Expected: missing module failure.

- [ ] **Step 3: Implement explicit coefficient state**

Register one scalar `torch.nn.Parameter` per retained operator using the manifest's exact safe state key. `forward(design)` stacks parameters in manifest order and returns `design @ coefficients`. Save schema, manifest payload/hash, state dict, compatibility tuple, and metadata. Load validates all hashes and exact state-key set before loading tensors.

- [ ] **Step 4: Run Gate A checkpoint tests**

Run the full Task 6 command. Expected: all pass.

- [ ] **Step 5: Commit Task 8**

```bash
git add tace/renormalization/__init__.py tace/renormalization/linear_checkpoint.py \
  test/test_stage186_gate_a_linear_semantic.py
git commit -m "Add explicit Gate A linear checkpoints"
```

### Task 9: Materialize And Verify The Complete Gate A Artifact Set

**Files:**
- Create: `benchmarks/oc20neb_tace_mace/make_rtece_stage186_gate_a.py`
- Modify: `test/test_stage186_gate_a_linear_semantic.py`
- Modify: `docs/superpowers/specs/2026-07-21-tece-renorm-core-proof-design.md`

**Interfaces:**
- Produces: `build_known_gate_a_fixture(seed=186) -> tuple[LinearSemanticProblem, OperatorManifest]`
- Produces CLI artifacts under `--output-root`: `operator_manifest.json`, `linear_problem.npz`, `projection_fit.json`, `retained_linear.pt`, `gate_a_summary.json`

- [ ] **Step 1: Write end-to-end materialization test**

```python
def test_gate_a_cli_materializes_replayable_claim_artifacts(tmp_path):
    summary = materialize_gate_a(tmp_path, seed=186, damping=1e-4)
    assert summary["schema_version"] == "rtece_stage186_gate_a_summary.v1"
    assert summary["c2a_status"] == "supported"
    assert summary["direct_wls_match_max_abs"] <= 1e-10
    for name in (
        "operator_manifest.json",
        "linear_problem.npz",
        "projection_fit.json",
        "retained_linear.pt",
        "gate_a_summary.json",
    ):
        assert (tmp_path / name).is_file()
```

The fixture must contain at least one retained and one deleted path, E/F rows, explicit zero-weight V rows, nonempty held-out rows, and a deleted relevant group whose removal raises irreducible residual.

- [ ] **Step 2: Run end-to-end test and confirm failure**

Run Gate A tests with `-k materializes`. Expected: missing script/function failure.

- [ ] **Step 3: Implement deterministic fixture and CLI**

The CLI accepts only `--output-root`, `--seed`, `--damping`, and `--rcond`. It writes canonical artifacts, reloads all artifacts from disk, replays the retained checkpoint, and only then writes the summary. `c2a_status` is `supported` only when:

```python
direct_wls_match_max_abs <= 1e-10
checkpoint_replay_max_abs <= 1e-10
exact_fit_residual_delta <= 1e-10
relevant_group_residual_increase > 0.0
```

Otherwise use `not_supported`; schema/load/solver failures are hard errors and do not write a favorable summary.

- [ ] **Step 4: Run focused and regression verification**

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest -q \
  test/test_stage186_protocol_foundations.py \
  test/test_stage186_gate_a_linear_semantic.py \
  test/test_stage180_minimal_renormalization_proof.py \
  test/test_stage181_conventional_datasets.py

/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest -q \
  test/test_rtece_scalar.py -k 'path_manifest or projection or checkpoint or wrapper_contract'

/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python \
  benchmarks/oc20neb_tace_mace/make_rtece_stage186_gate_a.py \
  --output-root /tmp/rtece-stage186-gate-a
```

Expected: tests pass; CLI prints a summary whose `c2a_status` is `supported`; no Stage180 artifact or source file is modified.

- [ ] **Step 5: Update the design record with evidence boundaries**

Append a Stage186 Gate A result section stating that exact coefficient-space downfolding is supported only for the controlled linear semantic fixture, while nonlinear TACE/TECE mapping and GN remain untested. Include artifact schema versions and the command above; do not paste machine-specific `/tmp` paths as durable evidence.

- [ ] **Step 6: Commit Task 9**

```bash
git add benchmarks/oc20neb_tace_mace/make_rtece_stage186_gate_a.py \
  test/test_stage186_gate_a_linear_semantic.py \
  docs/superpowers/specs/2026-07-21-tece-renorm-core-proof-design.md
git commit -m "Materialize Stage186 Gate A proof"
```

## Plan Completion Gate

Before starting the separate Gate B plan:

1. `git diff --check` is clean.
2. All Task 9 verification commands pass.
3. A fresh `gpt-5.6-terra xhigh` reviewer confirms protocol/spec compliance.
4. A different fresh `gpt-5.6-terra xhigh` reviewer confirms code quality and test sufficiency.
5. The branch is pushed and the two source documents are re-read to reprioritize Gate B.

Gate B is a separate plan covering opt-in nonlinear operator amplitudes, `gn_parameter_manifest.json`, matrix-free damped GN, v2 teacher-cache consumption, fixed-budget matched controls, seven-arm wrappers, physical validation, and Stage186 summary statistics.
