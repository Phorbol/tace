# TECE Renormalization Core Proof Design

## Status And Scope

This specification defines Approach A: repair the experimental protocol and
test the central TECE/TACE renormalization claim before expanding datasets,
optimizing kernels, or making cross-engine Pareto claims.

It refines the original `TECE Renormalized Distillation Design` and supersedes
the scientific interpretation of Stage180. Stage180 remains useful evidence
that an additional same-student E/F prefit can improve optimization, but its
current initializer does not compute or write a Schur-complement or
Gauss-Newton downfolding solution. It therefore must not be cited as proof of
TECE renormalization.

Stage185 rMD17 work is paused, not deleted. It resumes only after the core
projection, data-split, and comparison contracts in this document pass.

## Research Objective

The objective is not merely to train a small fast potential. It is to test
whether a deployment-conditioned compiler can map

\[
(T,\mu,\mathcal O,C_{\mathrm{HW}})
\longrightarrow
\{S_0\subset S_1\subset\cdots\subset S_K\}
\]

where:

- \(T\) is a trained TACE/TECE teacher;
- \(\mu\) is a declared deployment distribution;
- \(\mathcal O\) is a weighted set of energy, force, virial, curvature, and
  physical observables;
- \(C_{\mathrm{HW}}\) is measured latency, throughput, and memory cost;
- each \(S_k\) is a real compact student with an auditable subset of semantic
  operators, not a masked full model.

The compiler claim is meaningful only if deleting operators is followed by a
quantified projection/downfolding step, the resulting coefficients are written
into the student, and the benefit survives equal-budget controls and held-out
evaluation.

## Core Falsifiable Claims

### C1: Semantic nesting

A stable ordered operator manifest can define a nested ACE-derived student
family whose members monotonically retain more radial, angular, correlation,
species, or edge-relational information while remaining conservative and
equivariant/invariant by construction.

### C2a: Exact semantic downfolding

For a teacher and student expressed in one explicit semantic linear basis,
weighted Schur downfolding reproduces the direct retained-space optimum and
quantifies the irreducible error caused by deleted operators.

### C2b: Nonlinear local initialization

For a fixed nonlinear student, a damped Gauss-Newton update improves a
predeclared held-out objective over random initialization and ordinary
first-order prefit controls with matched data and teacher-oracle exposure.
This supports a local initializer, not TECE operator downfolding, unless a
versioned teacher-path-to-student-Jacobian mapping is also demonstrated.

### C3: Distillation closure

Teacher E/F/(V) residual distillation gives a repeatable held-out gain beyond
ordinary optimization, and a factorial comparison determines whether that gain
depends on GN initialization. Distillation cannot be used to hide an
architecture whose irreducible projection error is too large.

### C4: Cost-accuracy ordering

The nested family produces measured nondominated points in error versus actual
hardware cost. The edge-relational T3 tier is promoted only when its marginal
accuracy/physics gain justifies its marginal measured cost over the atomic
moment L2 tier.

Stage186 tests C1, C2a, C2b, and C3. C4 is a subsequent Stage187 nested-family
test after the core protocol is frozen; one fixed Stage186 student cannot prove
a Pareto ordering.

Failure of C2a, C2b, or C3 is a valid scientific result. It would mean the
present semantic basis or local linearization is insufficient, and the project
should revise that basis before further throughput optimization.

## Non-Goals For This Stage

- Do not claim a complete TECE/TACE compiler before a teacher semantic tape or
  explicitly defined linear semantic supernet exists.
- Do not claim NEP, DPA, or rTECE ranking from unmatched engines or timing
  layers.
- Do not optimize Triton, nvalchemi-toolkit-ops, graph construction, or fused
  force/virial kernels unless required to make the core proof executable.
- Do not tune specifically for C/N or turn rattle-relax into a binary gate.
- Do not select architectures from a test set.
- Do not enlarge the final MLP merely to reach a target parameter count.
- Do not resume the broad rMD17 matrix until the core proof protocol passes.

## Existing Evidence And Corrections

The current branch already supports three useful conclusions:

1. Early scalarization gives a large model-only throughput increase relative to
   compact TACE, but cached-graph atoms/s is not full periodic MD throughput.
2. Trainable low-rank species/radial/local-L0 representation layers improve the
   very small fixed-feature endpoint without requiring a large final head.
3. Stage183 defines a genuine capacity ladder from L0 through atomic L1/L2 and
   edge-relational T3. On 3BPA, L2 is currently the measured model-only Pareto
   point; T3 adds large cost with little ID force-RMSE gain and worse dihedral
   force RMSE.

The following evidence must be relabeled or repaired:

- Stage180 `renorm init` is an ordinary 512-step same-student E/F prefit. It
  supports an initialization/optimization claim, not a downfolding claim.
- The Stage180 scratch arm lacks an equal-budget prefit control.
- Stage182/183 used `test_300K.xyz` for validation and then reported it as the
  ID test set. Those ID rows are validation metrics, not clean test metrics.
- The current teacher residual cache records data but is not consumed by the
  official training loss.
- rTECE, NEP/GPUMD, and DeepMD/DPA timing paths measure different execution
  layers and cannot yet share one strict cross-engine Pareto frontier.

## Nested Model Contract

### Operator identity

Every candidate operator has a stable semantic path ID. At minimum the ID
encodes:

- radial family and mode;
- center/neighbor species factorization;
- angular or Cartesian moment order;
- correlation/contraction order;
- atomic versus edge-relational placement;
- source/target and cavity convention for edge paths;
- cutoff and normalization convention.

Use separate canonical hashes for the ordered operator manifest, feature/units
schema, model configuration, implementation revision, and teacher
checkpoint/config. Checkpoint loading recomputes the compatibility tuple and
fails on any semantic/config mismatch. A migration is an explicit
schema-versioned file that names old/new hashes and the deterministic state
transformation; changing only a stored hash is forbidden.

### Strict nesting

The first proof family uses strict retained prefixes or explicitly declared
nested groups:

\[
\mathcal P_{L0}\subset\mathcal P_{L1}\subset
\mathcal P_{L2}\subset\mathcal P_{T3}.
\]

The scalar head architecture is held fixed during a path-retention comparison.
Increasing descriptor capacity must come from declared representation paths,
trainable radial/species projections, or low-rank local mixing, not an unrelated
head-width increase.

### Semantic honesty

The current rTECE descriptors are ACE-inspired hand-defined moment and
edge-sketch operators. They are not automatically coordinate-aligned with an
arbitrary nonlinear TACE/TECE teacher.

The first exact proof may therefore use a linear semantic proxy/supernet whose
candidate operators and coefficients are fully observable. This can prove the
downfolding machinery and model-selection logic. A later nonlinear proof may
use teacher output Jacobians. Neither result may be described as full
TACE/TECE path compilation until the teacher exposes a stable semantic path
tape and a documented mapping to student operators.

## Exact Linear Downfolding

Let \(A\) be the candidate semantic design matrix evaluated on the fitting
structures. Its rows contain observable derivatives, not only scalar energies.
Let \(W\) be the diagonal or block-diagonal observable/sample weight matrix,
and let \(\Phi=W^{1/2}A\) be the whitened design matrix:

\[
A =
\begin{bmatrix}
A_E\\
A_F\\
A_V
\end{bmatrix},
\qquad
G=A^TWA=\Phi^T\Phi.
\]

Here \(A_F=-\partial A_E/\partial\mathbf r\), and \(A_V\) is obtained
from the same strained/PBC-aware energy basis. If virials are unavailable in
the first fixture, the corresponding virial weight in \(W\) is explicitly zero
in the artifact; virials are not silently omitted.

Partition candidate coefficients into retained \(R\) and deleted \(D\):

\[
\theta=(\theta_R,\theta_D),\qquad
G=\begin{bmatrix}G_{RR}&G_{RD}\\G_{DR}&G_{DD}\end{bmatrix}.
\]

For a teacher represented in the candidate basis, the retained optimum is

\[
\beta_R^*=\theta_R+G_{RR}^{+}G_{RD}\theta_D,
\]

where \(G_{RR}^{+}\) is a rank-revealing pseudoinverse after a declared gauge or
null-space policy. This unregularized solution is the exact Schur claim.

Numerically damped fits use the explicitly anchored objective

\[
\min_{\beta_R}
\left\|W^{1/2}
\left(A_R\beta_R-A_R\theta_R-A_D\theta_D\right)\right\|_2^2
+\lambda\|\beta_R-\theta_R\|_2^2,
\]

which gives

\[
\beta_{R,\lambda}^*=\theta_R+
(G_{RR}+\lambda I)^{-1}G_{RD}\theta_D.
\]

The implementation uses a stable solve and records conditioning, numerical
rank, gauge, damping, and residuals. It must never form an explicit inverse.
For \(\lambda=0\), report the exact Schur residual. For \(\lambda>0\), report
the anchored augmented objective and the unregularized fit/held-out observable
residual separately; do not call the augmented value an exact Schur residual.

The proof is valid only when \(\beta_R^*\) is written into the retained student
state and the saved checkpoint reproduces the predicted projection residual.
A diagnostic JSON followed by ordinary gradient training is not downfolding.

## Nonlinear Gauss-Newton Initialization

For a nonlinear student \(f_S(x;\beta)\), stack teacher and student E/F/(V)
observables under the same weights. Around \(\beta_0\), define student Jacobian
\(J_S\) and residual \(r=y_T-y_S(\beta_0)\). One damped local update is

\[
\Delta\beta=
(J_S^T WJ_S+\lambda I)^{-1}J_S^TWr,
\qquad
\beta_1=\beta_0+\Delta\beta.
\]

Large systems may use matrix-free JVP/VJP and an iterative solver. The artifact
must record solver tolerance, iterations, damping, initial/final residual, and
the exact parameter subset updated. A line search or trust-region acceptance
rule rejects a step that raises the fitting objective.

This is called `GN initialization`, never `gn_renorm`, and not exact TECE Schur
downfolding unless the Jacobian columns have a demonstrated, versioned
teacher/student semantic correspondence.

## Stage186 Experimental Design

Stage186 replaces the misleading Stage180 proof label. It has two gates.

### Gate A: exact synthetic/linear semantic proof

Use a deterministic small fixture or linear ACE-like semantic supernet for
which retained/deleted paths and teacher coefficients are known. The canonical
`linear_problem.npz` artifact contains \(A_E,A_F,A_V,W,\theta\),
retained/deleted path indices, fit/held-out row indices, dtype, units, and
schema/hash metadata. `operator_manifest.json` maps each retained coefficient
to one checkpoint state entry. The fixture generator fixes the gauge, solver
tolerance, and damping policy.

Required checks:

1. closed-form downfolding matches direct weighted least squares;
2. saved retained checkpoint reproduces the analytic prediction;
3. projection residual agrees with the reported Schur residual;
4. deleting a known relevant group raises irreducible error;
5. all results are invariant to batching and serialization;
6. damped and unregularized result fields cannot be confused by schema.

Gate A must pass before expensive cluster runs.

### Gate B: fixed-student nonlinear comparison

Use one frozen student manifest, one immutable data split, one teacher, and at
least three seeds. Run these arms:

| arm | initialization/treatment | purpose |
|---|---|---|
| `scratch_dft` | random initialization, DFT objective | base optimization control |
| `scratch_distill` | same random initialization, DFT plus teacher-residual objective | isolates ordinary distillation |
| `dft_prefit_dft` | ordinary DFT E/F prefit, with all prefit updates counted | isolates the historical Stage180 mechanism |
| `teacher_prefit_dft` | ordinary first-order prefit on the same teacher oracle set as GN, followed by DFT objective | matched-oracle GN control |
| `truncation_copy` | direct retained coefficient/weight copy when semantically defined | tests deletion without compensation |
| `gn_dft` | accepted GN coefficients written into the student, followed by DFT objective | tests C2b |
| `gn_distill` | the identical GN initialization plus teacher-residual objective | tests the GN/distillation interaction in C3 |

`truncation_copy` is applicable only when teacher and student share exact
semantic path IDs and tensor shapes and the checkpoint exposes a deterministic
coefficient-slice mapping with no nonlinear remapping. Otherwise the arm is
`not_applicable` with the failed predicate recorded; no favorable ad hoc
baseline replaces it.

The core factorial contrasts are `scratch_distill - scratch_dft`,
`gn_dft - scratch_dft`, and `gn_distill - gn_dft`, plus the interaction between
initialization and distillation. `teacher_prefit_dft` is the required
matched-oracle comparator for `gn_dft`; `dft_prefit_dft` preserves the Stage180
historical question.

Every arm has a fixed `total_gradient_updates` ledger. Prefit updates are
subtracted from the common main-training budget. The arms use the same main
batches, sampler policy, optimizer family, learning-rate schedule, warmup,
checkpoint selection rule, and model architecture. Early stopping is disabled
for the primary fixed-budget comparison; a separately labeled deployment-tuning
run may use one frozen early-stopping rule.

For each seed, `scratch_dft`, `scratch_distill`, `dft_prefit_dft`,
`teacher_prefit_dft`, `gn_dft`, and `gn_distill` start from one
byte-identical model/E0 checkpoint and base-state hash. `gn_dft` and
`gn_distill` additionally share the same post-GN checkpoint. Before any GN
output is computed, `gn_parameter_manifest.json` enumerates every allowed
fully qualified state-dict key and tensor slice; wildcards are forbidden. The
default subset is retained operator amplitudes and scalar readout parameters,
excluding E0, cutoff, ZBL/baseline, and unrelated hidden parameters.

`teacher_prefit_dft` and GN consume the same ordered initializer structures,
observable components, teacher outputs, and query count. Before non-control
runs, a training-only calibration freezes the primary initializer compute
currency, `student_autodiff_work_units`. Forward, backward, JVP, and VJP
calls are weighted by median device time on fixed Stage186 tensor shapes;
linear-solver host time is added in the same wall-time units. Both arms receive
one common frozen budget `B_init` and consumed work must agree within 2%.

The run manifest also records unweighted call counts, E/F/V scalar observations,
full data passes, solver iterations, total initializer wall time, and peak
memory. If state/oracle/parameter-manifest hashes differ or primary work differs
by more than 2%, C2b is `inconclusive`. The complete ledger remains
reportable even when a run is invalidated.

## Data And Split Contract

### 3BPA core proof

- Build training and validation subsets only from `train_300K.xyz`.
- Use `split_seed=186`. Preserve source order and canonical frame identities.
  In the absence of authoritative trajectory-group metadata, divide the 500
  source frames into five contiguous 100-frame windows. Select validation window
  `int(SHA256(split_seed || source_hash), 16) mod 5` and embargo the five
  nearest source frames at each train/validation boundary. If authoritative
  trajectory groups exist, hold out whole groups instead and record the
  superseding rule before any model arm runs.
- Record source trajectory/frame IDs when available. The manifest stores exact
  train/embargo/validation indices and hashes normalized arrays, labels, units,
  and the original source file.
- Keep `test_300K.xyz` untouched until final ID evaluation.
- Use `test_600K.xyz`, `test_1200K.xyz`, and `test_dih.xyz` only as declared OOD
  evaluations, never for model selection.
- Use the identical split for every arm and compatible community baseline.

The split artifact is generated and committed before any non-control arm runs.
Model selection uses the exact normalized E/F RMSE objective defined below.
Test metrics are materialized only after the selection rule, weights, and
practical-effect thresholds are frozen.

### rMD17 transfer

rMD17 resumes after Stage186. It must not use adjacent contiguous temporal
blocks as the only generalization test. The transfer protocol uses recognized
published splits when available or frozen nonadjacent windows separated by
temporal gaps, with units converted from kcal/mol to eV exactly once and
recorded in the manifest.

### Reproducibility metadata

Every run records:

- source file hashes, split indices, units, E0 method, and element set;
- teacher checkpoint/config hash and teacher code revision;
- student manifest/config hash and repository revision;
- environment and accelerator identity;
- seed, data order, total updates, schedule, warmup, and early stopping;
- observable weights and normalization constants.

Average per-element E0 values are fitted by weighted least squares on the
training split only when user E0 values are absent. Validation/test energies do
not enter E0 fitting.

## Teacher Residual Distillation Contract

The cache schema contains at least:

- schema version, canonical structure hash, and explicit join key;
- teacher model/config/code hashes;
- units, cutoff, dtype, cell, PBC, and edge convention;
- teacher total energy, forces, and nullable virial with availability flags;
- reference DFT values and explicit teacher-minus-reference residuals;
- source split, frame identity, and generation command.

Records join by canonical structure hash and reject duplicates or missing
structures; file order is never the join contract. For each observable \(O\),

\[
r_T^O=y_T^O-y_{\mathrm{DFT}}^O,
\]

and the absolute-output student uses

\[
L=\lambda_{\mathrm{DFT}}\|f_S-y_{\mathrm{DFT}}\|_W^2+
\lambda_T\|(f_S-y_{\mathrm{DFT}})-r_T\|_W^2.
\]

The weights, normalization, and per-element E0 gauge are frozen in the run
manifest. The same training-split E0 gauge is applied to DFT and teacher
energies. At inference, the checkpoint emits the absolute student energy and
needs no teacher/cache. Under squared loss this residual form is equivalent to
a declared blended target; it is not claimed as a distinct algorithmic benefit.

The official training path must consume the cache and log separate DFT and
teacher E/F/(V) terms. Cache production without loss consumption does not count
as distillation. Stage186 is molecular E/F-only and explicitly sets virial
availability false and \(W_V=0\); periodic E/F/V distillation is a later
applicability gate.

Atomic energy is not treated as a unique teacher truth unless its gauge and
definition are fixed. Primary distillation targets are total/relative energy,
force, optional virial/HVP, and explicitly defined semantic scalar paths.

## Evaluation Contract

### Numerical accuracy

Report distributions and aggregate values for both DFT and teacher targets:

- E MAE, RMSE, and maximum absolute error in meV/atom;
- F MAE, RMSE, and maximum component/vector error in meV/A;
- relative energies, path/image differences, and barriers where applicable;
- virial/stress MAE, RMSE, and maximum error when labels exist.

RMSE is the primary scalar ranking statistic because force/energy tails matter,
but MAE and maxima remain mandatory diagnostics. No single F MAE row determines
model selection.

### Physical generalization

- Dimer scans span declared element pairs and approximately 0.5 to 5 times a
  covalent-radius reference. Report curve error, force error, smoothness,
  short-range monotonicity/repulsion, and finite values. Use the repository's
  ZBL basis for an explicit short-range baseline rather than an unrelated
  softplus overlap term.
- Rattle followed by relaxation reports continuous final RMSD, energy change,
  final force distribution, force tail, convergence steps, and failure reason.
  C/N and other element groups are diagnostics, not training-specific gates.
- PBC image invariance, conservative force finite differences, finite-strain
  virial/stress, and checkpoint/calculator equivalence are mandatory before a
  periodic deployment claim.
- HVP/curvature and short NVE drift are later production tests; they are
  required before claiming stable long-time MD, not before Gate A.

### Hardware cost taxonomy

Each timing row declares exactly one layer:

1. descriptor/kernel only;
2. prebuilt-graph model E/F/(V);
3. full calculator including graph/update;
4. full periodic MD including integrator and update policy.

Report atoms/s, configurations/s, latency, peak allocated/reserved memory,
atom count, edge count, batch size, precision, device, warmup, repeats, and
whether forces/virials are included. Scaling tests use multiple atom counts and
identify the saturation region.

Cross-engine NEP/DPA/rTECE Pareto ranking is prohibited until data, cutoff,
precision, hardware, observables, and timing layer match. Until then, rows are
shown in separate panels as engineering context.

## Acceptance And Falsification Rules

### C1 semantic and physical consistency

The Stage186 manifest freezes dtype-specific tolerances before any arm runs.
Default float64 fixture tolerances are \(10^{-8}\) eV for transformed energy
and \(10^{-7}\) eV/A for transformed forces; float32 tolerances are
\(10^{-5}\) eV and \(10^{-4}\) eV/A. Tests cover random SO(3), reflection,
translation, atom permutation, edge reorder, and serialization. Conservative
force finite differences must satisfy the larger of \(10^{-5}\) eV/A absolute
or \(10^{-4}\) relative error in float64; float32 uses a preregistered
\(5\times10^{-3}\) relative tolerance.

PBC image invariance and finite-strain calculator stress are tested on separate
periodic fixtures even though the Stage186 3BPA training objective is molecular
and sets \(W_V=0\). PBC image tests use the transformed energy/force tolerances
above; finite-strain stress uses \(10^{-4}\) relative error in float64 and
\(5\times10^{-3}\) in float32. Native trainable virial support remains
`not_evaluated` until the cache, model, and loss all expose it. A failed
applicable invariance or conservative-force check makes C1 `not_supported`;
a missing required periodic deployment check blocks only the deployment claim.

### Preregistered primary endpoints

The validation and held-out teacher objective is

\[
J^2=w_E\widehat{\mathrm{RMSE}}_E^2+
w_F\widehat{\mathrm{RMSE}}_F^2+
w_V\widehat{\mathrm{RMSE}}_V^2,
\]

Define DFT training-only scales after the training-split E0 fit:

\[
s_E=\max\left(\operatorname{std}_{\mathrm{train}}(E/N),1\ {\rm meV/atom}\right),
\qquad
s_F=\max\left(\sqrt{\operatorname{mean}_{\mathrm{train}}F_\alpha^2},
1\ {\rm meV/A}\right).
\]

Then \(\widehat{\mathrm{RMSE}}_E=\mathrm{RMSE}_E/s_E\) and
\(\widehat{\mathrm{RMSE}}_F=\mathrm{RMSE}_F/s_F\). Stage186 uses
\(w_E=w_F=1/2\) and \(w_V=0\). `J_teacher` is the primary C2b projection
endpoint and `J_DFT` is the primary deployment endpoint. E/F MAE, individual
RMSEs, maxima, OOD sets, dimer scans, and rattle-relax remain mandatory
secondary/safety diagnostics but cannot independently flip a primary claim.

Using only the three `scratch_dft` controls, freeze the relative practical
threshold before non-control outcomes are read:

\[
\delta=\max\left(0.01,\exp\left[
1.96\,\operatorname{sd}(\log J_{\mathrm{scratch}})
\right]-1\right).
\]

The `J_DFT` non-inferiority margin is \(\delta\). Relative safety margins are
5% for E/F maximum errors and dimer-curve RMSE, 10% for final rattle-relax RMSD,
and 5% for final-force p95/max. New nonfinite values, dimer discontinuities, or
failed C1 checks have zero tolerance. Arm comparisons use paired data order and
a hierarchical bootstrap over seeds and held-out structures. Start with three
seeds; expand the affected contrast to five when the 95% interval intersects
\(\delta\).

### C2a exact semantic downfolding

C2a is `supported` only when unregularized Gate A coefficients match direct
weighted least squares, the saved checkpoint reproduces them, and fit/held-out
residuals agree to the declared float64 solver tolerance. Damped results are
reported separately and cannot substitute for this test.

### C2b nonlinear initialization

`gn_dft` must beat both `scratch_dft` and the matched-oracle
`teacher_prefit_dft` on `J_teacher` by \(\delta\), while `J_DFT` remains within
its frozen non-inferiority margin. Data/oracle exposure and compute ledgers must
pass. If GN only improves early convergence but not the fixed-budget endpoint,
report optimization acceleration rather than better projection.

### C3 distillation closure

Ordinary distillation is supported only by the preregistered
`scratch_distill - scratch_dft` contrast. A GN-specific distillation interaction
is supported only when `gn_distill - gn_dft` exceeds \(\delta\) and the
difference-of-differences against the scratch contrast has the expected sign.
All contrasts use the same total update and teacher-query rules and must not fit
only cache-training structures.

### Stage187 architecture promotion

C4 is not decided in Stage186. Stage187 trains preregistered nested manifests
with the frozen protocol, selects paths from validation results only, and emits
one final test-only Pareto report. T3 edge sketches are not privileged: if L2
dominates T3, L2 remains the Pareto student and the edge basis is redesigned or
rejected for that deployment distribution.

## Required Artifacts

Stage186 produces machine-readable, schema-versioned artifacts:

- `operator_manifest.json` with ordered path groups, state mapping, and hashes;
- `linear_problem.npz` plus `projection_fit.json` with rows, weights,
  retained/deleted indices, rank, gauge, damping, solver status, and residuals;
- `data_manifest.json` with canonical frame IDs, split/embargo indices,
  hashes, units, and E0 fit;
- a versioned teacher cache with join and availability metadata;
- initialized checkpoints containing the complete compatibility tuple;
- one `train_summary.json` per applicable arm and seed, including
  `total_gradient_updates` and checkpoint-selection accounting;
- `gn_parameter_manifest.json` with exact state keys/slices and base state hash;
- `oracle_compute_ledger.json` for teacher/data/JVP/VJP/time/memory exposure;
- DFT/teacher/OOD/physical benchmark JSON files;
- hardware timing JSON files separated by timing layer;
- `wrapper_audit.json` with static and `sbatch --test-only` results;
- `stage186_summary.json` with cross-seed statistics, primary contrasts,
  and per-claim decisions using the complete closed enum.

Schema violations, nonfinite values, structure/hash mismatches, solver
nonconvergence, and missing observable units are hard errors. A run must not
silently fall back from GN initialization to ordinary prefit.

## Scientific Minimum Versus Production Readiness

The scientific minimum for Approach A is Gate A plus a fair Gate B comparison
that decides C1, C2a, C2b, and the preregistered C3 contrasts. It does not
require a fused production MD engine or a cross-engine Pareto ranking.

User-facing production readiness additionally requires:

- package-native training, checkpoint loading, inference, and ASE calculator;
- one PBC geometry implementation shared by energy, force, and virial;
- correct neighbor/image shifts and periodic graph updates;
- native virial/stress validated by finite strain;
- scalable data loading, shuffling, batching, scheduling, warmup, early
  stopping, and resume behavior;
- calculator versus direct-model parity and portable checkpoints;
- full periodic MD timing, memory scaling, and basic stability tests.

The branch currently has substantial parts of the package training/calculator
surface, but it is not yet production-ready for general periodic long-time MD.
Production closure follows the core scientific proof, except for correctness
fixes needed to make its metrics trustworthy.

## Execution Priority

1. Repair the projection active-set schema drift and lock tests green.
2. Materialize the leakage-free 3BPA split and immutable manifests.
3. Implement and test Gate A exact linear downfolding.
4. Implement nonlinear GN state initialization with explicit solver artifacts.
5. Extend and connect the teacher cache to the official E/F/(V) loss.
6. Run the seven-arm, matched-oracle, fixed-budget, three-seed Stage186 comparison.
7. Run preregistered numerical and physical diagnostics; decide C1-C3 explicitly.
8. If seed uncertainty straddles a decision threshold, expand only the affected
   contrast to five seeds.
9. After Stage186, run Stage187 nested L0/L1/L2/T3 selection on validation data,
   freeze the frontier, and evaluate its final test-only Pareto rows.
10. Only then resume rMD17 transfer, matched NEP/DPA baselines, full atom-count
    scaling, and kernel/provider optimization.

## Agent And Review Policy

The main `gpt-5.6-sol` high-reasoning agent owns theory, experimental design,
priority decisions, interpretation, and changes to this specification.

Normal implementation and routine repository work are delegated to
`gpt-5.6-terra` workers with disjoint file ownership. Workers must follow the
specification, existing repository structure, tests-first workflow, and SAI
wrapper constraints. They must not reinterpret the scientific claim.

Before integration, separate `gpt-5.6-terra` reviewers inspect:

1. spec conformance and scientific-control leakage;
2. code quality, tests, schemas, and failure handling.

The main agent resolves disagreements, re-reads `TECE_design_space.md` and
`rTECE_review.md` after each completed stage, and reprioritizes according to the
core claims rather than stage-number momentum.

## Cluster Constraints

SAI wrappers must not use `--export`, shell `export`, `#SBATCH --mem`,
`--mem=`, `#SBATCH --cpus-per-task`, `--cpus-per-task`, or `set -u`. Stage186
generates one wrapper per arm and seed. The static audit scans both `#SBATCH`
lines and shell bodies for every forbidden token, and each accepted
`sbatch --test-only` result is recorded before submission. Downloaded datasets
are prepared on the login node when network access is required; compute-node
jobs remain offline.

## Decision After Stage186

The stage reports each claim with the closed enum `supported`,
`not_supported`, `inconclusive`, `not_applicable`, or `not_evaluated`. It also
reports one overall decision:

- `supported`: C1, C2a, C2b, and the declared C3 contrast survive fair controls;
  proceed to Stage187 transfer and measured Pareto compilation;
- `partially_supported`: C2a works but nonlinear initialization, distillation,
  or deployment correctness is not supported; revise the semantic mapping or
  training closure before architecture expansion;
- `not_supported`: C2a fails in the controlled semantic fixture, so stop
  presenting renormalization as the organizing mechanism;
- `inconclusive`: a preregistered test could not distinguish the arms or a
  required artifact/control failed for a reason that does not test the claim.

This decision is made from frozen acceptance rules, not from the most favorable
single seed or metric.
