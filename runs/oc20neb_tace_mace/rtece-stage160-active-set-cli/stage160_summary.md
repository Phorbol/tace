# Stage160 Active-Set Projection CLI

Stage160 connects the Stage159 row-level active-set ranking into the reusable projection diagnostic CLI. This is a small but important compiler step: projection JSONs can now carry explicit marginal gain/cost rows instead of requiring manual table assembly.

## Code Change

`benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py` now accepts:

- `--active-set-baseline-candidate`
- `--active-set-energy-weight`
- `--active-set-force-weight`
- `--active-set-projection-weight`

When a baseline is provided, the output JSON includes:

- `active_set`: the baseline and ranking weights;
- `active_set_rows`: candidates ranked by weighted marginal gain per cost, with marginal paths, gain components, cost proxy, gain/cost, and promotion flag.

The existing `rows` output remains unchanged apart from the extra top-level fields, so old projection consumers can continue using the original ranking.

## First-Rung Holdout Diagnostic

File: `stage160_train_limit64_holdout_active_set_projection.json`

Setup: DFT labels from mixed train, 64 configs, odd/even holdout split, baseline `atomic.radial_density`, candidates adding species, vector, direct edge, and vector+direct.

| rank | candidate | marginal paths | E holdout RMSE | F holdout RMSE | gain/cost | promoted |
|---:|---|---|---:|---:|---:|---|
| 1 | `vector` | `atomic.vector_norm` | 3.834 | 1.408 | 0.253 | yes |
| 2 | `vector_direct` | `atomic.vector_norm`, `edge.direct.radial` | 3.390 | 1.387 | 0.245 | yes |
| 3 | `direct` | `edge.direct.radial` | 4.554 | 1.388 | 0.152 | yes |
| 4 | `species` | `atomic.species_basis_density` | 24.601 | 1.418 | -0.617 | no |

The in-sample `limit32` diagnostic drove energy projection RMSE close to zero, which is not meaningful. The holdout run is the useful result: it confirms that active-set ranking must use held-out projection error, otherwise high-dimensional species descriptors can look falsely good.

## Current-Baseline Edge-Frame Energy Diagnostic

File: `stage160_current_baseline_limit64_holdout_edgeframe_energy_projection.json`

Setup: current Stage157 path set as baseline, adding Stage158 edge-frame `uv`, `uQu`, and `uv+uQu` paths. Energy-only holdout was used because full force projection was too expensive with the current full Jacobian implementation.

| rank | candidate | marginal paths | E holdout RMSE | energy gain | gain/cost | promoted |
|---:|---|---|---:|---:|---:|---|
| 1 | `uv` | target/source vector projections | 11.239 | 26.616 | 13.308 | yes |
| 2 | `uv_uqu` | vector + quadrupole projections | 14.299 | 23.556 | 5.889 | yes |
| 3 | `uqu` | target/source quadrupole projections | 7954.103 | -7916.249 | -3958.124 | no |
| baseline | `stage157_baseline` | current Stage157 paths | 37.854 | 0.000 | 0.000 | - |

This differs from the Stage158 trained-model result, where `uv` worsened DFT E/F RMSE. The clean interpretation is not that `uv` is definitely promotable; it is that `uv` has energy-projection information but fails after nonlinear force-coupled training, or under force/Jacobian constraints. That makes it a downfolding/training-coupling question, not a pure semantic-path question.

## Force Projection Bottleneck

Two current-baseline force-aware runs were interrupted after spending minutes in `_force_descriptor_matrix` autograd Jacobian construction, including a 16-config smoke. This is now a concrete tooling bottleneck. Full force Jacobian projection is too expensive for routine active-set selection at Stage157 descriptor scale.

Next implementation priority should be a sampled force-component or JVP/VJP-based projection diagnostic, so candidate path selection can include force sensitivity without materializing a full force descriptor matrix.

## Priority Update

1. Keep the CLI active-set contract and use holdout projection by default for candidate selection.
2. Do not promote `species_basis_density` based on descriptor residual alone; holdout E/F rejects it in the first-rung diagnostic.
3. Do not blindly retrain `uv` just because energy-only projection likes it; first add a cheaper force-aware projection path or a Schur/Gauss-Newton refit diagnostic that can explain the Stage158 training/projection disagreement.
4. The next stage should implement force-row sampling or JVP/VJP active-set scoring, then re-run current-baseline `uv/uQu` selection with force evidence.
