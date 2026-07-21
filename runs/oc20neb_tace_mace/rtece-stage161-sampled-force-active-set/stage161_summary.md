# Stage161 Sampled-Force Active-Set Projection

Stage160 connected active-set ranking to the projection CLI, but current-baseline force-aware projection was blocked by full force-Jacobian materialization. Stage161 implements a sampled force-component route using JVP rows. This keeps the document-aligned active-set/downfolding workflow usable at Stage157 descriptor scale.

## Code Change

`benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py` now supports:

- `_force_descriptor_matrix(..., force_row_indices=...)`: computes selected force descriptor rows using forward-mode JVP and matches the full Jacobian rows on small tests.
- `--force-component-sample-count N`: samples force components deterministically while preserving fit/eval split semantics.
- force sampling metadata in projection JSON: `force_component_sampled`, `force_target_num_components`, `force_sampled_num_components`, `force_fit_num_components`, `force_eval_num_components`.
- `--active-set-gain-mode absolute|relative`: absolute is the old behavior; relative scores fractional RMSE gains so raw E/F units do not dominate the active-set score by accident.

## Current-Baseline Diagnostic

File: `stage161_current_baseline_limit64_holdout_edgeframe_sampled_force_relative_projection.json`

Setup:

- configs: mixed train `:64`
- baseline: Stage157 path set
- candidates: Stage158 edge-frame `uv`, `uqu`, `uv_uqu`
- energy: DFT holdout split, odd/even configs
- force: DFT force holdout split, 96 sampled force components from 12,744 total components
- active-set score: relative gains, energy weight 1, force weight 1

| candidate | E RMSE | sampled F RMSE | E relative gain | F relative gain | weighted gain | gain/cost | promoted |
|---|---:|---:|---:|---:|---:|---:|---|
| `uv` | 11.239 | 85.290 | 0.703 | -0.079 | 0.624 | 0.312 | yes |
| `uv_uqu` | 14.299 | 103.795 | 0.622 | -0.313 | 0.309 | 0.077 | yes |
| `uqu` | 7954.103 | 89.967 | -209.123 | -0.139 | -209.262 | -104.631 | no |
| baseline | 37.854 | 79.022 | 0.000 | 0.000 | 0.000 | 0.000 | - |

## Interpretation

The sampled-force diagnostic resolves the Stage160 bottleneck and clarifies the Stage158 disagreement:

1. `uv` has strong energy-baseline information: holdout E RMSE improves from 37.854 to 11.239 meV/atom in the projection diagnostic.
2. `uv` is not force-neutral: sampled force RMSE worsens from 79.022 to 85.290 meV/A. That is a 7.9% relative force regression.
3. `uv_uqu` is worse: it keeps energy improvement but worsens sampled force RMSE by 31.3% and is dominated in E/F rank.
4. `uqu` is rejected cleanly: catastrophic holdout energy projection and worse force.
5. Therefore the Stage158 trained-model failure is consistent with force-coupled nonlinear training punishing `uv`, not with `uv` containing no useful energy information.

## Force-Weight Sensitivity

Offline re-ranking of the same JSON gives:

| force weight | `uv` promoted | `uv_uqu` promoted | note |
|---:|---|---|---|
| 1 | yes | yes | energy gain dominates force regression |
| 5 | yes | no | `uv_uqu` rejected; `uv` still positive |
| 10 | no | no | force penalty dominates `uv` energy gain |

This means `uv` should not be promoted blindly. It is a candidate for a controlled training experiment only if the next run explicitly tracks force degradation and uses a loss/scoring regime that states how much force regression is acceptable for raw-energy baseline repair.

## Priority Update

1. Keep sampled force/JVP projection as the default force-aware active-set diagnostic for high-dimensional current-baseline candidates.
2. Use relative-gain active-set scoring for mixed E/F decisions; raw absolute gain is retained only for backward compatibility and narrow single-metric diagnostics.
3. Do not promote `uqu` or `uv_uqu` from the current evidence.
4. Treat `uv` as a conditional candidate: useful for energy baseline, risky for force. The next training experiment should either use a force-protected objective or compare `uv` against an alternative representation increment that improves energy without worsening sampled force.
5. The next algorithmic step should be Schur/Gauss-Newton or short refit diagnostics for `uv`, because projection says energy improves but Stage158 full training says final E/F worsens.
