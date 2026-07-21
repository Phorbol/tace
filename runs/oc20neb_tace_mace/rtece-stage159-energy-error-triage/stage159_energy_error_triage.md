# Stage159 Energy Error Triage

This note responds to the observation that the current rTECE student still has poor energy MAE/RMSE. It keeps the diagnosis tied to `TECE_design_space.md` and `rTECE_review.md`: raw total energy, case-level gauge, relative NEB shape, force error, and throughput must be separated before changing the architecture.

## Compared Runs

| split | configs | raw E RMSE | raw E MAE | raw E max | global-offset E RMSE | group-offset E RMSE | first-anchor E RMSE | relative-image E RMSE | barrier E RMSE | F RMSE | F MAE | F max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage157 rel0.25 mixed train `:512` | 512 | 25.162 | 17.831 | 131.409 | 24.720 | 7.011 | 11.321 | 10.680 | 14.788 | 84.952 | 42.421 | 1879.658 |
| Stage157 rel0.25 DFT valid `:1024` | 1024 | 52.538 | 35.443 | 204.986 | 51.602 | 4.769 | 11.461 | 7.104 | 11.403 | 94.159 | 41.823 | 2092.961 |

Units are meV/atom for energy and meV/A for forces.

## What This Says

1. The user observation is correct. Raw energy MAE/RMSE remains poor: the current best high-throughput Stage157 row is still `52.538` meV/atom E RMSE on DFT validation.
2. This is not the original `energy_per_atom_shift` bug. The model uses least-squares per-element atomic energies and `energy_per_atom_shift = 0.0`. A single global offset correction barely changes the error.
3. The dominant raw-energy failure is low-frequency case/slab/adsorbate baseline error. On DFT valid, group-mean correction drops E RMSE from `52.538` to `4.769` meV/atom; on the mixed train window it drops from `25.162` to `7.011` meV/atom.
4. It is not pure validation-set extrapolation either. The train-window raw E RMSE is still `25.162` meV/atom, so the current descriptor/training objective does not fully fit case baselines even in-distribution.
5. Relative NEB shape is much better than raw total energy but not solved: relative-image/barrier RMSE remains about `7-15` meV/atom depending on split. Force RMSE is still high at about `85-94` meV/A, with force tails above `1.8-2.1` eV/A.
6. Stage158 hand-added edge-frame `u dot v` and `u^T Q u` paths did not improve the Stage157 Pareto point. This argues against adding more manual paths without active-set/downfolding evidence.

## Root-Cause Hypothesis

The current model has crossed the line from the early fixed-descriptor demo into a learnable T3 scalarized student, but the retained front-loaded statistics are still too weak for OC20NEB's heterogeneous slab plus adsorbate baselines. Per-element E0s explain composition, but not surface/site/adsorbate environment offsets. The remaining error is therefore a representation/distillation/data-manifold problem, not a scalar loss-weight or global gauge problem.

## Stage158 Active-Set Backtest

Using the new row-level active-set ranking utility with Stage157 rel0.25 as baseline and Stage158 edge-frame projection rows as candidates, all three hand-added candidates are rejected:

| candidate | marginal paths | weighted marginal gain | cost proxy | gain/cost | promoted |
|---|---|---:|---:|---:|---|
| uQu | target/source quadrupole projections | -6.292 | 8.0 | -0.786 | no |
| uv+uQu | target/source vector and quadrupole projections | -11.897 | 9.0 | -1.322 | no |
| uv | target/source vector projections | -19.632 | 1.0 | -19.632 | no |

This backtest agrees with the Stage158 manual interpretation: these paths are semantically valid TECE/rTECE scalar sketches, but they are not current Pareto-promotable increments under the observed OC20NEB energy/force/relative metrics.

## Priority Update

Do not spend the next stage on another relative-loss scalar sweep. The document-aligned next step is cost-aware active-set/downfolding:

1. Rank candidate TECE scalar paths by marginal improvement in energy/force projection error per hardware cost.
2. Add only promoted representation increments, especially learnable low-rank radial/species/edge-relational gates that can express case baselines without persistent equivariant state.
3. Keep reporting raw E MAE/RMSE/max, F MAE/RMSE/max, group-offset, first-anchor, relative-image/barrier, and physical tests together.
4. Separately compare train-window and valid-window metrics for every promoted point so we can distinguish in-distribution representation failure from data/teacher-label coverage failure.

## Caveat

The train-window diagnostic fell back to CPU (`device: cpu`) although `--device cuda` was requested, so its throughput is not comparable. The error metrics are still usable for this triage.
