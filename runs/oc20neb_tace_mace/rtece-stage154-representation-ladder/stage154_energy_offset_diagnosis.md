# Stage154 Energy Offset Diagnosis

This note records the Stage153 evidence behind the Stage154 representation ladder. It is meant to keep the next architecture decision tied to `TECE_design_space.md` and `rTECE_review.md`.

## Observation

The current rTECE scalar student still has poor raw absolute energy error on the OC20NEB validation set:

| Stage153 row | raw E RMSE (meV/atom) | raw E MAE (meV/atom) | F RMSE (meV/A) | F MAE (meV/A) |
|---|---:|---:|---:|---:|
| DFT-only labels | 133.835 | 99.787 | 110.465 | 50.928 |
| mixed labels | 126.883 | 99.102 | 105.745 | 46.922 |

DFT-only labels did not repair the energy error, so the narrow hypothesis "mixed teacher labels are the dominant cause" is rejected.

## Residual Decomposition

For the mixed Stage153 row on DFT validation:

| metric | value |
|---|---:|
| raw E RMSE | 126.883 meV/atom |
| global-offset-corrected E RMSE | 125.799 meV/atom |
| case/group-mean-offset-corrected E RMSE | 6.189 meV/atom |
| first-image-anchor E RMSE | 14.774 meV/atom |
| relative-image E RMSE | 8.546 meV/atom |
| barrier E RMSE | 16.089 meV/atom |

For the DFT-only Stage153 row:

| metric | value |
|---|---:|
| raw E RMSE | 133.835 meV/atom |
| global-offset-corrected E RMSE | 133.742 meV/atom |
| case/group-mean-offset-corrected E RMSE | 6.352 meV/atom |
| first-image-anchor E RMSE | 14.911 meV/atom |
| relative-image E RMSE | 8.188 meV/atom |
| barrier E RMSE | 16.007 meV/atom |

Thus the raw energy error is dominated by a case-level low-frequency offset, not by a single global shift and not by the NEB intra-path relative shape.

## E0 Status

The current Lightning rTECE training path fits least-squares per-element atomic energies on the training samples, then reloads the data with those E0s. The Stage153 benchmark uses `energy_reference = per_element_atomic_energies` and `energy_per_atom_shift = 0.0`.

This means the remaining raw energy problem is not the original review bug of using only one global per-atom shift. The likely failure is that per-element E0s are insufficient for OC20NEB case-level baselines across slabs, adsorbates, and reaction environments.

## Document-Grounded Interpretation

`TECE_design_space.md` decomposes deployment error into teacher error, projection error, distillation error, numerical error, and integrator error. Stage153 mainly points to projection/distillation/data-manifold error, not a simple label or E0 toggle.

`rTECE_review.md` says NEB evaluation should report relative energy, path/barrier errors, force errors, high-force tails, and deployment metrics instead of only absolute energy MAE. The Stage153 split confirms why: raw absolute energy is currently measuring a case-offset failure that is nearly absent after group anchoring.

## Stage154 Question

Stage154 asks whether front-loaded representation capacity can reduce this case-level offset while preserving high throughput:

1. raise atomic cross-radial shell sketch rank;
2. raise radial rank and species basis channels;
3. test a higher-capacity front-loaded point;
4. add one edge-relational cavity-vector increment.

The promotion rule is not "lowest raw E RMSE only". A useful point must improve raw E RMSE/MAE/max or force RMSE/MAE/max enough to justify measured atoms/s and memory cost, without degrading relative-image and barrier metrics.
