# Stage162 Energy Root-Cause Note

This note responds to the current observation that rTECE energy MAE/RMSE is still poor. It keeps the priority tied to `TECE_design_space.md` and `rTECE_review.md`: raw energy, case/slab/adsorbate gauge, relative NEB shape, force tails, and throughput must be separated before changing the architecture.

## Current Evidence

Representative Stage157 DFT-valid model: `stage157_direct_b32_rel0p25_mixed2048`.

- raw E RMSE/MAE/max: `52.538 / 35.443 / 204.986` meV/atom
- group-offset E RMSE: `4.769` meV/atom
- first-image-anchor E RMSE: `11.461` meV/atom
- relative-image/barrier E RMSE: `7.104 / 11.403` meV/atom
- F RMSE/MAE/max: `94.159 / 41.823 / 2092.961` meV/A
- throughput in the current prebuilt graph benchmark: `5.00e5` atoms/s

Across Stage157 relative-energy weights, group-offset correction removes about `90.9-91.7%` of raw E RMSE. The largest repeated case offsets are tens to about 200 meV/atom, led by `dissociation_id_212_7241_50_011-0_neb1.0`.

## Interpretation

The user observation is correct: raw absolute energy is not acceptable. However, the decomposition says the dominant failure is not the old `energy_per_atom_shift` bug and not a single global E0 shift. The model already uses least-squares per-element atomic energies with `energy_per_atom_shift = 0.0`; a global offset barely helps.

The main error mode is a low-frequency case/site/slab/adsorbate baseline that per-element E0s cannot represent. After removing a per-case mean, the remaining NEB shape error is much smaller but still nonzero, and force RMSE/max remain weak. This matches `rTECE_review.md`: E0/gauge, physical closure, and real distillation are P0/P1 issues. It also matches `TECE_design_space.md`: the compiler objective must rank retained TECE semantic paths by physical error reduction under hardware cost, not by descriptor residual alone.

## Connection To Stage160-161

Stage160/161 active-set projection gives the first useful architecture clue. Adding the `uv` edge-frame relational scalar path to the Stage157 baseline improves holdout projected E RMSE from `37.854` to `11.239` meV/atom, but worsens sampled F RMSE from `79.022` to `85.290` meV/A. `uv_uqu` is dominated and `uqu` is rejected.

Therefore energy can be repaired by document-consistent edge-relational scalar information, but naive promotion hurts force. The next architecture experiment should not be another final-MLP widening sweep. It should be a force-protected representation increment or short refit/Gauss-Newton diagnostic that tests whether energy-baseline repair can coexist with force RMSE.

## Next Priority

1. Keep raw E RMSE/MAE/max in the primary table, but always report group-offset, first-anchor, relative-image/barrier, and F RMSE/MAE/max beside it.
2. Do not add per-case IDs or C/N-specific patches. That would explain the current data but violate the deployable TECE compiler goal.
3. Test a symmetry-preserving, learnable front-loaded baseline path: low-rank species/radial/site-conditioned scalar summaries and the `uv` relational scalar under a force-protected objective.
4. Use Stage161 sampled-force active-set scoring as the pre-training filter. Candidate promotion should require energy gain without unacceptable sampled-force regression.
5. Compare any promoted Stage162/163 model against NEP/DPA-like community baselines on E/F RMSE/max, relative NEB energy, dimer scan, rattle-relax, and throughput scaling.
