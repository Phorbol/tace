# Stage170 Energy RMSE Diagnosis

The current rTECE absolute energy error is genuinely poor. The best recent Stage157 row (`stage157_direct_b32_rel0p25_mixed2048`) has DFT E RMSE `52.538` meV/atom, E MAE `35.443` meV/atom, and E max `204.986` meV/atom. Its force metrics are much more competitive: F RMSE `94.159` meV/A, F MAE `41.823` meV/A.

This is not the original global `energy_per_atom_shift` bug by itself. Current rTECE training fits per-element `atomic_energies` by default, the model forward adds `atomic_reference_energy`, and the Stage157 run used 38 fitted element references. Stage150 also showed that residual per-element calibration fitted on valid `0:128` did not transfer to valid `128:1024`.

The strongest decomposition is Stage162/165: raw E RMSE is `52-58` meV/atom, but after an oracle per-NEB-case offset it drops to about `4.8` meV/atom; relative-image RMSE is about `7.1` meV/atom and barrier RMSE is about `11.1-11.4` meV/atom. So the model captures part of the local NEB shape but misses a deployable low-frequency case/site/adsorbate energy baseline.

Stage169 makes the failure sharper. Current implemented rTECE semantic descriptor groups can fit the 22 case offsets in-sample, but LOOCV rejects them as deployable offset coordinates: intercept-only LOOCV is `60.482` meV/atom, while all current descriptor groups overfit to train RMSE `0.007` and LOOCV `108.732` meV/atom.

Compared with Stage145 community baselines, NEP smoke has E RMSE `23.846` meV/atom and F RMSE `143.037` meV/A; rTECE is faster and has better F RMSE in Stage157, but its raw E RMSE is more than 2x worse. The DPA1-zero-attention smoke has poor E RMSE (`325.281` meV/atom), so it is not a useful energy target yet.

## Diagnosis

- The energy issue is real and must stay on the Pareto scorecard through E/F RMSE, E/F MAE, and max errors.
- It is not solved by another global/per-element E0 toggle.
- It is not mainly a relative NEB curve-shape failure.
- It is not fixed by the current implemented rTECE descriptors as simple low-frequency baseline features.
- The most likely missing piece is a deployable low-frequency chemical/site/adsorbate representation plus true teacher projection/downfolding.

## Next Priority

Stage171 should not add benchmark-only group offsets. It should target the Stage165 case-offset residual with TECE-aligned candidate scalar/edge paths selected by teacher response, then compare the same architecture trained from scratch against renormalized/distilled initialization. This keeps the route aligned with `rTECE_review.md`: per-element reference plus residual learning, teacher cache, semantic projection, and Schur/Gauss-Newton downfolding.
