# Stage153 DFT-Only Energy Control Summary

Stage153 tested whether the persistent rTECE absolute-energy error is mainly caused by the 0.75 teacher / 0.25 DFT mixed-label contract. Both controls used the same Stage152 low-cost semantic front (`t2_l2_atomic_cross`), the same 51.4k-parameter rTECE scalar architecture, the same 20k-step Lightning training setup, fitted per-element E0s, and the same DFT/teacher validation benchmarks.

## Jobs

| job | role | state | exit | elapsed | max RSS |
|---:|---|---|---|---|---:|
| 687455 | DFT-only base2048 | COMPLETED | 0:0 | 00:10:28 | 1734692K |
| 687456 | mixed base2048 | COMPLETED | 0:0 | 00:10:22 | 1609732K |

## Training Contract

| variant | train labels | train configs | best step | best valid loss | E/F weights | scheduler |
|---|---|---:|---:|---:|---|---|
| `stage153_l2_dft_only_base2048` | `dft_energy` / `dft_forces` remapped to `energy` / `forces` | 2048 | 16384 | 0.117725 | 1 / 10 | warmup 500 + plateau |
| `stage153_l2_mixed_base2048` | existing 0.75 teacher / 0.25 DFT target | 2048 | 19712 | 0.108605 | 1 / 10 | warmup 500 + plateau |

The DFT-only remap smoke verified that output `energy` and `forces` match the DFT fields exactly, while original teacher labels are preserved.

## DFT-Valid Benchmark

Errors are meV/atom for energy and meV/A for force.

| variant | atoms/s | E RMSE | E MAE | E bias | E max | F RMSE | F MAE | F max | rel image RMSE | barrier RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DFT-only | 1.024e6 | 133.835 | 99.787 | 5.000 | 359.323 | 110.465 | 50.928 | 1881.008 | 8.188 | 16.007 |
| mixed | 1.023e6 | 126.883 | 99.102 | -16.553 | 325.773 | 105.745 | 46.922 | 1641.924 | 8.546 | 16.089 |

## Teacher-Valid Benchmark

| variant | atoms/s | E RMSE | E MAE | E bias | E max | F RMSE | F MAE | F max | rel image RMSE | barrier RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DFT-only | 1.024e6 | 113.861 | 86.970 | 1.024 | 318.083 | 107.788 | 51.813 | 1614.828 | 7.533 | 13.747 |
| mixed | 1.024e6 | 107.619 | 82.136 | -20.529 | 284.533 | 103.735 | 48.348 | 1585.007 | 7.835 | 13.516 |

## Interpretation

- DFT-only labels do not repair the absolute-energy problem. On DFT-valid, DFT-only is worse than mixed in both E RMSE (`133.8` vs `126.9`) and F RMSE (`110.5` vs `105.7`).
- Throughput is unchanged at about `1.02e6 atoms/s`, so the paired comparison isolates label/objective rather than hardware cost.
- Relative NEB metrics stay much smaller than absolute E RMSE. The model still captures part of the path shape, but case-level absolute energy remains a weak axis.
- This rejects the narrow hypothesis that mixed teacher labels are the dominant cause of the Stage149 energy error. The next mainline should treat the remaining error as representation/projection/data-manifold limited.

## Document-Grounded Next Priority

`TECE_design_space.md` says projection error and training/distillation error must be separated; Stage153 now shows that simply switching the training target to DFT does not remove the error. `rTECE_review.md` says path selection should use active-set / Schur / Gauss-Newton refits plus real marginal hardware cost. Therefore the next stage should not be another E0 toggle or teacher-weight sweep.

Stage154 should test a front-loaded representation expansion ladder over the current T2 L2 front:

1. Add selected L3/L4 Cartesian scalar paths or higher-order scalar interactions only where Stage152/153 residuals justify them.
2. Keep the path manifest explicit and measure E RMSE/MAE/max, F RMSE/MAE/max, relative NEB/barrier, and atoms/s for every increment.
3. Use Schur/GN or active-set residual diagnostics to rank candidate groups before full training.
4. Keep edge-relational paths optional and costed; Stage152 showed they are not yet a clean Pareto win.
