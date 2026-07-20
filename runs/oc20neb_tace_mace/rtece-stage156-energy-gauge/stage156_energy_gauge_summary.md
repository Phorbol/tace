# Stage156 Energy Gauge Diagnostic Summary

Stage156 responds to the observed poor rTECE raw energy MAE/RMSE after Stage155. The diagnostic is intentionally separated from force benchmarking: it tests whether raw absolute energy error is dominated by transferable energy gauge/E0 terms or by the learned residual PES itself.

## Why This Stage

rTECE_review.md Section 7 warned that a global per-atom energy shift is insufficient for multi-element data and that per-element E0s should be fitted before asking the small model to absorb composition differences. The current training code already uses per-element least-squares atomic energies, so the remaining question is whether residual raw energy error is still gauge-like. The same review also says NEB selection should not use only absolute energy MAE, but should separately track relative image energy, barrier error, force RMSE/tails, and deployment physical metrics.

## Engineering Change

- Added RTECEScalarModel.forward(graph, compute_forces=False) for energy-only inference. Default behavior remains compute_forces=True, so training, ASE calculator force paths, and existing benchmarks keep the same output contract.
- Updated analyze_rtece_energy_gauge.py with --energy-only; force metrics are reported as NaN instead of silently fabricating force errors.
- Added regression tests for energy-only forward equality and energy-only metric summaries.

## Medium-Window Diagnostic

Calibration window: valid.extxyz :32; evaluation window: valid.extxyz 128:192. This is a medium interactive diagnostic, not the final full-window benchmark. Errors are meV/atom.

| model | calibration | E RMSE | E MAE | E max | E bias | relative-image RMSE | barrier RMSE | group-offset RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Stage155 direct | none | 52.364 | 35.910 | 96.916 | 19.951 | 6.357 | 7.383 | 4.865 |
| Stage155 direct | global residual | 63.081 | 62.831 | 80.619 | -28.064 | 6.357 | 7.383 | 4.865 |
| Stage155 direct | per-element residual | 39.859 | 28.956 | 73.273 | 12.867 | 6.357 | 7.383 | 4.865 |
| Stage155 quad+direct | none | 59.655 | 37.766 | 112.473 | 26.745 | 4.801 | 6.249 | 3.302 |
| Stage155 quad+direct | global residual | 62.878 | 36.466 | 119.023 | 35.599 | 4.801 | 6.249 | 3.302 |
| Stage155 quad+direct | per-element residual | 62.011 | 39.048 | 116.832 | 28.052 | 4.801 | 6.249 | 3.302 |

## Interpretation

1. The user's observation is correct: raw absolute energy MAE/RMSE is still poor. Full Stage155 DFT-valid metrics are still 71.830/50.575 meV/atom for direct and 68.099/44.976 meV/atom for quad+direct.
2. The error is not a simple global shift. On the medium window, global residual calibration worsens Stage155 direct and barely helps only MAE for quad+direct.
3. Per-element residual calibration helps Stage155 direct substantially, from 52.364 to 39.859 E RMSE, but does not help quad+direct, from 59.655 to 62.011. Therefore the remaining raw E problem is partly energy gauge/composition transfer and partly model/data-dependent low-frequency residual.
4. Relative/path energy is much better than raw absolute energy: relative-image RMSE is about 4.8-6.4 meV/atom and barrier RMSE about 6.2-7.4 meV/atom in this window. This supports the document's instruction to evaluate NEB models with relative energy and barrier metrics, not raw absolute energy alone.
5. Quad+direct is better for relative/path metrics and group-offset residual, while direct is better after residual per-element E0 calibration and is faster in Stage155 throughput. This remains a real Pareto split, not a single winner.

## Priority Update

Immediate priority is not widening the final MLP. The next model/training work should test a cleaner energy-baseline contract: source-aware or case/composition-stratified E0 fitting, teacher/DFT mixed-label E0 consistency, and loss terms that separate absolute energy gauge from relative NEB energy. In parallel, physical triage should continue on the promoted Stage155 points so we do not optimize raw E while damaging dimer/rattle-relax behavior.

The full 128/256 energy-gauge diagnostic should be run via sbatch GPU or a batched graph evaluator; interactive CPU Python loops are too slow for that window.
