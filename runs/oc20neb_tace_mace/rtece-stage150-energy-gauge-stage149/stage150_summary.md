# Stage150 Energy Gauge Diagnostic Summary

Stage150 reruns the Stage146-style residual gauge diagnostic on the two completed Stage149 checkpoints. Calibration uses valid configs 0:128 and evaluation uses disjoint valid configs 128:1024. The calibrated rows are diagnostics only and are not replacement benchmark scores.

## Results

| model | calibration | E RMSE | E MAE | E max | E bias | F RMSE | rel image RMSE | barrier RMSE | group offset RMSE | first anchor RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| L2 anchor | none | 95.983 | 65.234 | 276.095 | 30.728 | 110.212 | 8.975 | 16.575 | 6.248 | 14.687 |
| L2 anchor | global | 93.410 | 66.803 | 264.860 | 13.480 | 110.212 | 8.975 | 16.575 | 6.248 | 14.687 |
| L2 anchor | per-element residual | 97.121 | 66.852 | 278.595 | 20.684 | 110.212 | 8.975 | 16.575 | 6.248 | 14.687 |
| T3 cavity vector | none | 111.934 | 87.196 | 297.823 | 43.646 | 107.988 | 10.165 | 20.600 | 6.854 | 18.322 |
| T3 cavity vector | global | 107.777 | 83.177 | 289.000 | 30.100 | 107.988 | 10.165 | 20.600 | 6.854 | 18.322 |
| T3 cavity vector | per-element residual | 108.816 | 84.988 | 293.031 | 33.860 | 107.988 | 10.165 | 20.600 | 6.854 | 18.322 |

## Interpretation

- Stage149 already used fitted per-element atomic energies, not the old global per-atom shift.
- A residual per-element calibration fitted on valid 0:128 does not transfer to valid 128:1024. This argues against treating the remaining absolute E RMSE as a simple missing E0 problem.
- Global residual shift removes some bias but leaves E RMSE around 93-108 meV/atom, so absolute energy error is still too large for thermodynamics/composition transfer.
- Relative image and barrier metrics are unchanged by energy gauge calibration, as expected. They remain much smaller than absolute E RMSE, so the model captures part of the NEB PES shape while missing case-level absolute energy structure.
- The T3 cavity-vector path gives slightly lower F RMSE than L2 but worse energy metrics and about half the throughput in Stage149. It is not a Pareto improvement under this setup.

## Next Priority

The next algorithmic step should not be another E0 toggle. Based on rTECE_review.md and TECE_design_space.md, the next useful experiment is a representation/data step: teacher-selected scalar paths or active-set projection on the residual, combined with broader teacher fake-label coverage. A clean Stage151 should ask whether the missing absolute energy structure is recoverable by selected scalar/path basis capacity under the same training contract, rather than by post-hoc residual calibration.
