# rTECE Stage131 Results

## Setup

Stage131 fixed the stage129 species24 current-Pareto atomic backbone and non-weighted teacher-rattle train set, then added minimal TECE/rTECE edge-relational scalar residual paths. Jobs 685123 and 685124 completed with exit code 0:0.

## Metrics

| variant | params | DFT E RMSE | DFT F RMSE | DFT F max | atoms/s | delta F RMSE vs stage129 | delta F max vs stage129 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| l1_active_species24_cavity_vec_residual_h64 | 29117 | 282.884 | 104.996 | 2296.722 | 567140 | 3.439 | -400.810 |
| l2_active_species24_cavity_vecq_residual_h64 | 29181 | 298.722 | 108.596 | 2399.430 | 413308 | 7.039 | -298.103 |

Stage129 species24 baseline: DFT E RMSE 273.849 meV/atom, DFT F RMSE 101.557 meV/A, DFT F max 2697.533 meV/A, throughput 1224418 atoms/s.

## Interpretation

- The edge-residual rows are not RMSE/throughput Pareto improvements versus stage129 species24.
- L1 cavity-vector residual has a real tail signal: DFT F max improves by about 401 meV/A, but DFT F RMSE worsens by 3.439 meV/A and throughput falls to 0.567M atoms/s.
- L2 quadrupole residual is dominated by L1 here; adding higher-L edge scalarization naively is not the next priority.
- Next step should be physical triage for the L1 row because the max-error improvement might correspond to better rattle/dimer behavior. If not, move away from this expensive edge residual implementation toward projection diagnostics or broader teacher trajectory/Jacobian distillation.
