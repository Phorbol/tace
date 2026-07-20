# rTECE Stage Sweep Summary: stage149-energy-loss-normalization-rerun

Primary metric: DFT force RMSE vs atoms/s. Missing benchmark rows are kept explicit. Absolute E RMSE is reported alongside relative image/barrier RMSE and case/first-anchor offsets to separate NEB PES shape from energy gauge.

| row | status | params | TECE axes | atoms/s | F RMSE | F MAE | F max | E RMSE | E MAE | E bias | E max | rel image RMSE | barrier RMSE | case offset RMSE | first anchor RMSE |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stage149_l2_anchor_lossfix | benchmark_found | 51367 | atomic.radial_density, atomic.species_basis_density, atomic.vector_norm, atomic.vector_cross_radial_dot, atomic.quadrupole_norm, atomic.quadrupole_cross_radial_frobenius | 1.023e+06 | 108.282 | 50.211 | 2354.453 | 92.671 | 61.149 | 28.249 | 276.097 | 8.507 | 15.963 | 5.970 | 13.897 |
| stage149_t3_cavity_vector_lossfix | missing_benchmark | NA | atomic.radial_density, atomic.species_basis_density, atomic.vector_norm, atomic.vector_cross_radial_dot, atomic.quadrupole_norm, atomic.quadrupole_cross_radial_frobenius, edge.cavity.vector_dot | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA | NA |

## Missing Benchmark Rows

- `stage149_t3_cavity_vector_lossfix`

## DFT Force RMSE Pareto Front

- `stage149_l2_anchor_lossfix`: atoms/s=1.023e+06, F RMSE=108.282

## Review Notes

- TECE_design_space: hardware Pareto frontier over physical error vs atom/s and memory
- rTECE_review: record force RMSE, high-force tail/max error, energy RMSE/bias/max separately
- rTECE_review: report relative image/barrier RMSE and case/first-anchor offsets so NEB PES shape is not conflated with energy gauge
- rTECE_review: keep dimer/rattle/element-stratified physical probes explicit rather than collapsing to one MAE
