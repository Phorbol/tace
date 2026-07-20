# Stage146 Energy Gauge Diagnostic

Calibration and evaluation windows are disjoint; per-element residual E0 here is a diagnostic for gauge transfer, not a final benchmark correction.

- model: `runs/oc20neb_tace_mace/rtece-stage155-edge-relational-ladder/edge_ladder_runs/stage155_t3_cavity_vector_quad_direct_mixed2048/stage155_t3_cavity_vector_quad_direct_mixed2048/rtece_scalar_best.pt`
- calibration: `:32`
- evaluation: `128:192`
- model energy reference: per-element=True, global_shift=0.0
- energy_only: True

| calibration | E RMSE meV/atom | E MAE meV/atom | E max meV/atom | E bias meV/atom | F RMSE meV/A | F MAE meV/A | relative image RMSE meV/atom | barrier RMSE meV/atom | group mean-offset RMSE meV/atom | first image-anchor RMSE meV/atom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 59.655 | 37.766 | 112.473 | 26.745 | nan | nan | 4.801 | 6.249 | 3.302 | 4.801 |
| global | 62.878 | 36.466 | 119.023 | 35.599 | nan | nan | 4.801 | 6.249 | 3.302 | 4.801 |
| per_element | 62.011 | 39.048 | 116.832 | 28.052 | nan | nan | 4.801 | 6.249 | 3.302 | 4.801 |
