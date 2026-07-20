# Stage146 Energy Gauge Diagnostic

Calibration and evaluation windows are disjoint; per-element residual E0 here is a diagnostic for gauge transfer, not a final benchmark correction.

- model: `runs/oc20neb_tace_mace/rtece-stage155-edge-relational-ladder/edge_ladder_runs/stage155_t3_cavity_vector_direct_mixed2048/stage155_t3_cavity_vector_direct_mixed2048/rtece_scalar_best.pt`
- calibration: `:32`
- evaluation: `128:192`
- model energy reference: per-element=True, global_shift=0.0
- energy_only: True

| calibration | E RMSE meV/atom | E MAE meV/atom | E max meV/atom | E bias meV/atom | F RMSE meV/A | F MAE meV/A | relative image RMSE meV/atom | barrier RMSE meV/atom | group mean-offset RMSE meV/atom | first image-anchor RMSE meV/atom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 52.364 | 35.910 | 96.916 | 19.951 | nan | nan | 6.357 | 7.383 | 4.865 | 6.357 |
| global | 63.081 | 62.831 | 80.619 | -28.064 | nan | nan | 6.357 | 7.383 | 4.865 | 6.357 |
| per_element | 39.859 | 28.956 | 73.273 | 12.867 | nan | nan | 6.357 | 7.383 | 4.865 | 6.357 |
