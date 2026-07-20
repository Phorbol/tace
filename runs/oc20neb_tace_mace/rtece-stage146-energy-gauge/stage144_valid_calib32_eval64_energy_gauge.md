# Stage146 Energy Gauge Diagnostic

Calibration and evaluation windows are disjoint; per-element residual E0 here is a diagnostic for gauge transfer, not a final benchmark correction.

- model: `runs/oc20neb_tace_mace/rtece-stage144-t3-cavity-vector/t3_cavity_vector_runs/t3_l2_cavity_vector_cond32_h64/t3_l2_cavity_vector_cond32_h64/rtece_scalar_best.pt`
- calibration: `:32`
- evaluation: `32:96`
- model energy reference: per-element=True, global_shift=0.0

| calibration | E RMSE meV/atom | E MAE meV/atom | E max meV/atom | E bias meV/atom | F RMSE meV/A | F MAE meV/A |
|---|---:|---:|---:|---:|---:|---:|
| none | 80.905 | 74.914 | 111.264 | -74.914 | 102.025 | 48.838 |
| global | 32.327 | 26.864 | 54.444 | -25.906 | 102.025 | 48.838 |
| per_element | 79.010 | 65.861 | 111.166 | -64.903 | 102.025 | 48.838 |
