# Stage146 Energy Gauge Diagnostic

Calibration and evaluation windows are disjoint; per-element residual E0 here is a diagnostic for gauge transfer, not a final benchmark correction.

- model: `runs/oc20neb_tace_mace/rtece-stage144-t3-cavity-vector/t3_cavity_vector_runs/t3_l2_cavity_vector_cond32_h64/t3_l2_cavity_vector_cond32_h64/rtece_scalar_best.pt`
- calibration: `:128`
- evaluation: `128:384`
- model energy reference: per-element=True, global_shift=0.0

| calibration | E RMSE meV/atom | E MAE meV/atom | E max meV/atom | E bias meV/atom | F RMSE meV/A | F MAE meV/A |
|---|---:|---:|---:|---:|---:|---:|
| none | 159.182 | 124.010 | 287.049 | -27.555 | 93.703 | 39.837 |
| global | 159.895 | 122.996 | 282.678 | -32.196 | 93.703 | 39.837 |
| per_element | 139.492 | 103.499 | 240.911 | -47.501 | 93.703 | 39.837 |
