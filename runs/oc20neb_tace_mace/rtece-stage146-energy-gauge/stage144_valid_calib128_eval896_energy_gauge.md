# Stage146 Energy Gauge Diagnostic

Calibration and evaluation windows are disjoint; per-element residual E0 here is a diagnostic for gauge transfer, not a final benchmark correction.

- model: `runs/oc20neb_tace_mace/rtece-stage144-t3-cavity-vector/t3_cavity_vector_runs/t3_l2_cavity_vector_cond32_h64/t3_l2_cavity_vector_cond32_h64/rtece_scalar_best.pt`
- calibration: `:128`
- evaluation: `128:1024`
- model energy reference: per-element=True, global_shift=0.0

| calibration | E RMSE meV/atom | E MAE meV/atom | E max meV/atom | E bias meV/atom | F RMSE meV/A | F MAE meV/A |
|---|---:|---:|---:|---:|---:|---:|
| none | 295.887 | 233.717 | 636.648 | -44.763 | 109.196 | 45.702 |
| global | 297.699 | 234.279 | 633.253 | -49.974 | 109.196 | 45.702 |
| per_element | 295.033 | 222.196 | 678.153 | -57.396 | 109.196 | 45.702 |
