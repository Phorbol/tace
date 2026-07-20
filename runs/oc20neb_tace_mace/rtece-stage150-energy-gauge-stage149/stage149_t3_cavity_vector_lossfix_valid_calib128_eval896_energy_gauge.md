# Stage146 Energy Gauge Diagnostic

Calibration and evaluation windows are disjoint; per-element residual E0 here is a diagnostic for gauge transfer, not a final benchmark correction.

- model: `runs/oc20neb_tace_mace/rtece-stage149-energy-loss-normalization-rerun/corrected_energy_loss_runs/stage149_t3_cavity_vector_lossfix/stage149_t3_cavity_vector_lossfix/rtece_scalar_best.pt`
- calibration: `:128`
- evaluation: `128:1024`
- model energy reference: per-element=True, global_shift=0.0

| calibration | E RMSE meV/atom | E MAE meV/atom | E max meV/atom | E bias meV/atom | F RMSE meV/A | F MAE meV/A | relative image RMSE meV/atom | barrier RMSE meV/atom | group mean-offset RMSE meV/atom | first image-anchor RMSE meV/atom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 111.934 | 87.196 | 297.823 | 43.646 | 107.988 | 50.879 | 10.165 | 20.600 | 6.854 | 18.322 |
| global | 107.777 | 83.177 | 289.000 | 30.100 | 107.988 | 50.879 | 10.165 | 20.600 | 6.854 | 18.322 |
| per_element | 108.816 | 84.988 | 293.031 | 33.860 | 107.988 | 50.879 | 10.165 | 20.600 | 6.854 | 18.322 |
