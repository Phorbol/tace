# Stage146 Energy Gauge Diagnostic

Calibration and evaluation windows are disjoint; per-element residual E0 here is a diagnostic for gauge transfer, not a final benchmark correction.

- model: `runs/oc20neb_tace_mace/rtece-stage149-energy-loss-normalization-rerun/corrected_energy_loss_runs/stage149_l2_anchor_lossfix/stage149_l2_anchor_lossfix/rtece_scalar_best.pt`
- calibration: `:128`
- evaluation: `128:1024`
- model energy reference: per-element=True, global_shift=0.0

| calibration | E RMSE meV/atom | E MAE meV/atom | E max meV/atom | E bias meV/atom | F RMSE meV/A | F MAE meV/A | relative image RMSE meV/atom | barrier RMSE meV/atom | group mean-offset RMSE meV/atom | first image-anchor RMSE meV/atom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| none | 95.983 | 65.234 | 276.095 | 30.728 | 110.212 | 51.319 | 8.975 | 16.575 | 6.248 | 14.687 |
| global | 93.410 | 66.803 | 264.860 | 13.480 | 110.212 | 51.319 | 8.975 | 16.575 | 6.248 | 14.687 |
| per_element | 97.121 | 66.852 | 278.595 | 20.684 | 110.212 | 51.319 | 8.975 | 16.575 | 6.248 | 14.687 |
