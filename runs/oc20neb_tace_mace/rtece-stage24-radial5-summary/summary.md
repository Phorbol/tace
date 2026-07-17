# Stage 24: Radial5 Gap-Fill Under Triton Element-Density

Stage 24 filled the radial-resolution gap between Stage 23 radial4 and Stage 22 radial8. It fixed `rtece_element_density`, force weight 30, mixed labels, and `analytic_element_triton_force`, then swept only `num_radial=5` head width.

## Prefix Results

| num radial | hidden | params | atoms/s | peak alloc MB | DFT F MAE | teacher F MAE |
|---:|---|---:|---:|---:|---:|---:|
| 4 | 16x16 | 449 | 42954558 | 436.4 | 43.91 | 47.77 |
| 5 | 16x16 | 481 | 38825023 | 495.4 | 39.02 | 43.07 |
| 5 | 20x20 | 681 | 38384295 | 495.4 | 49.64 | 53.20 |
| 5 | 24x24 | 913 | 37893470 | 495.4 | 46.57 | 50.23 |
| 6 | 24x24 | 961 | 34333375 | 554.9 | 42.87 | 46.67 |
| 8 | 24x24 | 1057 | 30655057 | 674.7 | 30.19 | 35.83 |

## Offset Robustness For Radial5/16x16

| extxyz index | configs | atoms/s | DFT F MAE | DFT F RMSE | DFT E MAE |
|---|---:|---:|---:|---:|---:|
| `:4096` | 4096 | 38423189 | 39.02 | 114.24 | 1449.6 |
| `4096:8192` | 4096 | 39634440 | 41.61 | 123.82 | 1533.4 |
| `8192:12288` | 1808 | 33595106 | 40.55 | 122.25 | 1358.6 |

## Interpretation

Radial5/16x16 is a useful middle Pareto point: it is much more accurate than radial4 while remaining faster than pair32 and radial8 element-density. Wider radial5 heads are dominated, which suggests the capacity bottleneck is not simply MLP width. The next priority should be fused descriptor construction or a robustness repeat for radial5/16x16 with a longer/seed-varied training run.
