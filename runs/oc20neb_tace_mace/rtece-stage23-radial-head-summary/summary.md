# Stage 23: Radial/Head Capacity Under Triton Element-Density

Stage 23 fixed the TECE-positive `rtece_element_density` descriptor family and the `analytic_element_triton_force` evaluator, then swept radial resolution and scalar head width. This tests a clean scalar-projection capacity axis: coarsen radial density resolution and shrink the readout while preserving conservative forces.

## Prefix Results

| num radial | hidden | params | atoms/s | peak alloc MB | DFT F MAE | teacher F MAE |
|---:|---|---:|---:|---:|---:|---:|
| 4 | 16x16 | 449 | 42305218 | 436.4 | 43.91 | 47.77 |
| 4 | 24x24 | 865 | 40459620 | 436.4 | 48.65 | 52.13 |
| 6 | 16x16 | 513 | 35899727 | 554.9 | 45.42 | 49.18 |
| 6 | 24x24 | 961 | 34333375 | 554.9 | 42.87 | 46.67 |
| 8 | 24x24 | 1057 | 30655057 | 674.7 | 30.19 | 35.83 |
| 8 | 32x32 | 1665 | 26033459 | 674.7 | 28.14 | 33.93 |

## Offset Robustness For Radial4/16x16

| extxyz index | configs | atoms/s | DFT F MAE | DFT F RMSE | DFT E MAE |
|---|---:|---:|---:|---:|---:|
| `:4096` | 4096 | 42954558 | 43.91 | 114.91 | 1430.0 |
| `4096:8192` | 4096 | 43229041 | 43.93 | 122.23 | 1516.6 |
| `8192:12288` | 1808 | 38643787 | 42.99 | 120.32 | 1349.6 |

## Interpretation

`num_radial=4, hidden=16x16` is a new extreme-throughput endpoint around 43M atoms/s, but it pays a large accuracy penalty. It is useful because it quantifies the high-throughput limit of the TECE scalar element-density projection, not because it is the best general model.

The non-monotonic results show that simply widening the head is not enough when the radial basis is too coarse. The next targeted experiment should either fill the gap with `num_radial=5` or reduce implementation cost by fusing descriptor construction, since Stage 22/23 profiles still leave density `index_add_` overhead visible.
