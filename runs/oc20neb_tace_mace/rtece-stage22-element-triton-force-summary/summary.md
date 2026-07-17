# Stage 22: Triton Element-Density Force Evaluator

Stage 22 applied the Stage 21 fused force-evaluator idea to the TECE-positive element-density descriptor. The trained checkpoints, descriptors, scalar MLP heads, and conservative chain rule are unchanged. The Triton kernel evaluates `(grad_rho + z_j grad_rho_z) dR/dr` and force accumulation directly.

## Results

| checkpoint | force mode | params | atoms/s | seconds/pass | peak alloc MB | peak reserved MB | DFT F MAE | DFT F RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| element-density 24x24 | analytic_density | 1057 | 15920358 | 0.015725 | 1404.9 | 1716 | 30.19297 | 111.49715 |
| element-density 24x24 | analytic_element_triton_force | 1057 | 30655057 | 0.008167 | 674.7 | 1084 | 30.19297 | 111.49715 |
| element-density 32x32 | analytic_density | 1665 | 14696599 | 0.017035 | 1404.9 | 1716 | 28.14186 | 109.92687 |
| element-density 32x32 | analytic_element_triton_force | 1665 | 26033459 | 0.009617 | 674.7 | 1084 | 28.14186 | 109.92687 |

## Profiler

For element-density 24x24 over 5 profiled passes, the old analytic path was led by `aten::mul` 24.01 ms, `index_add_` 9.32 ms, `index` 7.51 ms, and `div` 5.96 ms. The Triton force path shifts this to `aten::mul` 7.10 ms, `index_add_` 6.46 ms, `_element_density_force_kernel` 4.55 ms, and smaller gather/radial elementwise terms.

## Interpretation

This is a positive evaluator result for the lower-error TECE scalar descriptor. The current Pareto front is now pair-32 Triton at about 31.1M atoms/s and 39.42 meV/A, element-density 24x24 Triton at about 30.7M atoms/s and 30.19 meV/A, and element-density 32x32 Triton at about 26.0M atoms/s and 28.14 meV/A.

The next priority should be fused descriptor construction or a disciplined head/radial-channel sweep under the same TECE scalar projection. Adding vector/quadrupole descriptors remains lower priority because those paths were dominated before fusion.
