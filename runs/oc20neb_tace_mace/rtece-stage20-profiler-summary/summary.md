# rTECE Scalar Profiling Summary

Scope: 4096 valid configs, prebuilt batched graph, 250355 atoms, 5 profiled passes on one V100. Times below are profiler inclusive device times over the profiled passes, so they should guide bottleneck ordering but not be summed as percentages.

## Pair 32x32 Analytic Pair

| op | calls | device time ms | CPU time ms |
|---|---:|---:|---:|
| aten::mul | 70 | 16.751 | 0.736 |
| aten::linear | 15 | 7.491 | 0.827 |
| aten::addmm | 15 | 7.491 | 0.528 |
| cuda mul elementwise kernel | 25 | 7.353 | 0.000 |
| volta_sgemm | 10 | 7.125 | 0.000 |
| aten::index_add_ | 25 | 6.227 | 0.305 |
| cuda index_add reduce kernel | 15 | 5.986 | 0.000 |
| aten::div | 40 | 5.958 | 0.426 |

## Element-Density 24x24 Analytic Density

| op | calls | device time ms | CPU time ms |
|---|---:|---:|---:|
| aten::mul | 85 | 24.006 | 0.882 |
| cuda mul elementwise kernel | 35 | 11.690 | 0.000 |
| aten::index_add_ | 30 | 9.321 | 0.333 |
| cuda index_add reduce kernel | 20 | 9.084 | 0.000 |
| aten::index | 25 | 7.513 | 0.789 |
| aten::div | 40 | 5.957 | 0.440 |
| aten::sum | 15 | 5.176 | 0.414 |
| aten::pow | 35 | 3.648 | 0.703 |

## Stage-20 Interpretation

- Both front points are dominated by radial/force elementwise kernels plus scatter/gather work, not by parameter count.
- Pair-32 still spends large device time in `aten::mul`, `aten::index_add_`, `aten::div`, and indexed gathers; MLP GEMM is visible but not the only bottleneck.
- Element-density-24 adds more `mul`, `index_add_`, and `index` work; this explains why its semantic gain costs throughput even with a smaller 24x24 head.
- The next implementation target should be a lower-level fused evaluator for radial basis/cutoff, density and element-density accumulation, descriptor-gradient edge scale, and force accumulation. MLP-only fusion or PyTorch tensor packing is too narrow.
