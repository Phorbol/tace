# Stage 21: Triton Pair Force Evaluator

Stage 21 tested the implementation priority identified by Stage 20 profiling: fuse the scalarized rTECE force evaluator rather than add more descriptors. The prototype keeps the pair-32 checkpoint, pair density descriptor, scalar MLP head, and conservative chain rule unchanged. Only the edge radial-derivative and force accumulation are moved into a Triton kernel.

## Result

| checkpoint | force mode | params | atoms/s | seconds/pass | peak alloc MB | peak reserved MB | DFT F MAE | DFT F RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pair-32 | analytic_pair | 1409 | 18622904 | 0.013443 | 1405.4 | 1716 | 39.42031 | 114.50856 |
| pair-32 | analytic_pair_triton_force | 1409 | 31097432 | 0.008051 | 675.2 | 1084 | 39.42031 | 114.50856 |

This moves the same pair-32 model from 18.6M to 31.1M atoms/s with numerically unchanged force error and lower memory.

## Profiler

Over 5 profiled passes, the old pair path was led by `aten::mul` 16.75 ms, `linear/addmm` 7.49 ms, `index_add_` 6.23 ms, and `div` 5.96 ms. The Triton path shifts the leading costs to `linear/addmm` 7.50 ms, `_pair_force_kernel` 4.28 ms, `aten::mul` 4.95 ms, and `index_add_` 3.33 ms.

## Interpretation

This is a positive evaluator result, not a new architecture. It supports the TECE/TACE thesis that once the equivariant state is renormalized down to scalar pair density, the remaining route to NEP/DPA-like throughput is a compact fused evaluator. The Pareto front now has a new maximum-throughput endpoint: pair-32 Triton at about 31.1M atoms/s and 39.42 meV/A force MAE.

Next priority is to apply the same fused-evaluator discipline to the TECE-positive element-density descriptor, because that is where the current scalar rTECE front gets substantially lower force error.
