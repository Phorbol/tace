# Stage 28: Fused Descriptor rTECE Pareto Rebenchmark

Stage 28 rebenchmarked the current element-density scalar rTECE front with the Stage 27 `analytic_element_triton_descriptor_force` evaluator. The model checkpoints were unchanged; only descriptor construction and force evaluation used the fused Triton implementation. All benchmarks use the 4096-config prefix, 250355 atoms, prebuilt batched graph, float32, and one V100.

## DFT Prefix Results

| point | num radial | hidden | params | atoms/s | peak alloc MB | DFT F MAE | DFT F RMSE | teacher F MAE | status |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| radial4h16 | 4 | 16x16 | 449 | 72644836 | 176.6 | 43.91 | 114.91 | 47.77 | max-throughput endpoint |
| radial5h16 seed2-long | 5 | 16x16 | 481 | 64515626 | 178.0 | 38.28 | 112.60 | 42.24 | seed-sensitive middle point |
| radial5h16 stage24 | 5 | 16x16 | 481 | 64463762 | 178.0 | 39.02 | 114.24 | 43.07 | dominated by seed2-long if best-seed protocol is allowed |
| radial8h24 | 8 | 24x24 | 1057 | 50929958 | 212.1 | 30.19 | 111.50 | 35.83 | lower-error front point |
| radial8h32 | 8 | 32x32 | 1665 | 40750024 | 242.7 | 28.14 | 109.93 | 33.93 | lowest-error scalar point |

## Interpretation

Fusing descriptor construction changes the quantitative front. The ultra-fast endpoint is now radial4/16x16 at 72.6M atoms/s, but the lower-error radial8/24x24 point also reaches 50.9M atoms/s with 30.19 meV/A DFT force MAE. This confirms that TECE architecture degradation and hardware renormalization must be treated as coupled axes: a less degraded descriptor can remain high-throughput if edge-state lifetime is eliminated.

The clean current Pareto front under the fused evaluator is radial4/16x16, radial5/16x16 as a seed-sensitive band, radial8/24x24, and radial8/32x32. The stage24 radial5 checkpoint is dominated by the seed2-long checkpoint if best-of-seed selection is allowed, but Stage 25-26 showed radial5 accuracy is not deterministic under the current selection protocol, so it should be reported as a band rather than a single guaranteed point.

The next experimental priority is to add a direct DFT-force validation selector or a best-of-N protocol with reported seed cost, then rerun radial5. The next implementation priority is to make the fused descriptor path the default for eligible element-density benchmarks and profile whether graph construction or neighbor-list building becomes the next dominant bottleneck.
