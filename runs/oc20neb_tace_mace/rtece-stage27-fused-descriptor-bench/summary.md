# Stage 27: Triton Fused Element-Density Descriptor

Stage 27 moved the TECE-positive element-density descriptor construction from PyTorch radial tensors plus `index_add_` into a fused Triton edge kernel. The new benchmark mode is `analytic_element_triton_descriptor_force`: the descriptor kernel builds packed `[rho, rho_z]` node descriptors, the scalar MLP remains unchanged, and the existing Triton force kernel still evaluates the conservative chain rule.

## Implementation Gate

- Added `packed_element_density_descriptors` as the CPU/PyTorch equivalence anchor for packed `[rho, rho_z]` descriptors.
- Added `element_density_descriptors_triton` and `RTECEScalarModel.forward_element_density_triton_descriptor_force_analytic_forces`.
- Exposed `analytic_element_triton_descriptor_force` in benchmark and profiler force-mode choices.
- Test gate: `test/test_rtece_scalar.py` passed, 38 tests.

## Benchmark

Checkpoint: `runs/oc20neb_tace_mace/rtece-scalar-678773/rtece_element_density/rtece_scalar_best.pt`, radial8/24x24, 1057 parameters. Benchmark window: DFT valid `:4096`, 250355 atoms, prebuilt batched graph, float32, V100.

| force mode | atoms/s | seconds/pass | peak alloc MB | peak reserved MB | DFT F MAE | DFT F RMSE | DFT E MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| `analytic_element_triton_force` | 29413444 | 0.008512 | 674.7 | 1084 | 30.19297 | 111.49715 | 1402.9 |
| `analytic_element_triton_descriptor_force` | 50975964 | 0.004911 | 212.1 | 402 | 30.19297 | 111.49715 | 1402.9 |

## Interpretation

This is a positive TECE hardware-renormalization result. It changes neither the learned checkpoint nor the descriptor semantics, but removes the PyTorch descriptor-construction bottleneck and shortens intermediate edge-state lifetime. The radial8/24x24 lower-error point now reaches about 51.0M atoms/s on this benchmark while preserving the 30.19 meV/A DFT force MAE.

The current priority should be to rebenchmark the existing radial4, radial5, and radial8 checkpoints with the fused descriptor+force mode and then update the Pareto front. If the same speedup holds for radial4/radial5, the scalar rTECE family may already exceed the previous ultra-fast endpoint while keeping a clean TECE interpretation; if the speedup is strongest for radial8, radial8/24x24 may become the new best throughput/error tradeoff.
