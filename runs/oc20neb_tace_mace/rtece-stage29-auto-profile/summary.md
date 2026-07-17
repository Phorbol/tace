# Stage 29: Default Auto Evaluator And Graph-Construction Bottleneck

Stage 29 made the Stage 27 fused element-density evaluator the default `auto` path for eligible rTECE element-density benchmarks and profilers. Eligibility is intentionally narrow: CUDA, float32, `use_element_density=True`, no density quadratic/vector/atomic moments, and no edge sketches. Other models remain on autograd unless a force mode is explicitly requested.

## Test Gate

`test/test_rtece_scalar.py` passed with 39 tests after adding the auto selector.

## Benchmark

Checkpoint: radial8/24x24 element-density, 1057 parameters. Window: DFT valid `:1024`, 59193 atoms, float32, one V100. Both benchmark rows requested `--force-mode auto` and resolved to `analytic_element_triton_descriptor_force`.

| mode | prebuilt graph | includes graph construction | atoms/s | configs/s | seconds/pass | peak alloc MB | DFT F MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| auto fused evaluator | yes | no | 34203147 | 591692 | 0.001731 | 63.2 | 33.17 |
| auto fused evaluator | no | yes | 8567 | 148 | 6.909510 | 17.3 | 33.17 |

## Profiler

The prebuilt-graph fused evaluator is now dominated by the fused descriptor kernel, force kernel, and the small MLP/autograd head. Across five profile passes on the 1024-config batched graph, top CUDA events were:

| op | device time us | count |
|---|---:|---:|
| `_element_density_descriptor_kernel` | 2747.9 | 5 |
| `_element_density_force_kernel` | 1084.9 | 5 |
| `aten::linear` / `aten::addmm` | 436.8 | 15 |
| `AddmmBackward0` / `aten::mm` | 382.7 | 15 |
| `SiluBackward0` / `aten::silu_backward` | 234.0 | 10 |

## Interpretation

The Stage 28 fused evaluator is now safely integrated as the default eligible element-density benchmark path. The major new finding is system-level: if graph construction and per-configuration execution are included, throughput collapses from 34.2M atoms/s to 8.6k atoms/s on the same 1024-config window. This is not a model-architecture limitation; it is an input-pipeline/runtime realization problem.

The next clean TECE/TACE renormalization step should therefore target batched graph construction, graph caching, or an MD-runtime integration path that keeps neighbor lists and edge buffers alive across steps. Further shrinking the scalar head or radial basis will not address this end-to-end bottleneck.
