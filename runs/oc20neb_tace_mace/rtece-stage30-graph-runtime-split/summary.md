# Stage 30: Graph Runtime Cost Split

Stage 30 split the Stage 29 end-to-end bottleneck into three runtime modes for the same radial8/24x24 element-density checkpoint. All rows use DFT valid `:1024`, 59193 atoms, float32, one V100, and `--force-mode auto`, which resolves to `analytic_element_triton_descriptor_force`.

## Results

| runtime mode | prebuilt graph | graph construction timed | batched model pass | atoms/s | configs/s | seconds/pass | DFT F MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| prebuilt batched graph | yes | no | yes | 36564104 | 632535 | 0.001619 | 33.17 |
| rebuild graphs, then batch | no | yes | yes | 9879 | 171 | 5.992034 | 33.17 |
| rebuild graph per config | no | yes | no | 8599 | 149 | 6.883463 | 33.17 |

## Interpretation

This stage separates two possible explanations for the Stage 29 endpoint collapse. If per-configuration model dispatch were dominant, rebuilding all graphs and then collating into one batched model pass would have recovered a large fraction of the 36.6M atoms/s prebuilt throughput. It did not: the batched rebuild path is only about 15% faster than the per-config rebuild path, and both are around four orders of magnitude slower than the prebuilt batched graph.

The dominant end-to-end bottleneck is therefore ASE neighbor-list and graph construction, not the scalar rTECE force pass and not primarily per-config Python model calls. In TECE terms, the next system-level renormalization is to keep the local graph/edge buffer as a persistent coarse-grained state across MD steps, with skin/cache updates, instead of reconstructing it from atom positions through ASE every step.

The next clean implementation target is a graph-cache benchmark: build an initial neighbor graph once, reuse it for repeated force passes, and separately measure an amortized rebuild/update interval. A later runtime target should bypass ASE neighbor-list construction entirely and connect the fused descriptor/force path to an MD-style neighbor-list provider.
