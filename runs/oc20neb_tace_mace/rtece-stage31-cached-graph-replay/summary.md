# Stage 31: Cached Graph Replay

This stage tests whether the Stage-30 bottleneck is the fused rTECE model pass or the lifetime of graph/edge buffers. The model and checkpoint are unchanged: `rtece_element_density`, `num_radial=8`, hidden `24x24`, `--force-mode auto`, one V100, DFT valid `:1024`, 1024 configs and 59193 atoms. `auto` resolves to `analytic_element_triton_descriptor_force`.

| runtime mode | prebuilt graph | graph construction timed | replay cached graph | atoms/s | configs/s | seconds/pass | peak alloc MB | DFT F MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prebuilt batched graph | yes | no | no | 36790610 | 636453 | 0.001609 | 63.2 | 33.17 |
| cached graph replay | no | no | yes | 36793984 | 636512 | 0.001609 | 64.6 | 33.17 |
| rebuild graphs, then batch | no | yes | no | 9885 | 171 | 5.987913 | 78.2 | 33.17 |

The replay mode keeps `z`, `edge_index`, and `batch` from one template graph, refreshes positions, then runs one batched fused conservative force pass. It is effectively identical to the prebuilt-graph throughput and force error. Rebuilding all graphs and then batching the model pass remains about 3700x slower.

Amortized throughput if a cached neighbor graph is rebuilt every `K` force steps, using `seconds_per_step = 0.001608768 + 5.987913 / K`:

| rebuild interval K | amortized atoms/s |
|---:|---:|
| 1 | 9883 |
| 10 | 98589 |
| 100 | 962677 |
| 1000 | 7791955 |
| 2000 | 12860426 |
| 5000 | 21092515 |
| 10000 | 26813771 |

Interpretation:

- The rTECE force pass itself is already in the intended high-throughput regime for this benchmark window.
- End-to-end throughput depends on making graph topology and edge buffers persistent state, not on further head/radial tuning.
- To cross 1e7 atoms-step/s under this measured cost model, the current ASE graph rebuild cost would need to be amortized over roughly 2000 force steps, or replaced by a much faster MD-runtime neighbor provider.
- The next priority is therefore graph lifetime and device-side neighbor integration, not another scalar architecture sweep.
