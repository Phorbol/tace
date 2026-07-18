# Stage 42: Grouped Direct-Radius Update Backend

Stage 42 implemented and benchmarked `torch_radius_nopbc_grouped`, a grouped direct-radius update backend for the current direct-distance rTECE branch. It preserves the same direct active-edge semantics as `torch_radius_nopbc`, but replaces the Python loop over many per-config radius checks with one padded batched radius computation over `(num_configs, max_atoms, max_atoms)`.

Implementation gate:

- Added `torch_radius_nopbc_grouped_graph(...)` and exposed `--graph-update-backend torch_radius_nopbc_grouped`.
- Added tests that the grouped graph matches the existing loop backend, that the update backend rebuilds the same directed edges, and that CLI help exposes the new backend.
- Full rTECE test file after the change: 52 passed.

GPU setup: radial8h24 `rtece_element_density`, direct-active initial graph, DFT valid `:1024`, 59193 atoms, one V100, float32, `--force-mode auto`, forced invalid cache with `trajectory_skin_margin=0.002`, `trajectory_displacement_std=0.001`, 20 force steps and 10 update events.

| update backend | mode | update total s | mean update s | seconds/pass | atom-step/s | peak alloc MB | DFT F MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| `torch_radius_nopbc` | update-only | 1.772007 | 0.177201 | 1.776811 | 666283 | 55.7 | n/a |
| `torch_radius_nopbc_grouped` | update-only | 0.012035 | 0.001204 | 0.017411 | 67994130 | 241.3 | n/a |
| `torch_radius_nopbc` | model+updates | 1.719516 | 0.171952 | 1.752224 | 675633 | 77.6 | 33.18 |
| `torch_radius_nopbc_grouped` | model+updates | 0.012803 | 0.001280 | 0.043798 | 27029982 | 263.1 | 33.18 |

Stage-42 interpretation against TECE/TACE:

- This is a positive system-renormalization result. It does not change descriptors, learned parameters, graph semantics, or conservative force evaluation. It only changes the realization of the retained direct active-edge topology update.
- Grouping removes the dominant Python/per-config small-kernel overhead exposed in Stage 41. Update-only cost improves by about 147x, and invalid-cache model+updates throughput improves by about 40x.
- The grouped backend still does not reach the cached-topology upper bound from Stage 41: 27.0M atom-step/s versus about 43.6M atom-step/s for model+cached-topology invalid updates on the same window. The remaining gap is now dense padded radius work plus allocation/materialization overhead, not ASE or per-config Python dispatch.
- Memory rises from about 78MB to 263MB for model+updates because the grouped path materializes padded distance/mask tensors. This is acceptable as a provider prototype but not the final implementation.
- Next priority: turn this into a cleaner provider by avoiding padded all-pairs materialization, either with a fused cell-list/radius kernel or a grouped-by-size backend. The current result is strong enough to deprioritize the old loop backend and confirms that provider realization is a first-class TECE/TACE deployment axis.
