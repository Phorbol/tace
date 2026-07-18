# Stage 44: Chunked Grouped Direct-Radius Backend

Stage 44 tested `torch_radius_nopbc_grouped_chunked`, a provider variant between Stage-42 all-config grouping and Stage-43 by-size grouping. It preserves the direct-distance active-edge semantics, keeps config order, and chunks consecutive configurations into padded radius batches. The default chunk size is 128 configs, which reduces peak padded tensor size without introducing by-size Python edge splitting.

Implementation gate:

- Added `torch_radius_nopbc_grouped_chunked_graph(...)` and exposed `--graph-update-backend torch_radius_nopbc_grouped_chunked`.
- Added tests that chunked grouped radius matches the loop backend, that the backend factory works, and that CLI help exposes the backend.
- Full rTECE test file after the change: 54 passed.

GPU setup: radial8h24 `rtece_element_density`, direct-active initial graph, DFT valid `:1024`, 59193 atoms, one V100, float32, `--force-mode auto`, forced invalid cache with `trajectory_skin_margin=0.002`, `trajectory_displacement_std=0.001`, 20 force steps and 10 update events.

| update backend | mode | update total s | mean update s | seconds/pass | atom-step/s | peak alloc MB | DFT F MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| `torch_radius_nopbc_grouped` | update-only | 0.012132 | 0.001213 | 0.017416 | 67977303 | 241.3 | n/a |
| `torch_radius_nopbc_grouped_chunked` | update-only | 0.033966 | 0.003397 | 0.037933 | 31208976 | 73.4 | n/a |
| `torch_radius_nopbc_grouped_by_size` | update-only | 1.040244 | 0.104024 | 1.045440 | 1132404 | 61.1 | n/a |
| `torch_radius_nopbc_grouped` | model+updates | 0.012259 | 0.001226 | 0.043711 | 27083689 | 263.1 | 33.18 |
| `torch_radius_nopbc_grouped_chunked` | model+updates | 0.035949 | 0.003595 | 0.065868 | 17973177 | 95.0 | 33.18 |
| `torch_radius_nopbc_grouped_by_size` | model+updates | 1.135829 | 0.113583 | 1.167079 | 1014379 | 82.6 | 33.18 |

Stage-44 interpretation against TECE/TACE:

- Chunked grouping is a useful provider Pareto point. It keeps the same rTECE direct active graph semantics and conservative scalar force path, but trades about 1.5x lower model+update throughput for about 2.8x lower peak allocation compared with all-config grouped radius.
- This resolves the Stage-43 ambiguity: reducing memory is possible without falling back to Python by-size edge splitting. The right intermediate path is coarse chunking or a fused provider, not fine Python grouping.
- The current direct-active invalid-cache provider front is now: `cached_topology` as a topology-reuse upper bound, `torch_radius_nopbc_grouped` for maximum update throughput, `torch_radius_nopbc_grouped_chunked` for lower memory, and `torch_radius_nopbc_grouped_by_size` as a diagnostic negative result.
- The next clean implementation target remains a fused direct-radius/cell-list provider. Stage 44 suggests that chunk granularity should be a tunable deployment parameter until that lower-level provider exists.
