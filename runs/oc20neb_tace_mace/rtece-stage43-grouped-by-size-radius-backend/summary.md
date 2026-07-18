# Stage 43: Grouped-By-Size Direct-Radius Backend

Stage 43 tested the Stage-42 follow-up idea: reduce the dense padded all-pairs memory of `torch_radius_nopbc_grouped` by grouping configurations with the same atom count, then running a smaller batched radius calculation per size group. This preserves the direct-distance rTECE graph semantics and conservative force path, but changes provider realization.

Implementation gate:

- Added `torch_radius_nopbc_grouped_by_size_graph(...)` and exposed `--graph-update-backend torch_radius_nopbc_grouped_by_size`.
- Added tests that grouped-by-size matches the loop backend edges, that the backend factory works, and that CLI help exposes the backend.
- Full rTECE test file after the change: 53 passed.

GPU setup: radial8h24 `rtece_element_density`, direct-active initial graph, DFT valid `:1024`, 59193 atoms, one V100, float32, `--force-mode auto`, forced invalid cache with `trajectory_skin_margin=0.002`, `trajectory_displacement_std=0.001`, 20 force steps and 10 update events.

| update backend | mode | update total s | mean update s | seconds/pass | atom-step/s | peak alloc MB | DFT F MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| `torch_radius_nopbc_grouped` | update-only | 0.012194 | 0.001219 | 0.017505 | 67628193 | 241.3 | n/a |
| `torch_radius_nopbc_grouped_by_size` | update-only | 1.046924 | 0.104692 | 1.052414 | 1124900 | 61.1 | n/a |
| `torch_radius_nopbc_grouped` | model+updates | 0.012321 | 0.001232 | 0.045059 | 26273400 | 263.1 | 33.18 |
| `torch_radius_nopbc_grouped_by_size` | model+updates | 1.163704 | 0.116370 | 1.195055 | 990632 | 82.6 | 33.18 |

Stage-43 interpretation against TECE/TACE:

- This is a negative provider result. Grouping by atom count reduces peak allocation by about 3.2x for model+updates, but it gives up the main Stage-42 win: update cost rises from about 1.2 ms/update to 116 ms/update.
- The result is still useful theoretically because it separates two system-renormalization costs: padded dense all-pairs memory versus Python-level grouping/edge-materialization overhead. In this implementation, launch and edge-splitting overhead dominate the saved pair work.
- The grouped-by-size backend should not replace `torch_radius_nopbc_grouped` for the current direct-active deployment front. It is a diagnostic branch showing that lowering memory with Python control flow is the wrong priority.
- Next priority: a fused direct-radius/cell-list provider that avoids both all-config max padding and Python per-size edge splitting. The provider should produce `edge_index` or direct edge buffers in one low-level path, closer to the fused descriptor/force kernels already validated for scalar rTECE.
