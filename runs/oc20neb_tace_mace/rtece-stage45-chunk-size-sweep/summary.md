# Stage 45: Chunk-Size Sweep For Chunked Direct-Radius Backend

GPU setup: radial8h24 `rtece_element_density`, direct-active initial graph, DFT valid `:1024`, 59193 atoms, one V100, float32, `--force-mode auto`, forced invalid cache with `trajectory_skin_margin=0.002`, `trajectory_displacement_std=0.001`, 20 force steps and 10 update events.

| update backend | chunk configs | mode | update total s | mean update s | seconds/pass | atom-step/s | peak alloc MB | peak reserved MB | DFT F MAE |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `torch_radius_nopbc_grouped` | all | update-only | 0.011970 | 0.001197 | 0.017392 | 68,070,707 | 241.3 | 346.0 | n/a |
| `torch_radius_nopbc_grouped_chunked` | 64 | update-only | 0.069001 | 0.006900 | 0.073236 | 16,165,038 | 61.3 | 106.0 | n/a |
| `torch_radius_nopbc_grouped_chunked` | 128 | update-only | 0.035014 | 0.003501 | 0.039086 | 30,288,403 | 73.4 | 124.0 | n/a |
| `torch_radius_nopbc_grouped_chunked` | 256 | update-only | 0.021553 | 0.002155 | 0.025552 | 46,331,185 | 110.0 | 174.0 | n/a |
| `torch_radius_nopbc_grouped_chunked` | 512 | update-only | 0.015826 | 0.001583 | 0.019991 | 59,220,668 | 174.8 | 230.0 | n/a |
| `torch_radius_nopbc_grouped` | all | model+updates | 0.012118 | 0.001212 | 0.043540 | 27,189,963 | 263.1 | 406.0 | 33.18 |
| `torch_radius_nopbc_grouped_chunked` | 64 | model+updates | 0.068641 | 0.006864 | 0.099996 | 11,839,051 | 83.0 | 126.0 | 33.18 |
| `torch_radius_nopbc_grouped_chunked` | 128 | model+updates | 0.037028 | 0.003703 | 0.066623 | 17,769,557 | 95.0 | 144.0 | 33.18 |
| `torch_radius_nopbc_grouped_chunked` | 256 | model+updates | 0.021643 | 0.002164 | 0.051295 | 23,079,365 | 130.5 | 214.0 | 33.18 |
| `torch_radius_nopbc_grouped_chunked` | 512 | model+updates | 0.016391 | 0.001639 | 0.046616 | 25,395,950 | 197.7 | 288.0 | 33.18 |

Interpretation:

- Chunk size is a real provider Pareto axis. Increasing `chunk_configs` monotonically increases update throughput in this run, while peak allocation rises from about 83MB at chunk64 model+updates to about 198MB at chunk512 and 263MB for all-config grouped radius.
- The force MAE is unchanged because graph semantics, scalar descriptor, checkpoint, and conservative fused force path are unchanged. This isolates graph-update realization from model projection error.
- Chunk64 already stays above the 1e7 atom-step/s target in the forced-invalid model+update benchmark with low memory. Chunk256 is the best balanced point here: about 85% of all-config grouped model+update throughput with about half the peak allocation.
- Until a fused direct-radius/cell-list provider exists, `--graph-update-chunk-configs` should be treated as a deployment parameter in the TECE/TACE Pareto table, not as a hidden benchmark constant.
