# Stage 36: Cached-Topology Update Backend

Stage 36 adds the first non-ASE graph update backend candidate. The `cached_topology` backend reuses the existing `z`, `edge_index`, and `batch` tensors and only refreshes positions through `replay_graph_positions`. It is therefore an upper-bound and provider-interface test, not a real neighbor-list rebuild.

Implementation checks:

- `--graph-update-backend {ase_neighborlist,cached_topology}` selects the trajectory graph update backend.
- `make_graph_update_backend(...)` builds either the existing ASE backend or the cached-topology position-refresh backend.
- `test/test_rtece_scalar.py`: 47 passed after the change.
- CPU smoke with forced skin invalidation used `cached_topology`, recorded 4 graph updates, and spent 13.7 us total in graph update timing.

GPU setup: radial8/24x24 `rtece_element_density`, DFT valid `:1024`, 59193 atoms, float32, one V100, `--force-mode auto`, `trajectory_displacement_std=0.001`, `trajectory_skin_margin=0.002`. Stage35 ASE rows are included as the direct invalid-cache baseline.

| backend | mode | force steps | rebuilds | graph update total s | mean update s | seconds/pass | atom-step/s | DFT F MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `cached_topology` | update-only invalid cache | 20 | 10 | 0.0000446 | 0.00000446 | 0.003787 | 312592485 | n/a |
| `cached_topology` | fused model + invalid updates | 20 | 10 | 0.0000529 | 0.00000529 | 0.032912 | 35970261 | 33.17 |
| `cached_topology` | fused model + invalid updates | 2000 | 1039 | 0.005264 | 0.00000507 | 3.196984 | 37030530 | 33.17 |
| `ase_neighborlist` | update-only invalid cache | 20 | 10 | 53.943 | 5.394 | 53.948049 | 21944 | n/a |
| `ase_neighborlist` | fused model + invalid updates | 20 | 10 | 54.377 | 5.438 | 54.416800 | 21755 | 33.17 |

Interpretation:

- The provider abstraction is now useful: the same invalid-cache benchmark can swap update backends and report update timing separately from model timing.
- Cached-topology position refresh reduces mean update timing from about 5.39 s per ASE rebuild to about 5 us per refresh, an upper-bound difference of roughly six orders of magnitude.
- With forced invalidation every two steps, fused model throughput recovers from 2.18e4 atom-step/s with ASE updates to 3.60e7 atom-step/s with cached-topology updates. The 2000-step row remains 3.70e7 atom-step/s even with 1039 update events.
- This is not a deployable invalid-neighbor solution because topology is not recomputed. It is the TECE/TACE system-level renormalization limit where retained edge topology is assumed valid and only positions flow through the scalar fused evaluator.
- The next priority is a safe fast neighbor-provider backend, such as a cell-list or MD-runtime neighbor update path, that preserves the provider interface while actually updating topology when the skin criterion fails. More scalar architecture sweeps remain lower priority until this backend exists.
