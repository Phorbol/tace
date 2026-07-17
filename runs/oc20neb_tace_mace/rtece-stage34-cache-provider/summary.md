# Stage 34: Graph-Cache Provider Boundary And Validity Timing

Stage 34 adds a reusable trajectory graph-cache provider boundary and separates O(N) cache-validity timing from fused model timing. This is the next system-renormalization step after Stage 33: the MD loop should query `needs_rebuild` from a cache/provider state, not hard-code a rebuild interval.

Implementation checks:

- `TrajectoryGraphCacheProvider(reference_positions, skin_margin)` owns reference positions, `needs_rebuild` checks, and rebuild bookkeeping.
- `--trajectory-validity-only` times only deterministic position generation plus provider validity checks; it does not run the model and does not rebuild graphs.
- Validity-only output sets `prediction_errors_available=false` and leaves MAE/RMSE fields as `null`, avoiding fake zero-error rows in Pareto tables.
- `test/test_rtece_scalar.py`: 45 passed after the change.

GPU setup: radial8/24x24 `rtece_element_density`, DFT valid `:1024`, 59193 atoms, float32, one V100, `--force-mode auto`, `trajectory_displacement_std=0.001`, `trajectory_skin_margin=0.004`, 2000 force steps/pass.

| mode | validity only | force steps | rebuild count | max displacement A | atom-step/s | seconds/pass | ms/step | peak alloc MB | DFT F MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| provider validity check | yes | 2000 | 0 | 0.001733 | 356832007 | 0.331770 | 0.166 | 24.5 | n/a |
| cached fused model + validity | no | 2000 | 0 | 0.001733 | 38889577 | 3.044158 | 1.522 | 69.8 | 33.17 |

Interpretation:

- The provider boundary makes graph validity an explicit stateful interface: `check(positions)` and `mark_rebuilt(positions, step, cause)`.
- The O(N) validity check is much cheaper than the fused model pass on this window, about 0.166 ms/step versus 1.522 ms/step. It is not the bottleneck, but it is still around 11% of the cached model step, so a production implementation should keep it device-side and avoid host synchronization.
- The skin margin is valid for this synthetic trajectory: max displacement is about 0.00173 A, below the half-skin threshold 0.002 A, so no rebuilds are triggered over 2000 steps.
- The next boundary is the graph update provider itself: when `needs_rebuild=True`, the current implementation still falls back to ASE graph reconstruction. The remaining gap to production throughput is therefore a fast neighbor update/rebuild path, not scalar descriptor semantics.
