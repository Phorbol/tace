# Stage 35: ASE Graph Update Backend Timing

Stage 35 adds a `GraphUpdateBackend` boundary and times graph update/rebuild separately from validity checks and fused model evaluation. The current backend is still `ase_neighborlist`, so this stage quantifies the remaining deployment gap rather than solving it.

Implementation checks:

- `GraphUpdateBackend(name, rebuild_fn)` records rebuild count, per-rebuild times, and total rebuild time.
- `--trajectory-update-only` runs position generation, cache validity checks, and provider graph updates/rebuilds without model evaluation.
- Normal trajectory replay now also routes rebuilds through the backend and records the same update timing fields.
- `test/test_rtece_scalar.py`: 46 passed after the change.

GPU setup: radial8/24x24 `rtece_element_density`, DFT valid `:1024`, 59193 atoms, float32, one V100, `--force-mode auto`, `trajectory_displacement_std=0.001`.

| mode | skin margin A | force steps | rebuilds | graph update total s | mean update s | seconds/pass | atom-step/s | DFT F MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| update-only valid cache | 0.004 | 2000 | 0 | 0.000 | n/a | 0.325606 | 363586905 | n/a |
| update-only invalid cache | 0.002 | 20 | 10 | 53.943 | 5.394 | 53.948049 | 21944 | n/a |
| fused model + invalid updates | 0.002 | 20 | 10 | 54.377 | 5.438 | 54.416800 | 21755 | 33.17 |

Interpretation:

- When the cache remains valid, update-only timing is essentially the Stage-34 validity cost: about 0.326 s for 2000 steps on 59193 atoms.
- When the cache is invalid every two steps, the ASE update backend dominates end-to-end cost. Ten graph rebuilds take 53.94-54.38 s, about 5.4 s per 1024-config rebuild.
- Adding 20 fused rTECE force evaluations on top of those rebuilds increases total time by only about 0.47 s. Therefore invalid-graph execution is graph-update-bound, not model-bound.
- This is the cleanest current deployment bottleneck: to make the scalar rTECE Pareto endpoint meaningful end-to-end, the provider backend must be replaced or accelerated. Further scalar architecture sweeps will not fix invalid-cache throughput.
