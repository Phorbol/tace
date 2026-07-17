# Stage 32: Synthetic Trajectory Replay

This stage turns the Stage-31 cached-graph estimate into an MD-style measured benchmark. The benchmark keeps the same radial8/24x24 `rtece_element_density` checkpoint, DFT valid `:1024` prefix, 59193 atoms, float32, one V100, and `--force-mode auto`, resolving to `analytic_element_triton_descriptor_force`. The new trajectory mode keeps graph topology and edge buffers persistent, updates positions with a deterministic small displacement, and optionally rebuilds the batched ASE graph every `K` force steps.

In trajectory mode, `atoms_per_second` is atom-step/s: `atoms * force_steps_per_pass / seconds_per_pass`.

| mode | force steps/pass | graph rebuild interval K | atom-step/s | seconds/pass | peak alloc MB | DFT F MAE at first frame |
|---|---:|---:|---:|---:|---:|---:|
| cached trajectory | 2000 | 0 | 39909204 | 2.966383 | 69.1 | 33.17 |
| trajectory + rebuild | 2001 | 2000 | 13329198 | 8.886145 | 105.0 | 33.17 |
| trajectory + rebuild | 1001 | 1000 | 7906961 | 7.493674 | 105.0 | 33.17 |
| trajectory + rebuild | 1001 | 500 | 4444052 | 13.332921 | 125.2 | 33.17 |
| trajectory + rebuild | 501 | 100 | 977274 | 30.345312 | 125.2 | 33.17 |

Derived cost split:

- Cached fused model force step: about `2.966383 / 2000 = 0.001483 s` for 59193 atoms, or about 39.9M atom-step/s.
- One full ASE graph rebuild for this 1024-config window: about 5.92-6.01 s, inferred from the K=2000/1000/500/100 rows.
- K=2000 crosses the 1e7 atom-step/s threshold at 13.33M; K=1000 does not, at 7.91M.

Interpretation:

- The Stage-31 amortized model is validated by a direct trajectory-style benchmark.
- The current rTECE scalar force evaluator is fast enough for the target class only if graph topology/edge buffers live for about 2000 force steps, or if the rebuild/update provider is made substantially faster than ASE neighbor-list reconstruction.
- This does not yet prove a production MD engine will safely use K=2000; it proves the quantitative target for a neighbor-skin/device-provider layer.
- The next highest-priority work is a displacement-aware neighbor-cache validity/update path, not another radial/head architecture sweep.
