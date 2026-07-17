# Stage 33: Skin-Aware Graph Cache Validity

This stage turns the fixed rebuild interval from Stage 32 into a displacement/skin validity probe. The benchmark now records the maximum displacement since the last graph build, the half-skin threshold, rebuild steps, rebuild causes, and rebuild count. The rebuild criterion is the standard neighbor-list rule: rebuild when `max_displacement_since_rebuild > 0.5 * skin_margin`.

Implementation checks:

- `graph_cache_displacement_probe(reference_positions, positions, skin_margin)` computes max displacement and half-skin rebuild necessity.
- `--trajectory-skin-margin` enables skin-triggered rebuilds in trajectory replay mode.
- JSON records `trajectory_skin_margin`, `trajectory_skin_threshold`, `trajectory_rebuild_count`, `trajectory_rebuild_steps`, `trajectory_rebuild_causes`, and `trajectory_max_displacement_since_rebuild_a`.
- `test/test_rtece_scalar.py`: 42 passed after the change.

GPU probe setup: radial8/24x24 `rtece_element_density`, DFT valid `:256`, 12770 atoms, float32, one V100, `--force-mode auto`, `trajectory_displacement_std=0.001`. This is a small-window validity probe; absolute atom-step/s is not directly comparable to the 1024-config Stage-32 rows because the smaller batch underutilizes the GPU.

| skin margin A | half-skin threshold A | force steps | rebuild count | rebuild steps | atom-step/s | first-frame DFT F MAE |
|---:|---:|---:|---:|---|---:|---:|
| 0.000 | n/a | 200 | 0 | [] | 14880257 | 34.14 |
| 0.008 | 0.004 | 200 | 0 | [] | 14623702 | 34.14 |
| 0.004 | 0.002 | 100 | 0 | [] | 14750307 | 34.14 |
| 0.002 | 0.001 | 20 | 10 | [1,3,5,7,9,11,13,15,17,19] | 15839 | 34.14 |

Interpretation:

- The skin-validity criterion works as a first-class edge-state lifetime control: margins above the measured max displacement avoid rebuilds and preserve cached-model throughput; an overly small margin forces frequent graph rebuilds and collapses throughput.
- For this synthetic trajectory, max displacement since rebuild is about 0.00173 A without rebuilds. Skin margins of 0.004 A and 0.008 A are valid; 0.002 A is too small and rebuilds every two steps.
- The production target is not a fixed K. It is a skin/displacement policy that maximizes edge-state lifetime subject to neighbor validity. Stage 32 says K around 2000 is needed for >1e7 atom-step/s on the 1024-config window; Stage 33 says such a K must be justified by actual displacement staying below half the skin margin.
- The next implementation target is a neighbor-provider boundary that uses this validity signal and avoids full ASE rebuild when the graph is still valid.
