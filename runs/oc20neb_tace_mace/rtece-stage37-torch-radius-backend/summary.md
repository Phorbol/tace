# Stage 37: Torch Radius Non-PBC Update Backend

Stage 37 adds and benchmarks `torch_radius_nopbc`, the first backend that actually recomputes `edge_index` without ASE. It rebuilds directed cutoff edges independently inside each batched configuration using torch tensor distance checks. It deliberately does not implement periodic boundary conditions, so it is a provider-boundary and performance probe, not a production-correct neighbor backend for the current PBC OC20NEB slabs.

Implementation checks:

- `--graph-update-backend {ase_neighborlist,cached_topology,torch_radius_nopbc}` selects the trajectory graph update backend.
- `torch_radius_nopbc_graph(template, positions, cutoff=...)` reuses `z` and `batch`, refreshes `pos`, and rebuilds per-config directed edges under the cutoff.
- `test/test_rtece_scalar.py`: 48 passed after the change.
- CPU smoke with forced skin invalidation used `torch_radius_nopbc`, recorded 4 graph updates, and spent 0.898 ms total in graph update timing on 2 configs / 100 atoms.

GPU setup: radial8/24x24 `rtece_element_density`, DFT valid `:1024`, 59193 atoms, float32, one V100, `--force-mode auto`, `trajectory_displacement_std=0.001`, `trajectory_skin_margin=0.002`. Stage35 and Stage36 rows are included as direct baselines.

| backend | mode | force steps | rebuild/update events | graph update total s | mean update s | seconds/pass | atom-step/s | DFT F MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `cached_topology` | update-only invalid cache | 20 | 10 | 0.0000446 | 0.00000446 | 0.003787 | 312592485 | n/a |
| `torch_radius_nopbc` | update-only invalid cache | 20 | 10 | 1.716636 | 0.171664 | 1.721270 | 687783 | n/a |
| `torch_radius_nopbc` | fused model + invalid updates | 20 | 10 | 1.729607 | 0.172961 | 1.762445 | 671714 | 33.17* |
| `ase_neighborlist` | update-only invalid cache | 20 | 10 | 53.943 | 5.394 | 53.948049 | 21944 | n/a |

`*` The trajectory benchmark still collects prediction errors at step 0 before the first forced rebuild, so this MAE is not evidence that the non-PBC rebuilt graph preserves model accuracy.

Topology comparison on the same `:1024` window:

| metric | value |
|---|---:|
| ASE edge entries | 1,221,178 |
| ASE unique edges | 1,193,050 |
| torch non-PBC unique edges | 825,510 |
| overlap with ASE unique edges | 825,510 |
| missing ASE/PBC unique edges | 367,540 |
| extra torch unique edges | 0 |
| raw ASE unique-edge recall | 0.6919 |
| active ASE direct-distance unique edges | 825,510 |
| active unique-edge recall | 1.0000 |
| inactive ASE direct-distance entries | 368,880 |
| duplicate ASE edge entries | 28,128 |
| mismatch configs | 1024 / 1024 |

Interpretation:

- This is the first measured non-ASE topology rebuild backend. It improves invalid-cache update-only throughput from 2.19e4 atom-step/s with ASE to 6.88e5 atom-step/s, about a 31x speedup.
- The backend is still far from the cached-topology upper bound: 0.172 s per update versus about 5 us for fixed-topology position refresh. The remaining cost is Python per-config looping and many small dense radius kernels, not scalar rTECE model evaluation.
- The raw topology comparison shows that `torch_radius_nopbc` recovers only 69.2% of ASE unique edges because the OC20NEB configs have full PBC. However, the current rTECE graph does not store periodic shift vectors, so model geometry uses direct coordinate differences. Under that implemented direct-distance semantics, `torch_radius_nopbc` recovers 100% of active unique edges inside the cutoff and drops 368,880 inactive ASE edge entries.
- This exposes a code/theory mismatch that is now a priority: either formalize the current rTECE endpoint as a direct-distance graph and remove ASE PBC overhead, or extend `RTECEGraph` and the fused evaluator to carry PBC shift/displacement state. A production PBC neighbor provider still needs a batched cell-list or MD-runtime neighbor interface; the current non-PBC torch backend is useful evidence, not the final backend.
