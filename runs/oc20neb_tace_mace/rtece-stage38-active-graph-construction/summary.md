# Stage 38: Direct Active Graph Construction Backend

Stage 38 formalizes the Stage-37 active-edge finding as a benchmark switch. The new `--graph-construction-backend torch_radius_nopbc` builds the initial and timed atom graphs with direct-distance cutoff edges, rather than ASE/PBC neighbor entries. It does not change the rTECE checkpoint, scalar descriptors, force mode, or model parameters.

Implementation checks:

- `benchmark_rtece_scalar.py` exposes `--graph-construction-backend {ase_neighborlist,torch_radius_nopbc}`.
- `atoms_to_torch_radius_nopbc_graph(...)` builds a single-configuration direct-distance graph.
- All benchmark graph construction paths now route through `build_atom_graph(...)`.
- `test/test_rtece_scalar.py`: 49 passed after the change.
- CPU smoke on 4 configs ran with `graph_construction_backend=torch_radius_nopbc` and wrote the backend into JSON.

GPU setup: radial8/24x24 `rtece_element_density`, DFT valid `:1024`, 59193 atoms, float32, one V100, `--force-mode auto`.

| graph construction backend | runtime mode | seconds/pass | atoms/s or atom-step/s | peak alloc MB | DFT F MAE | DFT F RMSE |
|---|---|---:|---:|---:|---:|---:|
| `ase_neighborlist` | prebuilt model pass | 0.001784 | 33186591 | 63.2 | 33.1687 | 109.3660 |
| `torch_radius_nopbc` | prebuilt model pass | 0.001416 | 41816583 | 57.2 | 33.1837 | 109.3663 |
| `ase_neighborlist` | rebuild graphs, then batch | 5.963333 | 9926 | 78.2 | 33.1687 | 109.3660 |
| `torch_radius_nopbc` | rebuild graphs, then batch | 0.429820 | 137716 | 60.1 | 33.1837 | 109.3663 |
| `ase_neighborlist` | cached trajectory, 2000 steps | 3.183696 | 37185080 | 69.8 | 33.1687 | 109.3660 |
| `torch_radius_nopbc` | cached trajectory, 2000 steps | 2.733743 | 43305459 | 63.7 | 33.1837 | 109.3663 |

Interpretation:

- Direct active graph construction is a real hardware-cost improvement under the current rTECE graph semantics. Prebuilt model throughput rises by about 1.26x, cached trajectory throughput by about 1.16x, and rebuild+batch throughput by about 13.9x.
- The force-error change is negligible for this benchmark: +0.015 meV/A force MAE and essentially unchanged RMSE. This is consistent with Stage37: the dropped ASE/PBC entries are inactive under the current direct-coordinate geometry, while duplicate ASE entries create only a tiny descriptor perturbation.
- This is not a claim that PBC physics is solved. It says the current rTECE implementation should stop paying for ASE/PBC entries that it cannot represent correctly without shift vectors.
- The next theory-clean choice is explicit: either freeze this endpoint as a direct-distance/local slab approximation and use direct active graphs by default for its Pareto rows, or extend `RTECEGraph` plus fused descriptor/force kernels with PBC shift/displacement state before claiming PBC-correct topology.
