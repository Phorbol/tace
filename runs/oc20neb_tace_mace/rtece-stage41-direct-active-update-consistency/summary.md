# Stage 41: Direct-Active Update-Backend Consistency

Stage 41 checked whether the Stage-37 update-backend conclusion still holds when the initial graph construction also uses the Stage-38/39 direct-active semantics. The checkpoint, scalar descriptor, force mode, validation window, and trajectory displacement match the previous radial8h24 trajectory probes; the key change is `--graph-construction-backend torch_radius_nopbc` for every row.

GPU setup: radial8h24 `rtece_element_density`, DFT valid `:1024`, 59193 atoms, one V100, float32, `--force-mode auto` resolving to `analytic_element_triton_descriptor_force`, `trajectory_displacement_std=0.001`.

| graph backend | update backend | mode | force steps | update events | update total s | seconds/pass | atom-step/s | DFT F MAE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `torch_radius_nopbc` | default/no rebuild | valid cached trajectory, skin 0.004 | 2000 | 0 | 0.000000 | 2.674001 | 44272977 | 33.18 |
| `torch_radius_nopbc` | `cached_topology` | invalid update-only, skin 0.002 | 20 | 10 | 0.000046 | 0.003753 | 315412557 | n/a |
| `torch_radius_nopbc` | `cached_topology` | invalid model+updates, skin 0.002 | 20 | 10 | 0.000052 | 0.027125 | 43645228 | 33.18 |
| `torch_radius_nopbc` | `torch_radius_nopbc` | invalid update-only, skin 0.002 | 20 | 10 | 1.723663 | 1.728869 | 684760 | n/a |
| `torch_radius_nopbc` | `torch_radius_nopbc` | invalid model+updates, skin 0.002 | 20 | 10 | 1.729914 | 1.762232 | 671796 | 33.18 |

Stage-41 interpretation against TECE/TACE:

- Direct-active initial graph semantics are consistent with the Stage-36/37 provider conclusion. Cached topology remains an upper bound near the fused model runtime, while real `torch_radius_nopbc` topology rebuild is still two orders of magnitude too slow for end-to-end high-throughput invalid-cache execution.
- The direct-active cached trajectory improves over the earlier ASE/PBC initial-graph trajectory because it carries fewer active edges and less memory, but it does not solve topology update when the skin validity criterion fails.
- This strengthens the current route: the clean rTECE Pareto front is now model-semantics complete for the direct-distance branch, and the limiting missing component is a batched/fused neighbor provider or MD-runtime neighbor-list interface.
- Next priority should be implementation work on a batched direct-active provider/cell-list boundary, not more radial/head sweeps. The separate PBC-correct branch remains valid but larger: it requires periodic shift/displacement state in `RTECEGraph` and the fused descriptor/force kernels.
