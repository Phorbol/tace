# Stage 55: Cell-List Descriptor+Force Oracle Entry

Stage 55 turns the Stage52 cell-list descriptor oracle into a real rTECE inference force mode. This is still a PyTorch/CPU-oriented oracle, not the final Triton/NVIDIA-ops backend, but it is now callable through the formal model, workflow, and benchmark paths.

## What changed

- Added `RTECEScalarModel.forward_element_density_cell_list_packed_analytic_forces(...)`. The method ignores the input `edge_index`, constructs direct-active no-PBC pairs from batch-sorted positions with the same cell-list semantics as Stage52, accumulates packed `[rho, rho_z]` descriptors, and applies the same conservative descriptor-gradient chain rule as `forward_element_density_packed_analytic_forces(...)`.
- Added workflow dispatch for force mode `analytic_element_cell_list_descriptor_force`.
- Added benchmark CLI support for the same force mode, so existing rTECE checkpoints can run through the normal benchmark script.
- Extended `rtece_route_contract(...)` so this mode reports `cell_list_fused_descriptor_oracle`, `cell_list_analytic_descriptor_force`, direct-active no-PBC graph semantics, and `streaming_cell_candidates_oracle` edge lifetime.

## Verification

- TDD red check: the new tests failed before implementation because the model method and workflow force mode did not exist.
- Targeted Stage55 tests: `2 passed, 68 deselected, 1 warning`.
- Full rTECE test file: `70 passed, 1 warning`.

## CPU oracle smoke

Setup: existing `rtece-stage23-radial4-hidden16` element-density checkpoint, first 8 configs from `teacher_valid.extxyz`, CPU float64, prebuilt `torch_radius_nopbc` graph, one timed pass.

| mode | atoms/s | seconds/pass | DFT F MAE meV/A | DFT F RMSE meV/A | note |
|---|---:|---:|---:|---:|---|
| `analytic_element_cell_list_descriptor_force` | 4028.8 | 0.099286 | 52.099328 | 96.659877 | ignores input edge list; rebuilds direct-active pairs from cell candidates |
| `analytic_element_packed` | 2093.3 | 0.191085 | 52.099328 | 96.659877 | materialized edge-list packed descriptor reference |

The error metrics are identical on the smoke window, confirming semantic equivalence to the packed edge-list path for this scalar endpoint. The CPU timing is only an oracle signal and should not be used as the high-throughput Pareto frontier.

## Interpretation

This stage narrows the remaining backend target. The scalar element-density endpoint no longer requires a public materialized `edge_index` for descriptor/force semantics; a backend can stream cell candidates, accumulate descriptors, then apply the descriptor-gradient force rule. The next useful implementation is a GPU fused cell-list descriptor+force kernel, not more hidden-size sweeps or another PyTorch graph-update variant.
