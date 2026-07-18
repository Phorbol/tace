# Stage 49: Triton Counted Direct-Radius Provider

Question: can the Stage48 Triton provider keep its high-throughput lower-level execution path while removing the full `2 x padded_pair_slots` int64 edge-buffer allocation?

Implementation:

- Added a count-only Triton pass over the same padded candidate slots.
- Reused the padded writer kernel with an explicit output stride so the second pass writes into an exact `2 x num_edges` edge buffer.
- Exposed `torch_radius_nopbc_triton_counted` through `--graph-update-backend` and the graph update backend factory.
- Added CPU fallback tests, backend factory tests, and CLI exposure tests.

GPU setup: radial8h24 `rtece_element_density`, direct-active initial graph, DFT valid `:1024`, 59193 atoms, one V100, float32, `--force-mode auto`, forced invalid cache with `trajectory_skin_margin=0.002`, `trajectory_displacement_std=0.001`, 20 force steps and 10 update events.

| update backend | mode | atom-step/s | seconds/pass | update total s | peak alloc MB | peak reserved MB | padded pair slots | padding overhead | directed edges | DFT F MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `torch_radius_nopbc_grouped` | update-only | 66,748,236 | 0.017736 | 0.012397 | 241.3 | 346.0 | 7,750,656 | 2.095 | 825,584 | n/a |
| `torch_radius_nopbc_grouped_chunked` chunk256 | update-only | 47,059,209 | 0.025157 | 0.021184 | 110.0 | 174.0 | 6,211,328 | 1.679 | 825,584 | n/a |
| `torch_radius_nopbc_ragged` | update-only | 39,427,477 | 0.030026 | 0.025690 | 383.9 | 492.0 | 3,699,489 | 1.000 | 825,584 | n/a |
| `torch_radius_nopbc_triton_padded` | update-only | 154,149,942 | 0.007680 | 0.003715 | 255.0 | 286.0 | 7,750,656 | 2.095 | 825,584 | n/a |
| `torch_radius_nopbc_triton_counted` | update-only | 146,335,457 | 0.008090 | 0.003444 | 43.7 | 74.0 | 7,750,656 | 2.095 | 825,584 | n/a |
| `torch_radius_nopbc_grouped` | model+updates | 27,909,688 | 0.042418 | 0.012639 | 263.1 | 406.0 | 7,750,656 | 2.095 | 825,584 | 33.18 |
| `torch_radius_nopbc_grouped_chunked` chunk256 | model+updates | 23,534,235 | 0.050304 | 0.022151 | 130.5 | 214.0 | 6,211,328 | 1.679 | 825,584 | 33.18 |
| `torch_radius_nopbc_ragged` | model+updates | 21,476,197 | 0.055124 | 0.026212 | 405.9 | 552.0 | 3,699,489 | 1.000 | 825,584 | 33.18 |
| `torch_radius_nopbc_triton_padded` | model+updates | 37,468,188 | 0.031596 | 0.003911 | 276.8 | 346.0 | 7,750,656 | 2.095 | 825,584 | 33.18 |
| `torch_radius_nopbc_triton_counted` | model+updates | 36,141,905 | 0.032756 | 0.003524 | 77.2 | 126.0 | 7,750,656 | 2.095 | 825,584 | 33.18 |

Interpretation:

- Counted is a positive memory-renormalization result. It preserves direct-active graph semantics and all model outputs while cutting peak allocated memory from 255.0 MB to 43.7 MB in update-only and from 276.8 MB to 77.2 MB in model+updates.
- The speed cost is small: counted keeps 94.9% of Triton padded update-only atom-step/s and 96.5% of model+updates atom-step/s.
- The result separates candidate-work cost from edge-buffer lifetime cost. Stage48 proved that PyTorch materialization was the main throughput bottleneck; Stage49 shows the remaining padded edge-buffer allocation can be removed without falling back to Torch-style overhead.
- The new Pareto front for invalid-cache direct-active updates is: `torch_radius_nopbc_triton_padded` when peak throughput is the only objective, `torch_radius_nopbc_triton_counted` when memory headroom or larger batch/system size matters.
- Next priority: test whether counted's exact edge buffer allows larger config batches or larger atom counts before OOM, then move to a cell-list or fused descriptor provider that removes the remaining padded candidate scan itself.

Verification:

- `python -m pytest test/test_rtece_scalar.py -q`: 59 passed, 1 warning.
- Slurm job `679167` produced all 10 Stage49 benchmark JSON outputs.
