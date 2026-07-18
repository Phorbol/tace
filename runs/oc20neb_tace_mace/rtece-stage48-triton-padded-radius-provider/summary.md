# Stage 48: Triton Padded Direct-Radius Provider

GPU setup: radial8h24 `rtece_element_density`, direct-active initial graph, DFT valid `:1024`, 59193 atoms, one V100, float32, `--force-mode auto`, forced invalid cache with `trajectory_skin_margin=0.002`, `trajectory_displacement_std=0.001`, 20 force steps and 10 update events.

| update backend | mode | atom-step/s | update total s | peak alloc MB | peak reserved MB | padded pair slots | padding overhead | directed edges | DFT F MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `torch_radius_nopbc_grouped` | update-only | 68,486,684 | 0.012020 | 241.3 | 346.0 | 7,750,656 | 2.095 | 825,584 | n/a |
| `torch_radius_nopbc_grouped_chunked` chunk256 | update-only | 48,274,213 | 0.020571 | 110.0 | 174.0 | 6,211,328 | 1.679 | 825,584 | n/a |
| `torch_radius_nopbc_ragged` | update-only | 39,868,611 | 0.025378 | 383.9 | 492.0 | 3,699,489 | 1.000 | 825,584 | n/a |
| `torch_radius_nopbc_triton_padded` | update-only | 141,308,212 | 0.004430 | 255.0 | 286.0 | 7,750,656 | 2.095 | 825,584 | n/a |
| `torch_radius_nopbc_grouped` | model+updates | 26,536,430 | 0.012534 | 263.1 | 406.0 | 7,750,656 | 2.095 | 825,584 | 33.18 |
| `torch_radius_nopbc_grouped_chunked` chunk256 | model+updates | 23,294,254 | 0.021821 | 130.5 | 214.0 | 6,211,328 | 1.679 | 825,584 | 33.18 |
| `torch_radius_nopbc_ragged` | model+updates | 21,688,771 | 0.025748 | 405.9 | 552.0 | 3,699,489 | 1.000 | 825,584 | 33.18 |
| `torch_radius_nopbc_triton_padded` | model+updates | 37,057,631 | 0.003897 | 276.8 | 346.0 | 7,750,656 | 2.095 | 825,584 | 33.18 |

Interpretation:

- This is the first positive lower-level graph-update provider result after Stage 47. It keeps the same direct-active semantics and emits the same 825,584 directed edges, but moves candidate filtering and edge writing into a Triton kernel.
- The Triton provider still scans the all-config padded candidate space, so it does not reduce `padded_pair_slots`. The gain comes from avoiding long-lived PyTorch distance, mask, and ragged-index tensors and writing active edges directly with an atomic counter.
- Update-only throughput improves about 2.1x over grouped and about 2.9x over chunk256. Model+updates improves to 37.1M atom-step/s, beating both grouped and chunk256 while preserving the same DFT force MAE.
- Peak allocation is still high because the current kernel preallocates a full `2 x padded_pair_slots` int64 edge buffer. The next provider target is a counted/two-pass or cell-list variant that keeps the Triton execution path but avoids full padded edge-buffer allocation.
