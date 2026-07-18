# Stage 47: Ragged Exact-Pair Direct-Radius Provider

GPU setup: radial8h24 `rtece_element_density`, direct-active initial graph, DFT valid `:1024`, 59193 atoms, one V100, float32, `--force-mode auto`, forced invalid cache with `trajectory_skin_margin=0.002`, `trajectory_displacement_std=0.001`, 20 force steps and 10 update events.

| update backend | mode | atom-step/s | update total s | peak alloc MB | peak reserved MB | chunks | padded pair slots | padding overhead | directed edges | DFT F MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `torch_radius_nopbc_grouped` | update-only | 66,746,689 | 0.012180 | 241.3 | 346.0 | 1 | 7,750,656 | 2.095 | 825,584 | n/a |
| `torch_radius_nopbc_grouped_chunked` chunk256 | update-only | 46,507,195 | 0.021406 | 110.0 | 174.0 | 4 | 6,211,328 | 1.679 | 825,584 | n/a |
| `torch_radius_nopbc_ragged` | update-only | 39,074,531 | 0.025768 | 383.9 | 492.0 | 1 | 3,699,489 | 1.000 | 825,584 | n/a |
| `torch_radius_nopbc_grouped` | model+updates | 26,766,448 | 0.012598 | 263.1 | 406.0 | 1 | 7,750,656 | 2.095 | 825,584 | 33.18 |
| `torch_radius_nopbc_grouped_chunked` chunk256 | model+updates | 22,949,920 | 0.021976 | 130.5 | 214.0 | 4 | 6,211,328 | 1.679 | 825,584 | 33.18 |
| `torch_radius_nopbc_ragged` | model+updates | 20,805,343 | 0.025858 | 405.9 | 552.0 | 1 | 3,699,489 | 1.000 | 825,584 | 33.18 |

Interpretation:

- The ragged provider is semantically correct and removes all padded pair-slot overhead: it evaluates exactly 3,699,489 per-config dense pair slots and emits the same 825,584 directed active edges.
- It is still not Pareto-improving. In PyTorch, exact ragged pair generation materializes long integer tensors such as graph ids, pair offsets, source indices, and destination indices. That raises peak allocation to about 406MB for model+updates, worse than both grouped and chunk256.
- Throughput is also worse than chunk256 despite fewer pair slots: 20.8M versus 22.9M atom-step/s for model+updates. The saved distance work is outweighed by ragged-index construction and materialization overhead.
- This is an important negative result for the TECE system-renormalization route. It rules out a pure PyTorch ragged exact-pair provider as the next front point and strengthens the need for a fused lower-level provider that generates active edges or descriptor inputs without materializing long-lived ragged index tensors.
