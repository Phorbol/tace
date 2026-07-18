# Stage 46: Direct-Radius Provider Work Metadata

GPU setup: radial8h24 `rtece_element_density`, direct-active initial graph, DFT valid `:1024`, 59193 atoms, one V100, float32, `--force-mode auto`, forced invalid cache with `trajectory_skin_margin=0.002`, `trajectory_displacement_std=0.001`, 20 force steps and 10 update events.

| update backend | chunk configs | mode | atom-step/s | update total s | peak alloc MB | chunks | padded pair slots | padding overhead | directed edges | DFT F MAE |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `torch_radius_nopbc_grouped` | all | update-only | 68,543,495 | 0.011981 | 241.3 | 1 | 7,750,656 | 2.095 | 825,584 | n/a |
| `torch_radius_nopbc_grouped_chunked` | 64 | update-only | 16,279,118 | 0.068561 | 61.3 | 16 | 4,956,992 | 1.340 | 825,584 | n/a |
| `torch_radius_nopbc_grouped_chunked` | 128 | update-only | 29,808,822 | 0.035617 | 73.4 | 8 | 5,365,888 | 1.450 | 825,584 | n/a |
| `torch_radius_nopbc_grouped_chunked` | 256 | update-only | 46,420,113 | 0.021265 | 110.0 | 4 | 6,211,328 | 1.679 | 825,584 | n/a |
| `torch_radius_nopbc_grouped_chunked` | 512 | update-only | 59,142,031 | 0.015863 | 174.8 | 2 | 7,574,528 | 2.047 | 825,584 | n/a |
| `torch_radius_nopbc_grouped` | all | model+updates | 27,338,063 | 0.012368 | 263.1 | 1 | 7,750,656 | 2.095 | 825,584 | 33.18 |
| `torch_radius_nopbc_grouped_chunked` | 64 | model+updates | 12,175,411 | 0.067447 | 83.0 | 16 | 4,956,992 | 1.340 | 825,584 | 33.18 |
| `torch_radius_nopbc_grouped_chunked` | 128 | model+updates | 17,863,813 | 0.035686 | 95.0 | 8 | 5,365,888 | 1.450 | 825,584 | 33.18 |
| `torch_radius_nopbc_grouped_chunked` | 256 | model+updates | 22,841,929 | 0.022921 | 130.5 | 4 | 6,211,328 | 1.679 | 825,584 | 33.18 |
| `torch_radius_nopbc_grouped_chunked` | 512 | model+updates | 24,931,106 | 0.017448 | 197.7 | 2 | 7,574,528 | 2.047 | 825,584 | 33.18 |

Static provider metadata for this window: `num_configs=1024`, `num_atoms=59193`, `max_atoms_per_config=87`, `exact_pair_slots=3699489`, and `num_directed_edges=825584`.

Interpretation:

- The provider tradeoff is now quantified. All-config grouped radius is fastest because it uses one large padded batch, but it evaluates about 2.10x the exact per-config pair slots and uses the most memory.
- Chunk64 evaluates the fewest padded pair slots, about 1.34x exact, but pays 16 chunk launches/materializations and is therefore much slower despite lower memory.
- Chunk256 remains the practical balanced point in the current PyTorch provider: about 22.8M atom-step/s for model+updates at 130.5MB peak allocation, versus 27.3M atom-step/s and 263.1MB for all-config grouping.
- The fused/cell-list provider target is now concrete: approach the exact 3.70M pair-slot or 0.826M edge scale while keeping launch count close to one and avoiding long-lived padded distance/mask tensors.
