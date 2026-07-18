# Stage 50: Triton Counted Batch-Size Stress

Question: does the Stage49 counted provider turn its lower edge-buffer memory into a better large-batch throughput point, or is it mainly a memory-headroom fallback behind the Stage48 padded provider?

Setup:

- Same radial8h24 `rtece_element_density` checkpoint and direct-active graph semantics as Stage49.
- Update-only trajectory replay, one V100, float32, 20 replay steps and 19 invalid-cache rebuilds.
- Compared only `torch_radius_nopbc_triton_padded` and `torch_radius_nopbc_triton_counted`.
- The OC20NEB valid extxyz contains 10000 configs, so 10000 is the real-data upper bound for this stress test.

| limit configs | backend | atoms | atom-step/s | seconds/pass | update enqueue s | peak alloc MB | peak reserved MB | directed edges | padded pair slots | counted/padded throughput | counted/padded alloc |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | `triton_padded` | 59,193 | 155,998,009 | 0.007589 | 0.003706 | 255.0 | 286.0 | 825,584 | 7,750,656 | n/a | n/a |
| 1024 | `triton_counted` | 59,193 | 146,506,776 | 0.008081 | 0.003414 | 43.7 | 74.0 | 825,584 | 7,750,656 | 0.939 | 0.171 |
| 2048 | `triton_padded` | 119,271 | 180,892,419 | 0.013187 | 0.008160 | 510.1 | 582.0 | 1,634,364 | 15,501,312 | n/a | n/a |
| 2048 | `triton_counted` | 119,271 | 172,418,404 | 0.013835 | 0.006429 | 86.7 | 158.0 | 1,634,364 | 15,501,312 | 0.953 | 0.170 |
| 4096 | `triton_padded` | 250,355 | 257,656,655 | 0.019433 | 0.014725 | 1,327.7 | 1,466.0 | 3,459,786 | 40,960,000 | n/a | n/a |
| 4096 | `triton_counted` | 250,355 | 207,028,037 | 0.024186 | 0.010465 | 181.3 | 322.0 | 3,459,786 | 40,960,000 | 0.804 | 0.137 |
| 8192 | `triton_padded` | 525,769 | 325,610,466 | 0.032294 | 0.026098 | 2,658.2 | 2,926.0 | 7,303,632 | 81,920,000 | n/a | n/a |
| 8192 | `triton_counted` | 525,769 | 249,657,496 | 0.042119 | 0.015261 | 382.2 | 650.0 | 7,303,632 | 81,920,000 | 0.767 | 0.144 |
| 10000 | `triton_padded` | 646,473 | 341,899,920 | 0.037817 | 0.031106 | 3,247.3 | 3,582.0 | 9,006,668 | 100,000,000 | n/a | n/a |
| 10000 | `triton_counted` | 646,473 | 258,546,783 | 0.050008 | 0.017702 | 471.3 | 806.0 | 9,006,668 | 100,000,000 | 0.756 | 0.145 |

Interpretation:

- Counted remains a strong memory point, not the large-batch throughput winner. Across 1024 to 10000 configs it uses only 13.7-17.1% of padded peak allocated memory. At 10000 configs, counted uses 471 MB allocated vs padded 3247 MB.
- Padded remains the real-data throughput front through the dataset limit. Its throughput rises to 342M atom-step/s at 10000 configs, while counted reaches 259M atom-step/s.
- The `graph_update_total_s` field is CPU-side enqueue timing and is not sufficient for GPU-kernel comparison. `seconds_per_pass` is the end-to-end update-only timing and shows counted's second pass costs more at large batch despite lower edge-buffer memory.
- Stage49's Pareto statement should be narrowed: counted is the memory-constrained Triton provider, while padded is the peak-throughput provider until memory pressure becomes binding.
- Next priority should not be another two-pass provider. For peak throughput, the clean TECE/TACE next step is a cell-list or fused descriptor provider that removes the remaining padded candidate scan and avoids the extra counted pass. For memory-bound deployment, a synthetic tiling/OOM benchmark can quantify when counted becomes necessary beyond the real 10000-config validation set.

Verification:

- Slurm job `679176` produced 1024/2048/4096/8192 outputs with status 0.
- Slurm job `679179` produced 10000 outputs with status 0.
