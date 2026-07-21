# Stage178 Benchmark Analysis

This is a stage-level evidence summary, not a final Pareto-front claim.

Rattle-relax policy: `continuous_rmsd_after_relax_not_binary_gate`.

## Stage178 Rows

| variant | params | F RMSE | E RMSE | F max | E max | rel-img RMSE | barrier RMSE | atoms/s 256 | atoms/s 1024 | peak alloc 1024 MB | rattle RMSD A | rattle max F eV/A |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stage178_l0_local_species | 20217 | 116.428 | 116.118 | 1704.400 | 300.837 | 12.302 | 16.274 | 1317471 | 1790966 | 2986.250 | 0.234 | 7.996 |
| stage178_l1_atomic_cross | 21079 | 121.981 | 150.241 | 4530.529 | 322.693 | 17.055 | 31.955 | 1223215 | 1596562 | 3168.874 | 0.305 | 0.214 |
| stage178_t3_minimal_cavity_direct | 21271 | 121.277 | 165.541 | 4502.970 | 284.876 | 16.818 | 32.106 | 638094 | 782853 | 3918.489 | 0.300 | 0.188 |

## Training

| variant | steps | best step | best valid loss |
|---|---:|---:|---:|
| stage178_l0_local_species | 19840 | 3840 | 0.157997 |
| stage178_l1_atomic_cross | 16064 | 64 | 0.171868 |
| stage178_t3_minimal_cavity_direct | 16064 | 64 | 0.175106 |

## Prior Stage177 Context

| candidate | family | F RMSE | E RMSE | F max | E max | atoms/s | note |
|---|---|---:|---:|---:|---:|---:|---|
| stage176_local_l0_rank3 | rtece | 141.997 | 294.931 | 5033.161 | 665.008 | 3988402 | atom_count_scaling_found_memory_missing |
| stage157_direct_b32_rel0p25_mixed2048 | rtece | 94.159 | 52.538 | 2092.961 | 204.986 | 500402 | prebuilt_graph_model_only |
| nep4_mixed_smoke | nep | 143.037 | 23.846 | 2867.353 | 81.018 | 103227 | community_engine_wall_time |
| deepmd_dpa_like_mixed_smoke | deepmd | 109.859 | 325.281 | 4429.181 | 867.313 | 10917 | community_engine_wall_time |

## Interpretation

- Stage178 tests front-loaded representation capacity, not final-head widening.
- Rattle-relax is interpreted as continuous relax RMSD and force-tail evidence; legacy boolean flags are retained only as provenance.
- The L0 local/species front is the current Stage178 best by DFT F RMSE, E RMSE, F max, and throughput.
- The L1/T3 increments improve the legacy rattle-relax flag and force MAE, but worsen E RMSE, F RMSE, and force max tails in this training recipe.
- Stage177 NEP/DPA rows remain smoke baselines under non-matching throughput protocols, so Stage178 cannot claim superiority over production NEP/DPA yet.

## Next Actions

- Replace boolean rattle gating in future reports with continuous RMSD/force-tail thresholds selected per deployment task.
- Run matched NEP/DPA1-0-layer training and throughput on the same data, same hardware scope, and same physical tests.
- Use Stage178 evidence to prioritize renormalized initialization or teacher residual projection before adding more edge paths.
- Promote peak_allocated_mb and peak_reserved_mb into the unified Pareto audit schema.
