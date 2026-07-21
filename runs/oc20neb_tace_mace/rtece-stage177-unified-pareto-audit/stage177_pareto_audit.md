# Stage177 Unified Pareto Audit

This is an evidence audit, not a final Pareto-front claim.

| candidate | family | rankable | F RMSE | E RMSE | F max | E max | rel-img RMSE | barrier RMSE | atoms/s | benchmark | physical | throughput |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| stage176_local_l0_rank3 | rtece | False | 141.997 | 294.931 | 5033.161 | 665.008 | 16.465 | 37.369 | 3988401.606 | found | found | atom_count_scaling_found_memory_missing |
| stage157_direct_b32_rel0p25_mixed2048 | rtece | False | 94.159 | 52.538 | 2092.961 | 204.986 | 7.104 | 11.403 | 500402.226 | found | missing_or_not_in_stage157_summary | prebuilt_graph_model_only |
| nep4_mixed_smoke | nep | True | 143.037 | 23.846 | 2867.353 | 81.018 | 10.068 | 19.344 | 103226.501 | found | found | community_engine_wall_time |
| deepmd_dpa_like_mixed_smoke | deepmd | True | 109.859 | 325.281 | 4429.181 | 867.313 | 17.009 | 39.474 | 10916.840 | found | found | community_engine_wall_time |

## Caveats

- `non_matching_throughput_protocols`
- `rankable_now_requires_matching_data_metric_physics_and_hardware_scope`
- `physical_triage_not_complete_for_all_rtece_rows`

## Next Required Actions

- `stage176_atom_count_throughput_scaling` (P0): TECE design-space ranking requires hardware cost across atom counts and memory; current Stage176 CUDA memory fields are missing.
- `stage178_representation_upgrade_after_local_l0_endpoint` (P0): Stage176 proves a fast local-L0 endpoint but fails force/energy/physical gates; next algorithm step should add document-grounded representation capacity, not final-head width.
- `community_baseline_protocol_alignment` (P1): Stage145 community atoms/s, Stage157 prebuilt-graph atoms/s, and Stage176 ASE/autograd atoms/s are not directly comparable yet.
