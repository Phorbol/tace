# rTECE Stage Sweep Summary: active-frontloaded-stage121

Primary metric: DFT force RMSE vs atoms/s. Missing benchmark rows are kept explicit.

| row | status | params | TECE axes | atoms/s | F RMSE | F MAE | F max | E RMSE | E MAE | E bias | E max |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| l1_active_species16_bneck16_h64 | benchmark_found | 9449 | angular_bandwidth_l_max, scalar_path_density, radial_rank, scalar_head_capacity, trainable_feature_extractor, trainable_species_basis, short_range_physical_prior, descriptor_bottleneck, stage114_ef_active_selection, frontloaded_representation_capacity, low_rank_neighbor_species_basis, cross_radial_invariants, trainable_cross_radial_projection, front_low_rank_path_mixer | 1.910e+06 | 113.073 | 33.998 | 4915.273 | 302.927 | 241.193 | -49.287 | 721.859 |
| l1_active_species16_bneck32_h64 | benchmark_found | 12841 | angular_bandwidth_l_max, scalar_path_density, radial_rank, scalar_head_capacity, trainable_feature_extractor, trainable_species_basis, short_range_physical_prior, descriptor_bottleneck, stage114_ef_active_selection, frontloaded_representation_capacity, low_rank_neighbor_species_basis, cross_radial_invariants, trainable_cross_radial_projection, front_low_rank_path_mixer | 1.901e+06 | 137.531 | 62.323 | 2726.624 | 464.156 | 399.730 | -338.229 | 832.070 |
| l2_active_species16_bneck16_h64 | benchmark_found | 9625 | angular_bandwidth_l_max, scalar_path_density, radial_rank, scalar_head_capacity, trainable_feature_extractor, trainable_species_basis, short_range_physical_prior, descriptor_bottleneck, stage114_ef_active_selection, frontloaded_representation_capacity, low_rank_neighbor_species_basis, cross_radial_invariants, trainable_cross_radial_projection, front_low_rank_path_mixer, atomic_l2_scalar_paths | 1.619e+06 | 131.949 | 58.665 | 2548.679 | 409.607 | 335.651 | -119.938 | 854.889 |
| l2_active_species16_bneck32_h64 | benchmark_found | 13193 | angular_bandwidth_l_max, scalar_path_density, radial_rank, scalar_head_capacity, trainable_feature_extractor, trainable_species_basis, short_range_physical_prior, descriptor_bottleneck, stage114_ef_active_selection, frontloaded_representation_capacity, low_rank_neighbor_species_basis, cross_radial_invariants, trainable_cross_radial_projection, front_low_rank_path_mixer, atomic_l2_scalar_paths | 1.610e+06 | 126.051 | 53.839 | 2467.245 | 317.564 | 244.302 | -42.891 | 723.848 |

## Missing Benchmark Rows

None

## DFT Force RMSE Pareto Front

- `l1_active_species16_bneck16_h64`: atoms/s=1.910e+06, F RMSE=113.073

## Review Notes

- TECE_design_space: hardware Pareto frontier over physical error vs atom/s and memory
- rTECE_review: record force RMSE, high-force tail/max error, energy RMSE/bias/max separately
- rTECE_review: keep dimer/rattle/element-stratified physical probes explicit rather than collapsing to one MAE
