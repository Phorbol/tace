# rTECE Stage Sweep Summary: descriptor-bottleneck-stage120

Primary metric: DFT force RMSE vs atoms/s. Missing benchmark rows are kept explicit.

| row | status | params | TECE axes | atoms/s | F RMSE | F MAE | F max | E RMSE | E MAE | E bias | E max |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| l0_species8_bneck16_h64 | benchmark_found | 7417 | angular_bandwidth_l_max, scalar_path_density, radial_rank, scalar_head_capacity, trainable_feature_extractor, trainable_species_basis, short_range_physical_prior, frontloaded_representation_capacity, low_rank_neighbor_species_basis, descriptor_bottleneck, front_low_rank_path_mixer | 2.135e+06 | 115.390 | 37.182 | 4835.172 | 302.912 | 243.452 | -37.946 | 713.779 |
| l0_species8_bneck32_h64 | benchmark_found | 9609 | angular_bandwidth_l_max, scalar_path_density, radial_rank, scalar_head_capacity, trainable_feature_extractor, trainable_species_basis, short_range_physical_prior, frontloaded_representation_capacity, low_rank_neighbor_species_basis, descriptor_bottleneck, front_low_rank_path_mixer | 2.137e+06 | 129.762 | 58.994 | 2523.890 | 383.948 | 334.812 | -137.005 | 729.060 |
| l2_species32_cavity_atomic_bneck16_h64 | benchmark_found | 13353 | angular_bandwidth_l_max, scalar_path_density, radial_rank, scalar_head_capacity, trainable_feature_extractor, trainable_species_basis, short_range_physical_prior, frontloaded_representation_capacity, cross_radial_invariants, trainable_cross_radial_projection, cavity_edge_relational_scalar_sketches, direct_edge_radial_path, atomic_l2_scalar_paths, descriptor_bottleneck, front_low_rank_path_mixer | 5.272e+05 | 137.316 | 61.053 | 2877.407 | 362.402 | 304.415 | -65.020 | 744.926 |
| l2_species32_cavity_atomic_bneck32_h64 | benchmark_found | 19033 | angular_bandwidth_l_max, scalar_path_density, radial_rank, scalar_head_capacity, trainable_feature_extractor, trainable_species_basis, short_range_physical_prior, frontloaded_representation_capacity, cross_radial_invariants, trainable_cross_radial_projection, cavity_edge_relational_scalar_sketches, direct_edge_radial_path, atomic_l2_scalar_paths, descriptor_bottleneck, front_low_rank_path_mixer | 5.273e+05 | 127.650 | 59.213 | 2475.742 | 375.218 | 322.074 | -68.802 | 636.863 |

## Missing Benchmark Rows

None

## DFT Force RMSE Pareto Front

- `l0_species8_bneck32_h64`: atoms/s=2.137e+06, F RMSE=129.762
- `l0_species8_bneck16_h64`: atoms/s=2.135e+06, F RMSE=115.390

## Review Notes

- TECE_design_space: hardware Pareto frontier over physical error vs atom/s and memory
- rTECE_review: record force RMSE, high-force tail/max error, energy RMSE/bias/max separately
- rTECE_review: keep dimer/rattle/element-stratified physical probes explicit rather than collapsing to one MAE
