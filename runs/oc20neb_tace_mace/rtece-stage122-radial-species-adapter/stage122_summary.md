# rTECE Stage Sweep Summary: radial-species-adapter-stage122

Primary metric: DFT force RMSE vs atoms/s. Missing benchmark rows are kept explicit.

| row | status | params | TECE axes | atoms/s | F RMSE | F MAE | F max | E RMSE | E MAE | E bias | E max |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| l0_pair_radial_species8_h64 | benchmark_found | 6673 | angular_bandwidth_l_max, scalar_path_density, radial_rank, scalar_head_capacity, trainable_feature_extractor, trainable_edge_species_radial_basis, short_range_physical_prior, stage122_radial_species_adapter, frontloaded_representation_capacity | 2.570e+06 | 134.670 | 49.482 | 4695.332 | 335.338 | 274.571 | 84.011 | 807.300 |
| l0_species8_radial_species8_h64 | benchmark_found | 11577 | angular_bandwidth_l_max, scalar_path_density, radial_rank, scalar_head_capacity, trainable_feature_extractor, trainable_edge_species_radial_basis, trainable_species_basis, short_range_physical_prior, stage122_radial_species_adapter, frontloaded_representation_capacity, low_rank_neighbor_species_basis | 2.040e+06 | 123.440 | 50.707 | 2905.086 | 318.140 | 244.790 | 18.573 | 734.122 |
| l1_active_radial_species8_h64 | benchmark_found | 17209 | angular_bandwidth_l_max, scalar_path_density, radial_rank, scalar_head_capacity, trainable_feature_extractor, trainable_edge_species_radial_basis, trainable_species_basis, short_range_physical_prior, stage122_radial_species_adapter, frontloaded_representation_capacity, low_rank_neighbor_species_basis, stage114_ef_active_selection, cross_radial_invariants, trainable_cross_radial_projection | 1.671e+06 | 112.838 | 46.315 | 1996.323 | 335.816 | 268.928 | -41.688 | 731.924 |
| l1_active_radial_species16_h64 | benchmark_found | 18953 | angular_bandwidth_l_max, scalar_path_density, radial_rank, scalar_head_capacity, trainable_feature_extractor, trainable_edge_species_radial_basis, trainable_species_basis, short_range_physical_prior, stage122_radial_species_adapter, frontloaded_representation_capacity, low_rank_neighbor_species_basis, stage114_ef_active_selection, cross_radial_invariants, trainable_cross_radial_projection | 1.661e+06 | 115.390 | 48.508 | 2364.083 | 333.318 | 262.331 | -31.542 | 741.273 |

## Physical Diagnostics

| row | focus | focus F RMSE | focus F max | physical score | physical gate | dimer gate | rattle gate |
|---|---|---:|---:|---:|---|---|---|
| l1_active_radial_species8_h64 | C_or_N | NA | NA | 17.615 | False | True | False |

## Missing Benchmark Rows

None

## DFT Force RMSE Pareto Front

- `l0_pair_radial_species8_h64`: atoms/s=2.570e+06, F RMSE=134.670
- `l0_species8_radial_species8_h64`: atoms/s=2.040e+06, F RMSE=123.440
- `l1_active_radial_species8_h64`: atoms/s=1.671e+06, F RMSE=112.838

## Review Notes

- TECE_design_space: hardware Pareto frontier over physical error vs atom/s and memory
- rTECE_review: record force RMSE, high-force tail/max error, energy RMSE/bias/max separately
- rTECE_review: keep dimer/rattle/element-stratified physical probes explicit rather than collapsing to one MAE
