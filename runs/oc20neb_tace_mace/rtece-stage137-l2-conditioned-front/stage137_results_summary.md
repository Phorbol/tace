# rTECE Stage Sweep Summary: stage137-l2-conditioned-front

Primary metric: DFT force RMSE vs atoms/s. Missing benchmark rows are kept explicit.

| row | status | params | TECE axes | atoms/s | F RMSE | F MAE | F max | E RMSE | E MAE | E bias | E max |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| l2_active_nrad12_species24_radial_species8_cross3_bneck32_h64 | benchmark_found | 21405 | angular_bandwidth_l_max, scalar_path_density, radial_rank, scalar_head_capacity, trainable_feature_extractor, trainable_edge_species_radial_basis, trainable_species_basis, short_range_physical_prior, descriptor_bottleneck, stage137_l2_conditioned_front, stage136_force_projection_guided, stage132_broad_teacher_anchor, stage135_l2_atomic_reference, controlled_L_A_axis, atomic_quadrupole_scalar_paths, cross_radial_invariants, trainable_cross_radial_projection, low_rank_neighbor_species_basis, front_representation_capacity_not_wider_head, front_low_rank_path_mixer | 1.061e+06 | 114.259 | 48.988 | 1861.035 | 309.728 | 236.931 | 9.894 | 691.876 |
| l2_active_nrad12_species24_radial_species8_cross3_cond32_h64 | benchmark_found | 51367 | angular_bandwidth_l_max, scalar_path_density, radial_rank, scalar_head_capacity, trainable_feature_extractor, trainable_edge_species_radial_basis, trainable_species_basis, short_range_physical_prior, stage137_l2_conditioned_front, stage136_force_projection_guided, stage132_broad_teacher_anchor, stage135_l2_atomic_reference, controlled_L_A_axis, atomic_quadrupole_scalar_paths, cross_radial_invariants, trainable_cross_radial_projection, low_rank_neighbor_species_basis, front_representation_capacity_not_wider_head, scalar_descriptor_conditioning, zero_init_residual_path_conditioner | 1.023e+06 | 107.516 | 45.846 | 1914.572 | 340.032 | 271.543 | -59.148 | 740.266 |

## Physical Diagnostics

| row | focus | focus F RMSE | focus F max | physical score | physical gate | dimer gate | rattle gate |
|---|---|---:|---:|---:|---|---|---|
| l2_active_nrad12_species24_radial_species8_cross3_bneck32_h64 | C_or_N | NA | NA | 12.861 | False | True | False |
| l2_active_nrad12_species24_radial_species8_cross3_cond32_h64 | C_or_N | NA | NA | 14.403 | False | True | False |

## Missing Benchmark Rows

None

## DFT Force RMSE Pareto Front

- `l2_active_nrad12_species24_radial_species8_cross3_bneck32_h64`: atoms/s=1.061e+06, F RMSE=114.259
- `l2_active_nrad12_species24_radial_species8_cross3_cond32_h64`: atoms/s=1.023e+06, F RMSE=107.516

## Review Notes

- TECE_design_space: hardware Pareto frontier over physical error vs atom/s and memory
- rTECE_review: record force RMSE, high-force tail/max error, energy RMSE/bias/max separately
- rTECE_review: keep dimer/rattle/element-stratified physical probes explicit rather than collapsing to one MAE
