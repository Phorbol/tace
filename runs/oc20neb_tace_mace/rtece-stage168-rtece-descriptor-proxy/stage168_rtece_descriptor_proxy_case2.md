# Stage168 rTECE Descriptor Proxy

This tests implemented rTECE semantic descriptor families against Stage165 oracle case offsets. case_id is used only for grouping offsets to structures; it is not used as a model feature.

Source variant: `stage157_direct_b32_rel0p25_mixed2048`. Target offset RMSE: `61.445` meV/atom.
Generalization status: `rtece_descriptor_proxy_not_supported_by_loocv`. Recommendation: `do_not_treat_current_rtece_descriptors_as_sufficient_energy_baseline_prioritize_teacher_coverage_and_new_scalar_paths`.

| feature set | features | train RMSE | train R2 | LOOCV RMSE | LOOCV R2 |
|---|---:|---:|---:|---:|---:|
| intercept_only | 0 | 57.732 | 0.000 | 60.482 | -0.098 |
| all_rtece_descriptor_groups | 237 | 0.007 | 1.000 | 108.732 | -2.547 |
| t3_atomic_l2_cross | 89 | 0.016 | 1.000 | 111.258 | -2.714 |
| t3_full_moment_shell | 49 | 0.053 | 1.000 | 138.244 | -4.734 |
| t3_cavity_edge | 49 | 0.317 | 1.000 | 142.760 | -5.115 |
| t4_pair_density | 17 | 22.518 | 0.848 | 238.393 | -16.051 |
| t3_element_density | 33 | 1.564 | 0.999 | 332.736 | -32.217 |

## Interpretation

If an implemented rTECE descriptor family improves LOOCV, the absolute-energy failure is likely a training, downfolding, or distillation issue. If it only reduces train error, current semantic paths are not sufficient deployable coordinates for the low-frequency energy baseline, and the next priority should be teacher coverage plus new high-value scalar or edge-relational paths.
