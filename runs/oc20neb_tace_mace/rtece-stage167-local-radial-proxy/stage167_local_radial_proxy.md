# Stage167 Local Radial Proxy

This tests one neighbor-pass local radial scalar proxies for Stage165 oracle case offsets. case_id is used only for grouping offsets to structures; it is not used as a model feature.

Source variant: `stage157_direct_b32_rel0p25_mixed2048`. Target offset RMSE: `61.445` meV/atom.
Generalization status: `local_radial_proxy_not_supported_by_loocv`. Recommendation: `do_not_promote_simple_local_radial_proxy_prioritize_teacher_coverage_or_higher_body_scalar_paths`.

| feature set | features | train RMSE | train R2 | LOOCV RMSE | LOOCV R2 |
|---|---:|---:|---:|---:|---:|
| composition | 29 | 0.004 | 1.000 | 59.076 | -0.047 |
| intercept_only | 0 | 57.732 | 0.000 | 60.482 | -0.098 |
| local_radial_plus_global | 684 | 0.000 | 1.000 | 168.080 | -7.476 |
| local_radial_plus_composition | 646 | 0.000 | 1.000 | 180.943 | -8.823 |
| local_radial | 617 | 0.000 | 1.000 | 191.033 | -9.949 |

## Interpretation

If local radial proxies improve LOOCV over Stage166 global proxies, a deployable scalar baseline or richer local radial path is worth promoting. If they only reduce train error, the case-offset mode requires better deployment coverage, higher-body scalar contractions, or teacher response distillation rather than another simple baseline.
