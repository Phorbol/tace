# Stage166 Deployable Proxy Baseline

This tests whether cheap deployable scalar proxies can explain the oracle case offsets from Stage165. case_id is used only for grouping offsets to structures; it is not used as a model feature.

Source variant: `stage157_direct_b32_rel0p25_mixed2048`. Target offset RMSE: `61.445` meV/atom.
Generalization status: `proxy_not_supported_by_loocv`. Recommendation: `do_not_add_global_proxy_baseline_yet_prioritize_richer_local_tece_paths_or_distillation_coverage`.

| feature set | features | train RMSE | train R2 | LOOCV RMSE | LOOCV R2 |
|---|---:|---:|---:|---:|---:|
| composition | 39 | 0.000 | 1.000 | 59.073 | -0.047 |
| intercept_only | 0 | 57.732 | 0.000 | 60.482 | -0.098 |
| composition_cell | 44 | 0.000 | 1.000 | 66.014 | -0.307 |
| composition_cell_geometry | 47 | 0.000 | 1.000 | 68.344 | -0.401 |

## Interpretation

A low train RMSE with poor LOOCV means these cheap global proxies can memorize the 22-case offset pattern but are not a deployable fix. A strong LOOCV result would justify adding an explicit low-frequency scalar baseline path to the student; a weak LOOCV result points back to richer local TECE scalar paths or teacher-generated coverage.
