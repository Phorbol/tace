# Stage174 Low-Frequency Feature Probe Results

- split: `group-loocv` by `case_id`
- target: `stage165_case_offset_residual_mev_atom`
- intercept baseline: `56.60319287094426` meV/atom RMSE
- recommendation: `do_not_promote_current_lowfreq_proxy_prioritize_richer_local_tece_front_or_distillation_coverage`

| feature family | features | requires tags | E RMSE | E MAE | E max | selected ridge | beats intercept |
|---|---:|---|---:|---:|---:|---:|---|
| pair_histogram | 479 | False | 60.263 | 44.101 | 199.804 | 0.01 | False |
| composition_fraction | 41 | False | 130.600 | 81.212 | 338.213 | 100 | False |
| tag_composition | 167 | True | 141.888 | 86.453 | 375.750 | 100 | False |
| geometry_z_profile | 51 | False | 143.249 | 80.848 | 385.120 | 100 | False |
| composition_tag_geometry | 186 | True | 150.821 | 84.912 | 409.238 | 100 | False |

## Interpretation

A feature family is architecture-promotable only if it beats the intercept-only group-heldout baseline without using non-deployable metadata as features.

No tested low-frequency proxy beats the intercept-only group-heldout baseline. This rejects a simple global composition/tag/geometry correction as the next architecture move. The next branch should use richer local TECE front coordinates or teacher-projected/distilled coverage rather than adding these proxy summaries directly to the student.

## Next Priority

- Do not add `case_id`, source metadata, or per-case offsets as model inputs.
- Do not promote the current composition/tag/geometry proxy families into the production student.
- Treat coarse pair histograms as a weak hint only: they are closest to the baseline but still do not pass the group-heldout gate.
- Move the next architecture experiment toward local, learnable TECE front coordinates and force-protected E/F active-set scoring.
