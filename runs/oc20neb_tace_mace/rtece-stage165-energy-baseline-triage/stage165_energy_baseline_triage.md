# Stage165 Energy Baseline Triage

This treats per-case/group offsets as an oracle diagnostic, not a deployable correction or model input.
The goal is to decide whether the bad raw energy needs a symmetry-preserving low-frequency baseline path, stronger front-loaded scalar representation, or true distillation rather than a larger final head.

| variant | raw E RMSE | group-offset RMSE | explained raw RMSE | relative image RMSE | barrier RMSE | F RMSE | cases | max case offset |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| stage157_direct_b32_rel0_mixed2048 | 58.067 | 4.811 | 0.993 | 7.207 | 11.171 | 95.224 | 22 | 214.954 |
| stage157_direct_b32_rel0p25_mixed2048 | 52.538 | 4.769 | 0.992 | 7.104 | 11.403 | 94.159 | 22 | 199.681 |
| stage157_direct_b32_rel1p0_mixed2048 | 53.229 | 4.774 | 0.992 | 7.087 | 11.139 | 95.898 | 22 | 195.994 |

## Largest Case Offsets

### stage157_direct_b32_rel0_mixed2048

| case | family | offset | abs offset |
|---|---|---:|---:|
| dissociation_id_212_7241_50_011-0_neb1.0 | dissociation | 0.214954 | 214.954 |
| dissociation_id_252_5588_26_011-1_neb1.0 | dissociation | 0.117926 | 117.926 |
| dissociation_id_195_6923_47_111-1_neb1.0 | dissociation | 0.100503 | 100.503 |
| dissociation_id_281_2644_33_200-4_neb1.0 | dissociation | 0.093770 | 93.770 |
| dissociation_id_271_7189_22_211-5_neb1.0 | dissociation | 0.075751 | 75.751 |

### stage157_direct_b32_rel0p25_mixed2048

| case | family | offset | abs offset |
|---|---|---:|---:|
| dissociation_id_212_7241_50_011-0_neb1.0 | dissociation | 0.199681 | 199.681 |
| dissociation_id_252_5588_26_011-1_neb1.0 | dissociation | 0.094946 | 94.946 |
| dissociation_id_195_6923_47_111-1_neb1.0 | dissociation | 0.089989 | 89.989 |
| dissociation_id_281_2644_33_200-4_neb1.0 | dissociation | 0.089665 | 89.665 |
| dissociation_id_236_2395_27_000-0_neb1.0 | dissociation | -0.079598 | 79.598 |

### stage157_direct_b32_rel1p0_mixed2048

| case | family | offset | abs offset |
|---|---|---:|---:|
| dissociation_id_212_7241_50_011-0_neb1.0 | dissociation | 0.195994 | 195.994 |
| dissociation_id_236_2395_27_000-0_neb1.0 | dissociation | -0.093498 | 93.498 |
| dissociation_id_252_5588_26_011-1_neb1.0 | dissociation | 0.091006 | 91.006 |
| dissociation_id_195_6923_47_111-1_neb1.0 | dissociation | 0.086812 | 86.812 |
| dissociation_id_281_2644_33_200-4_neb1.0 | dissociation | 0.085812 | 85.812 |

## Interpretation

If the explained raw-RMSE fraction remains near one across relative-loss variants, the energy problem is dominated by low-frequency case/site/adsorbate baseline modes. That points to deployable baseline features or TECE semantic scalar paths, not per-case IDs and not final-head widening.
