# rTECE Energy Offset Summary

This separates raw absolute energy error from case/group offsets, relative NEB shape, force error, and throughput.

| row | raw E RMSE | E MAE | E max | group-offset E RMSE | removed by group offset | first-anchor E RMSE | rel image RMSE | barrier RMSE | F RMSE | F max | atoms/s | top case offsets meV/atom |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| rel0 | 58.067 | 40.886 | 219.783 | 4.811 | 91.7% | 10.989 | 7.207 | 11.171 | 95.224 | 2095.459 | 500308 | dissociation_id_212_7241_50_011-0_neb1.0:+214.954, dissociation_id_252_5588_26_011-1_neb1.0:+117.926, dissociation_id_195_6923_47_111-1_neb1.0:+100.503, dissociation_id_281_2644_33_200-4_neb1.0:+93.770, dissociation_id_271_7189_22_211-5_neb1.0:+75.751 |
| rel0p25 | 52.538 | 35.443 | 204.986 | 4.769 | 90.9% | 11.461 | 7.104 | 11.403 | 94.159 | 2092.961 | 500402 | dissociation_id_212_7241_50_011-0_neb1.0:+199.681, dissociation_id_252_5588_26_011-1_neb1.0:+94.946, dissociation_id_195_6923_47_111-1_neb1.0:+89.989, dissociation_id_281_2644_33_200-4_neb1.0:+89.665, dissociation_id_236_2395_27_000-0_neb1.0:-79.598 |
| rel1p0 | 53.229 | 36.156 | 201.196 | 4.774 | 91.0% | 11.320 | 7.087 | 11.139 | 95.898 | 2150.411 | 499833 | dissociation_id_212_7241_50_011-0_neb1.0:+195.994, dissociation_id_236_2395_27_000-0_neb1.0:-93.498, dissociation_id_252_5588_26_011-1_neb1.0:+91.006, dissociation_id_195_6923_47_111-1_neb1.0:+86.812, dissociation_id_281_2644_33_200-4_neb1.0:+85.812 |

Interpretation rule: if group-offset RMSE is much smaller than raw E RMSE, the dominant raw-energy error is a case/slab/adsorbate baseline term; representation changes must still be checked against relative-image/barrier and force RMSE/max.
