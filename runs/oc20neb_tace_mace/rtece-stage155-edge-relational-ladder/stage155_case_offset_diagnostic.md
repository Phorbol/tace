# Stage155 Case Offset Diagnostic

This diagnostic compares Stage153 mixed L2 atomic baseline and Stage154 T3 cavity-vector group mean energy offsets on the same DFT validation window.

## Summary

- cases: `22`
- Stage153 raw E RMSE: `126.883` meV/atom; group-offset RMSE: `6.189` meV/atom
- Stage154 T3 raw E RMSE: `94.167` meV/atom; group-offset RMSE: `5.859` meV/atom
- C/N cases: `21`; non-C/N cases: `1`

## Largest Remaining Stage154 Case Offsets

| case | Stage153 abs offset | Stage154 abs offset | improvement | C | N | O | atoms |
|---|---:|---:|---:|---:|---:|---:|---:|
| `dissociation_id_252_5588_26_011-1_neb1.0` | 313.3 | 274.7 | 38.6 | 2 | 0 | 2 | 85 |
| `dissociation_id_295_8139_40_211-10_neb1.0` | 146.0 | 185.6 | -39.6 | 2 | 0 | 2 | 54 |
| `dissociation_id_212_7241_50_011-0_neb1.0` | 208.7 | 178.2 | 30.5 | 0 | 1 | 0 | 79 |
| `dissociation_id_266_9844_9_111-2_neb1.0` | 34.5 | 168.9 | -134.4 | 1 | 0 | 1 | 58 |
| `dissociation_id_127_9635_0_100-1_neb1.0` | 142.6 | 147.2 | -4.6 | 0 | 0 | 1 | 66 |
| `dissociation_id_236_2395_27_000-0_neb1.0` | 277.0 | 130.9 | 146.2 | 2 | 0 | 1 | 69 |
| `dissociation_id_195_6923_47_111-1_neb1.0` | 74.8 | 126.1 | -51.3 | 0 | 1 | 1 | 82 |
| `dissociation_id_281_2644_33_200-4_neb1.0` | 96.0 | 90.3 | 5.6 | 2 | 0 | 1 | 53 |

## Best Offset Improvements

| case | Stage153 abs offset | Stage154 abs offset | improvement | C | N | O |
|---|---:|---:|---:|---:|---:|---:|
| `dissociation_id_236_2395_27_000-0_neb1.0` | 277.0 | 130.9 | 146.2 | 2 | 0 | 1 |
| `dissociation_id_239_6570_8_211-5_neb1.0` | 135.2 | 21.3 | 114.0 | 1 | 0 | 0 |
| `dissociation_id_294_2408_32_222-0_neb1.0` | 119.8 | 9.3 | 110.5 | 2 | 0 | 1 |
| `dissociation_id_104_2948_48_111-1_neb1.0` | 110.3 | 3.0 | 107.4 | 1 | 1 | 0 |
| `dissociation_id_293_11099_47_100-0_neb1.0` | 131.3 | 40.6 | 90.7 | 0 | 1 | 1 |
| `dissociation_id_245_139_42_211-1_neb1.0` | 59.9 | 2.3 | 57.6 | 2 | 0 | 0 |
| `dissociation_id_219_10897_5_100-0_neb1.0` | 59.8 | 5.4 | 54.5 | 1 | 0 | 1 |
| `dissociation_id_258_5496_39_100-1_neb1.0` | 92.2 | 47.1 | 45.1 | 2 | 0 | 2 |

## Worst Offset Regressions

| case | Stage153 abs offset | Stage154 abs offset | improvement | C | N | O |
|---|---:|---:|---:|---:|---:|---:|
| `dissociation_id_266_9844_9_111-2_neb1.0` | 34.5 | 168.9 | -134.4 | 1 | 0 | 1 |
| `dissociation_id_182_2807_43_111-2_neb1.0` | 29.2 | 88.6 | -59.4 | 2 | 0 | 1 |
| `dissociation_id_195_6923_47_111-1_neb1.0` | 74.8 | 126.1 | -51.3 | 0 | 1 | 1 |
| `dissociation_id_295_8139_40_211-10_neb1.0` | 146.0 | 185.6 | -39.6 | 2 | 0 | 2 |
| `dissociation_id_271_7189_22_211-5_neb1.0` | 60.2 | 81.6 | -21.4 | 2 | 0 | 1 |
| `dissociation_id_154_1581_1_111-0_neb1.0` | 43.6 | 53.7 | -10.1 | 1 | 0 | 1 |
| `dissociation_id_127_9635_0_100-1_neb1.0` | 142.6 | 147.2 | -4.6 | 0 | 0 | 1 |
| `dissociation_id_281_2644_33_200-4_neb1.0` | 96.0 | 90.3 | 5.6 | 2 | 0 | 1 |

## Interpretation

Stage154 improves raw energy mainly by reducing several large case-level offsets, but substantial case offsets remain. This supports Stage155's edge-path marginal test: direct radial and cavity quadrupole increments should be judged by raw E RMSE, group/case offset RMSE, relative NEB metrics, force RMSE/max, and measured atoms/s together.
