# Stage181 3BPA Distribution Coverage Comparison

- baseline: same rTECE student trained on `train_300K` (job `688836`, benchmark `688878`)
- coverage arm: same rTECE student trained on `train_mixedT` (job `689295`, benchmark `689302`)
- mixedT steps / best step / best valid loss: `18096` / `14096` / `0.193109`
- architecture is held fixed: T3 scalarized vector-moment student, 21,079 params, descriptor dim 223

| split | train | E RMSE | E max | F RMSE | F max | atoms/s | peak MB |
|---|---|---:|---:|---:|---:|---:|---:|
| `test_300K` | `300K` | 4.209 | 11.442 | 135.767 | 1163.953 | 1648463 | 618.7 |
| `test_300K` | `mixedT` | 5.854 | 15.765 | 139.229 | 902.757 | 1761891 | 618.7 |
| `test_300K` | `mixed/300 ratio` | 1.391 |  | 1.025 | 0.776 | 1.069 |  |
| `test_600K` | `300K` | 4.502 | 16.353 | 211.201 | 3631.354 | 1573112 | 620.6 |
| `test_600K` | `mixedT` | 6.482 | 18.795 | 192.916 | 1444.686 | 1764258 | 620.6 |
| `test_600K` | `mixed/300 ratio` | 1.440 |  | 0.913 | 0.398 | 1.122 |  |
| `test_1200K` | `300K` | 8.900 | 31.399 | 361.980 | 3290.784 | 1800502 | 606.1 |
| `test_1200K` | `mixedT` | 9.205 | 32.007 | 300.193 | 2564.484 | 1797775 | 606.1 |
| `test_1200K` | `mixed/300 ratio` | 1.034 |  | 0.829 | 0.779 | 0.998 |  |
| `test_dih` | `300K` | 16.151 | 28.698 | 212.427 | 1654.093 | 1774193 | 618.4 |
| `test_dih` | `mixedT` | 9.706 | 14.903 | 150.794 | 984.470 | 1769121 | 618.4 |
| `test_dih` | `mixed/300 ratio` | 0.601 |  | 0.710 | 0.595 | 0.997 |  |

## Interpretation

- `train_mixedT` improves the strongest OOD force RMSE: `test_1200K` 361.980 -> 300.193 meV/A and `test_dih` 212.427 -> 150.794 meV/A.
- It does not uniformly improve every metric: ID `test_300K` force RMSE is slightly worse, and energy RMSE worsens on `test_300K`/`test_600K`/`test_1200K` despite force-tail improvements.
- This supports the TECE-design view that deployment distribution coverage is part of the low-cost projection target, but the remaining force tails mean architecture/curvature distillation is still needed.
