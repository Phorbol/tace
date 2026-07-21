# Stage181 3BPA Scratch Summary

- train job: `688836`
- benchmark job: `688878`
- variant: `stage181_3bpa_l1_local_l0_student`
- train/valid configs: `500` / `64`
- steps / best step: `20000` / `19936`
- best valid loss: `0.180432`

| split | E RMSE meV/atom | E max | F RMSE meV/A | F max | atoms/s | peak MB |
|---|---:|---:|---:|---:|---:|---:|
| `test_300K` | 4.209 | 11.442 | 135.767 | 1163.953 | 1648463 | 618.7 |
| `test_600K` | 4.502 | 16.353 | 211.201 | 3631.354 | 1573112 | 620.6 |
| `test_1200K` | 8.900 | 31.399 | 361.980 | 3290.784 | 1800502 | 606.1 |
| `test_dih` | 16.151 | 28.698 | 212.427 | 1654.093 | 1774193 | 618.4 |

## Interpretation

- Energy error is small on 3BPA compared with OC20NEB, which supports using molecular MD as a cleaner expression/generalization benchmark.
- Force RMSE and force tails worsen strongly under temperature and dihedral OOD, so the current student is not yet a physically robust molecular MD endpoint.
- The next scientific comparison should be the same fixed architecture trained from scratch versus TECE projection/GN-renormalized initialization on these deployment splits.
