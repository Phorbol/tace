# Stage180 Renorm Comparison

- initializer job: `689028`
- renorm train job: `689033`
- renorm DFT benchmark job: `689120`
- renorm physical/scaling job: `689145`
- valid loss ratio renorm/scratch: `0.625`

## DFT/Teacher Metrics

| arm | target | E RMSE | E max | F RMSE | F max | rel-image RMSE | barrier RMSE | atoms/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `scratch` | dft | 150.241 | 322.693 | 121.982 | 4530.538 | 17.055 | 31.954 | 1223288 |
| `scratch` | teacher | 139.614 | 305.322 | 120.288 | 4501.704 | 16.900 | 30.671 | 1258339 |
| `renorm` | dft | 65.755 | 166.064 | 100.827 | 1638.535 | 8.657 | 12.314 | 1228316 |
| `renorm` | teacher | 57.993 | 148.693 | 97.098 | 1566.623 | 8.121 | 10.655 | 1252128 |

## Physical Diagnostics

| arm | dimer lift | dimer short force | rattle RMSD mean/max | rattle fmax max | physical gate |
|---|---:|---:|---:|---:|---|
| `scratch` | -0.016 | 0.087 | 0.305/0.665 | 0.214 | False |
| `renorm` | 20.156 | 206.981 | 0.124/0.163 | 10.833 | False |

## Scaling

| arm | configs | atoms/s | peak MB |
|---|---:|---:|---:|
| `scratch` | 32 | 327129 | 126.7 |
| `scratch` | 128 | 888157 | 297.7 |
| `scratch` | 512 | 1524481 | 1803.9 |
| `scratch` | 1024 | 1682683 | 3168.9 |
| `renorm` | 32 | 301008 | 126.7 |
| `renorm` | 128 | 875655 | 297.7 |
| `renorm` | 512 | 1525372 | 1803.9 |
| `renorm` | 1024 | 1686207 | 3168.9 |

## Interpretation

- Renorm-init improves matched DFT and teacher E/F RMSE, E/F max tails, NEB relative-image RMSE, and barrier RMSE at essentially unchanged throughput/memory.
- Dimer scan becomes strongly short-range repulsive, while scratch is nearly flat at short range; this supports the initializer adding a physically useful local curvature/repulsion component.
- Rattle-relax is mixed: final RMSD improves, but max force tail becomes much worse. This is not a binary blocker, but it identifies the next algorithmic target: force-tail/curvature distillation or short-range/ZBL-consistent regularization.
- Full completion still requires the teacher residual distillation arm and a cleaner Schur/GN coefficient initializer rather than the current projection-constrained E/F prefit proxy.
