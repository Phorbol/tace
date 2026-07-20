# Stage134 Balanced Teacher-Relax Distillation Results

- jobs: prepare `685204`, train `685205`, physical `685206`; all `COMPLETED 0:0`
- data: `2048` base configs + `320` teacher-relax trajectory configs = `2368` train configs / `126386` atoms
- weight policy: base energy weight high, teacher trajectory energy low, teacher trajectory force high; energy/force means normalized to 1
- architecture: fixed `l1_active_nrad12_species24_radial_species8_cross3_h64`, `28925` params

## Metrics

| row | atoms/s | E RMSE | E max | F MAE | F RMSE | F max | C/N RMSD | rattle fmax | benchmark | physical |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stage134 balanced teacher-relax | 1215642 | 337.270 | 742.225 | 47.766 | 107.020 | 2455.000 | 0.1172 | 9.946 | False | False |
| stage133 teacher-relax | 1213704 | 363.615 | 806.456 | 44.212 | 106.766 | 2098.922 | 0.1132 | 8.763 | False | False |
| stage132 broad-rattle atomic | 1217882 | 293.391 | 697.301 | 46.503 | 108.355 | 1840.539 | 0.1031 | 10.559 | True | False |

## Interpretation

DFT energy anchoring partially reduces the stage133 energy drift, but it does not recover the stage132 benchmark Pareto point and it worsens force max / rattle indicators relative to stage133.

The source-weight policy improves E RMSE from stage133 363.6 to 337.3 meV/atom, but F max rises to 2455 meV/A and physical score regresses. The remaining gap is therefore not just source energy/force balance.

## Next Priority

Stop iterating source-weight heuristics on the fixed atomic front. The next TECE-aligned move should be a representation/projection step with clean theory: controlled L_A/cross-radial expansion or low-rank trainable front mixer, coupled to projection diagnostics and the same RMSE/max/physical/throughput gates.
