# Stage133 Teacher-Relax Trajectory Distillation Results

- jobs: prepare `685197`, train `685198`, physical `685199`; all `COMPLETED 0:0`
- data: `2048` base configs + `320` teacher-relax trajectory configs = `2368` train configs / `126386` atoms
- architecture: fixed `l1_active_nrad12_species24_radial_species8_cross3_h64`, `28925` params

## Metrics

| row | atoms/s | E RMSE | E max | F MAE | F RMSE | F max | C/N RMSD | rattle fmax | benchmark | physical |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stage133 teacher-relax atomic | 1213704 | 363.615 | 806.456 | 44.212 | 106.766 | 2098.922 | 0.1132 | 8.763 | False | False |
| stage132 broad-rattle atomic | 1217882 | 293.391 | 697.301 | 46.503 | 108.355 | 1840.539 | 0.1031 | 10.559 | True | False |

## Interpretation

Teacher-relax trajectory coverage is not a clean Pareto improvement over the stage132 fixed atomic row: it slightly improves F RMSE and rattle force-tail score, but worsens energy RMSE/max, F max, C/N relax RMSD, and benchmark gate status at essentially unchanged throughput.

The lower best valid loss and lower physical score indicate that trajectory labels add useful force-field-manifold information, but the degraded DFT energy metrics show that simply adding short teacher trajectories shifts the target distribution and does not close the DFT-comparable PES.

## Next Priority

Stop increasing same-family fake labels alone. Next TECE-consistent step should either introduce explicit Sobolev/Jacobian weighting with DFT/teacher balance or test a front-representation axis that is still theoretically clean: higher controlled L_A/cross-radial paths or low-rank trainable feature mixers, evaluated by RMSE/max/physical tests and throughput scaling.
