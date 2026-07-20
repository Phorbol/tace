# Stage135 L2 Atomic Projection Results

- jobs: train `685231`, physical `685234`; all `COMPLETED 0:0`
- data: fixed Stage132 broad-teacher-rattle augmented train, `2560` configs
- architecture: `l2_active_nrad12_species24_radial_species8_cross3_h64`, `29885` params, `L_A=2`, no edge paths, ZBL, fixed `64,64` head
- training: 20k steps, best step `19840`, best valid loss `0.093676`

## Metrics

| row | atoms/s | E RMSE | E max | F MAE | F RMSE | F max | C/N RMSD | rattle fmax | benchmark | physical |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stage135 L2 atomic | 1058909 | 294.945 | 720.404 | 47.506 | 112.960 | 2649.912 | 0.1096 | 10.754 | False | False |
| stage132 L1 broad-rattle atomic | 1217882 | 293.391 | 697.301 | 46.503 | 108.355 | 1840.539 | 0.1031 | 10.559 | True | False |
| stage134 balanced teacher-relax L1 | 1215642 | 337.270 | 742.225 | 47.766 | 107.020 | 2455.000 | 0.1172 | 9.946 | False | False |

## Interpretation

The controlled `L_A=2` atomic quadrupole row improved validation loss but did not improve the Pareto frontier. Relative to the Stage132 L1 atomic anchor, throughput drops from 1.218M to 1.059M atoms/s, DFT F RMSE rises from 108.35 to 112.96 meV/A, and F max rises from 1840.54 to 2649.91 meV/A. C/N rattle RMSD also increases slightly and rattle fmax remains around 10.75 eV/A.

This means the next TECE coordinate should not be naive higher angular bandwidth alone. For this data slice, L2 atomic scalarization adds cost and apparent validation capacity without reducing the force-tail or physical-relax failure that matters for deployment.

## Next Priority

Run projection diagnostics specifically comparing the Stage132 L1 path set against the Stage135 L2 quadrupole paths, then prioritize a low-rank trainable front mixer or better Sobolev/teacher trajectory target if the residual is label/optimization dominated. Do not add edge relational sketches or larger L before the projection residual justifies their hardware cost.
