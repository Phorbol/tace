# Stage132 Results Summary

- stage: `stage132_broad_teacher_distill_projection_check`
- slurm: all five jobs `COMPLETED 0:0`
- data: 2048 base configs + 512 teacher-rattle configs = 2560 configs / 135986 atoms

| row | E RMSE meV/atom | F RMSE meV/A | F max meV/A | atoms/s | rattle fmax eV/A | C/N RMSD A | physical gate |
|---|---:|---:|---:|---:|---:|---:|---:|
| l1_active_nrad12_species24_radial_species8_cross3_h64 | 293.391 | 108.355 | 1840.539 | 1217882 | 10.559 | 0.103 | False |
| l1_active_species24_cavity_vec_residual_h64 | 307.302 | 109.173 | 2129.123 | 568751 | 10.134 | 0.102 | False |

## Interpretation

Broader teacher-rattle coverage improves high-force-tail/physical indicators versus the corresponding narrower-data baselines, but it does not close the rattle force-spike failure and does not improve RMSE Pareto quality.

Do not keep increasing edge L or only adding small local fake-label patches. The next TECE-consistent step should test true Sobolev/Jacobian or relaxation-trajectory distillation on the fixed atomic front, then decide whether a trainable front representation axis is needed from residual/projection diagnostics.
