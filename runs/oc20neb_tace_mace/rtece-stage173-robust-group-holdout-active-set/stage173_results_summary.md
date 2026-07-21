# Stage173 Robust Group-Heldout Active-Set Results

Stage173 reruns the Stage172 case-group-heldout residual projection with fold-local feature standardization, an intercept column, a ridge grid, and an intercept-only group-heldout baseline.

## Job Status

- Slurm job: `688196`
- State: `COMPLETED 0:0`
- Elapsed: `00:00:53`
- Raw projection JSON remains uncommitted under `projection_outputs/`.

## Main Result

- split: `group-loocv` by `case_id`
- groups: `11`
- robust options: intercept=`True`, standardize=`True`, ridge grid=`[1e-08, 1e-06, 0.0001, 0.01, 1.0, 100.0]`
- intercept-only group-heldout baseline: `56.603` meV/atom RMSE

| candidate | dim | E RMSE | E MAE | E max | selected ridge | beats intercept | descriptor residual |
|---|---:|---:|---:|---:|---:|---|---:|
| t2_edge_frame_plus_direct | 338 | 104.198 | 64.951 | 311.434 | 1 | False | 1.91973e-07 |
| t2_edge_frame_projection | 336 | 103.786 | 64.774 | 311.517 | 1 | False | 2.75754e-07 |
| t2_cavity_quadrupole | 332 | 105.414 | 65.303 | 313.118 | 1 | False | 0.0106391 |
| t2_cavity_vector | 331 | 109.869 | 68.267 | 336.223 | 1 | False | 0.0108591 |
| t1_l2_atomic_cross | 330 | 118.454 | 73.185 | 354.239 | 1 | False | 0.0111739 |
| t1_l1_atomic_cross | 315 | 94.046 | 69.699 | 370.327 | 1 | False | 0.0270998 |
| t1_l0_species_radial | 300 | 106.292 | 58.376 | 349.740 | 100 | False | 0.108093 |

## Active-Set Gate

| candidate | promoted | rejection reasons | E gain vs t1_l0 | gain/cost |
|---|---|---|---:|---:|
| t1_l1_atomic_cross | False | intercept_baseline_not_beaten | 12.246 | 0.817717 |
| t2_edge_frame_projection | False | intercept_baseline_not_beaten | 2.505 | 0.0703445 |
| t2_edge_frame_plus_direct | False | intercept_baseline_not_beaten | 2.093 | 0.0557953 |
| t2_cavity_quadrupole | False | intercept_baseline_not_beaten | 0.878 | 0.0282026 |
| t2_cavity_vector | False | energy_gain_required, intercept_baseline_not_beaten | -3.577 | -0.114614 |
| t1_l2_atomic_cross | False | energy_gain_required, intercept_baseline_not_beaten | -12.162 | -0.4046 |

## Interpretation

1. Stage172 was partly numerically ill-conditioned: the best group-heldout residual RMSE falls from about `1235` meV/atom in Stage172 to about `94` meV/atom after intercept, standardization, and ridge sweep.
2. The corrected diagnostic is still negative for deployable path promotion: every candidate is worse than the intercept-only group-heldout baseline (`56.603` meV/atom RMSE).
3. This means the current semantic rTECE path descriptors can reduce error relative to the weak `t1_l0_species_radial` row, but they do not provide a useful unseen-case low-frequency energy coordinate.
4. The next TECE-aligned branch should stop promoting Stage171/172/173 winners as training architectures and instead add learnable low-frequency chemistry/site/front representations, then re-run E/F RMSE, E/F max, relative NEB, dimer, rattle/relax, and throughput scaling.
