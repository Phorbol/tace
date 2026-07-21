# Stage171 Residual Active-Set Results

Stage171 ran as Slurm job `688065` and completed with exit code `0:0` in `00:00:23`.

## Main Result

- target: `stage165-case-offset` / `meV_total`
- configs: `512` across `11` cases
- split: config-heldout with `energy_eval_stride=4`; this is not case-heldout.

| candidate | dim | E RMSE | E MAE | E max | descriptor residual |
|---|---:|---:|---:|---:|---:|
| t2_edge_frame_plus_direct | 338 | 0.469801 | 0.321583 | 2.06117 | 1.91973e-07 |
| t2_edge_frame_projection | 336 | 0.470047 | 0.321747 | 2.06205 | 2.75754e-07 |
| t2_cavity_quadrupole | 332 | 0.485453 | 0.328996 | 2.16146 | 0.0106391 |
| t2_cavity_vector | 331 | 0.485188 | 0.334216 | 2.00767 | 0.0108591 |
| t1_l2_atomic_cross | 330 | 0.471206 | 0.32803 | 1.86357 | 0.0111739 |
| t1_l1_atomic_cross | 315 | 0.575518 | 0.391074 | 2.50927 | 0.0270998 |
| t1_l0_species_radial | 300 | 0.564364 | 0.392341 | 2.45225 | 0.108093 |

## Interpretation

- The seen-case config-heldout residual can be fitted by deployable rTECE descriptors; best E RMSE is about `0.470 meV/atom`.
- `t1_l2_atomic_cross` has the strongest marginal gain per cost; `t2_edge_frame_plus_direct` gives the best absolute residual RMSE.
- This does not contradict Stage169: Stage169 tested case-level LOOCV and found current descriptors do not generalize offsets to unseen cases.
- Next priority is a case-heldout/group-LOOCV version of this active-set projection before turning these candidates into training rows.
