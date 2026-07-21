# Stage172 Group-Heldout Residual Active-Set Results

Stage172 ran as Slurm job `688116` and completed with exit code `0:0` in `00:03:16`.

## Main Result

- target: `stage165-case-offset` / Stage165 case-offset residual
- split: `group-loocv` by `case_id`
- configs: `512` across `11` groups
- uses group as feature: `False`

| candidate | dim | group-LOOCV E RMSE | E MAE | E max | descriptor residual |
|---|---:|---:|---:|---:|---:|
| t2_edge_frame_projection | 336 | 1234.926 | 449.970 | 4562.048 | 2.75754e-07 |
| t2_edge_frame_plus_direct | 338 | 1236.450 | 450.417 | 4567.714 | 1.91973e-07 |
| t2_cavity_quadrupole | 332 | 1392.079 | 503.278 | 5125.834 | 0.0106391 |
| t2_cavity_vector | 331 | 1589.391 | 561.920 | 5867.335 | 0.0108591 |
| t1_l2_atomic_cross | 330 | 1596.181 | 564.085 | 5894.602 | 0.0111739 |
| t1_l1_atomic_cross | 315 | 2506.573 | 839.453 | 9347.984 | 0.0270998 |
| t1_l0_species_radial | 300 | 2941.872 | 974.406 | 10951.203 | 0.108093 |

## Interpretation

- This is a negative result: Stage171 config-heldout residual projection did not survive group/case-heldout evaluation.
- Best group-LOOCV E RMSE is about `1235 meV/atom`, versus about `0.47 meV/atom` in Stage171 config-heldout.
- The active-set ranking is only relative to the weak `t1_l0_species_radial` baseline; it must not be used as a deployment architecture choice until an intercept/standardized/ridge-swept baseline is included.
- The result supports the Stage170 diagnosis: current rTECE descriptors do not provide a deployable low-frequency case/site/adsorbate energy coordinate.

## Next Priority

Stage173 should make the group-heldout projection statistically robust: add an intercept baseline, feature standardization, ridge sweep, and a require-beat-intercept active-set gate. If that still fails, the algorithmic branch should move to learnable low-frequency chemical/site representation modules instead of training the Stage171 winner.
