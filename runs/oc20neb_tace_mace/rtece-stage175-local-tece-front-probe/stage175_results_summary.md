# Stage175 Local TECE Front Probe Results

- split: `group-loocv` by `case_id`
- target: `stage165_case_offset_residual_mev_atom`
- intercept baseline: `56.60319287094426` meV/atom RMSE
- recommendation: `promote_best_supported_local_tece_front_module`

| feature family | features | Lmax | rank | E RMSE | E MAE | E max | selected ridge | beats intercept |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| local_l0_rank3 | 90 | 0 | 3 | 44.547 | 35.694 | 103.368 | 100 | True |
| local_l0_l1_l2_rank3 | 180 | 2 | 3 | 53.855 | 46.318 | 125.862 | 100 | True |
| local_l0_l1_rank3 | 135 | 1 | 3 | 55.418 | 46.577 | 165.814 | 100 | True |
| local_l0_l1_l2_rank4 | 320 | 2 | 4 | 67.832 | 54.951 | 230.337 | 100 | False |

## Interpretation

At least one local TECE front feature family beats the intercept baseline; the best supported row is a candidate for the next trainable rTECE front module.

## Next Priority

- Promote the best supported local front only as an architecture candidate, not as a post-hoc correction.
- Start with the low-rank L0 local density front; the fixed L1/L2 additions did not improve the group-heldout energy-offset RMSE enough to justify their extra feature/cost burden.
- Convert the diagnostic all-distance implementation into a normal neighbor-pass trainable rTECE front before any throughput claim.
- Re-test the promoted front with full DFT/teacher E/F RMSE, E/F max, relative NEB, dimer scan, rattle-relax, and atom-count throughput scaling.
