# Stage136 L2 Projection Diagnostic Results

- jobs: 64-config `685255` and smoke16 `685271`; both `COMPLETED 0:0`
- reference: Stage135 `L_A=2` atomic quadrupole path set, nrad12/species24/cross3
- candidate of interest: Stage132 `L_A=1` atomic subset
- method: descriptor projection plus held-out energy/force label projection, eval stride 4

## Valid DFT64

| candidate | dim | descriptor residual | E/atom RMSE | E underdetermined | F RMSE | F max | EF rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_l2_reference | 330 | 3.55698e-09 | 0.0274401 | True | 0.102403 | 1.12825 | 2 |
| drop_quadrupole_cross | 327 | 0.000739406 | 46.6956 | True | 0.102882 | 1.13157 | 4 |
| drop_quadrupole_norm | 318 | 0.010622 | 0.0188461 | True | 0.113776 | 1.23441 | 3 |
| stage132_l1_atomic | 315 | 0.0117352 | 0.0146767 | True | 0.1125 | 1.27496 | 1 |

## Train Mixed64

| candidate | dim | descriptor residual | E/atom RMSE | E underdetermined | F RMSE | F max | EF rank |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_l2_reference | 330 | 5.9839e-09 | 228.251 | True | 0.265764 | 4.94995 | 4 |
| drop_quadrupole_cross | 327 | 0.00259031 | 139.035 | True | 0.269275 | 4.9779 | 2 |
| drop_quadrupole_norm | 318 | 0.0294524 | 54.7345 | True | 0.266105 | 4.87601 | 1 |
| stage132_l1_atomic | 315 | 0.0320434 | 44.066 | True | 0.269599 | 4.96001 | 3 |

## Interpretation

On valid DFT64, the full L2 reference has the best force projection: F RMSE `0.1024` versus `0.1125` for the Stage132 L1 subset, and F max `1.128` versus `1.275`. That means the L2 quadrupole paths do contain force-response information in the current active coordinate system.

This resolves the Stage135 negative result: naive L2 training failed to reach the Pareto frontier, but not because L2 information is absent. The likely issue is optimization/distillation/regularization or learned-front conditioning: the student can fit lower validation loss yet produce worse force tails and lower throughput.

Energy projection in this run is underdetermined because descriptor dimension is much larger than the number of fit configs. The energy ranks should be treated as weak evidence until rerun with larger fit sets or stronger low-rank/ridge structure.

## Next Priority

Do not simply discard L2, and do not continue naive full-L2 rows. The next TECE-aligned architecture step should retain L2 force information through a lower-rank or conditioned front representation, or add projection/Sobolev regularization during distillation, then re-evaluate RMSE/max/physical/throughput gates.
