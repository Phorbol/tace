# Stage152 Train-Side E/F Active-Set Projection Summary

Stage152 follows Stage151. Stage151 showed that DFT-valid absolute energy is easy for the Stage143 semantic basis, but train-side absolute energies are not: full-reference projection on mixed-train structures remains around `93-102 meV/atom` for DFT, teacher, and mixed labels. Stage152 adds force-Jacobian projection on the train manifold so path selection is based on a Sobolev E/F target rather than energy-only or valid-only evidence.

## Jobs

| job | role | state | exit | elapsed | status |
|---:|---|---|---|---|---|
| 687449 | DFT E/F projection, limit128 | COMPLETED | 0:0 | 00:04:38 | force smoke only; energy underdetermined |
| 687450 | mixed E/F projection, limit128 | COMPLETED | 0:0 | 00:04:37 | force smoke only; energy underdetermined |
| 687451 | teacher E/F projection, limit128 | COMPLETED | 0:0 | 00:04:36 | force smoke only; energy underdetermined |
| 687452 | DFT E/F projection, limit512 | COMPLETED | 0:0 | 00:17:10 | primary |
| 687453 | mixed E/F projection, limit512 | COMPLETED | 0:0 | 00:17:24 | primary |
| 687454 | teacher E/F projection, limit512 | COMPLETED | 0:0 | 00:17:06 | primary |

## Method Note

The first `limit128` E/F smoke is useful for force rows, but its graph-level energy projection is underdetermined: the fit split has only about 96 structures while candidate dimensions are around 300-334 plus element-count baselines. Stage152 therefore treats the `limit512` reruns as the primary E/F active-set evidence. At `limit512`, every primary row has `energy_underdetermined=false`; the full-reference candidate has 12 energy degrees of freedom after the element baseline.

## Limit512 Primary Results

Energy errors are eV/atom; force errors are eV/A.

### Manual Candidates

| target | candidate | dim | E/atom RMSE | F RMSE | energy rank | force rank | combined rank |
|---|---|---:|---:|---:|---:|---:|---:|
| DFT | `t2_l0_species_radial` | 300 | 0.13415 | 0.41078 | 7 | 19 | 18 |
| DFT | `t2_l1_atomic_cross` | 315 | 0.16043 | 0.38695 | 18 | 8 | 17 |
| DFT | `t2_l2_atomic_cross` | 330 | 0.13686 | 0.38309 | 9 | 6 | 3 |
| DFT | `t3_l2_cavity_vector` | 331 | 0.13748 | 0.38297 | 12 | 4 | 7 |
| DFT | `t3_full_reference` | 334 | 0.12931 | 0.39715 | 3 | 12 | 6 |
| teacher | `t2_l0_species_radial` | 300 | 0.11520 | 0.40692 | 6 | 19 | 18 |
| teacher | `t2_l1_atomic_cross` | 315 | 0.13599 | 0.38345 | 18 | 8 | 17 |
| teacher | `t2_l2_atomic_cross` | 330 | 0.11958 | 0.37944 | 9 | 6 | 4 |
| teacher | `t3_l2_cavity_vector` | 331 | 0.12738 | 0.37932 | 14 | 4 | 10 |
| teacher | `t3_full_reference` | 334 | 0.11430 | 0.39342 | 4 | 12 | 6 |
| mixed | `t2_l0_species_radial` | 300 | 0.11862 | 0.40772 | 6 | 19 | 18 |
| mixed | `t2_l1_atomic_cross` | 315 | 0.14120 | 0.38416 | 18 | 8 | 17 |
| mixed | `t2_l2_atomic_cross` | 330 | 0.12246 | 0.38018 | 10 | 6 | 4 |
| mixed | `t3_l2_cavity_vector` | 331 | 0.12849 | 0.38006 | 13 | 4 | 8 |
| mixed | `t3_full_reference` | 334 | 0.11653 | 0.39419 | 4 | 12 | 7 |

### Best Combined Rows

| target | best combined candidate | dim | E/atom RMSE | F RMSE | note |
|---|---|---:|---:|---:|---|
| DFT | `prefix_006` / `t2_l2_atomic_cross` | 330 | 0.13686 | 0.38309 | robust low-cost atomic L2 front |
| teacher | `single_delete_edge_direct_radial` | 332 | 0.11928 | 0.37903 | includes edge cavity paths, so higher hardware cost |
| mixed | `single_delete_edge_direct_radial` | 332 | 0.12245 | 0.37978 | similar to teacher, but costlier than T2 L2 |

## Interpretation

- The train-side absolute-energy difficulty persists when the energy projection is no longer underdetermined. Mixed full-reference E/atom RMSE is `0.11653 eV/atom`, matching the earlier Stage143 train-side signal.
- Force projection is much less sensitive to full edge-relational reference paths than energy. The force-best region is around `0.379-0.383 eV/A`, and `t2_l2_atomic_cross` is close to the best force rows without edge buffers.
- `t3_full_reference` improves energy slightly over `t2_l2_atomic_cross` on mixed/teacher labels, but worsens force RMSE and carries more edge-relational cost. It is not a clean next Pareto point.
- `t3_l2_cavity_vector` also does not justify its throughput hit here: it has similar force to T2 L2, worse energy, and requires edge-relational work.
- The best combined teacher/mixed row, `single_delete_edge_direct_radial`, keeps edge cavity paths but drops direct radial. This is scientifically interesting but not yet a deployment recommendation, because the marginal hardware cost of the retained edge paths must beat the small force gain over T2 L2.

## Next Priority

1. Treat `t2_l2_atomic_cross` / `prefix_006` as the current low-cost semantic front for the next supervised/distilled control. It is the stable E/F compromise across DFT, teacher, and mixed targets.
2. Run a DFT-only supervised control with the same production training framework for this T2 L2 front. This directly tests whether the poor Stage149 energy is due to mixed objective/data distribution or unavoidable architecture projection.
3. Keep edge-relational candidates as a measured optional increment, not the default next step. If retested, benchmark `single_delete_edge_direct_radial`-style retained cavity paths against T2 L2 with real throughput and physical probes.
4. Do not widen only the final MLP head. Any capacity increase should target front-loaded TECE representation: learnable radial/channel projection, species-conditioned radial adapters, or Schur-selected scalar paths.
5. Continue to report E RMSE/MAE/max, F RMSE/MAE/max, relative NEB/barrier, dimer smoothness, and rattle-relax metrics separately.
