# rTECE stage128 physical triage results

## Design intent

Stage128 follows the stage127 split between a force-RMSE branch and an energy/robustness branch. It evaluates the three stage127 non-dominated candidates with the existing rTECE physical-test tools: CHNO dimer scans over 0.5-5.0 covalent-radius scale and C_or_N-window rattle+relax on validation configs 58:66. The stage keeps the architecture fixed and tests whether the RMSE/throughput candidates remain reasonable under physical external diagnostics.

## Completed sbatch jobs

| job | row | state | elapsed |
| --- | --- | --- | ---: |
| 685054 | l1_active_nrad12_species20_radial_species8_cross3_h64 | COMPLETED 0:0 | 00:00:13 |
| 685055 | l1_active_nrad12_species24_radial_species8_cross3_h64 | COMPLETED 0:0 | 00:00:12 |
| 685056 | l1_active_nrad12_species24_radial_species8_cross4_h64 | COMPLETED 0:0 | 00:00:11 |

## Physical Pareto metrics

| row | phys score | gate | bench | dimer | rattle | F RMSE | E RMSE | F max | E max | C/N RMSD | rattle max fmax | atoms/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| l1_active_nrad12_species24_radial_species8_cross3_h64 | 13.897 | 0 | 1 | 1 | 0 | 99.480 | 307.299 | 2429.936 | 671.251 | 0.112 | 9.971 | 1232241.120 |
| l1_active_nrad12_species20_radial_species8_cross3_h64 | 14.580 | 0 | 1 | 1 | 0 | 104.793 | 277.925 | 2263.224 | 649.159 | 0.107 | 10.809 | 1288754.340 |
| l1_active_nrad12_species24_radial_species8_cross4_h64 | 14.743 | 0 | 1 | 1 | 0 | 103.283 | 277.843 | 2355.617 | 609.100 | 0.118 | 10.971 | 1224723.791 |

## Physical Pareto front

| row | phys score | atoms/s |
| --- | ---: | ---: |
| l1_active_nrad12_species20_radial_species8_cross3_h64 | 14.580 | 1288754.340 |
| l1_active_nrad12_species24_radial_species8_cross3_h64 | 13.897 | 1232241.120 |

## Interpretation

1. The dimer test no longer identifies a short-range physical failure for these active-set models. All three candidates have 7/7 short-range repulsive dimer pairs, no nonfinite energy/force values, and positive short-minus-long energy lift for C/N/CHNO pairs.

2. The earlier C/N RMSD concern is not the dominant failure in this stage128 window. Mean C_or_N rattle final RMSD is 0.107 A for species20/cross3, 0.112 A for species24/cross3, and 0.118 A for species24/cross4. These are all below the 0.35 A stage128 focus threshold and below the historical 0.247 A concern.

3. The remaining physical gate failure is force-spike based, not RMSD based. All three rows fail rattle_gate because max fmax is about 10 eV/A, far above the 1.0 eV/A diagnostic threshold. This indicates that the next physical task should analyze which rattled configs create spikes and whether teacher-labeled rattle distillation reduces them, rather than treating C/N RMSD as a hard architecture blocker.

4. species24/cross3 remains the best balanced row. It has the lowest physical score, best DFT/teacher force RMSE, and the lowest rattle max fmax among the three candidates. species20/cross3 remains a throughput Pareto row, because it is faster and has slightly lower C/N RMSD but worse force RMSE and larger force spikes.

5. species24/cross4 is not supported by this physical triage despite its better energy RMSE/E max from stage127. It is dominated in physical score and throughput by species24/cross3, so cross4 should not become the default architecture unless a later energy-specific deployment objective requires it.

## Next priority

Stage129 should follow the TECE document's projection/distillation argument: keep the species24/cross3 and species20/cross3 active-set architectures fixed, add teacher-labeled rattle configurations around the stage128 stress window, and test whether the force-spike/rattle gate improves without C/N-specific architecture changes. This separates data/distribution error from intrinsic representation bias and keeps the path aligned with systematic TECE downfolding rather than ad hoc element specialization.
