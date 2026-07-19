# rTECE stage125 front capacity ladder results

## Design intent

Stage125 follows the stage124 priority update: edge-only residual sketches were not a clean Pareto improvement, while the useful signal came from coupled atomic+edge downfolding. This stage therefore keeps the L1 active atomic backbone only and moves capacity into front representation rank: radial rank, learnable species basis rank, center-neighbor radial-species adapter rank, learnable cross-radial projection rank, and bottleneck rows as low-rank path mixers. The final scalar head width remains fixed at `64,64`.

## Completed sbatch jobs

| job | row | state |
| --- | --- | --- |
| 685034 | l1_active_species24_radial_species12_cross4_h64 | COMPLETED 0:0 |
| 685032 | l1_active_species32_radial_species16_cross4_h64 | COMPLETED 0:0 |
| 685031 | l1_active_species32_radial_species16_cross4_bneck32_h64 | COMPLETED 0:0 |
| 685033 | l1_active_species48_radial_species24_cross5_bneck48_h64 | COMPLETED 0:0 |

## DFT benchmark metrics

Stage123 L1 active anchor for reference: F RMSE 112.838 meV/A, E RMSE 335.816 meV/atom, F max 1996.322 meV/A, throughput 1.669M atoms/s.
Stage124 best edge tradeoff for reference: F RMSE 110.805 meV/A, E RMSE 310.071 meV/atom, F max 2251.594 meV/A, throughput 0.535M atoms/s.

| row | params | repr params | readout params | F RMSE | F MAE | F max | E RMSE | E MAE | E bias | E max | atoms/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| l1_active_species24_radial_species12_cross4_h64 | 30033 | 5328 | 24705 | 107.444 | 45.945 | 2394.173 | 302.809 | 233.133 | -80.893 | 717.002 | 1206744.735 |
| l1_active_species32_radial_species16_cross4_h64 | 37889 | 7040 | 30849 | 108.006 | 44.942 | 2359.164 | 318.058 | 254.581 | -32.005 | 696.753 | 1060022.066 |
| l1_active_species32_radial_species16_cross4_bneck32_h64 | 26721 | 20320 | 6401 | 109.921 | 45.886 | 2519.195 | 314.024 | 258.848 | -0.137 | 674.847 | 1062288.241 |
| l1_active_species48_radial_species24_cross5_bneck48_h64 | 57153 | 49728 | 7425 | 108.399 | 48.380 | 1836.125 | 322.994 | 262.089 | -20.645 | 734.968 | 734186.960 |

## Teacher-label benchmark metrics

| row | F RMSE | F MAE | F max | E RMSE | E MAE | E bias | E max | atoms/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| l1_active_species24_radial_species12_cross4_h64 | 104.718 | 47.414 | 2436.302 | 288.979 | 218.853 | -84.869 | 708.370 | 1209846.421 |
| l1_active_species32_radial_species16_cross4_h64 | 104.478 | 46.084 | 2401.294 | 302.454 | 239.888 | -35.982 | 686.536 | 1058501.729 |
| l1_active_species32_radial_species16_cross4_bneck32_h64 | 108.458 | 47.739 | 2187.266 | 296.669 | 243.486 | -4.113 | 633.607 | 1064113.036 |
| l1_active_species48_radial_species24_cross5_bneck48_h64 | 104.962 | 49.316 | 1659.003 | 308.003 | 247.236 | -24.622 | 724.751 | 733871.939 |

## Training controls

All four rows used 2048 train configs, 256 validation configs, batch size 8, 20000 max steps, LR warmup 500, plateau LR scheduling, and early-stopping patience 400. Best checkpoints landed between 18176 and 19968 steps, so these are real training runs rather than short smoke tests.

## Interpretation

1. Front capacity is a better next axis than edge-only residuals. The best stage125 row, `l1_active_species24_radial_species12_cross4_h64`, improves DFT F RMSE to 107.444 meV/A and E RMSE to 302.809 meV/atom while retaining 1.207M atoms/s. That is better F/E RMSE than the stage123 anchor and much faster than the stage124 all-scope edge tradeoff.

2. Bigger is not monotonically better. Increasing to species32/radial_species16 improves neither F RMSE nor E RMSE relative to species24/radial_species12, and the largest bottleneck row improves F max error substantially but gives worse average force/energy RMSE and lower throughput. This is consistent with active-set/downfolding: the useful degrees of freedom are sparse and rank-limited, not simply proportional to parameter count.

3. The bottleneck rows are useful diagnostics, not current winners. `bneck32` removes much of the readout parameter growth and keeps throughput near 1.06M atoms/s, but it loses RMSE versus the non-bottleneck species24 row. `bneck48` has the best DFT F max error at 1836.125 meV/A, which may matter for stability/relaxation, but its E RMSE and throughput are worse.

4. Current Pareto interpretation: for RMSE-focused selection, `l1_active_species24_radial_species12_cross4_h64` is the new stage125 candidate. For physical robustness/max-force selection, the largest bottleneck row deserves dimer/rattle follow-up because it lowers force max error below the stage123 and stage124 references.

## Next priority

The next stage should not blindly increase rank. It should take the stage125 species24/radial_species12/cross4 row as the RMSE anchor and test controlled distillation/data enrichment: teacher-labeled rattle/relax trajectories and dimer scans for the C/N sensitivity issue, plus a small rank-neighborhood sweep around species16-24 and cross3-4. This keeps the search tied to TECE active-set downfolding instead of drifting into heuristic width tuning.
