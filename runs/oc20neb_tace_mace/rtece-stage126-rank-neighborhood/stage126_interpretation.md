# rTECE stage126 rank-neighborhood results

## Design intent

Stage126 follows the stage125 result that front representation capacity is useful but not monotonic. Instead of increasing model size, this stage performs a lower-rank active-set neighborhood around the stage125 anchor. It keeps the L1 active atomic paths and fixed `64,64` scalar head, removes edge paths, fixes cross-radial rank to 3, and varies only `num_radial`, learnable species basis rank, and radial-species adapter rank.

## Completed sbatch jobs

| job | row | state |
| --- | --- | --- |
| 685037 | l1_active_nrad10_species16_radial_species8_cross3_h64 | COMPLETED 0:0 |
| 685038 | l1_active_nrad12_species16_radial_species8_cross3_h64 | COMPLETED 0:0 |
| 685040 | l1_active_nrad12_species24_radial_species8_cross3_h64 | COMPLETED 0:0 |
| 685041 | l1_active_nrad12_species16_radial_species12_cross3_h64 | COMPLETED 0:0 |
| 685039 | l1_active_nrad12_species24_radial_species12_cross3_h64 | COMPLETED 0:0 |

## DFT benchmark metrics

Stage125 anchor reference: F RMSE 107.444 meV/A, E RMSE 302.809 meV/atom, F max 2394.173 meV/A, E max 717.002 meV/atom, throughput 1.207M atoms/s.

| row | params | repr params | readout params | F RMSE | F MAE | F max | E RMSE | E MAE | E bias | E max | atoms/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| l1_active_nrad10_species16_radial_species8_cross3_h64 | 19587 | 3522 | 16065 | 118.227 | 48.317 | 1899.649 | 301.989 | 243.178 | 32.292 | 711.039 | 1506841.338 |
| l1_active_nrad12_species16_radial_species8_cross3_h64 | 21973 | 3604 | 18369 | 106.524 | 43.465 | 2023.923 | 293.286 | 222.399 | -23.585 | 667.404 | 1365038.095 |
| l1_active_nrad12_species24_radial_species8_cross3_h64 | 28925 | 4412 | 24513 | 99.480 | 42.842 | 2429.937 | 307.299 | 242.118 | -32.523 | 671.251 | 1232755.256 |
| l1_active_nrad12_species16_radial_species12_cross3_h64 | 22877 | 4508 | 18369 | 103.663 | 44.938 | 2403.838 | 316.104 | 252.266 | -63.361 | 647.757 | 1352510.949 |
| l1_active_nrad12_species24_radial_species12_cross3_h64 | 29829 | 5316 | 24513 | 104.993 | 44.368 | 1935.396 | 315.348 | 244.944 | -23.355 | 739.728 | 1214485.203 |

## Teacher-label benchmark metrics

| row | F RMSE | F MAE | F max | E RMSE | E MAE | E bias | E max | atoms/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| l1_active_nrad10_species16_radial_species8_cross3_h64 | 115.985 | 49.634 | 2066.347 | 285.468 | 229.567 | 28.316 | 669.799 | 1508154.299 |
| l1_active_nrad12_species16_radial_species8_cross3_h64 | 104.317 | 44.908 | 1925.458 | 277.160 | 211.781 | -27.561 | 641.871 | 1363521.153 |
| l1_active_nrad12_species24_radial_species8_cross3_h64 | 97.057 | 44.196 | 2472.065 | 289.699 | 226.755 | -36.499 | 661.449 | 1232327.680 |
| l1_active_nrad12_species16_radial_species12_cross3_h64 | 102.169 | 46.631 | 2064.376 | 300.775 | 239.973 | -67.337 | 637.540 | 1351756.497 |
| l1_active_nrad12_species24_radial_species12_cross3_h64 | 102.661 | 45.708 | 2018.041 | 299.905 | 231.698 | -27.331 | 729.511 | 1216012.828 |

## Training controls

All five rows used 2048 train configs, 256 validation configs, batch size 8, 20000 max steps, LR warmup 500, plateau LR scheduling, and early-stopping patience 400. Best checkpoints landed between 12800 and 19968 steps.

## Interpretation

1. Stage126 finds a better RMSE Pareto point than stage125. `l1_active_nrad12_species24_radial_species8_cross3_h64` reaches DFT F RMSE 99.480 meV/A and E RMSE 307.299 meV/atom at 1.233M atoms/s with 28.9k parameters. Compared with the stage125 anchor, force RMSE improves strongly and throughput is slightly higher, while energy RMSE is only modestly worse.

2. The useful capacity is species rank, not radial-species adapter rank. Holding `num_radial=12` and `cross3`, increasing species rank from 16 to 24 with adapter rank 8 improves F RMSE from 106.524 to 99.480. Increasing adapter rank from 8 to 12 at species16 worsens F RMSE to 103.663 and energy RMSE to 316.104. At species24, adapter rank 12 is also worse than adapter rank 8.

3. Radial rank 10 is too aggressive for average force RMSE but useful for max-force robustness. The nrad10/species16/adapter8 row has F RMSE 118.227 but F max 1899.649, lower than the stage125 anchor and close to the stage123 anchor. This suggests it may be stable for some dynamics but is not the RMSE-frontier candidate.

4. Cross-radial rank 3 is enough in this neighborhood. The best stage126 row uses cross3 and beats the stage125 cross4 anchor on force RMSE and throughput. This is direct evidence for active-set downfolding: deleting the fourth cross-radial sketch is not costly here and may improve generalization.

## Next priority

The next architecture sweep should be even more local: keep `num_radial=12`, `species_basis_channels=24`, `radial_species_adapter_channels=8`, compare cross2/cross3/cross4 and a smaller species20 row. In parallel, the physical-validation branch should test the stage126 winner and the nrad10 max-force row on dimer scans and teacher-labeled rattle/relax trajectories, especially for C/N adsorbates.
