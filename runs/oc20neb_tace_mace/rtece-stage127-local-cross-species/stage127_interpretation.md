# rTECE stage127 local cross/species results

## Design intent

Stage127 is a local TECE active-set/downfolding sweep around the stage126 winner. It keeps the L1 active atomic paths, `num_radial=12`, `radial_species_adapter_channels=8`, `radial_species_adapter_scope=all`, no edge paths, ZBL baseline, and a fixed `64,64` scalar head. The only intended architecture coordinates are the learnable species rank and the cross-radial invariant rank.

This directly follows the TECE design-space rule that the student should be selected by semantic path/rank projection under hardware cost, not by unconstrained readout widening.

## Completed sbatch jobs

| job | row | state | elapsed |
| --- | --- | --- | ---: |
| 685045 | l1_active_nrad12_species20_radial_species8_cross3_h64 | COMPLETED 0:0 | 00:08:40 |
| 685046 | l1_active_nrad12_species24_radial_species8_cross2_h64 | COMPLETED 0:0 | 00:08:22 |
| 685047 | l1_active_nrad12_species24_radial_species8_cross3_h64 | COMPLETED 0:0 | 00:08:37 |
| 685048 | l1_active_nrad12_species24_radial_species8_cross4_h64 | COMPLETED 0:0 | 00:08:46 |

## DFT benchmark metrics

Stage126 force-RMSE winner reference: `l1_active_nrad12_species24_radial_species8_cross3_h64`, F RMSE 99.480 meV/A, E RMSE 307.299 meV/atom, F max 2429.937 meV/A, E max 671.251 meV/atom, throughput 1.233M atoms/s.

| row | params | F RMSE | F MAE | F max | E RMSE | E MAE | E bias | E max | atoms/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| l1_active_nrad12_species20_radial_species8_cross3_h64 | 25449 | 104.793 | 43.585 | 2263.224 | 277.925 | 210.098 | -21.797 | 649.159 | 1288754.340 |
| l1_active_nrad12_species24_radial_species8_cross2_h64 | 28785 | 104.586 | 42.161 | 2121.428 | 307.950 | 219.985 | -81.562 | 737.737 | 1232145.810 |
| l1_active_nrad12_species24_radial_species8_cross3_h64 | 28925 | 99.480 | 42.842 | 2429.936 | 307.299 | 242.118 | -32.523 | 671.251 | 1232241.120 |
| l1_active_nrad12_species24_radial_species8_cross4_h64 | 29129 | 103.283 | 42.346 | 2355.617 | 277.843 | 211.358 | -24.691 | 609.100 | 1224723.791 |

## Teacher-label benchmark metrics

| row | F RMSE | F MAE | F max | E RMSE | E MAE | E bias | E max | atoms/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| l1_active_nrad12_species20_radial_species8_cross3_h64 | 102.635 | 44.671 | 2465.867 | 261.559 | 194.735 | -25.773 | 607.387 | 1286338.436 |
| l1_active_nrad12_species24_radial_species8_cross2_h64 | 101.579 | 43.710 | 2163.557 | 292.168 | 205.414 | -85.538 | 727.520 | 1224869.990 |
| l1_active_nrad12_species24_radial_species8_cross3_h64 | 97.057 | 44.196 | 2472.063 | 289.699 | 226.756 | -36.499 | 661.449 | 1232301.820 |
| l1_active_nrad12_species24_radial_species8_cross4_h64 | 100.946 | 44.039 | 2397.744 | 260.816 | 195.902 | -28.667 | 599.889 | 1224892.735 |

## Training controls

| row | best step | best valid loss | steps | train configs | valid configs | scheduler | warmup | early stop patience |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| l1_active_nrad12_species20_radial_species8_cross3_h64 | 19712 | 0.115125 | 20000 | 2048 | 256 | plateau | 500 | 400 |
| l1_active_nrad12_species24_radial_species8_cross2_h64 | 16896 | 0.099780 | 20000 | 2048 | 256 | plateau | 500 | 400 |
| l1_active_nrad12_species24_radial_species8_cross3_h64 | 19456 | 0.094367 | 20000 | 2048 | 256 | plateau | 500 | 400 |
| l1_active_nrad12_species24_radial_species8_cross4_h64 | 17664 | 0.086091 | 20000 | 2048 | 256 | plateau | 500 | 400 |

## Interpretation

1. The stage126 cross3 row remains the best force-RMSE point in this local neighborhood: DFT F RMSE 99.480 meV/A at 1.232M atoms/s. This supports the current active-set/downfolding choice that three cross-radial invariant sketches are a good force-oriented rank near species24/adapter8.

2. cross4 is not a force-RMSE improvement, but it improves energy metrics strongly: DFT E RMSE drops from 307.299 to 277.843 meV/atom and E max from 671.251 to 609.100 meV/atom, while throughput changes only from 1.232M to 1.225M atoms/s. This means cross4 should not be discarded globally; it is an energy/robustness candidate even though it is not the force-RMSE front.

3. species20/cross3 is a higher-throughput energy-strong downrank row: 1.289M atoms/s with E RMSE 277.925 meV/atom and E max 649.159 meV/atom, but F RMSE worsens to 104.793 meV/A. It is a plausible deployment row if force RMSE can tolerate about 5.3 meV/A degradation versus the stage126 force-RMSE winner.

4. cross2 is dominated in this batch. It has similar throughput to cross3 but worse force RMSE, worse energy RMSE, worse energy bias, and worse E max. The second-to-third cross-radial sketch is therefore not redundant under this training/data setting.

5. None of the four rows early-stopped before max steps; best validation checkpoints landed between 16896 and 19712 steps. The run therefore confirms the intended 20k-step training control, but also suggests the active-set rows are still optimization-limited enough that the next comparison should keep training controls fixed or use a longer-data distillation stage before making final architecture pruning decisions.

## Next priority

Use stage127 to split the Pareto front into two candidate branches: a force-RMSE branch retaining species24/cross3, and an energy/robustness branch comparing species20/cross3 against species24/cross4 with physical external tests. The next algorithmic step should not widen the final MLP; it should either add teacher-labeled rattle/relax data under the same active-set rows, or introduce a documented front representation path that can explain the energy-force tradeoff, then evaluate DFT/teacher RMSE, max errors, dimer smoothness, rattle/relax RMSD, and atom-count throughput scaling.
