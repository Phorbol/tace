# rTECE stage124 residual edge ladder results

## Design intent

Stage124 follows the stage123 priority update: keep the L1 active atomic backbone and add edge-relational scalar sketches as residual paths. This isolates the marginal value of edge paths instead of replacing the atomic active set. The tested ladder is `edge.direct.radial`, then `edge.cavity.vector_dot`, then `edge.cavity.quadrupole_frobenius`, with an `edge` vs `all` radial species adapter scope comparison at the richest cavity row.

This remains aligned with TECE_design_space and rTECE_review: path density, angular bandwidth, and path-scoped trainable representation are treated as explicit compiler axes; final MLP head width is held fixed at `64,64`.

## Completed sbatch jobs

| job | row | state | scope |
| --- | --- | --- | --- |
| 685018 | l1_active_edge_direct_radial_species8_h64 | COMPLETED 0:0 | edge |
| 685017 | l1_active_edge_cavity_vec_radial_species8_h64 | COMPLETED 0:0 | edge |
| 685015 | l1_active_edge_cavity_vecq_radial_species8_h64 | COMPLETED 0:0 | edge |
| 685016 | l1_active_all_cavity_vecq_radial_species8_h64 | COMPLETED 0:0 | all |

## DFT benchmark metrics

Stage123 L1 active anchor for reference: F RMSE 112.838 meV/A, E RMSE 335.816 meV/atom, F max 1996.322 meV/A, throughput 1.669M atoms/s.

| row | params | F RMSE meV/A | F MAE meV/A | F max meV/A | E RMSE meV/atom | E MAE meV/atom | E bias meV/atom | E max meV/atom | atoms/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| l1_active_edge_direct_radial_species8_h64 | 17337 | 117.972 | 50.803 | 2857.474 | 333.827 | 240.884 | -85.513 | 798.850 | 1054341.264 |
| l1_active_edge_cavity_vec_radial_species8_h64 | 17401 | 118.984 | 51.608 | 2243.217 | 335.695 | 252.302 | -8.358 | 788.558 | 773901.670 |
| l1_active_edge_cavity_vecq_radial_species8_h64 | 17465 | 121.113 | 53.558 | 2294.011 | 333.999 | 240.578 | -79.644 | 816.838 | 558262.272 |
| l1_active_all_cavity_vecq_radial_species8_h64 | 17465 | 110.805 | 44.344 | 2251.594 | 310.071 | 251.454 | -15.370 | 692.070 | 535106.630 |

## Teacher-label benchmark metrics

| row | F RMSE meV/A | F MAE meV/A | F max meV/A | E RMSE meV/atom | E MAE meV/atom | E bias meV/atom | E max meV/atom | atoms/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| l1_active_edge_direct_radial_species8_h64 | 115.284 | 51.671 | 2485.130 | 318.042 | 230.190 | -89.489 | 791.859 | 1053995.818 |
| l1_active_edge_cavity_vec_radial_species8_h64 | 117.229 | 52.831 | 1924.466 | 318.513 | 236.184 | -12.334 | 747.317 | 773628.492 |
| l1_active_edge_cavity_vecq_radial_species8_h64 | 118.531 | 54.420 | 1921.664 | 318.599 | 230.307 | -83.620 | 809.847 | 557134.099 |
| l1_active_all_cavity_vecq_radial_species8_h64 | 107.851 | 46.061 | 2073.506 | 292.590 | 236.091 | -19.346 | 650.829 | 535294.999 |

## Interpretation

1. Edge-scope residual paths alone are not enough. `edge.direct.radial`, `edge.cavity.vector_dot`, and `edge.cavity.vector_dot+quadrupole` all have worse DFT F RMSE than the stage123 L1 active anchor and lower throughput. Direct edge radial slightly improves E RMSE/MAE but degrades force RMSE, force max error, and throughput, so it is not a clean Pareto improvement.

2. The `all`-scope cavity vector+quadrupole row is a real accuracy-throughput tradeoff point. It improves DFT F RMSE from 112.838 to 110.805 meV/A and E RMSE from 335.816 to 310.071 meV/atom, but throughput drops from 1.669M to 0.535M atoms/s and F max error worsens from 1996.322 to 2251.594 meV/A.

3. The scope comparison is the important signal. The same richest edge residual path with `edge` scope is poor (F RMSE 121.113, E RMSE 333.999, 0.558M atoms/s), while `all` scope is substantially better. This suggests the edge residual path only helps when the trainable radial species adapter also remains active on the atomic backbone; the useful representation is coupled atomic+edge downfolding, not edge-only correction.

4. For the main Pareto search, edge relational sketches are not the high-throughput endpoint. They may be a mid-accuracy point, but their current cost is too high for the NEP/DPA1-style target. The next architecture priority should be a coupled frontloaded representation ladder: increase trainable atomic/radial/species capacity into the tens-of-k parameters while retaining sparse path semantics, then add only the minimal edge residual row if RMSE warrants the throughput cost.

## Next priority

Stage125 should not keep expanding edge sketches blindly. The clean next step is a capacity ladder on the L1 active atomic backbone: larger radial species adapter rank, learnable species basis rank, and optionally a small learnable low-rank moment/channel mixer before scalarization. The acceptance criterion should be E/F RMSE and max error vs atoms/s, plus later dimer/rattle physical tests for the surviving Pareto candidates.
