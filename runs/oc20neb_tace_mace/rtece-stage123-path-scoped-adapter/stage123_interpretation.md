# rTECE stage123 path-scoped radial species adapter results

## Design intent

Stage123 tests the TECE/rTECE design-space question that stage122 left ambiguous: the trainable radial species adapter is no longer only a global front-end knob. It is now an explicit path-group allocation parameter (`radial_species_adapter_scope = all | atomic | edge`) in the production rTECE config, manifest, Lightning/CLI training path, and Slurm wrapper path.

This follows the TECE_design_space compiler/downfolding view: capacity should be allocated to a selected TECE subspace, not merely added by widening the final scalar head. It also responds to rTECE_review: the fixed-feature bottleneck and missing edge-relational semantics must be tested as controlled architecture axes.

## Completed sbatch jobs

| job | row | state | scope | role |
| --- | --- | --- | --- | --- |
| 684995 | l1_active_atomic_radial_species8_h64 | COMPLETED 0:0 | atomic | stage122 regression anchor; only atomic paths |
| 684997 | l1_active_all_radial_species8_h64 | COMPLETED 0:0 | all | stage122 global-adapter anchor; only atomic paths, so equivalent to atomic scope |
| 684996 | l2_cavity_edge_radial_species8_h64 | COMPLETED 0:0 | edge | edge-relational/cavity sketch probe |

## DFT benchmark metrics

| row | params | F RMSE meV/A | F MAE meV/A | F max meV/A | E RMSE meV/atom | E MAE meV/atom | E bias meV/atom | E max meV/atom | atoms/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| l1_active_atomic_radial_species8_h64 | 17209 | 112.838 | 46.315 | 1996.323 | 335.816 | 268.928 | -41.688 | 731.923 | 1670311.742 |
| l1_active_all_radial_species8_h64 | 17209 | 112.838 | 46.315 | 1996.322 | 335.816 | 268.928 | -41.688 | 731.924 | 1668735.197 |
| l2_cavity_edge_radial_species8_h64 | 16737 | 118.413 | 50.711 | 2759.700 | 357.834 | 256.972 | -88.221 | 837.338 | 580130.373 |

## Teacher-label benchmark metrics

| row | F RMSE meV/A | F MAE meV/A | F max meV/A | E RMSE meV/atom | E MAE meV/atom | E bias meV/atom | E max meV/atom | atoms/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| l1_active_atomic_radial_species8_h64 | 110.166 | 47.587 | 2065.189 | 318.918 | 255.361 | -45.664 | 721.706 | 1668511.114 |
| l1_active_all_radial_species8_h64 | 110.166 | 47.587 | 2065.191 | 318.918 | 255.362 | -45.664 | 721.707 | 1673082.890 |
| l2_cavity_edge_radial_species8_h64 | 115.367 | 51.586 | 2387.355 | 341.707 | 246.573 | -92.197 | 827.121 | 579735.393 |

## Interpretation

1. The production engineering path is now real, not demo-only. The same scope parameter is checkpointed/manifested, accepted by the training CLI and Lightning backend, emitted by the Pareto wrapper generator, and used by real Slurm jobs. Full `test/test_rtece_scalar.py` passed before submission.

2. `atomic` and `all` are numerically identical for the L1 active rows because the selected scalar paths are all atomic. This is useful: it verifies backward compatibility with stage122 and confirms that the scoped adapter route does not perturb existing atomic-only behavior.

3. The edge/cavity row is currently not on the Pareto front for this OC20NEB 2048-config distillation setup. It has worse DFT F RMSE than the L1 active row (118.413 vs 112.838 meV/A), worse DFT E RMSE (357.834 vs 335.816 meV/atom), worse max force error (2759.700 vs 1996.323 meV/A), and much lower throughput (0.580M vs 1.67M atoms/s).

4. This should not be read as disproving edge-relational sketches in the TECE/rTECE design. It says this specific downfolded edge-only capacity allocation is dominated. The row changes multiple things at once: it removes the L1 active atomic vector/cross-radial paths and pays the edge-sketch cost. The next clean test should add edge paths on top of the L1 active atomic backbone instead of replacing it.

## Priority update

The next architecture sweep should keep the stage121/122 active atomic backbone and add a minimal edge-relational residual ladder: L1 active + edge.direct.radial, L1 active + edge.cavity.vector_dot, and L1 active + edge.cavity.vector_dot + quadrupole_frobenius, with `radial_species_adapter_scope=edge` and `all`. This isolates whether edge sketches add residual accuracy per throughput cost. If they remain dominated, the main Pareto path should focus on trainable low-rank atomic/radial/species representation and larger-but-frontloaded parameter counts, not edge-state lifetime or kernels.
