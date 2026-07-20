# Stage154 Representation Ladder Plan

Stage154 follows Stage153. Stage153 showed that switching the same T2 L2 rTECE front from mixed labels to DFT-only labels does not repair absolute energy: DFT-valid E RMSE remains 126-134 meV/atom and DFT-only is slightly worse than mixed. The next axis is therefore front-loaded representation/projection capacity, not another E0 or teacher-weight toggle.

## Document Alignment

- `TECE_design_space.md`: projection error and training/distillation error must be separated; lowering projection error requires choosing better basis/path/rank/placement under measured hardware cost.
- `rTECE_review.md`: path selection must use physical sensitivity and real marginal cost; Schur/GN or active-set evidence should decide whether added groups are worth keeping.
- Current code constraint: Cartesian rTECE supports `moment_l_max` only up to 2, so this stage does not pretend to test L3/L4. It tests supported front-loaded capacity knobs.

## Rows

| row | total params | representation params | readout params | purpose |
|---|---:|---:|---:|---|
| `stage154_l2_k6_mixed2048` | 54499 | 27490 | 27009 | increase atomic cross-radial shell sketch rank 3 -> 6 |
| `stage154_l2_radial16_k6_species32_mixed2048` | 85951 | 43838 | 42113 | raise radial rank and learnable species basis on the atomic L2 front |
| `stage154_l2_radial16_k8_species48_adapter16_mixed2048` | 153769 | 93608 | 60161 | high-capacity front-loaded upper point for projection-error testing |
| `stage154_t3_cavity_vector_k6_mixed2048` | 54628 | 27555 | 27073 | optional edge-relational cavity increment with real throughput cost |

## Decision Rule

The accepted next Pareto move must reduce E RMSE/MAE/max and/or F RMSE/MAE/max enough to justify its measured atoms/s and memory cost. Relative image/barrier RMSE remains reported separately because Stage149-153 repeatedly showed relative path shape can be much better than absolute energy.
