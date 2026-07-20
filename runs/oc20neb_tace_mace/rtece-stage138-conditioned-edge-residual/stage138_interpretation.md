# Stage138 conditioned edge residual interpretation

## Document alignment

- `TECE_design_space.md` frames rTECE/T2-T3 as early scalarization plus a small number of edge-relational total-m=0 sketches: a missing middle layer between NEP/MTP/DPA1-0 scalar endpoints and full TECE.
- `rTECE_review.md` flags the older `edge_sketch8/16` semantics as P0 because repeated/weak edge features are not a clean TECE path registry. Stage138 therefore tests named `edge.cavity.vector_dot` and `edge.direct.radial` paths on top of the Stage137 conditioned L2 front.
- This is deliberately not a wider final-head test. The added capacity lives in the TECE/rTECE front path space and pays real edge cost.

## Full sbatch contract

- Training job: `685348`, completed `0:0`, elapsed `00:30:35`.
- Physical job: `685421`, completed `0:0`, elapsed `00:00:19`.
- Train/valid configs: `2560/256`; max steps `20000`; batch `8`; valid batch `16`; best step `18560`; best valid loss `0.1028339490`.
- Energy zero: fitted per-element E0s, no `energy_per_atom_shift` offset. Short-range setting: ZBL path configured, but recorded strength is `0.0`; do not treat this as an active ZBL-baseline validation.
- Slurm wrappers avoid `--export`, `--mem`, and `--cpus-per-task`.

## Pareto comparison

| label | params | atoms/s | E RMSE | E max | F RMSE | F max | C/N RMSD | dimer | rattle | physical |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stage132_l1_atomic_anchor | 28925 | 1217882 | 293.391 | 697.301 | 108.355 | 1840.539 | 0.103133 | pass | fail | fail |
| stage135_l2_atomic | 29885 | 1058909 | 294.945 | 720.404 | 112.960 | 2649.912 | 0.109592 | pass | fail | fail |
| stage137_l2_bneck32 | 21405 | 1061041 | 309.728 | 691.876 | 114.259 | 1861.035 | 0.109978 | pass | fail | fail |
| stage137_l2_cond32 | 51367 | 1023496 | 340.032 | 740.266 | 107.516 | 1914.572 | 0.111161 | pass | fail | fail |
| stage138_l2_cond32_edge_residual | 51754 | 518939 | 329.414 | 763.132 | 112.703 | 1834.768 | 0.106088 | pass | fail | fail |

Units: E metrics are meV/atom, F metrics are meV/A, C/N RMSD is A.

## Interpretation

- Stage138 is not a new Pareto endpoint. It reaches `518,939 atoms/s`, about `0.507x` Stage137 cond32 and `0.426x` Stage132, while DFT F RMSE is worse than Stage137 cond32 and Stage132.
- The positive signal is narrow: F max improves slightly versus Stage137 cond32 (`1914.572 -> 1834.768 meV/A`) and C/N rattle RMSD improves (`0.111161 -> 0.106088 A`). Dimer remains physically sane under the current sanity gate, but this is not evidence of an active ZBL correction because the recorded ZBL strength is `0.0`.
- The negative signal dominates: E RMSE/max remain weak (`329.414 meV/atom`, `763.132 meV/atom`), F MAE remains high, rattle convergence still fails, and the edge residual halves throughput.
- Therefore the current sparse edge residual contains some tail/physical information, but under the present training/data/distillation setup it is not worth its hardware cost. The next priority should be distillation/data-limited diagnosis: teacher rattle-relax trajectory labels and/or Sobolev/force-Jacobian projection, using the Stage132/137 atomic fronts as throughput anchors.

