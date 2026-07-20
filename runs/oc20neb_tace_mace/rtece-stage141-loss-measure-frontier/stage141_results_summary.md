# Stage141 loss-measure frontier results

Stage141 fixed the Stage140 L=2 conditioned rTECE architecture and varied only teacher-relax force multiplier / global energy weight. The goal was to separate TECE operator capacity from Sobolev/deployment-measure weighting.

## Jobs

- Training: 685515, 685516, 685517 all `COMPLETED 0:0`, elapsed 9:51-10:00 on `16v100n14`.
- Physical triage: 685523, 685524, 685525 all `COMPLETED 0:0`, elapsed 7-9 s on `16v100n14`.

## Metrics

| variant | atoms/s | E RMSE | E max | F RMSE | F max | C/N RMSD | rattle fmax | bench gate | physical gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `fm1p5_ew2_h64` | 1023638 | 326.159 | 649.572 | 98.473 | 1867.594 | 0.106 | 9.159 | False | False |
| `fm2p0_ew3_h64` | 1024942 | 329.670 | 684.796 | 101.214 | 1732.664 | 0.104 | 9.452 | False | False |
| `fm2p5_ew2_h64` | 1021520 | 299.524 | 657.389 | 99.973 | 1848.652 | 0.097 | 9.547 | True | False |

## Stage140 Reference

| row | atoms/s | E RMSE | E max | F RMSE | F max | C/N RMSD | rattle fmax | bench gate | physical gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| `stage139_l2_cond32_relax_distill` | 1024003 | 388.377 | 794.805 | 103.073 | 1709.661 | 0.101 | 9.575 | False | False |
| `stage140_forceonly_ew1` | 1022239 | 324.591 | 679.403 | 104.090 | 1953.044 | 0.106 | 9.850 | False | False |
| `stage140_forceonly_ew2` | 1014674 | 306.507 | 677.391 | 96.851 | 2006.939 | 0.095 | 8.727 | True | False |

## Interpretation

- `fm2p5_ew2_h64` is the Stage141 benchmark incumbent: it is the only Stage141 row passing the current benchmark gate, with E RMSE 299.524 meV/atom, F RMSE 99.973 meV/A, F max 1848.652 meV/A, C/N RMSD 0.0966 A, and ~1.02M atoms/s.
- Relative to Stage140 ew2, `fm2p5_ew2_h64` improves benchmark-side E RMSE, E max, F max, and benchmark score, but it worsens F RMSE, C/N RMSD, rattle fmax, and physical score. So this is not a physical-frontier replacement for Stage140 ew2.
- Increasing global energy weight from 2 to 3 did not improve energy RMSE here; `fm2p0_ew3_h64` has lower F max but worse E RMSE and worse physical score.
- Lowering teacher-relax force multiplier to 1.5 slightly improves Stage141 F RMSE but misses the E RMSE gate and does not improve rattle fmax.
- All rows still fail physical closure because rattle relaxed fmax remains around 9 eV/A. Dimer scans pass and are finite/repulsive, so the failure is not a short-range nonfinite pathology; it is a relaxation-force-tail/local-curvature issue.

## Next Priority

Keep `fm2p5_ew2_h64` as the benchmark-side fixed-architecture loss-measure incumbent, but keep Stage140 ew2 as the physical-rattle reference. Stage142 should add teacher-relax trajectory volume / curvature-tail coverage around the same deployment manifold before spending architecture complexity, then compare a minimal representation/projection capacity increment only if the fmax tail persists. This follows `TECE_design_space.md` Sobolev metric separation and `rTECE_review.md` distillation/physical validation priorities.
