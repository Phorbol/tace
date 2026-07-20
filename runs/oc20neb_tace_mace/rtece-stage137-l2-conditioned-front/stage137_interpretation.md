# Stage137 L2 Conditioned Front Interpretation

- jobs: train `685308`/`685309`, physical `685311`/`685312`; all `COMPLETED 0:0`
- data: fixed Stage132 broad-teacher-rattle augmented train, `2560` configs
- question: whether Stage136 force-projection-positive L2 atomic paths can be made trainable through front low-rank mixing or residual descriptor conditioning without widening the final head

## Metrics

| row | params | atoms/s | E RMSE | E max | F MAE | F RMSE | F max | C/N RMSD | physical |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Stage132 L1 atomic | 28925 | 1217882 | 293.391 | 697.301 | 46.503 | 108.355 | 1840.539 | 0.1031 | False |
| Stage135 naive L2 | 29885 | 1058909 | 294.945 | 720.404 | 47.506 | 112.960 | 2649.912 | 0.1096 | False |
| Stage137 L2 bneck32 | 21405 | 1061041 | 309.728 | 691.876 | 48.988 | 114.259 | 1861.035 | 0.1100 | False |
| Stage137 L2 cond32 | 51367 | 1023496 | 340.032 | 740.266 | 45.846 | 107.516 | 1914.572 | 0.1112 | False |

## Interpretation

- `cond32` is the useful Stage137 signal: F RMSE `107.516` meV/A, slightly better than Stage132 L1 atomic `108.355` and clearly better than Stage135 naive L2 `112.960`. This supports the Stage136 diagnosis that L2 atomic quadrupole descriptors carry usable force-response information if the front representation is conditioned.
- `cond32` is not a clean Pareto endpoint: throughput drops from Stage132 `1.218M` atoms/s to `1.023M`, E RMSE worsens from `293.391` to `340.032` meV/atom, E max worsens to `740.266`, and rattle physical still fails with C/N RMSD `0.1112` A.
- `bneck32` is mostly a force-tail diagnostic: F max `1861.035` is close to Stage132 `1840.539` and far better than Stage135 `2649.912`, but F RMSE `114.259` is not competitive. Low-rank mixing alone discards too much average force information.
- The TECE interpretation is therefore nuanced: naive L2 is wrong, L2+front conditioning is partially right for force RMSE, but scalar-only atomic L2 still lacks the energy/physical closure demanded by the review document.

## Next Priority

1. Keep `cond32` as a force-RMSE diagnostic row, not as the current deployment winner.
2. Do not continue increasing conditioner width or final-head width; that would drift back into parameter tuning.
3. Separate the next error source: energy reference/head calibration versus force/Sobolev projection versus missing relational edge scalar sketches.
4. The next architecture stage should test either a minimal Stage136-guided edge-relational scalar sketch or a Sobolev/teacher-Jacobian distillation target on the Stage132/137 active basis, while keeping E/F RMSE/max, throughput, dimer and rattle as separate axes.
