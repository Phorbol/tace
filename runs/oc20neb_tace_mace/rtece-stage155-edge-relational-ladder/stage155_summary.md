# Stage155 Edge-Relational Ladder Summary

Stage155 tested marginal edge-relational paths after Stage154 showed that `edge.cavity.vector_dot` was the first clear raw-energy improvement and generic L2 capacity did not repair the energy problem. All rows used the same mixed base2048 training contract, 20k max steps, warmup 500 plus plateau scheduling, early stopping patience 400, E/F weights 1/10, per-element fitted E0s, ZBL short-range baseline, and 1024-config DFT/teacher validation benchmarks.

## Job Status

| job | row | state | exit | elapsed | max RSS |
|---:|---|---|---|---:|---:|
| 687464 | `stage155_t3_cavity_vector_direct_mixed2048` | COMPLETED | 0:0 | 00:18:03 | 1517128K |
| 687462 | `stage155_t3_cavity_vector_quad_mixed2048` | COMPLETED | 0:0 | 00:18:31 | 1720332K |
| 687463 | `stage155_t3_cavity_vector_quad_direct_mixed2048` | COMPLETED | 0:0 | 00:18:38 | 1478040K |

## DFT-Valid Metrics

Errors are meV/atom for energy and meV/A for force.

| row | params | atoms/s | E RMSE | E MAE | E bias | E max | F RMSE | F MAE | F max | group-offset E RMSE | relative-image RMSE | barrier RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage154 T3 vector baseline | 54628 | 5.050e5 | 94.167 | 66.090 | 39.241 | 286.219 | 102.467 | 46.740 | 1875.799 | 5.859 | 7.537 | 14.814 |
| `+ edge.direct.radial` | 54886 | 4.995e5 | 71.830 | 50.575 | 19.167 | 270.534 | 100.744 | 46.546 | 1946.920 | 5.773 | 8.123 | 12.830 |
| `+ edge.cavity.quadrupole_frobenius` | 54757 | 3.795e5 | 83.772 | 58.415 | 30.140 | 273.542 | 96.274 | 45.083 | 2059.967 | 5.100 | 7.093 | 13.607 |
| `+ quadrupole + direct` | 55015 | 3.798e5 | 68.099 | 44.976 | 18.996 | 249.199 | 98.655 | 45.411 | 2375.238 | 5.575 | 7.029 | 13.407 |

## Teacher-Valid Metrics

| row | atoms/s | E RMSE | E MAE | E max | F RMSE | F MAE | F max | relative-image RMSE | barrier RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `+ edge.direct.radial` | 4.997e5 | 52.733 | 36.050 | 176.623 | 97.526 | 47.076 | 1896.662 | 7.256 | 10.529 |
| `+ edge.cavity.quadrupole_frobenius` | 3.798e5 | 62.724 | 48.175 | 183.793 | 93.743 | 45.884 | 2102.094 | 6.152 | 11.215 |
| `+ quadrupole + direct` | 3.798e5 | 45.041 | 27.976 | 165.274 | 95.085 | 45.132 | 2270.713 | 6.062 | 11.156 |

## Interpretation

1. `edge.direct.radial` is the cleanest Stage155 marginal win for raw energy. It cuts DFT-valid E RMSE from Stage154 T3 vector `94.167` to `71.830` meV/atom with almost no throughput penalty relative to the T3 vector baseline (`4.995e5` vs `5.050e5` atoms/s).
2. `edge.cavity.quadrupole_frobenius` improves force RMSE and relative/path metrics, but has a large throughput cost (`3.795e5` atoms/s) and does not improve raw energy as much as direct radial.
3. The combined `quadrupole + direct` row gives the best DFT-valid E RMSE/MAE/max and teacher-valid E RMSE/MAE/max, but costs about 25% throughput relative to direct-only and worsens F max.
4. The current Pareto update is therefore not "add all edge paths". It is a two-point decision:
   - high-throughput T3 energy point: `edge.cavity.vector_dot + edge.direct.radial`;
   - more accurate but slower path/energy point: `edge.cavity.vector_dot + edge.cavity.quadrupole_frobenius + edge.direct.radial`.

## Document-Grounded Decision

This is now much closer to the TECE/rTECE design target than the earlier scalar endpoint: the retained model uses immediate scalarization, no persistent equivariant node state, a small scalar head, ZBL baseline, and a sparse edge-relational semantic basis. The main remaining gap to the document is not another manual path sweep, but replacing hand-selected increments with teacher projection / Schur-GN active-set selection and then validating the promoted points with physical dimer/rattle-relax tests plus NEP/DPA-style baselines.

Next priority:

1. promote `edge.direct.radial` as the default extra path for the high-throughput T3 student;
2. run physical triage on Stage155 direct and Stage155 quad+direct against Stage154 vector;
3. decompose Stage155 case offsets to verify whether direct radial repairs the same cases identified in `stage155_case_offset_diagnostic.md`;
4. compare the promoted Stage155 points with NEP/DPA-like baselines on the same E/F RMSE, E/F max, relative NEB, physical rattle-relax, and throughput axes.
