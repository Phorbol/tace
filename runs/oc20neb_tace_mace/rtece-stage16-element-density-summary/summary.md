# TECE Distillation Matrix Summary

## Students

| variant | force mode | atoms/s | configs/s | peak alloc MB | params | teacher E MAE | teacher F MAE | DFT E MAE | DFT F MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rtece_pair_32x32_force30_mixed | analytic_pair | 18622904.499 | 304685.015 | 1405.407 | 1409 | 1402.619 | 43.142 | 1407.123 | 39.420 |
| rtece_pair_64x64_force30_mixed | analytic_pair | 16230621.222 | 265545.424 | 1405.419 | 4865 | 1406.052 | 40.694 | 1410.556 | 36.127 |
| rtece_element_density_32x32_force30 | analytic_density | 14696599.227 | 240447.646 | 1404.937 | 1665 | 1404.869 | 33.928 | 1409.373 | 28.142 |
| rtece_element_density_64x64_force30 | analytic_density | 13057248.202 | 213626.605 | 1404.951 | 5377 | 1404.116 | 39.860 | 1408.620 | 35.114 |
| rtece_density_quadratic_32x32_force30 | analytic_density | 12100417.457 | 197972.119 | 1404.937 | 1665 | 1391.818 | 49.332 | 1395.726 | 46.390 |
| rtece_density_quadratic_64x64_force30 | analytic_density | 11041829.353 | 180652.805 | 1404.951 | 5377 | 1391.134 | 53.144 | 1395.176 | 50.557 |

## DFT Force Pareto Front

| variant | force mode | atoms/s | DFT F MAE | teacher F MAE | params |
|---|---|---:|---:|---:|---:|
| rtece_pair_32x32_force30_mixed | analytic_pair | 18622904.499 | 39.420 | 43.142 | 1409 |
| rtece_pair_64x64_force30_mixed | analytic_pair | 16230621.222 | 36.127 | 40.694 | 4865 |
| rtece_element_density_32x32_force30 | analytic_density | 14696599.227 | 28.142 | 33.928 | 1665 |

## Teacher Force Pareto Front

| variant | force mode | atoms/s | DFT F MAE | teacher F MAE | params |
|---|---|---:|---:|---:|---:|
| rtece_pair_32x32_force30_mixed | analytic_pair | 18622904.499 | 39.420 | 43.142 | 1409 |
| rtece_pair_64x64_force30_mixed | analytic_pair | 16230621.222 | 36.127 | 40.694 | 4865 |
| rtece_element_density_32x32_force30 | analytic_density | 14696599.227 | 28.142 | 33.928 | 1665 |

## TECE Source-Document Review

- Projection: did reducing persistent angular state improve real throughput and memory?
- Renormalization/distillation: is teacher-label error low enough to justify the deleted paths?
- Hardware cost: did atoms/s and peak memory improve, not just parameter count?
- Next priority: architecture projection, distillation loss, fusion/export, or benchmark methodology.

## Element-Density Offset-Window Robustness

| extxyz index | configs | atoms/s | DFT F MAE | DFT F RMSE | DFT E MAE | interpretation |
|---|---:|---:|---:|---:|---:|---|
| `:4096` | 4096 | 14696599.227 | 28.142 | 109.927 | 1409.373 | prefix benchmark; new lower-error rTECE Pareto point |
| `4096:8192` | 4096 | 14729931.636 | 30.747 | 117.549 | 1495.246 | stable offset window; same throughput class |
| `8192:12288` | 1808 | 13884470.569 | 31.182 | 116.158 | 1353.039 | shorter tail window; force-error regime remains stable |

## Stage-16 Interpretation

- `rtece_element_density` restores a TECE semantic group that `rtece_pair` deleted: neighbor-element-conditioned radial density, `sum_j z_j R_n(r_ij)`, while still avoiding persistent vector/quadrupole state, edge sketches, and full force autograd.
- This is a positive architecture-projection result. The 32x32 point improves DFT force MAE from 39.42 to 28.14 meV/A and teacher force MAE from 43.14 to 33.93 meV/A versus pair-32, at a throughput cost from 18.62M to 14.70M atoms/s.
- The point is Pareto-positive in both DFT and teacher force fronts: it is slower than pair-64 but substantially more accurate, and offset windows keep DFT force MAE around 28-31 meV/A.
- The 64x64 element-density run is not useful: 35.11 meV/A at 13.06M atoms/s is dominated by the 32x32 element-density point.
- Compared with Stage 15, the lesson is clear: adding semantic information about neighbor chemistry is useful, while a naive scalar polynomial `rho^2` is dominated. The next architecture step should preserve this semantic ordering.
- Next priority: test one low-order geometric moment scalar that has an analytic force path, preferably vector-moment norm only, and keep 32x32/force30 as the first setting. Do not widen the MLP first.
