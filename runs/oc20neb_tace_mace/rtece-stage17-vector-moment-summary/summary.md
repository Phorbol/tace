# TECE Distillation Matrix Summary

## Students

| variant | force mode | atoms/s | configs/s | peak alloc MB | params | teacher E MAE | teacher F MAE | DFT E MAE | DFT F MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rtece_pair_32x32_force30_mixed | analytic_pair | 18622904.499 | 304685.015 | 1405.407 | 1409 | 1402.619 | 43.142 | 1407.123 | 39.420 |
| rtece_pair_64x64_force30_mixed | analytic_pair | 16230621.222 | 265545.424 | 1405.419 | 4865 | 1406.052 | 40.694 | 1410.556 | 36.127 |
| rtece_element_density_32x32_force30 | analytic_density | 14696599.227 | 240447.646 | 1404.937 | 1665 | 1404.869 | 33.928 | 1409.373 | 28.142 |
| rtece_density_quadratic_32x32_force30 | analytic_density | 12100417.457 | 197972.119 | 1404.937 | 1665 | 1391.818 | 49.332 | 1395.726 | 46.390 |
| rtece_density_quadratic_64x64_force30 | analytic_density | 11041829.353 | 180652.805 | 1404.951 | 5377 | 1391.134 | 53.144 | 1395.176 | 50.557 |
| rtece_vector_moments_32x32_force30 | analytic_density | 6673067.885 | 109176.514 | 3401.881 | 1665 | 1406.192 | 38.827 | 1410.696 | 34.413 |

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

## Stage-17 Interpretation

- `rtece_vector_moments` tests the next TECE semantic block after element-conditioned density: a rotationally invariant vector moment norm, `||sum_j R_n(r_ij) u_ij||^2`, with an analytic chain-rule force path.
- The result is negative. At 32x32/force30 it reaches 34.41 meV/A DFT force MAE and 38.83 meV/A teacher force MAE, but only 6.67M atoms/s and 3.4GB peak allocation.
- It is dominated by `rtece_element_density_32x32_force30`, which is both faster (14.70M atoms/s) and more accurate (28.14 meV/A DFT force MAE), and it falls below the >1e7 atoms/s target class.
- The likely bottleneck is not parameter count: vector-moment and element-density both have 1665 parameters. The cost comes from carrying vector moment state and the transverse derivative term in the analytic force path under current PyTorch scatter kernels.
- Do not run 64x64 vector moments or quadrupole moments in this unfused implementation. The next priority should be fusion/export or a cheaper element-conditioned scalar kernel around the current pair/element-density front.
