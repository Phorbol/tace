# TECE Distillation Matrix Summary

## Students

| variant | force mode | atoms/s | configs/s | peak alloc MB | params | teacher E MAE | teacher F MAE | DFT E MAE | DFT F MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rtece_pair_32x32_force30_mixed | analytic_pair | 18622904.499 | 304685.015 | 1405.407 | 1409 | 1402.619 | 43.142 | 1407.123 | 39.420 |
| rtece_pair_64x64_force30_mixed | analytic_pair | 16230621.222 | 265545.424 | 1405.419 | 4865 | 1406.052 | 40.694 | 1410.556 | 36.127 |
| rtece_element_density_16x16_force30 | analytic_density | 16019208.056 | 262086.542 | 1404.933 | 577 | 1407.836 | 42.985 | 1412.459 | 38.686 |
| rtece_element_density_24x24_force30 | analytic_density | 15920357.686 | 260469.274 | 1404.935 | 1057 | 1399.078 | 35.829 | 1402.892 | 30.193 |
| rtece_element_density_32x32_force30 | analytic_density | 14696599.227 | 240447.646 | 1404.937 | 1665 | 1404.869 | 33.928 | 1409.373 | 28.142 |
| rtece_density_quadratic_32x32_force30 | analytic_density | 12100417.457 | 197972.119 | 1404.937 | 1665 | 1391.818 | 49.332 | 1395.726 | 46.390 |
| rtece_density_quadratic_64x64_force30 | analytic_density | 11041829.353 | 180652.805 | 1404.951 | 5377 | 1391.134 | 53.144 | 1395.176 | 50.557 |
| rtece_vector_moments_32x32_force30 | analytic_density | 6673067.885 | 109176.514 | 3401.881 | 1665 | 1406.192 | 38.827 | 1410.696 | 34.413 |

## DFT Force Pareto Front

| variant | force mode | atoms/s | DFT F MAE | teacher F MAE | params |
|---|---|---:|---:|---:|---:|
| rtece_pair_32x32_force30_mixed | analytic_pair | 18622904.499 | 39.420 | 43.142 | 1409 |
| rtece_pair_64x64_force30_mixed | analytic_pair | 16230621.222 | 36.127 | 40.694 | 4865 |
| rtece_element_density_24x24_force30 | analytic_density | 15920357.686 | 30.193 | 35.829 | 1057 |
| rtece_element_density_32x32_force30 | analytic_density | 14696599.227 | 28.142 | 33.928 | 1665 |

## Teacher Force Pareto Front

| variant | force mode | atoms/s | DFT F MAE | teacher F MAE | params |
|---|---|---:|---:|---:|---:|
| rtece_pair_32x32_force30_mixed | analytic_pair | 18622904.499 | 39.420 | 43.142 | 1409 |
| rtece_pair_64x64_force30_mixed | analytic_pair | 16230621.222 | 36.127 | 40.694 | 4865 |
| rtece_element_density_24x24_force30 | analytic_density | 15920357.686 | 30.193 | 35.829 | 1057 |
| rtece_element_density_32x32_force30 | analytic_density | 14696599.227 | 28.142 | 33.928 | 1665 |

## TECE Source-Document Review

- Projection: did reducing persistent angular state improve real throughput and memory?
- Renormalization/distillation: is teacher-label error low enough to justify the deleted paths?
- Hardware cost: did atoms/s and peak memory improve, not just parameter count?
- Next priority: architecture projection, distillation loss, fusion/export, or benchmark methodology.

## Element-Density 24x24 Offset-Window Robustness

| extxyz index | configs | atoms/s | DFT F MAE | DFT F RMSE | DFT E MAE | interpretation |
|---|---:|---:|---:|---:|---:|---|
| `:4096` | 4096 | 15920357.686 | 30.193 | 111.497 | 1402.892 | prefix benchmark; new intermediate front point |
| `4096:8192` | 4096 | 15990835.256 | 33.416 | 120.714 | 1486.758 | stable offset window; same throughput class |
| `8192:12288` | 1808 | 15018695.130 | 32.846 | 118.971 | 1371.768 | shorter tail window; force-error regime remains stable |

## Stage-18 Interpretation

- This stage keeps the Stage-16 semantic descriptor fixed and varies only the scalar head capacity. That is a valid parameter-axis refinement after the architecture projection proved useful.
- `rtece_element_density_24x24_force30` is Pareto-positive: it sits between pair-64 and element-density-32, at 15.92M atoms/s and 30.19 meV/A DFT force MAE.
- `rtece_element_density_16x16_force30` is dominated by pair-64: it is slightly slower and clearly less accurate.
- The current rTECE front is now four points: pair-32 for maximum throughput, pair-64, element-density-24, and element-density-32 for the lowest force MAE in this scalarized branch.
- Next priority should be implementation cost, not more descriptor semantics: export/fuse pair and element-density analytic kernels, or test a directly fused evaluator that avoids generic PyTorch scatter overhead.
