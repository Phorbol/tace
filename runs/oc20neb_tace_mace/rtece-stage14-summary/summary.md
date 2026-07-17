# TECE Distillation Matrix Summary

## Students

| variant | force mode | atoms/s | configs/s | peak alloc MB | params | teacher E MAE | teacher F MAE | DFT E MAE | DFT F MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rtece_pair_32x32_force10_mixed | analytic_pair | 18627754.258 | 304764.360 | 1405.407 | 1409 | 1403.952 | 47.493 | 1408.457 | 44.297 |
| rtece_pair_32x32_force30_mixed | analytic_pair | 18622904.499 | 304685.015 | 1405.407 | 1409 | 1402.619 | 43.142 | 1407.123 | 39.420 |
| rtece_pair_64x64_force30_mixed | analytic_pair | 16230621.222 | 265545.424 | 1405.419 | 4865 | 1406.052 | 40.694 | 1410.556 | 36.127 |
| rtece_pair_64x64_force30_teacher | analytic_pair | 16153061.503 | 264276.487 | 1405.419 | 4865 | 1410.190 | 40.802 | 1414.692 | 36.296 |

## TECE Source-Document Review

- Projection: did reducing persistent angular state improve real throughput and memory?
- Renormalization/distillation: is teacher-label error low enough to justify the deleted paths?
- Hardware cost: did atoms/s and peak memory improve, not just parameter count?
- Next priority: architecture projection, distillation loss, fusion/export, or benchmark methodology.
