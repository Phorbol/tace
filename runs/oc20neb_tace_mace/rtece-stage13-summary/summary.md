# TECE Distillation Matrix Summary

## Students

| variant | force mode | atoms/s | configs/s | peak alloc MB | params | teacher E MAE | teacher F MAE | DFT E MAE | DFT F MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rtece_pair_32x32_mixed | analytic_pair | 18627754.258 | 304764.360 | 1405.407 | 1409 | 1403.952 | 47.493 | 1408.457 | 44.297 |
| rtece_pair_64x64_force30 | analytic_pair | 16230621.222 | 265545.424 | 1405.419 | 4865 | 1406.052 | 40.694 | 1410.556 | 36.127 |
| rtece_pair_stage11_64x64 | analytic_pair | 16091762.816 | 263273.593 | 1405.419 | 4865 | 1403.909 | 51.219 | 1408.547 | 47.932 |
| rtece_pair_64x64_teacher | analytic_pair | 16064669.378 | 262830.324 | 1405.419 | 4865 | 1400.015 | 47.370 | 1404.519 | 43.891 |
| rtece_pair_128x64_mixed | analytic_pair | 15370090.556 | 251466.481 | 1405.437 | 9601 | 1405.387 | 57.013 | 1410.038 | 54.221 |

## TECE Source-Document Review

- Projection: did reducing persistent angular state improve real throughput and memory?
- Renormalization/distillation: is teacher-label error low enough to justify the deleted paths?
- Hardware cost: did atoms/s and peak memory improve, not just parameter count?
- Next priority: architecture projection, distillation loss, fusion/export, or benchmark methodology.
