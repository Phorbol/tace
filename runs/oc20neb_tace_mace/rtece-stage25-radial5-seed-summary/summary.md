# Stage 25: Radial5/16x16 Seed Robustness

Stage 25 repeated the Stage 24 radial5/16x16 candidate with explicit seeds. The architecture and evaluator were fixed: `rtece_element_density`, `num_radial=5`, `hidden=16x16`, force weight 30, train=512, valid=256, and `analytic_element_triton_force`.

## Results

| seed | best valid loss | atoms/s | DFT F MAE | teacher F MAE | DFT E MAE |
|---:|---:|---:|---:|---:|---:|
| pre-seed plumbing | 3.4773 | 38825023 | 39.02 | 43.07 | 1449.6 |
| 1 | 3.4156 | 38876690 | 42.48 | 46.52 | 1430.1 |
| 2 | 3.4345 | 38774313 | 45.78 | 49.21 | 1429.4 |

## Interpretation

Throughput is stable, but force accuracy has significant seed sensitivity. The Stage 24 seed0 result is promising but optimistic. Radial5/16x16 should be reported as a candidate band around 39-46 meV/A at about 38.8M atoms/s, not a single proven Pareto point.

The next priority should be either stronger selection/training robustness for radial5/16x16 or fused descriptor construction for the already stable radial8 element-density points.
