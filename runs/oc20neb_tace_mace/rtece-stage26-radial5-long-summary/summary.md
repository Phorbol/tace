# Stage 26: Radial5/16x16 Longer-Training Robustness

Stage 26 tested whether the Stage 25 seed sensitivity of the radial5/16x16 middle point could be reduced by a longer training and a larger validation selection window. The architecture and evaluator were fixed: `rtece_element_density`, `num_radial=5`, `hidden=16x16`, 481 parameters, train=512, valid=512, force weight 30, energy weight 1, and `analytic_element_triton_force`.

## Results

| seed | best step | best valid loss | atoms/s | DFT F MAE | DFT F RMSE | teacher F MAE | DFT E MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2100 | 4.4110 | 38606435 | 56.03 | 124.13 | 58.99 | 1409.5 |
| 1 | 600 | 4.4038 | 38660247 | 44.59 | 115.36 | 48.41 | 1403.4 |
| 2 | 900 | 4.4078 | 38870724 | 38.28 | 112.60 | 42.24 | 1428.1 |

## Interpretation

Longer training and a 512-config validation selector did not stabilize radial5/16x16 accuracy. Throughput remains architecture-determined and robust at about 38.6-38.9M atoms/s, but DFT force MAE spans 38.28-56.03 meV/A across seeds. The best-validation losses are tightly clustered around 4.40 while DFT force MAE differs by nearly 18 meV/A, so the current mixed-label proxy still does not select the best low-precision endpoint reliably.

This changes the status of radial5/16x16 from a clean middle Pareto point to a seed-sensitive candidate band. The TECE explanation remains useful: radial5 is the minimal radial-resolution recovery beyond the radial4 ultra-fast endpoint, but the current distillation/selection protocol is not a reliable renormalization map for that band.

The next priority should shift away from more head-width or radial-count sweeps. The two cleaner directions are:

1. improve selection by benchmarking/checkpointing against the actual DFT-force target on a larger validation slice, then use best-of-N seeds only if reported as an explicit stochastic protocol;
2. implement fused descriptor construction for the already stable element-density family, especially radial8, to move the lower-error Pareto points toward the radial4/radial5 throughput regime without adding semantic complexity.
