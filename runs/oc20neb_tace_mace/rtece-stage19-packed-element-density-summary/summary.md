# rTECE Packed Element-Density Force Path Summary

## A/B Benchmark

| label | force mode | atoms/s | seconds/pass | peak alloc MB | peak reserved MB | DFT F MAE |
|---|---|---:|---:|---:|---:|---:|
| element_density_24x24_analytic_density | analytic_density | 15920357.686 | 0.015725 | 1404.935 | 1716.000 | 30.193 |
| element_density_24x24_analytic_element_packed | analytic_element_packed | 15416267.648 | 0.016240 | 1404.935 | 2032.000 | 30.193 |
| element_density_32x32_analytic_density | analytic_density | 14696599.227 | 0.017035 | 1404.937 | 1716.000 | 28.142 |
| element_density_32x32_analytic_element_packed | analytic_element_packed | 14325009.288 | 0.017477 | 1404.937 | 2032.000 | 28.142 |

## Stage-19 Interpretation

- `analytic_element_packed` is numerically equivalent to the existing element-density analytic path, but it is not faster on the 4096-config V100 benchmark.
- For 24x24, throughput drops from 15.92M to 15.42M atoms/s; for 32x32, it drops from 14.70M to 14.33M atoms/s.
- Peak allocated memory is unchanged, but reserved memory rises from 1716 MB to 2032 MB in both comparisons.
- This rejects PyTorch-level descriptor packing as an implementation-cost optimization. The next implementation step should be a true fused/exported evaluator for pair and element-density descriptors, not another tensor packing refactor.
