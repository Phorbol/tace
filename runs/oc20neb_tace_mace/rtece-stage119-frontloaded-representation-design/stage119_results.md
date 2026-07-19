# Stage119 full sbatch results: frontloaded representation ladder

Date: 2026-07-19 to 2026-07-20
Branch: tece-renorm-distill
Code commits: d70f5e0 frontloaded representation ladder, baa507d learnable species basis edge-sketch fix

## Slurm jobs

Original full jobs submitted from no-export wrappers:

| Job | Variant | State | Note |
| --- | --- | --- | --- |
| 683137 | l0_species8_learnembed_h64 | COMPLETED | valid result |
| 683141 | l1_species16_cross_learnembed_h64 | COMPLETED | valid result |
| 683139 | l2_species16_atomic_cross_learnembed_h64 | COMPLETED | valid result |
| 683138 | l2_species16_cavity_edge_learnembed_h64 | FAILED | superseded by 683167; missing species_basis_embedding in edge moments |
| 683140 | l2_species32_cavity_edge_learnembed_h64 | FAILED | superseded by 683169; missing species_basis_embedding in edge moments |
| 683142 | l2_species32_cavity_atomic_cross_learnembed_h64 | FAILED | superseded by 683168; missing species_basis_embedding in edge moments |

Replacement full jobs after baa507d:

| Job | Variant | State | Elapsed |
| --- | --- | --- | --- |
| 683167 | l2_species16_cavity_edge_learnembed_h64 | COMPLETED | 00:12:36 |
| 683169 | l2_species32_cavity_edge_learnembed_h64 | COMPLETED | 00:12:34 |
| 683168 | l2_species32_cavity_atomic_cross_learnembed_h64 | COMPLETED | 00:14:24 |

All wrappers were submitted with plain `sbatch <wrapper>` and no Slurm command-line `--export`, `--mem`, or `--cpus-per-task`.

## DFT benchmark metrics

Evaluated on `valid.extxyz[:1024]`, force mode `autograd`, dtype `float32`, graph construction excluded by prebuilt batched graph. Units: E in meV/atom, F in meV/A.

| Variant | Params | E RMSE | F RMSE | E MAE | F MAE | E max | F max | Atoms/s | sec/pass | peak MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| l0_species8_learnembed_h64 | 9,833 | 304.855 | 111.498 | 247.199 | 33.129 | 714.506 | 4862.450 | 2,403,125 | 0.02463 | 1371.3 |
| l1_species16_cross_learnembed_h64 | 15,465 | 331.310 | 124.506 | 262.422 | 51.523 | 752.109 | 2277.234 | 1,910,366 | 0.03099 | 2497.8 |
| l2_species16_atomic_cross_learnembed_h64 | 16,169 | 352.637 | 133.264 | 274.813 | 58.142 | 804.338 | 2453.533 | 1,610,762 | 0.03675 | 2924.0 |
| l2_species16_cavity_edge_learnembed_h64 | 14,993 | 389.226 | 132.682 | 292.976 | 55.336 | 973.094 | 2908.185 | 635,381 | 0.09316 | 4594.6 |
| l2_species32_cavity_edge_learnembed_h64 | 24,801 | 337.667 | 131.500 | 257.307 | 56.049 | 742.024 | 2507.054 | 569,071 | 0.10402 | 5996.7 |
| l2_species32_cavity_atomic_cross_learnembed_h64 | 26,233 | 370.302 | 126.181 | 279.393 | 54.420 | 877.678 | 2357.520 | 525,940 | 0.11255 | 6546.7 |

## Teacher benchmark metrics

Same model checkpoints evaluated against teacher labels. Units match the DFT table.

| Variant | Params | E RMSE | F RMSE | E MAE | F MAE | E max | F max | Atoms/s | sec/pass | peak MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| l0_species8_learnembed_h64 | 9,833 | 288.747 | 112.965 | 230.086 | 39.232 | 704.618 | 4824.005 | 2,404,039 | 0.02462 | 1371.3 |
| l1_species16_cross_learnembed_h64 | 15,465 | 315.029 | 122.173 | 247.060 | 52.899 | 745.118 | 2319.362 | 1,909,682 | 0.03100 | 2497.8 |
| l2_species16_atomic_cross_learnembed_h64 | 16,169 | 336.422 | 131.085 | 259.542 | 58.943 | 797.347 | 2261.372 | 1,610,677 | 0.03675 | 2924.0 |
| l2_species16_cavity_edge_learnembed_h64 | 14,993 | 375.592 | 131.604 | 283.465 | 57.013 | 962.877 | 2347.137 | 633,617 | 0.09342 | 4594.6 |
| l2_species32_cavity_edge_learnembed_h64 | 24,801 | 321.539 | 130.789 | 246.120 | 57.552 | 713.375 | 2508.158 | 566,553 | 0.10448 | 5996.7 |
| l2_species32_cavity_atomic_cross_learnembed_h64 | 26,233 | 356.220 | 124.340 | 265.213 | 55.173 | 870.686 | 2042.704 | 523,855 | 0.11300 | 6546.7 |

## Interpretation against the TECE and rTECE documents

`TECE_design_space.md` identifies the main rTECE target as T2: one fused moment pass, small L_A, no persistent high-ell state, a small scalar head, and a small number of edge relational m_total=0 sketches. Stage119 tested this premise with learnable low-rank species basis and learnable radial mixing, then added atomic cross-radial and cavity edge relational paths.

The result is negative for the current implementation and training recipe: adding atomic or cavity relational paths did not improve DFT or teacher F RMSE over the L0 learnable species baseline, while cavity-edge paths reduced throughput by roughly 3.8x to 4.6x and raised memory sharply. Because teacher and DFT benchmarks show the same ranking, this is not merely a DFT label alignment issue. The current student does not convert the extra paths into better force fidelity.

This does not falsify the TECE design idea. It narrows the failure mode to the gap emphasized in `rTECE_review.md`: current path selection is still hand-specified, not teacher-projected or renormalized; edge paths have no learned path gating or POD initialization; and the implementation still uses expensive autograd edge sketches rather than a compact static edge-gradient realization.

## Priority change

Do not keep increasing hand-selected angular or cavity paths in the current head-dominated form. The next stage should move from `more paths` to `selected and initialized paths`:

1. Add a front bottleneck or low-rank path mixer before the scalar head so parameter growth is not dominated by descriptor_dim times head_width.
2. Run teacher sensitivity or projection to choose atomic and edge path subsets instead of treating all L1/L2/cavity paths as equally useful.
3. Add a supervised-vs-teacher-distilled comparison using the same TACE training workflow and the same Stage119 variants, because Stage119 currently only proves that naive full-step training does not make extra paths useful.
4. Keep L0 learnable species/radial as the current Pareto baseline for high-throughput student comparison.
5. Only return to cavity-edge expansion after path projection or distillation shows it reduces projection error enough to pay for its 3.8x to 4.6x throughput cost.

## Verification

Local tests before submission: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py -q` -> 246 passed, 9 warnings.
