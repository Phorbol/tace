# Stage154 Representation Ladder Summary

Stage154 tested whether the Stage153 raw absolute-energy failure is repaired by front-loaded representation capacity rather than by another E0 or teacher-weight toggle. All rows used the same mixed base2048 training contract, 20k max steps, warmup 500 plus plateau scheduling, early stopping patience 400, E/F weights 1/10, per-element fitted E0s, ZBL short-range baseline, and 1024-config DFT/teacher validation benchmarks.

## Job Status

| job | row | state | exit | elapsed | max RSS |
|---:|---|---|---|---:|---:|
| 687458 | `stage154_l2_k6_mixed2048` | COMPLETED | 0:0 | 00:12:55 | 1388620K |
| 687459 | `stage154_l2_radial16_k6_species32_mixed2048` | COMPLETED | 0:0 | 00:12:55 | 1440832K |
| 687457 | `stage154_l2_radial16_k8_species48_adapter16_mixed2048` | COMPLETED | 0:0 | 00:15:44 | 1404992K |
| 687460 | `stage154_t3_cavity_vector_k6_mixed2048` | COMPLETED | 0:0 | 00:17:49 | 1416040K |

## DFT-Valid Metrics

Errors are meV/atom for energy and meV/A for force.

| row | params | atoms/s | E RMSE | E MAE | E bias | E max | F RMSE | F MAE | F max | group-offset E RMSE | relative-image RMSE | barrier RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage153 mixed baseline | 51367 | 1.023e6 | 126.883 | 99.102 | -16.553 | 325.773 | 105.745 | 46.922 | 1641.924 | 6.189 | 8.546 | 16.089 |
| `stage154_l2_k6_mixed2048` | 54499 | 9.509e5 | 140.753 | 97.713 | -9.858 | 378.823 | 113.503 | 50.727 | 2455.936 | 6.562 | 8.536 | 17.573 |
| `stage154_l2_radial16_k6_species32_mixed2048` | 85951 | 7.101e5 | 132.437 | 77.955 | 18.889 | 356.545 | 109.619 | 49.529 | 1803.391 | 5.971 | 8.034 | 15.469 |
| `stage154_l2_radial16_k8_species48_adapter16_mixed2048` | 153769 | 5.650e5 | 155.269 | 128.755 | -19.536 | 367.172 | 110.114 | 48.771 | 1784.316 | 6.947 | 9.054 | 18.717 |
| `stage154_t3_cavity_vector_k6_mixed2048` | 54628 | 5.050e5 | 94.167 | 66.090 | 39.241 | 286.219 | 102.467 | 46.740 | 1875.799 | 5.859 | 7.537 | 14.814 |

## Interpretation

1. The raw energy problem is real, but Stage153/154 show it is dominated by case-level offsets. Stage153 mixed had raw E RMSE `126.883` meV/atom, while case/group-offset-corrected E RMSE was only `6.189` meV/atom.
2. Purely increasing L2 atomic front capacity did not repair raw E RMSE. The 85.9k radial/species point improved E MAE and relative metrics, but E RMSE stayed worse than Stage153 mixed. The 153.8k high-capacity L2 point overfit or misallocated capacity.
3. The first clear architecture-side energy improvement came from the T3 edge-cavity vector path: raw E RMSE `94.167`, E MAE `66.090`, E max `286.219`, F RMSE `102.467`, relative-image RMSE `7.537`, barrier RMSE `14.814`.
4. The cost is substantial. The T3 edge-cavity row runs at `5.05e5` atoms/s, about half of the Stage153 mixed baseline. It is a better accuracy point, not a free throughput point.

## Document-Grounded Decision

This supports the `TECE_design_space.md` claim that rTECE should not be treated as a pure scalar endpoint: a small number of edge-relational scalar sketches can carry information that pure atomic scalar moments miss. It also supports `rTECE_review.md`: path groups must be selected with measured projection/error benefit and real hardware cost.

The next mainline should not be "add more generic parameters". It should be:

1. promote `edge.cavity.vector_dot` as a real T3 candidate;
2. run an edge-relational cost ladder with only a few semantically distinct paths, not a broad MLP expansion;
3. decompose the remaining raw E RMSE by `case_id`, composition, adsorbate elements, and slab family;
4. test whether teacher rattle/relax augmented labels reduce the remaining case offsets without C/N-specific overfitting;
5. compare this T3 point against NEP/DPA-like baselines on the same DFT-valid E/F RMSE, E/F max, relative NEB, physical rattle-relax, and throughput axes.
