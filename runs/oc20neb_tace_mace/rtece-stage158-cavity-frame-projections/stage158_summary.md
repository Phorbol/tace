# Stage158 Cavity Edge-Frame Projection Summary

Stage158 implements and tests the rTECE document-specified edge-frame projection paths `u dot v` and `u^T Q u`. The purpose was to answer whether these missing total-m=0 cavity edge-relational scalar sketches fix the Stage157 raw-energy weakness without changing the training contract.

## Job Status

| variant | job | state | elapsed | best step | best valid loss |
|---|---:|---|---:|---:|---:|
| `stage158_cavity_uquproj_b32_rel0p25_mixed2048` | 687541 | COMPLETED | 00:22:12 | 15296 | 0.076571 |
| `stage158_cavity_uv_uquproj_b32_rel0p25_mixed2048` | 687542 | COMPLETED | 00:22:00 | 19136 | 0.086851 |
| `stage158_cavity_uvproj_b32_rel0p25_mixed2048` | 687540 | COMPLETED | 00:19:57 | 17344 | 0.084868 |

## DFT Benchmark

Baseline is Stage157 `stage157_direct_b32_rel0p25_mixed2048`, the best Stage157 raw-energy row. Errors are meV/atom for energy and meV/A for forces; benchmark is `valid.extxyz :1024`, graph construction excluded.

| model | E RMSE | E MAE | E max | rel-img RMSE | barrier RMSE | group-offset RMSE | F RMSE | F MAE | F max | atoms/s | params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `stage157_direct_b32_rel0p25_mixed2048` | 52.538 | 35.443 | 204.986 | 7.104 | 11.403 | 4.769 | 94.159 | 41.823 | 2092.961 | 500402 | 54886 |
| `stage158_cavity_uquproj_b32_rel0p25_mixed2048` | 55.098 | 35.705 | 191.412 | 8.354 | 13.348 | 5.588 | 99.123 | 46.277 | 2130.153 | 331263 | 55144 |
| `stage158_cavity_uv_uquproj_b32_rel0p25_mixed2048` | 60.252 | 45.617 | 186.756 | 7.354 | 12.826 | 5.401 | 102.024 | 47.787 | 2620.883 | 328409 | 55402 |
| `stage158_cavity_uvproj_b32_rel0p25_mixed2048` | 69.400 | 47.173 | 210.051 | 7.647 | 12.902 | 5.341 | 98.612 | 45.722 | 2154.208 | 494683 | 55144 |

## Delta Versus Stage157 rel0.25

| variant | dE RMSE | dE MAE | dE max | drel-img | dbarrier | dF RMSE | dF max | throughput ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `stage158_cavity_uquproj_b32_rel0p25_mixed2048` | 2.559 | 0.263 | -13.574 | 1.250 | 1.946 | 4.964 | 37.191 | 0.662 |
| `stage158_cavity_uv_uquproj_b32_rel0p25_mixed2048` | 7.714 | 10.175 | -18.230 | 0.250 | 1.423 | 7.865 | 527.921 | 0.656 |
| `stage158_cavity_uvproj_b32_rel0p25_mixed2048` | 16.862 | 11.730 | 5.065 | 0.543 | 1.500 | 4.453 | 61.246 | 0.989 |

## Teacher Benchmark

| variant | E RMSE | E MAE | E max | rel-img RMSE | barrier RMSE | F RMSE | F MAE | F max | atoms/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `stage158_cavity_uquproj_b32_rel0p25_mixed2048` | 32.336 | 22.307 | 97.503 | 7.347 | 10.840 | 95.752 | 46.158 | 2164.201 | 331359 |
| `stage158_cavity_uv_uquproj_b32_rel0p25_mixed2048` | 37.810 | 29.869 | 104.151 | 6.350 | 10.429 | 98.038 | 47.490 | 2465.767 | 328557 |
| `stage158_cavity_uvproj_b32_rel0p25_mixed2048` | 46.879 | 32.410 | 138.296 | 6.696 | 10.457 | 95.412 | 46.050 | 2196.336 | 494270 |

## Interpretation

1. Stage158 is a negative but useful architecture result. None of the edge-frame projection rows improves the Stage157 rel0.25 Pareto point on DFT E RMSE or F RMSE.
2. `uQu` lowers DFT E max from 204.986 to 191.412 meV/atom, and combined lowers it to 186.756, but both worsen E RMSE, F RMSE, relative-image RMSE, and barrier RMSE. This is not a promotion-quality tradeoff.
3. `u dot v` keeps throughput close to Stage157 but substantially worsens raw E RMSE to 69.400 meV/atom. The quadrupole projection rows also reduce benchmark throughput to about 0.66x Stage157.
4. The clean conclusion is not that rTECE edge relations are wrong; it is that these hand-added full-radial-mean edge-frame projections are not the current bottleneck solution. The next document-aligned step should use teacher-selected or learned low-rank edge-relational paths, or move to a projection/active-set criterion before training, rather than adding more manual projections.

## Priority Update

- Do not promote Stage158 projection paths into the current Pareto front.
- Keep Stage157 rel0.25 as the current direct high-throughput baseline for this branch.
- Next algorithmic stage should target teacher-conditioned active path selection or learned low-rank edge relational gates, because fixed hand-selected `u dot v/uQu` paths add cost without solving energy RMSE.
