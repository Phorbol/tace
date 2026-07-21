# Stage157 Relative NEB Loss Summary

Stage157 tests whether the observed poor raw absolute energy MAE/RMSE can be improved by adding a gauge-invariant NEB relative-energy objective while keeping the Stage155 high-throughput direct T3 architecture fixed.

## Job Status

| variant | job | state | elapsed | relative weight | best step | best valid loss |
|---|---:|---|---:|---:|---:|---:|
| `stage157_direct_b32_rel0_mixed2048` | 687503 | COMPLETED | 00:18:57 | 0.000 | 15680 | 0.077487 |
| `stage157_direct_b32_rel0p25_mixed2048` | 687504 | COMPLETED | 00:19:31 | 0.250 | 16896 | 0.075936 |
| `stage157_direct_b32_rel1p0_mixed2048` | 687505 | COMPLETED | 00:19:25 | 1.000 | 16896 | 0.074437 |

All three Slurm jobs completed with exit code 0:0. The committed artifacts are lightweight JSON/Markdown summaries only; checkpoints remain local run artifacts.

## DFT Benchmark

Errors are measured on `valid.extxyz :1024`. Energy units are meV/atom; force units are meV/A. Throughput excludes graph construction and uses the prebuilt batched graph path, matching the current rTECE benchmark convention.

| variant | rel w | E RMSE | E MAE | E max | E bias | rel-img RMSE | barrier RMSE | group-offset RMSE | F RMSE | F MAE | F max | atoms/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `stage157_direct_b32_rel0_mixed2048` | 0.000 | 58.067 | 40.886 | 219.783 | 15.895 | 7.207 | 11.171 | 4.811 | 95.224 | 42.690 | 2095.459 | 500308 |
| `stage157_direct_b32_rel0p25_mixed2048` | 0.250 | 52.538 | 35.443 | 204.986 | 9.872 | 7.104 | 11.403 | 4.769 | 94.159 | 41.823 | 2092.961 | 500402 |
| `stage157_direct_b32_rel1p0_mixed2048` | 1.000 | 53.229 | 36.156 | 201.196 | 7.580 | 7.087 | 11.139 | 4.774 | 95.898 | 42.285 | 2150.411 | 499833 |

## Teacher Benchmark

| variant | rel w | E RMSE | E MAE | E max | E bias | rel-img RMSE | barrier RMSE | group-offset RMSE | F RMSE | F MAE | F max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `stage157_direct_b32_rel0_mixed2048` | 0.000 | 35.233 | 26.937 | 125.874 | 11.919 | 6.231 | 8.841 | 4.109 | 91.233 | 42.765 | 1854.185 |
| `stage157_direct_b32_rel0p25_mixed2048` | 0.250 | 28.340 | 19.935 | 111.076 | 5.896 | 6.170 | 9.068 | 4.086 | 90.453 | 42.103 | 1851.688 |
| `stage157_direct_b32_rel1p0_mixed2048` | 1.000 | 28.853 | 20.850 | 107.284 | 3.604 | 6.152 | 8.834 | 4.069 | 92.142 | 42.549 | 1909.137 |

## Comparison To Stage155 Direct

| model | E RMSE | E MAE | E max | rel-img RMSE | barrier RMSE | F RMSE | F MAE | F max | atoms/s | params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `stage155_t3_cavity_vector_direct_mixed2048` | 71.830 | 50.575 | 270.534 | 8.123 | 12.830 | 100.744 | 46.546 | 1946.920 | 499526 | 54886 |
| `stage157_direct_b32_rel0_mixed2048` | 58.067 | 40.886 | 219.783 | 7.207 | 11.171 | 95.224 | 42.690 | 2095.459 | 500308 | 54886 |
| `stage157_direct_b32_rel0p25_mixed2048` | 52.538 | 35.443 | 204.986 | 7.104 | 11.403 | 94.159 | 41.823 | 2092.961 | 500402 | 54886 |
| `stage157_direct_b32_rel1p0_mixed2048` | 53.229 | 36.156 | 201.196 | 7.087 | 11.139 | 95.898 | 42.285 | 2150.411 | 499833 | 54886 |

## Interpretation

1. The user observation is correct: raw absolute energy remains poor. The best Stage157 DFT raw E RMSE is 52.538 meV/atom, still far from a high-accuracy potential and still accompanied by E max above 200 meV/atom.
2. Relative energy loss helps absolute energy modestly but does not solve the core representation problem. Moving from rel0 to rel0.25 improves DFT raw E RMSE by 5.529 meV/atom and E max by 14.797 meV/atom, but relative-image RMSE only changes from 7.207 to 7.104 meV/atom and barrier RMSE slightly worsens from 11.171 to 11.403 meV/atom. rel1.0 is similar.
3. The teacher benchmark shows the same pattern: raw E RMSE improves from 35.233 to about 28.3-28.9 meV/atom, but relative-image/barrier errors barely move. This means the objective is reducing low-frequency energy residuals more than adding missing local PES shape information.
4. Compared with Stage155 direct, the batch32/ZBL/current-training stack is a real improvement: DFT E RMSE drops from 71.830 to 52.538-58.067 meV/atom and F RMSE from 100.744 to 94.159-95.898 meV/A while keeping about 5.0e5 atoms/s in this benchmark. But this is still not a clean Pareto-front answer because absolute energy and force tails remain weak.
5. The document-grounded priority should therefore shift away from sweeping relative-loss weights. The next useful axis is the TECE_design_space/rTECE_review axis: add learnable, symmetry-preserving front-loaded representation modules and true edge-relational scalar sketches, then re-test energy/force RMSE, max errors, relative path/barrier, and physical extrapolation.

## Priority Update

- Do not treat raw E MAE/RMSE as a disposable metric. Stage156 showed a gauge component; Stage157 shows the residual is not only gauge. Keep reporting E RMSE/MAE/max, F RMSE/MAE/max, relative image/barrier, and physical tests together.
- Stop spending stages on more relative-loss scalar weights unless a new diagnostic suggests the relative objective is inactive. It is active, but it is not the dominant bottleneck.
- Implement the next rTECE representation rung from the documents: non-duplicated edge-relational m_total=0 sketches, learnable radial/species low-rank mixing with constraints, and possibly Lmax/path-count controls that expose a systematic Pareto ladder rather than a one-off hand-picked descriptor.
- Keep using the mature TACE/rTECE Lightning training path with warmup, plateau scheduling, early stopping, correct E0 handling, ZBL, batch-level relative metadata when needed, and production ASE calculator compatibility.
