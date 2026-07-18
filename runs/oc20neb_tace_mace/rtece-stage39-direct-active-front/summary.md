# Stage 39: Direct-Active Scalar rTECE Pareto Front

Stage 39 rebenchmarks the Stage-28 scalar rTECE front under the Stage-38 direct-active graph semantics. The checkpoints, scalar descriptors, force mode, labels, and validation windows are unchanged; only graph construction changes from ASE/PBC neighbor entries to direct-distance active edges via `--graph-construction-backend torch_radius_nopbc`.

GPU setup: DFT and teacher valid `:4096`, 250355 atoms, float32, one V100, `--force-mode auto` resolving to `analytic_element_triton_descriptor_force`, `measure_passes=5`.

| point | radial | hidden | params | direct-active atoms/s | DFT F MAE | teacher F MAE | peak alloc MB |
|---|---:|---|---:|---:|---:|---:|---:|
| radial4h16 | 4 | 16x16 | 449 | 87762046 | 43.85 | 47.71 | 150.8 |
| radial5h16_stage24 | 5 | 16x16 | 481 | 86308965 | 38.72 | 42.80 | 152.3 |
| radial5h16_seed2_long | 5 | 16x16 | 481 | 82331741 | 38.26 | 42.22 | 152.3 |
| radial8h24 | 8 | 24x24 | 1057 | 64507178 | 30.20 | 35.84 | 186.4 |
| radial8h32 | 8 | 32x32 | 1665 | 48192947 | 27.87 | 33.73 | 217.0 |

Comparison against Stage-28 ASE/PBC graph front on DFT valid `:4096`:

| point | ASE/PBC atoms/s | direct-active atoms/s | speedup | ASE/PBC DFT F MAE | direct-active DFT F MAE | delta MAE |
|---|---:|---:|---:|---:|---:|---:|
| radial4h16 | 72644836 | 87762046 | 1.21x | 43.91 | 43.85 | -0.06 |
| radial5h16_stage24 | 64463762 | 86308965 | 1.34x | 39.02 | 38.72 | -0.30 |
| radial5h16_seed2_long | 64515626 | 82331741 | 1.28x | 38.28 | 38.26 | -0.02 |
| radial8h24 | 50929958 | 64507178 | 1.27x | 30.19 | 30.20 | +0.01 |
| radial8h32 | 40750024 | 48192947 | 1.18x | 28.14 | 27.87 | -0.27 |

Interpretation:

- Direct-active graph construction improves every point on the scalar rTECE front by about 1.18-1.34x without a meaningful force-error penalty.
- The current high-throughput endpoint is now radial4h16 at 87.8M atoms/s and 43.85 meV/A DFT force MAE.
- The current lower-error scalar endpoint is radial8h32 at 48.2M atoms/s and 27.87 meV/A DFT force MAE.
- The best balanced direct-active point remains radial8h24: 64.5M atoms/s at 30.20 meV/A DFT force MAE, with much lower scalar-head cost than radial8h32.
- This confirms that Stage38 was not a radial8h24-specific artifact. Direct-active graph construction should be the default semantics for the current high-throughput rTECE direct-distance branch.
- A PBC-correct branch remains separate: it must add periodic displacement/shift state to `RTECEGraph` and the fused evaluator before it can be compared fairly.
