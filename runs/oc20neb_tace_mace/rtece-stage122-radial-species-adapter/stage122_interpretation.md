# Stage122 Radial Species Adapter Interpretation

Date: 2026-07-20

Code baseline: `4a8419d` (`Add rTECE radial species adapter stage122`)

## What Changed

Stage122 tests a front-representation change rather than a wider readout head. The new model path adds a zero-initialized, trainable center-neighbor species-conditioned radial adapter before moment aggregation and edge sketches:

- controlled knob: `radial_species_adapter_channels`
- placement: edge radial basis before `compute_atomic_moments`, `edge_relational_sketches`, and descriptors
- initialization: identity at step zero, so the model starts from the fixed radial basis
- intent: test whether fixed radial/species features are a real representation bottleneck

This follows the direction in `TECE_design_space.md` that the rTECE/TACE family should be explored as a controlled compiler over physical operator paths, not as a pure final-MLP size sweep. It also directly addresses the `rTECE_review.md` concern that the previous best rows were too close to a fixed-feature scalar endpoint.

## Full sbatch Runs

The completed Slurm jobs were:

- `684956`: `l0_species8_radial_species8_h64`
- `684957`: `l1_active_radial_species8_h64`
- `684958`: `l1_active_radial_species16_h64`
- `684959`: `l0_pair_radial_species8_h64`

All four jobs completed with exit code `0:0`. Submission used generated wrapper sbatch files and did not use command-line `sbatch --export`.

## Results

Primary comparison should use DFT force RMSE and throughput, while keeping energy error and max errors visible.

| row | params | atoms/s | F RMSE | F MAE | F max | E RMSE | E MAE | E bias | E max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| l0_pair_radial_species8_h64 | 6673 | 2.570e6 | 134.670 | 49.482 | 4695.332 | 335.338 | 274.571 | 84.011 | 807.300 |
| l0_species8_radial_species8_h64 | 11577 | 2.040e6 | 123.440 | 50.707 | 2905.086 | 318.140 | 244.790 | 18.573 | 734.122 |
| l1_active_radial_species8_h64 | 17209 | 1.671e6 | 112.838 | 46.315 | 1996.323 | 335.816 | 268.928 | -41.688 | 731.924 |
| l1_active_radial_species16_h64 | 18953 | 1.661e6 | 115.390 | 48.508 | 2364.083 | 333.318 | 262.331 | -31.542 | 741.273 |

Stage122 best by DFT F RMSE is `l1_active_radial_species8_h64`:

- F RMSE: `112.838` meV/A
- atoms/s: `1.671e6`
- E RMSE: `335.816` meV/atom
- F max: `1996.323` meV/A

Against the previous local references:

- Stage119 L0 reference: `111.498` F RMSE, `304.855` E RMSE, `2.403e6` atoms/s
- Stage121 best: `113.073` F RMSE, `302.927` E RMSE, `1.910e6` atoms/s

Stage122 slightly improves force RMSE relative to Stage121, but loses throughput and worsens energy RMSE. It does not beat the Stage119 throughput/accuracy point and should not be counted as a new Pareto-front model.

## Physical Diagnostics

Stage122 best was also evaluated with a real sbatch physical diagnostic job:

- job `684971`: `stage122_best_physical_no_export.sbatch`
- target row: `l1_active_radial_species8_h64`
- dimer pairs: `C-N`, `C-O`, `N-H`, `O-H`, `Cu-O`
- rattle-relax: first 8 teacher-valid configurations, `0.05` A rattle, 20 LBFGS steps

The dimer scan was finite and short-range forces were repulsive for all tested pairs, but the `Cu-O` short-minus-long energy lift was slightly negative (`-0.033` eV), which is a warning that short-range shape is not fully physical.

The rattle-relax test is more concerning:

- `C_or_N` mean final RMSD: `0.1975` A
- max final RMSD: `0.2358` A
- converged fraction: `0.0`
- max fmax: `4.089` eV/A
- physical gate: `False`

This confirms that the small Stage122 validation RMSE gain is not enough to claim better physical generalization. It supports the review-document warning that RMSE, dimer smoothness, and rattle-relax stability must remain separate acceptance axes.

## Interpretation

The useful signal is not that this exact adapter is the next deployment model. The signal is that putting trainable capacity before scalarization can recover a little force RMSE at a moderate parameter count (`17k`), while previous descriptor bottleneck/front mixer variants did not move the force RMSE front as cleanly.

The negative result is equally important: the current adapter is still too expensive because it forces autograd force evaluation and uses the ASE neighborlist graph path. It also worsens energy RMSE and fails the rattle-relax diagnostic, so it is not yet a clean TECE/rTECE renormalized model.

This suggests the next priority should be a compiler-style front basis, not another final-head sweep:

1. Keep `L_max`, active scalar paths, edge-relational sketches, radial rank, and species/radial learnable basis as explicit axes.
2. Rank candidate front paths using DFT/teacher E/F residual projection and max-error-sensitive diagnostics.
3. Implement analytic-force support only for the retained learnable front paths that actually improve the Pareto frontier.
4. Re-run the same full sbatch matrix with E/F RMSE, E/F max error, dimer smoothness, and rattle-relax metrics recorded separately.

## Current Decision

Stage122 is a valid full experiment and a useful representation-bottleneck probe, but it is not a new production Pareto point. The next algorithmic step should combine this front-adapter idea with TECE path selection: use a teacher/residual-projected active basis first, then add learnable radial/species capacity only on selected paths so the extra expressivity is cheaper and more interpretable.
