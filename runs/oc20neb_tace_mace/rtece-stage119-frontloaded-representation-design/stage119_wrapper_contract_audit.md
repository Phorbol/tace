# rTECE Wrapper Contract Audit: frontloaded-representation-stage119

- contract pass: `True`
- failed rows: None

| row | pass | forbidden sbatch | missing exports | mismatched exports | hidden | Lmax |
|---|---|---|---|---|---:|---:|
| l0_species8_learnembed_h64 | True | None | None | None | 64,64 | 0 |
| l1_species16_cross_learnembed_h64 | True | None | None | None | 64,64 | 1 |
| l2_species16_atomic_cross_learnembed_h64 | True | None | None | None | 64,64 | 2 |
| l2_species16_cavity_edge_learnembed_h64 | True | None | None | None | 64,64 | 2 |
| l2_species32_cavity_edge_learnembed_h64 | True | None | None | None | 64,64 | 2 |
| l2_species32_cavity_atomic_cross_learnembed_h64 | True | None | None | None | 64,64 | 2 |

## Review Basis

- SAI: no sbatch --export, --mem, or --cpus-per-task in wrappers
- TECE_design_space Stage E: comparable hardware Pareto rows require identical data/training/evaluation contracts
- rTECE_review: report E/F RMSE, high-force tails, physical diagnostics only after contract-equivalent runs
