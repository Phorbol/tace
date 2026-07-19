# Stage130 Sobolev-weighted results

Two real sbatch jobs completed after fixing the weighted-extxyz label preservation bug: species20 job 685084 and species24 job 685083. Both used max_steps=20000, Lightning, batch_size=8, warmup=500, plateau scheduler, and early_stopping_patience=400.

| row | params | DFT E RMSE | DFT F RMSE | DFT F max | atoms/s | delta F RMSE vs stage129 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| l1_active_nrad12_species20_radial_species8_cross3_h64 | 25449 | 293.637 | 119.029 | 2396.761 | 1280703 | 11.376 |
| l1_active_nrad12_species24_radial_species8_cross3_h64 | 28925 | 305.026 | 120.172 | 2300.489 | 1222159 | 18.615 |

Interpretation:

- Standard DFT force RMSE got worse for both rows, so this is not a clean Pareto-front improvement.
- Species20 slightly improves DFT energy RMSE versus stage129, but F RMSE worsens from 107.653 to 119.029 meV/A.
- Species24 worsens E/F RMSE, but DFT F max error drops from 2697.533 to 2300.489 meV/A, so the weighting may be buying tail control at the cost of average force accuracy.
- Next stage should run physical triage on these checkpoints. If dimer/rattle gates do not improve, weighting-only changes should be deprioritized in favor of representation/projection changes aligned with TECE edge-relational scalar sketches or broader teacher trajectory/Jacobian distillation.
