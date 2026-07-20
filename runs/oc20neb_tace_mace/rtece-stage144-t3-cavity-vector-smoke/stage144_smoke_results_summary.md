# Stage144 T3 Cavity Vector Smoke Summary

- status: `completed`
- train job: `687409` COMPLETED 0:0 in 00:04:47
- physical job: `687416` COMPLETED 0:0 in 00:00:09
- scope: 512 train configs, 64 valid configs, 2000 max steps; best step 64
- path manifest: `c2aba169fc550b16`

## Metrics

| metric | value |
|---|---:|
| DFT E RMSE (meV/atom) | 252.941 |
| DFT E max (meV/atom) | 344.755 |
| DFT F RMSE (meV/A) | 131.072 |
| DFT F max (meV/A) | 2719.591 |
| atoms/s | 343818.152 |
| C/N mean final RMSD (A) | 0.091 |
| rattle max fmax (eV/A) | 1.018 |
| dimer max force (eV/A) | 62.311 |

## Gates

- benchmark gate: `False`
- dimer gate: `True`
- rattle gate: `False`
- physical gate: `False`

## Interpretation

- This smoke proves the Stage144 minimal T3 cavity-vector workflow now trains, benchmarks, and runs dimer/rattle physical triage after the previous import/page-wait failure.
- The 2000-step/512-config run is not a final Pareto result and should not be used to accept or reject edge.cavity.vector_dot accuracy.
- Benchmark and rattle gates fail at smoke quality: DFT F RMSE is above the configured 120 meV/A threshold and rattle max fmax is slightly above 1.0 eV/A, while dimer scan passes without nonfinite values and all short-range pair forces are repulsive.
- The next research step remains the full Stage144 run, then comparison to Stage139/137 and the Stage145 NEP/DPA baselines under the same RMSE/max-error/physical protocol.
