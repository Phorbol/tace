# rTECE Physical Pareto Summary

## Ranked Rows

| variant | gate | score | atoms/s | DFT F MAE | dimer repulsive | C_or_N RMSD | max fmax | params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| l1_active_radial_species8_h64 | 0 | 17.615 | 1671492.012 | 46.315 | 1.000 | 0.198 | 4.089 | 17209 |

## Physical Pareto Front

| variant | score | atoms/s | DFT F MAE | C_or_N RMSD | max fmax |
|---|---:|---:|---:|---:|---:|
| l1_active_radial_species8_h64 | 17.615 | 1671492.012 | 46.315 | 0.198 | 4.089 |

## Gate Definition

- `physical_score` is lower-is-better: benchmark score using DFT force RMSE when available plus optional E RMSE and E/F max-error terms, then normalized `C_or_N` rattle RMSD, normalized rattle max fmax, dimer force/nonfinite penalties, and short-range dimer energy-shape penalty.
- `physical_gate_pass` requires benchmark, dimer, and rattle gates to pass at the configured thresholds.
- The rattle focus label is a configurable stress-test dimension, not an architecture-specific or element-specialized training objective.
- This score is a checkpoint-selection aid, not a replacement for the TECE path manifest or full Pareto table.
