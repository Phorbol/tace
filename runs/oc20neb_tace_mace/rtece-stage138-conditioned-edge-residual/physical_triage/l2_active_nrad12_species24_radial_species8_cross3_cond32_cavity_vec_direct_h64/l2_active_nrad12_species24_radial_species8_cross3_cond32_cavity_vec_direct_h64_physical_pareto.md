# rTECE Physical Pareto Summary

## Ranked Rows

| variant | gate | score | atoms/s | DFT F MAE | dimer repulsive | C_or_N RMSD | max fmax | params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| l2_active_nrad12_species24_radial_species8_cross3_cond32_cavity_vec_direct_h64 | 0 | 13.913 | 518938.763 | 47.322 | 1.000 | 0.106 | 9.931 | 51754 |

## Physical Pareto Front

| variant | score | atoms/s | DFT F MAE | C_or_N RMSD | max fmax |
|---|---:|---:|---:|---:|---:|
| l2_active_nrad12_species24_radial_species8_cross3_cond32_cavity_vec_direct_h64 | 13.913 | 518938.763 | 47.322 | 0.106 | 9.931 |

## Gate Definition

- `physical_score` is lower-is-better: benchmark score using DFT force RMSE when available plus optional E RMSE and E/F max-error terms, then normalized `C_or_N` rattle RMSD, normalized rattle max fmax, dimer force/nonfinite penalties, and short-range dimer energy-shape penalty.
- `physical_gate_pass` requires benchmark, dimer, and rattle gates to pass at the configured thresholds.
- The rattle focus label is a configurable stress-test dimension, not an architecture-specific or element-specialized training objective.
- This score is a checkpoint-selection aid, not a replacement for the TECE path manifest or full Pareto table.
