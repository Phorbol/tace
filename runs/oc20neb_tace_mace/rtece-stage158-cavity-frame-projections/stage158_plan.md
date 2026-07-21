# Stage158 Cavity Edge-Frame Projection Plan

Stage157 confirmed the user observation that raw absolute energy remains poor: the best DFT E RMSE is still 52.538 meV/atom. Relative NEB loss improves the low-frequency energy residual but barely moves relative-image/barrier errors, so Stage158 changes representation rather than loss weights.

## Document Basis

- `TECE_design_space.md` defines rTECE as sparse atomic contractions plus sparse edge-relational scalar ECE sketches, especially total-m=0 edge-frame contractions.
- `rTECE_review.md` Section 4.4 names `u dot v`, `u^T Q u`, `v_i dot v_j`, and `Q_i:Q_j` cavity edge paths as the missing source-target relational layer.
- Stage157 says the next priority is not more relative-loss sweeps but learnable/symmetry-preserving front-loaded representation modules and true edge-relational scalar sketches.

## Rows

| variant | increment over Stage157 rel0.25 direct | purpose |
|---|---|---|
| `stage158_cavity_uvproj_b32_rel0p25_mixed2048` | `edge.cavity.target_vector_projection,edge.cavity.source_vector_projection` | Add the u dot v_i\j and u dot v_j\i edge-frame vector projections from rTECE_review Section 4.4. |
| `stage158_cavity_uquproj_b32_rel0p25_mixed2048` | `edge.cavity.target_quadrupole_projection,edge.cavity.source_quadrupole_projection` | Add the u^T Q_i\j u and u^T Q_j\i u edge-frame quadrupole projections from rTECE_review Section 4.4. |
| `stage158_cavity_uv_uquproj_b32_rel0p25_mixed2048` | `edge.cavity.target_vector_projection,edge.cavity.source_vector_projection,edge.cavity.target_quadrupole_projection,edge.cavity.source_quadrupole_projection` | Test the combined edge-frame projection rung before moving to more expensive learned or higher-body contractions. |

All rows keep the Stage157 best training contract: mixed2048, valid256, benchmark1024, batch32, Lightning, warmup500, plateau scheduler, early stopping 400, ZBL, relative energy weight 0.25, species learnable embedding, learnable radial mixing, learnable atomic cross-radial projection, and residual descriptor conditioner.

## Metrics

Report DFT and teacher E RMSE/MAE/max, F RMSE/MAE/max, relative image RMSE, barrier RMSE, group-offset RMSE, parameter count, and atoms/s. Do not promote a row on E MAE alone.
