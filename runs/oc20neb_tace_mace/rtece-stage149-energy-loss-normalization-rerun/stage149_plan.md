# Stage149 Energy-Loss Normalization Rerun

## Question
After fixing batched energy loss normalization from total-batch atoms to per-configuration natoms, do the Stage142 L2 anchor and Stage144 minimal T3 cavity-vector rTECE points recover absolute energy RMSE without destroying force RMSE, relative NEB shape, throughput, or physical probes?

## Why this stage exists
- `loss_for_batch` previously divided every configuration energy error by the total atoms in the whole batch.
- Stage140-144 Lightning runs used `batch_size=8`, so their effective energy objective was much weaker than intended.
- This must be rerun before deciding whether the large absolute E RMSE is an architecture limit, a source/case energy reference issue, or mainly an optimization bug.

## Rows
| row | source | scalar paths | wrapper |
|---|---|---|---|
| `stage149_l2_anchor_lossfix` | `stage142_l2_active_nrad12_species24_radial_species8_cross3_cond32_h64` | `atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius` | `runs/oc20neb_tace_mace/rtece-stage149-energy-loss-normalization-rerun/wrappers/stage149_l2_anchor_lossfix/rtece_scalar_matrix_no_export.sbatch` |
| `stage149_t3_cavity_vector_lossfix` | `stage144_t3_l2_cavity_vector_cond32_h64` | `atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot` | `runs/oc20neb_tace_mace/rtece-stage149-energy-loss-normalization-rerun/wrappers/stage149_t3_cavity_vector_lossfix/rtece_scalar_matrix_no_export.sbatch` |

## Acceptance Metrics
- Report E/F MAE, RMSE, max, and E bias, with RMSE as the primary accuracy comparator.
- Preserve Stage148 energy decomposition: raw, global offset, case mean-offset, first-image anchor, relative image, and barrier errors.
- Run throughput and physical probes before moving back to architecture-capacity changes.
