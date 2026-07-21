# Stage173 Robust Group-Heldout Residual Active-Set Plan

- stage: `stage173_robust_group_holdout_active_set`
- target: `stage165_case_offset_residual_mev_atom`
- split: `group-loocv` by `case_id`
- robust projection: intercept=True, standardize=True
- ridge grid: `1e-08,1e-06,0.0001,0.01,1,100`
- configs: `/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz`
- Stage165 source: `runs/oc20neb_tace_mace/rtece-stage165-energy-baseline-triage/stage165_energy_baseline_triage.json`
- limit configs: `512`

## Question

After Stage172 rejected the raw group-heldout descriptor projection, does a statistically safer projection with fold-local standardization, intercept, ridge sweep, and intercept-only baseline still reject the current deployable rTECE path coordinates for unseen case/site/adsorbate energy offsets?

## Candidate Ladder

| candidate | tier | marginal paths | edge paths | frame projections |
|---|---|---|---:|---:|
| t1_l0_species_radial | T1_scalar_endpoint | atomic.radial_density, atomic.species_basis_density | 0 | 0 |
| t1_l1_atomic_cross | T1_atomic_scalar_moments | atomic.vector_norm, atomic.vector_cross_radial_dot | 0 | 0 |
| t1_l2_atomic_cross | T1_atomic_scalar_moments | atomic.quadrupole_norm, atomic.quadrupole_cross_radial_frobenius | 0 | 0 |
| t2_cavity_vector | T2_rtece_edge_relational | edge.cavity.vector_dot | 1 | 0 |
| t2_cavity_quadrupole | T2_rtece_edge_relational | edge.cavity.quadrupole_frobenius | 2 | 0 |
| t2_edge_frame_projection | T2_rtece_edge_frame_projection | edge.cavity.target_vector_projection, edge.cavity.source_vector_projection, edge.cavity.target_quadrupole_projection, edge.cavity.source_quadrupole_projection | 6 | 4 |
| t2_edge_frame_plus_direct | T2_rtece_residual_supernet | edge.direct.radial | 7 | 4 |

## Review Basis

- Stage170: raw absolute E RMSE is dominated by low-frequency case/site/adsorbate offsets, not by a simple E0 toggle.
- Stage171: config-heldout residual projection can interpolate seen cases and is not a deployment proof.
- Stage172: raw group-heldout projection produced catastrophic unseen-case RMSE and must be checked against intercept/standardization/ridge baselines before architecture conclusions.
- rTECE_review.md: random configuration splits can overestimate generalization; group-heldout and baseline comparisons are required.
- TECE_design_space.md: active-set pruning must separate projection error from optimization/distillation error before promoting path groups.
