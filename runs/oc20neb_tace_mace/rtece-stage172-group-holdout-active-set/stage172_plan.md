# Stage172 Group-Heldout Residual Active-Set Plan

- stage: `stage172_group_holdout_active_set`
- target: `stage165_case_offset_residual_mev_atom`
- split: `group-loocv` by `case_id`
- configs: `/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz`
- Stage165 source: `runs/oc20neb_tace_mace/rtece-stage165-energy-baseline-triage/stage165_energy_baseline_triage.json`
- limit configs: `512`

## Question

Do the Stage171 deployable TECE/rTECE scalar path subsets explain the Stage165/170 low-frequency energy residual on unseen case groups, or did the config-heldout result only interpolate case-specific offsets?

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
- Stage171: config-heldout residual projection reached about 0.47 meV/atom, but that split can still interpolate seen case groups.
- rTECE_review.md: group-heldout/path-heldout splits are required because random configuration splits can overestimate deployment generalization.
- TECE_design_space.md stage B: active-set Pareto search must select deployable semantic path groups before full distillation/training.
- Stage172 keeps case_id as group metadata only; it is never passed as a descriptor, baseline feature, or model input.
