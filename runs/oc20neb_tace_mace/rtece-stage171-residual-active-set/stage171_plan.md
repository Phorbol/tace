# Stage171 Residual Active-Set Plan

- stage: `stage171_residual_active_set`
- target: `stage165_case_offset_residual_mev_atom`
- configs: `/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz`
- Stage165 source: `runs/oc20neb_tace_mace/rtece-stage165-energy-baseline-triage/stage165_energy_baseline_triage.json`
- limit configs: `512`
- case id policy: `case_id_group_id_image_id_are_grouping_or_split_metadata_only`

## Question

Can a deployable TECE/rTECE scalar path subset explain the Stage165/170 case-offset residual that dominates absolute energy RMSE, without adding non-deployable case-id offsets or widening only the final head?

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

- Stage170: current absolute E RMSE is dominated by case/site/adsorbate low-frequency offsets, not a simple E0 toggle.
- TECE_design_space.md stage B: projection/importance and active-set search should choose the low-cost subspace before full distillation.
- TECE_design_space.md T2: the most valuable rTECE tier is sparse atomic scalar contractions plus 8-32 edge-relational m_total=0 sketches.
- rTECE_review.md: do not add case-id features; use explicit semantic scalar paths, teacher/cache/downfolding, and deployable route manifests.
