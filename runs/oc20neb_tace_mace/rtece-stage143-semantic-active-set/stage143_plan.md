# Stage143 Semantic Active-Set Projection Plan

- stage: `stage143_semantic_active_set`
- semantics: `semantic_path_active_set_projection_after_stage142`
- deployment measure: `stage142_force_only_teacher_relax_distribution`
- train configs: `runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/weighted_train_base2048_plus_teacher_relax640_forceonly_eanchor.extxyz`
- valid configs: `/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz`
- limit configs: `512`
- force projection: `deferred_force_descriptor_jacobian_cost`

## Question

Stage142 showed that more same-window teacher-relax force-only coverage is not enough. Use a Schur-complement-style projection diagnostic over explicit TECE semantic path groups to decide which atomic and edge-relational paths deserve the next training budget; this is not another same-window teacher-relax expansion. The default uses 512 configs so the energy projection fit is not below the 300-334 descriptor dimensions.

## Reference Supernet

atomic.radial_density, atomic.species_basis_density, atomic.vector_norm, atomic.vector_cross_radial_dot, atomic.quadrupole_norm, atomic.quadrupole_cross_radial_frobenius, edge.cavity.vector_dot, edge.cavity.quadrupole_frobenius, edge.direct.radial

## Candidates

| candidate | tier | Lmax | marginal paths | edge scalar cost |
|---|---|---:|---|---:|
| t2_l0_species_radial | T2_scalar_endpoint | 0 | atomic.radial_density, atomic.species_basis_density | 0 |
| t2_l1_atomic_cross | T2_atomic_moment_scalarized | 1 | atomic.vector_norm, atomic.vector_cross_radial_dot | 0 |
| t2_l2_atomic_cross | T2_atomic_moment_scalarized | 2 | atomic.quadrupole_norm, atomic.quadrupole_cross_radial_frobenius | 0 |
| t3_l2_cavity_vector | T3_rtece_edge_relational | 2 | edge.cavity.vector_dot | 1 |
| t3_l2_cavity_vector_quadrupole_direct | T3_rtece_edge_relational | 2 | edge.cavity.quadrupole_frobenius, edge.direct.radial | 4 |

## Review Basis

- TECE_design_space: choose a low-cost subspace by projection error, Sobolev E/F sensitivity, and hardware cost, not by widening the final head.
- TECE_design_space: rTECE is specifically the missing middle tier with sparse edge-relational scalar ECE sketches after immediate scalarization.
- rTECE_review: after Stage142 physical closure remains false, priority moves to semantic path registry, projection diagnostics, and active-set selection.
- Stage142: same-window teacher-relax coverage did not create a new Pareto point; the next falsification should evaluate operator basis value.
