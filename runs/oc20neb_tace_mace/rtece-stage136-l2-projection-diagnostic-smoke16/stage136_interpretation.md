# Stage136 L2 Projection Diagnostic Plan

- stage: `stage136_l2_projection_diagnostic`
- semantics: `stage135_l2_reference_vs_stage132_l1_projection_residual`
- train configs: `runs/oc20neb_tace_mace/rtece-stage132-broad-teacher-distill/augmented_train_base2048_plus_teacher_rattle512.extxyz`
- valid configs: `/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz`
- limit configs: `16`
- active row ranks: nrad `12`, species `24`, cross `3`

## Question

After Stage135 showed lower validation loss but worse throughput/F-RMSE/max, measure projection error and held-out E/F label residuals for the Stage132 L1 atomic subspace inside the Stage135 L_A=2 quadrupole reference. This separates true L2 projection value from optimization/distillation failure.

## Candidates

| candidate | paths |
|---|---|
| full_l2_reference | atomic.radial_density, atomic.species_basis_density, atomic.vector_norm, atomic.vector_cross_radial_dot, atomic.quadrupole_norm, atomic.quadrupole_cross_radial_frobenius |
| stage132_l1_atomic | atomic.radial_density, atomic.species_basis_density, atomic.vector_norm, atomic.vector_cross_radial_dot |
| drop_quadrupole_cross | atomic.radial_density, atomic.species_basis_density, atomic.vector_norm, atomic.vector_cross_radial_dot, atomic.quadrupole_norm |
| drop_quadrupole_norm | atomic.radial_density, atomic.species_basis_density, atomic.vector_norm, atomic.vector_cross_radial_dot, atomic.quadrupole_cross_radial_frobenius |

## Review Basis

- TECE_design_space: deleted TECE coordinates should be evaluated as projection/distillation error before choosing the next retained path set.
- rTECE_review: projection/importance diagnostics are higher priority than further kernel work when physical closure is still failing.
- Stage135: naive L_A=2 atomic quadrupole training lowered validation loss but worsened F RMSE/max and throughput, so L2 value must be diagnosed separately from optimization and label effects.
- Stage113: force-aware projection made L2 norm paths look force-relevant in a small k=2 smoke; Stage136 repeats the question on the current nrad12/species24/cross3 active row.
