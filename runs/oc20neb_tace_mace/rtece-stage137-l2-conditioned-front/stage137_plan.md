# Stage137 L2 Conditioned Front Plan

- stage: `stage137_l2_conditioned_front`
- row set: `stage137-l2-conditioned-front`
- semantics: `stage136_force_projection_guided_l2_low_rank_conditioning`
- train file: `runs/oc20neb_tace_mace/rtece-stage132-broad-teacher-distill/augmented_train_base2048_plus_teacher_rattle512.extxyz`
- train limit: `2560` configs

## Question

Stage136 force projection says L2 atomic quadrupole descriptors contain useful force-response information, but Stage135 naive full-L2 training is not Pareto. On the fixed Stage132 broad-teacher-rattle data and fixed 64,64 head, can low-rank descriptor mixing or zero-initialized residual descriptor conditioning convert that force projection signal into lower DFT E/F RMSE/max or better rattle physical robustness without edge-sketch cost?

## Rows

| variant | params | representation params | readout params | front control | scalar paths |
|---|---:|---:|---:|---|---|
| l2_active_nrad12_species24_radial_species8_cross3_bneck32_h64 | 21405 | 15004 | 6401 | bottleneck=32 | atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius |
| l2_active_nrad12_species24_radial_species8_cross3_cond32_h64 | 51367 | 25894 | 25473 | conditioner=residual_mlp:32 | atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius |

## Review Basis

- TECE_design_space: keep the degradation axis explicit as L_A=2 immediate scalarization, sparse scalar paths, no persistent high-l node state, no attention, and no per-edge CxC matrices.
- TECE_design_space: parameter growth should enter the representation/path compiler, not just the final scalar head.
- rTECE_review: compare E/F RMSE and max tails separately from physical probes; do not use a single MAE gate as the research objective.
- Stage120b/121: generic bottlenecks on older L2/cavity rows were not Pareto; Stage137 retests bottleneck/conditioning only on the Stage132/135 active atomic basis isolated by Stage136 projection evidence.
- Stage125: front representation capacity can improve RMSE when rank is allocated to active paths, but bigger ranks are not monotonic; Stage137 keeps the matrix narrow and diagnostic.
