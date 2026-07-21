# Stage180 Minimal Renormalization Proof

This stage is a proof scaffold, not a final renormalization result.

## Fixed Student

- fixed student paths: `atomic.radial_density,atomic.species_basis_density,atomic.local_l0_lowrank_density,atomic.vector_norm,atomic.vector_cross_radial_dot`
- reference proxy paths: `atomic.radial_density,atomic.species_basis_density,atomic.local_l0_lowrank_density,atomic.vector_norm,atomic.vector_cross_radial_dot,edge.cavity.vector_dot,edge.direct.radial`

## Comparison Arms

| arm | runnable now | blocking tooling | purpose |
|---|---:|---|---|
| `scratch_same_student` | True | none | Control arm: train the fixed student path set from random initialization. |
| `linear_projection_diagnostic` | True | none | Quantify projection/downfolding gap from reference proxy paths onto the fixed student path set. |
| `renorm_initialized_same_student` | False | checkpoint_initialization_from_projection_coefficients, train_entrypoint_init_checkpoint_or_init_state | Train the same fixed student path set from a projection/GN initialized checkpoint. |
| `renorm_initialized_teacher_residual_distill` | False | checkpoint_initialization_from_projection_coefficients, train_entrypoint_init_checkpoint_or_init_state, teacher_residual_cache_or_extxyz_labels, distillation_loss_mixing_real_and_teacher_labels | Use the same renormalized initialization plus teacher E/F residual distillation labels. |

## Success Criteria

- renorm_initialized_same_student beats scratch_same_student under matched E/F/relative/physical metrics
- or the result is recorded as falsifying the current renormalization recipe

## Metrics Required

- E MAE/RMSE/max
- F MAE/RMSE/max
- relative image RMSE
- barrier RMSE
- dimer scan smoothness
- rattle-relax final RMSD and force tail
- atoms/s scaling
- peak allocated/reserved memory
