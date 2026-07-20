# Stage151 DFT-vs-Mixed Energy Projection Diagnostic

- stage: `stage151_dft_vs_mixed_projection`
- purpose: diagnose why Stage149 rTECE absolute energy MAE/RMSE remains poor
- valid configs: `/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz`
- mixed train configs: `runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/weighted_train_base2048_plus_teacher_relax640_forceonly_eanchor.extxyz`
- primary metric: held-out E/atom RMSE under linear projection with element-count baseline
- secondary smoke: small held-out E/F projection on DFT valid

## Question

Stage149 fixed batched energy-loss normalization and fitted per-element atomic energies, but the benchmark still has about 90-110 meV/atom absolute E RMSE. Stage150 showed that post-hoc global or residual per-element gauge calibration does not remove this error. This stage asks whether the remaining error is caused by a simple missing energy zero, an unclean mixed-label energy contract, or insufficient TECE semantic path capacity.

## Reference Basis

The reference supernet matches Stage143:

`atomic.radial_density, atomic.species_basis_density, atomic.vector_norm, atomic.vector_cross_radial_dot, atomic.quadrupole_norm, atomic.quadrupole_cross_radial_frobenius, edge.cavity.vector_dot, edge.cavity.quadrupole_frobenius, edge.direct.radial`

Manual candidates:

| candidate | retained paths |
|---|---|
| `t2_l0_species_radial` | radial density + species basis density |
| `t2_l1_atomic_cross` | L0 + vector norm/cross radial |
| `t2_l2_atomic_cross` | L1 + quadrupole norm/cross radial |
| `t3_l2_cavity_vector` | L2 + edge cavity vector dot |
| `t3_full_reference` | all reference paths |

The wrappers also request `single_delete` and `prefix` auto-candidates to expose path-level active-set signals.

## Interpretation Rules

- If DFT valid energy projection remains low-error but mixed-train projection remains high-error, Stage149 energy failure is primarily a data-label contract/distillation issue, not just architecture capacity.
- If both DFT and mixed energy projections are high-error, the current semantic basis lacks the absolute energy structure and the next model step should expand front-loaded representations rather than widen the final head.
- If DFT energy is good but the E/F smoke ranks different paths than energy-only, the next Pareto sweep should use Sobolev E/F active-set ranking before training.

## Document Basis

- `TECE_design_space.md`: choose low-cost student subspaces by projection error, physical target and hardware cost; do not hand-pick paths only by heuristic.
- `TECE_design_space.md`: atomic-energy distillation has gauge freedom, so total E/F and relational response should dominate the closed-loop objective.
- `rTECE_review.md`: Stage branch must move from scalar student infrastructure toward teacher cache, projection Gram/Schur diagnostics and active-set basis selection.
- `rTECE_review.md`: energy RMSE/bias/max and relative NEB energies must be separated; Stage150 already ruled out a simple residual E0 fix.
