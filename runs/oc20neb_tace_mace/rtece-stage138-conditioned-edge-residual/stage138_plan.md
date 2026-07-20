# Stage138 Conditioned Edge Residual Plan

- stage: `stage138_conditioned_edge_residual`
- variant: `l2_active_nrad12_species24_radial_species8_cross3_cond32_cavity_vec_direct_h64`
- train data: `runs/oc20neb_tace_mace/rtece-stage132-broad-teacher-distill/augmented_train_base2048_plus_teacher_rattle512.extxyz` (`2560` configs)
- training contract: `20000` steps, batch `8`, valid batch `16`, plateau LR, warmup `500`, early stopping patience `400`
- sbatch policy: wrapper uses in-script environment assignments; no `--export`, `--mem`, or `--cpus-per-task`

## Question

Does adding the minimal TECE/rTECE cavity-vector edge relational residual to the Stage137 conditioned L2 atomic front improve E/F RMSE, max-error tails, or physical rattle/dimer behavior enough to justify the known edge-throughput cost?

## Architecture Delta

Stage138 starts from the Stage137 `cond32` L2 atomic front and adds only `edge.cavity.vector_dot` plus `edge.direct.radial`. It keeps the same `64,64` head, ZBL baseline, learnable radial/species front, and descriptor conditioner. It intentionally does not add edge quadrupole because Stage131 showed the L2 edge-quadrupole residual was dominated by the L1 cavity-vector residual.

```text
atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot,edge.direct.radial
```

## Decision Rule

- Prefer RMSE and max-error over MAE-only comparisons: E RMSE, E max, F RMSE, F max are primary.
- Edge residual is justified only if force-tail or physical robustness improves enough to compensate throughput loss.
- If E RMSE remains poor and physical still fails, next priority moves to Sobolev/teacher-Jacobian or relaxation-trajectory distillation rather than more edge width/head width.

## Document Basis

- TECE_design_space.md: rTECE/scalar-sketched TECE is sparse atomic contraction plus sparse edge-relational scalar ECE sketches.
- TECE_design_space.md: the useful T2/T3 design space is small L=0/1/2 moments plus few m_total=0 edge sketches, no persistent high-l state, no per-edge CxC matrix.
- rTECE_review.md: current prototype is valuable but not yet full renormalization/distillation; compare E/F RMSE/max and physical gates, and do not chase kernel speed before semantic closure.

## Slurm

- train wrapper: `runs/oc20neb_tace_mace/rtece-stage138-conditioned-edge-residual/conditioned_edge_wrappers/rtece_scalar_matrix_no_export.sbatch`
- run root: `runs/oc20neb_tace_mace/rtece-stage138-conditioned-edge-residual/conditioned_edge_runs`
