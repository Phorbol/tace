# Stage139 Conditioned Relax Distillation Plan

- stage: `stage139_conditioned_relax_distill`
- row set: `stage139-conditioned-relax-distill`
- semantics: `fixed_stage137_l2_conditioned_front_on_stage134_dft_anchor_teacher_relax_weighted_train`
- train file: `runs/oc20neb_tace_mace/rtece-stage134-balanced-teacher-relax/weighted_train_base2048_plus_teacher_relax320_eanchor.extxyz`
- train configs: `2368`

## Question

Does DFT-anchored teacher-relax trajectory distillation become useful when the student has Stage137 L2 conditioned-front representation capacity, or do RMSE/max/physical failures persist, indicating architectural projection error rather than only data/distillation error?

## Rows

| variant | params estimate | representation params | role |
|---|---:|---:|---|
| l2_active_nrad12_species24_radial_species8_cross3_bneck32_h64 | 21405 | 15004 | bottleneck L2 front |
| l2_active_nrad12_species24_radial_species8_cross3_cond32_h64 | 51367 | 25894 | residual conditioned L2 front |

## Design Basis

- This is a fixed-architecture data/distillation ablation, not edge widening or final-head widening.
- It reuses the full rTECE matrix training workflow: fitted per-element E0s, `energy_per_atom_shift=0`, batch 8, valid batch 16, 20k max steps, warmup 500, plateau scheduler, early stopping patience 400.
- It uses the Stage134 weighted teacher-relax training set to keep DFT structures as energy anchors while letting teacher relaxation trajectories contribute force/Sobolev-like deployment information.

## Review Basis

- TECE_design_space: deployment distribution and Sobolev/teacher force targets should be tested before adding uncontrolled edge/head capacity.
- TECE_design_space: distillation error and projection error must be separated; a fixed architecture/data ablation is the clean test.
- rTECE_review: teacher E/F distillation should be a formal workflow with physical gates and E/F RMSE/max, not just MAE or demo labels.
- Stage138: minimal edge residual has weak tail signal but halves throughput, so the next variable is distillation/data on the atomic conditioned front.
- Stage133/134: teacher-relax data was tested only on the L1 atomic row; Stage139 checks whether the more trainable Stage137 L2 front can use it.
