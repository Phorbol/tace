# Stage132 Broad Teacher-Rattle Distillation Plan

- stage: `stage132_broad_teacher_distill_projection_check`
- row set: `stage132-broad-teacher-distill`
- semantics: `broader_teacher_fake_labels_on_deployment_rattle_window`
- source window: `0:32`
- teacher rattles: `512` configs
- augmented train limit: `2560` configs

## Question

Does broader teacher-rattle deployment coverage reduce the current high-force physical tail, separating projection error from distillation error, when the stage129 atomic Pareto architecture and the stage131 minimal L1 edge-residual architecture are held fixed?

## Rows

| variant | params | representation params | role |
|---|---:|---:|---|
| l1_active_nrad12_species24_radial_species8_cross3_h64 | 28925 | 4412 | fixed architecture from the stage129/stage131 decision boundary; only broader deployment-distribution teacher-rattle coverage changes |
| l1_active_species24_cavity_vec_residual_h64 | 29117 | 4412 | fixed architecture from the stage129/stage131 decision boundary; only broader deployment-distribution teacher-rattle coverage changes |

## Steps

### make_broad_deployment_rattles

```bash
${TACE_PYTHON:-python} benchmarks/oc20neb_tace_mace/make_rattle_distill_configs.py --input /home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz --output runs/oc20neb_tace_mace/rtece-stage132-broad-teacher-distill/rattled_valid_start0_limit32_copies16_std0p05.extxyz --summary runs/oc20neb_tace_mace/rtece-stage132-broad-teacher-distill/rattled_valid_start0_limit32_copies16_std0p05_summary.json --start-config 0 --limit-configs 32 --copies-per-config 16 --rattle-std-a 0.05 --seed 20260720
```

### teacher_label_broad_deployment_rattles

```bash
${TACE_PYTHON:-python} benchmarks/oc20neb_tace_mace/distill_tace_labels.py --model runs/oc20neb_tace_mace/675969_resume/checkpoints_epoch/TACE-0-100000-0.1329.ckpt --input runs/oc20neb_tace_mace/rtece-stage132-broad-teacher-distill/rattled_valid_start0_limit32_copies16_std0p05.extxyz --output runs/oc20neb_tace_mace/rtece-stage132-broad-teacher-distill/teacher_labeled_broad_rattle512.extxyz --summary runs/oc20neb_tace_mace/rtece-stage132-broad-teacher-distill/teacher_labeled_broad_rattle512_summary.json --device cuda --default-dtype float32 --reference-prefix source_
```

### concat_base_and_broad_teacher_rattles

```bash
${TACE_PYTHON:-python} benchmarks/oc20neb_tace_mace/concat_extxyz_datasets.py --input runs/oc20neb_tace_mace/tece-distill-20260717/mixed_train_tw0.75.extxyz --source-label base_mixed_train_tw0p75 --input runs/oc20neb_tace_mace/rtece-stage132-broad-teacher-distill/teacher_labeled_broad_rattle512.extxyz --source-label teacher_labeled_broad_rattle512 --input-limit 2048 --input-limit -1 --output runs/oc20neb_tace_mace/rtece-stage132-broad-teacher-distill/augmented_train_base2048_plus_teacher_rattle512.extxyz --summary runs/oc20neb_tace_mace/rtece-stage132-broad-teacher-distill/augmented_train_base2048_plus_teacher_rattle512_summary.json
```

## Review Basis

- TECE_design_space: first separate projection error from distillation error before adding new operator axes.
- TECE_design_space: deployment distribution mu and physical Sobolev metrics determine which TECE coordinates are worth retaining.
- rTECE_review: teacher E/F distillation closure and physical tests are P1 requirements, not optional benchmark decoration.
- Stage129/131: tiny local fake-label patch and naive higher-L edge residuals did not form a clean RMSE/throughput/physical Pareto improvement.
