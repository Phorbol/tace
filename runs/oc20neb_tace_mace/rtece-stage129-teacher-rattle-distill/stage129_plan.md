# Stage129 Teacher-Rattle Distillation Plan

- stage: `stage129_teacher_rattle_distill_current_pareto`
- row set: `stage129-current-pareto`
- semantics: `teacher_fake_labels_on_stage128_rattle_window`
- source window: `58:66`
- augmented train limit: `2176` configs

## Rows

| variant | params | representation params | role |
|---|---:|---:|---|
| l1_active_nrad12_species20_radial_species8_cross3_h64 | 25449 | 4008 | fixed stage128 physical Pareto architecture; only the deployment-distribution teacher-rattle coverage changes relative to stage127 |
| l1_active_nrad12_species24_radial_species8_cross3_h64 | 28925 | 4412 | fixed stage128 physical Pareto architecture; only the deployment-distribution teacher-rattle coverage changes relative to stage127 |

## Steps

### make_stage128_window_rattles

```bash
${TACE_PYTHON:-python} benchmarks/oc20neb_tace_mace/make_rattle_distill_configs.py --input /home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz --output runs/oc20neb_tace_mace/rtece-stage129-teacher-rattle-distill/rattled_valid_start58_limit8_copies16_std0p05.extxyz --summary runs/oc20neb_tace_mace/rtece-stage129-teacher-rattle-distill/rattled_valid_start58_limit8_copies16_std0p05_summary.json --start-config 58 --limit-configs 8 --copies-per-config 16 --rattle-std-a 0.05 --seed 20260720
```

### teacher_label_stage128_window_rattles

```bash
${TACE_PYTHON:-python} benchmarks/oc20neb_tace_mace/distill_tace_labels.py --model runs/oc20neb_tace_mace/675969_resume/checkpoints_epoch/TACE-0-100000-0.1329.ckpt --input runs/oc20neb_tace_mace/rtece-stage129-teacher-rattle-distill/rattled_valid_start58_limit8_copies16_std0p05.extxyz --output runs/oc20neb_tace_mace/rtece-stage129-teacher-rattle-distill/teacher_labeled_stage128_window_rattles.extxyz --summary runs/oc20neb_tace_mace/rtece-stage129-teacher-rattle-distill/teacher_labeled_stage128_window_rattles_summary.json --device cuda --default-dtype float32 --reference-prefix source_
```

### concat_base_and_teacher_rattles

```bash
${TACE_PYTHON:-python} benchmarks/oc20neb_tace_mace/concat_extxyz_datasets.py --input runs/oc20neb_tace_mace/tece-distill-20260717/mixed_train_tw0.75.extxyz --source-label base_mixed_train_tw0p75 --input runs/oc20neb_tace_mace/rtece-stage129-teacher-rattle-distill/teacher_labeled_stage128_window_rattles.extxyz --source-label teacher_labeled_stage128_rattles --input-limit 2048 --input-limit -1 --output runs/oc20neb_tace_mace/rtece-stage129-teacher-rattle-distill/augmented_train_base2048_plus_teacher_rattle128.extxyz --summary runs/oc20neb_tace_mace/rtece-stage129-teacher-rattle-distill/augmented_train_base2048_plus_teacher_rattle128_summary.json
```

### make_current_pareto_train_wrappers

```bash
${TACE_PYTHON:-python} benchmarks/oc20neb_tace_mace/make_rtece_pareto_sweep.py --output-dir runs/oc20neb_tace_mace/rtece-stage129-teacher-rattle-distill/current_pareto_wrappers --run-root runs/oc20neb_tace_mace/rtece-stage129-teacher-rattle-distill/current_pareto_runs --train-file runs/oc20neb_tace_mace/rtece-stage129-teacher-rattle-distill/augmented_train_base2048_plus_teacher_rattle128.extxyz --train-valid-file /home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz --dft-valid-file /home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz --teacher-valid-file runs/oc20neb_tace_mace/tece-distill-20260717/teacher_valid.extxyz --row-set stage129-current-pareto --limit-configs 2176 --valid-limit-configs 256 --bench-limit-configs 1024 --max-steps 20000 --batch-size 8 --valid-batch-size 16 --lr-warmup-steps 500 --early-stopping-patience 400
```

## Review Basis

- TECE_design_space: deployment error separates projection error from distillation error.
- TECE_design_space stage D: teacher E/F distillation should precede architecture-specific hacks.
- rTECE_review: teacher cache/distillation closure is P1 and must be evaluated with E/F RMSE and physical tests.
- Stage128: dimer and C/N RMSD are not dominant; rattle force spikes are the current falsification target.
