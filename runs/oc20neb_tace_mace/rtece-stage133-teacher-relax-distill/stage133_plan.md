# Stage133 Teacher-Relax Trajectory Distillation Plan

- stage: `stage133_teacher_relax_trajectory_distill`
- row set: `stage133-teacher-relax-distill`
- semantics: `teacher_energy_force_labels_on_teacher_lbfgs_relaxation_trajectory`
- source window: `0:32`
- teacher trajectory frames: `320` configs
- augmented train limit: `2368` configs

## Question

Does short teacher PES relaxation manifold coverage reduce distillation error and physical force-tail failures for the fixed current-Pareto atomic rTECE front, separating projection error from distillation error before adding Lmax or edge-relational representation axes?

## Rows

| variant | params | representation params | role |
|---|---:|---:|---|
| l1_active_nrad12_species24_radial_species8_cross3_h64 | 28925 | 4412 | fixed stage129/stage132 atomic Pareto backbone; only teacher PES relaxation trajectory coverage changes |

## Steps

### make_teacher_relax_trajectory_labels

```bash
${TACE_PYTHON:-python} benchmarks/oc20neb_tace_mace/make_teacher_relax_distill_configs.py --model runs/oc20neb_tace_mace/675969_resume/checkpoints_epoch/TACE-0-100000-0.1329.ckpt --input /home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz --output runs/oc20neb_tace_mace/rtece-stage133-teacher-relax-distill/teacher_relax_valid_start0_limit32_copies2_std0p05_steps4.extxyz --summary runs/oc20neb_tace_mace/rtece-stage133-teacher-relax-distill/teacher_relax_valid_start0_limit32_copies2_std0p05_steps4_summary.json --start-config 0 --limit-configs 32 --copies-per-config 2 --rattle-std-a 0.05 --seed 20260720 --relax-max-steps ${RELAX_MAX_STEPS:-4} --device cuda --default-dtype float32 --nl-backend matscipy --reference-prefix source_
```

### concat_base_and_teacher_relax_trajectory

```bash
${TACE_PYTHON:-python} benchmarks/oc20neb_tace_mace/concat_extxyz_datasets.py --input runs/oc20neb_tace_mace/tece-distill-20260717/mixed_train_tw0.75.extxyz --source-label base_mixed_train_tw0p75 --input runs/oc20neb_tace_mace/rtece-stage133-teacher-relax-distill/teacher_relax_valid_start0_limit32_copies2_std0p05_steps4.extxyz --source-label teacher_relax_trajectory320 --input-limit 2048 --input-limit -1 --output runs/oc20neb_tace_mace/rtece-stage133-teacher-relax-distill/augmented_train_base2048_plus_teacher_relax320.extxyz --summary runs/oc20neb_tace_mace/rtece-stage133-teacher-relax-distill/augmented_train_base2048_plus_teacher_relax320_summary.json
```

## Review Basis

- TECE_design_space: deployment Sobolev metrics and projection/distillation error separation should drive row priority.
- TECE_design_space: full distillation should follow path projection, not uncontrolled head widening.
- rTECE_review: teacher E/F/V distillation and physical tests are P1 for a real rTECE, while kernel-only speedups are lower priority.
- Stage132: broader single-point teacher fake labels did not produce a clean RMSE/throughput/physical Pareto improvement.
