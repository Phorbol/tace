# Stage142 Teacher-Relax Coverage Plan

- stage: `stage142_teacher_relax_coverage`
- row set: `stage142-teacher-relax-coverage`
- semantics: `dft_energy_anchor_plus_force_only_teacher_relax_coverage`
- source window: `0:64`
- teacher trajectory frames: `640` configs
- augmented train limit: `2688` configs
- source energy multipliers: `{'base_mixed_train_tw0p75': 1.25, 'teacher_relax_trajectory640': 0.0}`
- source force multipliers: `{'teacher_relax_trajectory640': 2.0}`

## Question

With the Stage140/141 L2 conditioned rTECE architecture fixed, does broader teacher-relax deployment-measure coverage reduce the rattle force-tail/fmax failure while keeping DFT energy comparability through force-only teacher labels and DFT-only energy anchoring?

## Rows

| variant | params | representation params | scalar paths |
|---|---:|---:|---|
| l2_active_nrad12_species24_radial_species8_cross3_cond32_h64 | 51367 | 25894 | atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius |

## Steps

### make_teacher_relax_trajectory_labels

```bash
${TACE_PYTHON:-python} benchmarks/oc20neb_tace_mace/make_teacher_relax_distill_configs.py --model runs/oc20neb_tace_mace/675969_resume/checkpoints_epoch/TACE-0-100000-0.1329.ckpt --input /home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz --output runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/teacher_relax_valid_start0_limit64_copies2_std0p05_steps4.extxyz --summary runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/teacher_relax_valid_start0_limit64_copies2_std0p05_steps4_summary.json --start-config 0 --limit-configs 64 --copies-per-config 2 --rattle-std-a 0.05 --seed 20260720 --relax-max-steps ${RELAX_MAX_STEPS:-4} --device cuda --default-dtype float32 --nl-backend matscipy --reference-prefix source_
```

### concat_base_and_teacher_relax_trajectory

```bash
${TACE_PYTHON:-python} benchmarks/oc20neb_tace_mace/concat_extxyz_datasets.py --input runs/oc20neb_tace_mace/tece-distill-20260717/mixed_train_tw0.75.extxyz --source-label base_mixed_train_tw0p75 --input runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/teacher_relax_valid_start0_limit64_copies2_std0p05_steps4.extxyz --source-label teacher_relax_trajectory640 --input-limit 2048 --input-limit -1 --output runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/augmented_train_base2048_plus_teacher_relax640.extxyz --summary runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/augmented_train_base2048_plus_teacher_relax640_summary.json
```

### apply_dft_anchor_force_only_teacher_relax_weights

```bash
${TACE_PYTHON:-python} benchmarks/oc20neb_tace_mace/apply_extxyz_sample_weights.py --input runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/augmented_train_base2048_plus_teacher_relax640.extxyz --output runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/weighted_train_base2048_plus_teacher_relax640_forceonly_eanchor.extxyz --summary runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/weighted_train_base2048_plus_teacher_relax640_forceonly_eanchor_summary.json --source-energy-multiplier base_mixed_train_tw0p75:1.25 --source-energy-multiplier teacher_relax_trajectory640:0.0 --source-force-multiplier teacher_relax_trajectory640:2.0 --normalize-energy-mean
```

## Review Basis

- TECE_design_space: separate projection/architecture error from optimization/distillation error before changing paths.
- TECE_design_space: deployment Sobolev measure should include force and local-curvature trajectory stability, not only train/valid RMSE.
- rTECE_review: teacher distillation must preserve explicit source semantics, E/F max errors, and physical dimer/rattle probes.
- Stage140: force-only teacher-relax with weighted E0s repaired the Stage139 energy drift and became the physical-rattle reference.
- Stage141: loss weighting alone did not close rattle fmax, so coverage is the next controlled variable before edge/L/head complexity.
