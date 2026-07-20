# Stage134 Balanced Teacher-Relax Distillation Plan

- stage: `stage134_balanced_teacher_relax_distill`
- row set: `stage134-balanced-teacher-relax-distill`
- semantics: `dft_energy_anchor_plus_teacher_relax_force_weighted_distillation`
- teacher trajectory frames: `320` configs
- augmented train limit: `2368` configs
- source energy multipliers: `{'base_mixed_train_tw0p75': 1.25, 'teacher_relax_trajectory320': 0.25}`
- source force multipliers: `{'teacher_relax_trajectory320': 2.0}`

## Question

Does source-balanced teacher-relax distillation repair the stage133 energy drift while preserving force-tail benefits, thereby testing distillation-measure error before blaming projection error or changing architecture?

## Rows

| variant | params | representation params | role |
|---|---:|---:|---|
| l1_active_nrad12_species24_radial_species8_cross3_h64 | 28925 | 4412 | fixed stage133 atomic backbone; only DFT-anchor/teacher-trajectory sample weights change |

## Steps

### make_teacher_relax_trajectory_labels

```bash
${TACE_PYTHON:-python} benchmarks/oc20neb_tace_mace/make_teacher_relax_distill_configs.py --model runs/oc20neb_tace_mace/675969_resume/checkpoints_epoch/TACE-0-100000-0.1329.ckpt --input /home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz --output runs/oc20neb_tace_mace/rtece-stage134-balanced-teacher-relax/teacher_relax_valid_start0_limit32_copies2_std0p05_steps4.extxyz --summary runs/oc20neb_tace_mace/rtece-stage134-balanced-teacher-relax/teacher_relax_valid_start0_limit32_copies2_std0p05_steps4_summary.json --start-config 0 --limit-configs 32 --copies-per-config 2 --rattle-std-a 0.05 --seed 20260720 --relax-max-steps ${RELAX_MAX_STEPS:-4} --device cuda --default-dtype float32 --nl-backend matscipy --reference-prefix source_
```

### concat_base_and_teacher_relax_trajectory

```bash
${TACE_PYTHON:-python} benchmarks/oc20neb_tace_mace/concat_extxyz_datasets.py --input runs/oc20neb_tace_mace/tece-distill-20260717/mixed_train_tw0.75.extxyz --source-label base_mixed_train_tw0p75 --input runs/oc20neb_tace_mace/rtece-stage134-balanced-teacher-relax/teacher_relax_valid_start0_limit32_copies2_std0p05_steps4.extxyz --source-label teacher_relax_trajectory320 --input-limit 2048 --input-limit -1 --output runs/oc20neb_tace_mace/rtece-stage134-balanced-teacher-relax/augmented_train_base2048_plus_teacher_relax320.extxyz --summary runs/oc20neb_tace_mace/rtece-stage134-balanced-teacher-relax/augmented_train_base2048_plus_teacher_relax320_summary.json
```

### apply_dft_anchor_teacher_relax_weights

```bash
${TACE_PYTHON:-python} benchmarks/oc20neb_tace_mace/apply_extxyz_sample_weights.py --input runs/oc20neb_tace_mace/rtece-stage134-balanced-teacher-relax/augmented_train_base2048_plus_teacher_relax320.extxyz --output runs/oc20neb_tace_mace/rtece-stage134-balanced-teacher-relax/weighted_train_base2048_plus_teacher_relax320_eanchor.extxyz --summary runs/oc20neb_tace_mace/rtece-stage134-balanced-teacher-relax/weighted_train_base2048_plus_teacher_relax320_eanchor_summary.json --source-energy-multiplier base_mixed_train_tw0p75:1.25 --source-energy-multiplier teacher_relax_trajectory320:0.25 --source-force-multiplier teacher_relax_trajectory320:2.0 --normalize-energy-mean
```

## Review Basis

- TECE_design_space: deployment Sobolev metric and projection/distillation error separation must precede architecture changes.
- rTECE_review: teacher distillation must be a formal data product with explicit target semantics and physical tests.
- Stage133: teacher-relax trajectories improved some force-tail indicators but introduced energy drift, so source weighting is the next isolated variable.
- Stage130: force-tail weighting alone was not a clean RMSE Pareto improvement, so stage134 couples weighting to DFT energy anchoring rather than repeating force-only weighting.
