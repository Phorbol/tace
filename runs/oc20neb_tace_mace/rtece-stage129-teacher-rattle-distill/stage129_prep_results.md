# Stage129 Prep Results

## Slurm

| job | state | exit | elapsed |
| --- | --- | --- | ---: |
| 685061 | COMPLETED | 0:0 | 00:00:12 |

## Dataset Contract

The corrected prep job creates the intended training window for teacher-rattle distillation:

- `rattled_valid_start58_limit8_copies16_std0p05.extxyz`: 128 teacher-label candidate geometries from validation configs 58:66, 16 rattles per source config.
- `teacher_labeled_stage128_window_rattles.extxyz`: the same 128 geometries relabeled by the TACE teacher, with source labels preserved under `source_`.
- `augmented_train_base2048_plus_teacher_rattle128.extxyz`: 2176 configs total, with base configs occupying indices 0:2048 and teacher-rattle configs occupying indices 2048:2176.

This fixes the earlier stage129 setup hazard where appending rattles after all 5000 base configs would have left the `LIMIT_CONFIGS=2176` training window without any rattle examples.
