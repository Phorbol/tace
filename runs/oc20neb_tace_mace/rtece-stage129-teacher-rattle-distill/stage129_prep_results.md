# Stage129 Prep Results

## Slurm

| job | state | exit | elapsed | role |
| --- | --- | --- | ---: | --- |
| 685059 | COMPLETED | 0:0 | 00:00:25 | initial prep, later invalidated because rattles were appended after all 5000 base configs while training limited to 2176 |
| 685061 | COMPLETED | 0:0 | 00:00:12 | fixed base limit, later invalidated because the shorter overwrite left stale trailing bytes and concat did not materialize calculator E/F labels |
| 685067 | COMPLETED | 0:0 | 00:00:12 | current valid prep after concat unlink + E/F materialization fix |

## Dataset Contract

The current prep job creates the intended training window for teacher-rattle distillation:

- `rattled_valid_start58_limit8_copies16_std0p05.extxyz`: 128 teacher-label candidate geometries from validation configs 58:66, 16 rattles per source config.
- `teacher_labeled_stage128_window_rattles.extxyz`: the same 128 geometries relabeled by the TACE teacher, with source labels preserved under `source_`.
- `augmented_train_base2048_plus_teacher_rattle128.extxyz`: 2176 configs total, with base configs occupying indices 0:2048 and teacher-rattle configs occupying indices 2048:2176.

## Preflight

ASE extxyz preflight after job 685067 read all 2176 frames and 113970 atoms. Key frames 0, 2047, 2048, and 2175 all have readable standard energy and forces; frame 2048 and 2175 are from `teacher_labeled_stage128_rattles`.

This closes the stage129 training-data gate: the next training jobs can actually see all teacher-rattle examples under `LIMIT_CONFIGS=2176`.
