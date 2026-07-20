# Stage153 Submission Status

## Submitted Jobs

| job | role | initial state | initial node | wrapper |
|---:|---|---|---|---|
| 687455 | DFT-only base2048 control | RUNNING | 16v100n01 | `wrappers/dft_only_base2048/rtece_scalar_matrix_no_export.sbatch` |
| 687456 | mixed base2048 paired control | RUNNING | 16v100n01 | `wrappers/mixed_base2048/rtece_scalar_matrix_no_export.sbatch` |

## Label Remap Smoke

The DFT-only wrapper uses `mix_tece_distill_labels.py` with `teacher_weight=0.0` and `teacher_prefix=stage153_mixed_`.
A two-config smoke remap verified:

- output `energy` equals input `dft_energy`;
- output `forces` equals input `dft_forces`;
- original `teacher_energy` is preserved;
- mixed target is preserved under `stage153_mixed_energy`.

## Current Interpretation Before Results

The persistent energy error is real. Stage149 left DFT E RMSE at about 93-107 meV/atom after energy-loss normalization was fixed. Stage150 ruled out a simple residual E0/gauge fix. Stage151/152 showed train-side absolute energy remains difficult under overdetermined projection fits, while force projection favors the low-cost T2 L2 atomic-cross front.

Stage153 therefore tests the next root-cause split: mixed distillation/data contract versus unavoidable student representation projection on DFT labels.
