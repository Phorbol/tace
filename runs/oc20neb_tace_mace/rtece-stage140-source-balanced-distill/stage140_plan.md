# Stage140 source-balanced distillation plan

## Why this stage

Stage139 showed a useful but incomplete signal: the L=2 conditioned-front `cond32` row improved DFT force RMSE/max and C/N rattle RMSD, but energy RMSE, max error, and bias became unacceptable. Before changing edge paths or head width, Stage140 tests whether the failure is a distillation-measure/E0-source problem.

This follows `TECE_design_space.md`: separate projection error from distillation error under a deployment Sobolev measure. It also follows `rTECE_review.md`: fix E0s/energy comparability and true distillation closure before spending effort on more kernel or edge complexity.

## Precondition fixed in code

Commit `9eafa37` changes rTECE atomic E0 fitting to weighted least squares. A frame with `energy_weight=0` no longer contributes to fitted atomic energies. Without this, a force-only teacher-relax dataset would still leak teacher energies through E0s.

## Data

- train file: `runs/oc20neb_tace_mace/rtece-stage140-source-balanced-distill/weighted_train_base2048_plus_teacher_relax320_forceonly_e0.extxyz`
- source counts: 2048 base/DFT mixed train configs + 320 teacher-relax trajectory configs
- teacher-relax energy multiplier: 0.0
- base/DFT energy multiplier: 1.25, normalized to mean energy weight 1.0
- teacher-relax force multiplier: 2.0, normalized to mean force weight 1.0

## Rows

| row | architecture | global energy weight | changed variable |
|---|---|---:|---|
| `l2_active_nrad12_species24_radial_species8_cross3_cond32_forceonly_ew1_h64` | Stage139 L=2 cond32 conditioned-front | 1.0 | teacher relax contributes only force/Sobolev signal; DFT/base structures anchor energy and weighted E0s |
| `l2_active_nrad12_species24_radial_species8_cross3_cond32_forceonly_ew2_h64` | Stage139 L=2 cond32 conditioned-front | 2.0 | same force-only teacher measure plus stronger global DFT energy anchor to test energy/force tradeoff |

## Fixed training contract

Both rows use 2368 train configs, 256 valid configs, 1024 benchmark configs, max 20000 steps, batch 8, valid batch 16, LR 1e-3, plateau scheduler, 500-step warmup, LR patience 25, early stopping patience 400, matscipy neighborlist, and autograd force mode. `short_range_repulsion_potential=zbl` is configured but `short_range_repulsion_strength=0.0`, so this is not an active ZBL ablation.

## Decision rule

- If force-only ew1 keeps Stage139 cond32 force RMSE/max gains while reducing energy RMSE/bias toward Stage137/132, Stage139 failure was mainly distillation-measure/E0 source leakage.
- If ew2 further reduces energy RMSE/max with modest force loss, use energy_weight as a Pareto control knob for the source-balanced student.
- If neither fixes energy while force remains good, next isolate architecture/projection error or per-source energy normalization beyond scalar E0s.
