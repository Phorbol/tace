# rTECE Stage131 Residual Edge Current-Pareto Plan

## Basis

Stage131 is an architecture-control experiment, not another weighting sweep. `TECE_design_space.md` identifies sparse edge-relational scalar ECE sketches as the intended rTECE middle tier between pure scalar endpoints and full TECE. `rTECE_review.md` flags the same missing path family as a P1 issue: true rTECE should include cavity moments, radial/POD-style low-rank sketches, and explicit path bookkeeping.

The immediate evidence chain is:

- Stage124: edge-only residual paths were dominated, but all-scope cavity coupling showed that coupled atomic+edge downfolding can recover accuracy at high cost.
- Stage129: teacher-rattle augmented data improved the current physical candidate without changing the architecture.
- Stage130: Sobolev/force-tail weighting was a negative control for standard RMSE and did not rescue the physical gate.

Therefore Stage131 keeps the current stage129/stage127 species24 L1 active atomic backbone and adds only minimal cavity/direct edge-relational scalar residual paths. It isolates representation value from sample weighting, final-head widening, and C/N-specific tuning.

## Data And Training

- Train file: `runs/oc20neb_tace_mace/rtece-stage129-teacher-rattle-distill/augmented_train_base2048_plus_teacher_rattle128.extxyz`
- Train configs: 2176
- Validation file: OC20NEB valid extxyz
- Train-valid window: 256 configs
- DFT benchmark window: 1024 configs
- Teacher benchmark window: 1024 configs
- Max steps: 20000
- Batch size: 8
- Validation batch size: 16
- LR warmup steps: 500
- Early stopping patience: 400
- Short-range baseline: ZBL

## Rows

1. `l1_active_species24_cavity_vec_residual_h64`
   - Paths: atomic radial/species/vector/cross-radial + cavity vector dot + direct edge radial
   - `moment_l_max=1`
   - Estimated parameters: 29117

2. `l2_active_species24_cavity_vecq_residual_h64`
   - Paths: atomic radial/species/vector/cross-radial + cavity vector dot + cavity quadrupole Frobenius + direct edge radial
   - `moment_l_max=2`
   - Estimated parameters: 29181

## Acceptance Readout

Primary comparison should use DFT E/F RMSE, DFT E/F max error, teacher E/F RMSE, and throughput atoms/s. MAE remains secondary. If either row is non-dominated or shows a meaningful max-error/physical signal, run the same physical triage used for stage128-130: C/N rattle-relax RMSD/fmax and dimer-scan smoothness where available.

## Decision Rule

If minimal edge residual improves RMSE or max-error at acceptable throughput cost, continue a controlled edge-sketch rank/path ladder. If it is dominated again, stop spending iterations on this edge path family and move to broader teacher trajectory/Jacobian distillation or projection diagnostics for which TECE coordinates the student actually misses.
