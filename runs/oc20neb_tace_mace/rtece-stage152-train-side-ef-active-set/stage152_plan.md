# Stage152 Train-Side E/F Active-Set Projection Plan

- stage: `stage152_train_side_ef_active_set`
- purpose: extend Stage151 from energy-only diagnostics to train-side Sobolev E/F path ranking
- configs: `runs/oc20neb_tace_mace/tece-distill-20260717/mixed_train_tw0.75.extxyz`
- limit configs: `128`
- split: deterministic held-out stride `4`
- targets:
  - DFT: `dft_energy`, `dft_forces`
  - teacher: `teacher_energy`, `teacher_forces`
  - mixed training objective: `energy`, `forces`

## Question

Stage151 showed that DFT-valid energy is linearly recoverable by the Stage143 semantic supernet at about `7 meV/atom`, while the train-side structures remain around `93-102 meV/atom` even for full-reference `dft_energy`, `teacher_energy`, and mixed `energy`. That result localizes the absolute-energy failure to a train-manifold representation/projection issue, but it is still energy-only.

Stage152 asks which semantic TECE paths are valuable when the held-out metric includes both energy and force Jacobian sensitivity on the train manifold. This follows the review requirement to use E/F Sobolev projection and active-set ranking rather than final-head widening or one-off path guesses.

## Limit512 Correction

The first E/F smoke uses `limit_configs=128`, which is acceptable for force rows but underdetermines the energy projection because the held-in graph count is below the 300-334 descriptor dimensions plus element baseline. The decisive E/F active-set comparison therefore adds a `limit_configs=512` rerun for the same three label targets. The 128-config outputs are kept as a force-Jacobian smoke; the 512-config outputs are the primary E/F active-set evidence.

## Reference Basis

`atomic.radial_density, atomic.species_basis_density, atomic.vector_norm, atomic.vector_cross_radial_dot, atomic.quadrupole_norm, atomic.quadrupole_cross_radial_frobenius, edge.cavity.vector_dot, edge.cavity.quadrupole_frobenius, edge.direct.radial`

Manual candidates:

| candidate | retained paths |
|---|---|
| `t2_l0_species_radial` | radial density + species basis density |
| `t2_l1_atomic_cross` | L0 + vector norm/cross radial |
| `t2_l2_atomic_cross` | L1 + quadrupole norm/cross radial |
| `t3_l2_cavity_vector` | L2 + edge cavity vector dot |
| `t3_full_reference` | all reference paths |

Auto candidates: `single_delete`, `prefix`.

## Interpretation Rules

- If the same train-side paths dominate DFT, teacher, and mixed E/F targets, the next training stage should use that path set as a robust semantic student front.
- If teacher and DFT rank different paths, the distillation target is changing the required operator basis; the next step should make teacher cache/projection fields explicit before more training.
- If force ranking prefers a smaller L1/L2 atomic path while energy needs full reference, the current bottleneck is case-level absolute energy rather than local force PES; training should separate absolute energy calibration from force/relative-energy optimization.
- If edge-relational paths are not on the E/F Pareto front, do not pay their throughput cost in the next architecture sweep.

## Document Basis

- `TECE_design_space.md`: optimize projection error plus hardware cost under the deployment measure, not just parameter count.
- `rTECE_review.md`: use semantic path groups, Gram/Schur-style projection, and E/F/V Sobolev targets; compare path deletion by scientific cost.
- Stage151: a valid-only energy projection is insufficient because train-side absolute energy is much harder than DFT-valid energy.
