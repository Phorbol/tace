# Stage175 Local TECE Front Probe Plan

- stage: `stage175_local_tece_front_probe`
- target: `stage165_case_offset_residual_mev_atom`
- split: `group-loocv` by `case_id`
- shell edges: `0,1.6,2.6,3.6,5`
- ridge grid: `1e-08,1e-06,0.0001,0.01,1,100`

## Question

After Stage174 rejected global composition/tag/geometry proxies, can a deployable local TECE-style species-conditioned moment scalar front provide an unseen-case energy-offset coordinate worth promoting?

## Feature Families

| family | Lmax | chemistry rank | semantics |
|---|---:|---:|---|
| local_l0_rank3 | 0 | 3 | local_species_conditioned_cartesian_moment_scalar_contractions |
| local_l0_l1_rank3 | 1 | 3 | local_species_conditioned_cartesian_moment_scalar_contractions |
| local_l0_l1_l2_rank3 | 2 | 3 | local_species_conditioned_cartesian_moment_scalar_contractions |
| local_l0_l1_l2_rank4 | 2 | 4 | local_species_conditioned_cartesian_moment_scalar_contractions |

## Review Basis

- Stage174: global deployable composition/tag/geometry proxies did not beat the group-heldout intercept baseline.
- TECE_design_space.md: rTECE should use one local moment pass, sparse atomic invariants, and selected low-cost scalar paths.
- rTECE_review.md: low-rank species chemistry basis is required to avoid element-collision failures without returning to one-hot large channels.
- Stage170/173: the raw energy issue is a low-frequency deployable representation problem, not a per-case metadata correction.
