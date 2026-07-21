# Stage176 Production Local L0 Front Plan

- stage: `stage176_production_local_l0_front`
- entrypoint: `tace.scripts.rtece_train_scalar`
- scalar paths: `atomic.radial_density,atomic.local_l0_lowrank_density`
- local L0 chemistry rank: `3`
- max steps: `2000`

## Question

Can the Stage175 group-heldout local L0 rank3 signal survive as a real trainable rTECE model path under the production training and ASE-compatible inference stack?

## Review Basis

- Stage175: local_l0_rank3 beat the group-heldout intercept baseline while higher L/rank variants degraded.
- TECE_design_space.md: high-throughput students should use one local moment pass and early scalar sufficient statistics.
- rTECE_review.md: chemical collisions require low-rank species/chemistry bases, and trainable fronts must stay in production package paths.
- Stage174: global composition/tag/geometry proxies were rejected, so this must be a deployable local model path rather than a metadata correction.
