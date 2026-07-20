# Stage157 Relative NEB Loss Plan

Stage156 showed that raw absolute energy RMSE remains poor while relative/path energy errors are much smaller. Stage157 therefore tests a training-objective axis, not a new architecture axis: keep the Stage155 high-throughput direct T3 architecture fixed and add a document-grounded relative NEB energy term.

## Rows

| row | relative energy weight | batch size | purpose |
|---|---:|---:|---|
| `stage157_direct_b32_rel0_mixed2048` | 0.0 | 32 | batch-size control for Stage155 direct architecture |
| `stage157_direct_b32_rel0p25_mixed2048` | 0.25 | 32 | weak relative/path energy supervision |
| `stage157_direct_b32_rel1p0_mixed2048` | 1.0 | 32 | strong relative/path energy supervision |

## Document Basis

- `rTECE_review.md` says multi-element absolute energy needs explicit E0/gauge handling and NEB should track relative image and barrier errors rather than absolute E MAE alone.
- `TECE_design_space.md` frames student renormalization as preserving deployment Sobolev/relative PES behavior under controlled deletion of expensive equivariant state.
- Stage156 found relative-image/barrier errors are much better than raw E, so the clean next test is whether training can emphasize this gauge-invariant PES shape without sacrificing force RMSE or throughput.

## Batch Coverage

The 2048-config mixed train slice has 200 groups, 196 multi-image groups. Random batch_size=32 gives about 0.994 useful batches and 4.15 extra same-group images per batch, so relative loss is expected to be active. Batch_size=8 would be too weak and likely produce a false negative.

## Promotion Rule

Promote only if relative-image/barrier RMSE improves without worsening DFT F RMSE/max or raw E max enough to leave the Pareto front. The `rel0` batch-size control is required before attributing changes to the relative loss.
