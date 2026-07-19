# Stage130 Sobolev-weighted distillation

## Rationale

Stage129 falsified the simplest deployment-patch hypothesis: adding 128 teacher-labeled rattle structures to the current two-row Pareto set did not improve the physical rattle triage and worsened the C/N-sensitive force maxima. Per TECE_design_space and rTECE_review, the next clean test is not an element-specific correction. It is to align the student training measure with a deployment/Sobolev objective before blaming the representation itself.

## Single controlled change

Architecture is fixed to the stage129 current Pareto rows:

- l1_active_nrad12_species20_radial_species8_cross3_h64
- l1_active_nrad12_species24_radial_species8_cross3_h64

The train file is the same 2048 base + 128 teacher-rattle set as stage129, but each configuration now carries extxyz info fields:

- energy_weight: scalar energy-loss weight, kept at 1.0 for this stage
- forces_weight: scalar force-loss weight, normalized to mean 1.0

Force weights are assigned without element labels:

- teacher_labeled_stage128_rattles source multiplier: 4.0
- global top 10 percent max-force tail multiplier: 2.0
- normalized force-weight mean: 1.0

## Interpretation target

If this improves F RMSE / F max error / rattle force maxima without hurting throughput, the stage129 failure was partly a deployment-measure mismatch. If it does not, the evidence shifts toward representation/projection limits and the next priority should be a real TECE representation step, such as controlled edge-relational scalar sketches or broader teacher trajectory distillation, not more local weighting.

## Metrics to compare

Use RMSE as the primary scalar ranking axis, but record all of:

- DFT E MAE/RMSE and E max error
- DFT F MAE/RMSE and F max error
- teacher E/F metrics for distillation closure
- atoms/s throughput and atom-count scaling where available
- dimer smoothness and rattle/relax physical triage

## Jobs

- failed first submit: species20 685081, species24 685082; root cause was weighted extxyz losing ASE calculator energy/forces.
- active resubmit: species20 Slurm 685084, species24 Slurm 685083.
