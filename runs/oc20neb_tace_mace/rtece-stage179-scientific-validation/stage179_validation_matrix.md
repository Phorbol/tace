# Stage179 Scientific Validation Matrix

Separate the scientific claims required for the TECE/rTECE renormalization program from useful engineering optimizations.

Current best Stage178 endpoint: `stage178_l0_local_species`.
Rattle-relax policy: `continuous_rmsd_after_relax_not_binary_gate`.

## Required Scientific Hypotheses

| id | status | why required | next verification |
|---|---|---|---|
| `early_scalarization_hardware_hypothesis` | `supported` | This is the basic hardware premise of scalar-sketched rTECE; without it the program collapses into ordinary small equivariant GNN tuning. | Promote throughput taxonomy to model-only, graph-amortized, full E/F/V, and full MD; then run matched NEP/DPA1-0-layer baselines. |
| `low_rank_chemistry_front_reduces_fixed_descriptor_bias` | `supported` | The review identified chemical collisions and fixed descriptors as likely causes of poor energy accuracy and weak extrapolation. | Run held-out composition/reaction splits and teacher fake-label augmentation to separate architecture bias from data scarcity. |
| `edge_relational_rtece_advantage` | `contradicted_or_unrealized_current_recipe` | This is the main proposed uniqueness of rTECE over standard NEP/MTP/DPA1-0 endpoints. | Test edge paths only after semantic projection, cavity/direct separation, and renormalized initialization are available. |
| `operator_space_renormalization_beats_from_scratch` | `missing` | This is the central claim that the project is a TECE renormalization compiler rather than an ad hoc student architecture search. | Stage180 should compare from-scratch, teacher truncation, GN/Schur initialization, and full distillation for the same student path manifest. |
| `sobolev_physics_metric_is_needed` | `supported_but_incomplete` | The objective is accuracy, extrapolation, and physical reasonableness under deployment distributions, not leaderboard MAE alone. | Add virial finite-strain tests and HVP or curvature probes before claiming long-time MD reliability. |
| `matched_nep_dpa_protocol` | `protocol_mismatched` | The project target is not just internal improvement; it must show where rTECE sits relative to standard high-throughput potentials. | Run matched NEP/DPA1-0-layer training, E/F/max/relative energy metrics, dimer/rattle tests, memory, and throughput scaling. |
| `user_ready_production_model` | `not_ready` | A deployable model family must not remain a demo or benchmark-only prototype. | Close PBC edge-vector semantics, virial/stress tests, calculator contract, package entrypoints, and a documented user workflow before release. |

## Evidence Details

### early_scalarization_hardware_hypothesis

Question: Can a TECE-derived student approach NEP/MTP/DPA1-0-style throughput by dropping persistent high-l edge state and scalarizing after one moment pass?

Evidence:
- TECE_design_space.md argues NEP/DPA1-0 are fast because geometry is compressed early into scalar sufficient statistics.
- rTECE_review.md judged the current prototype as validating early scalarization as a throughput direction, while not full MD ready.
- Stage178 L0 local/species endpoint reaches 1790966 atoms/s at limit1024 in the current protocol.

Evidence limits:
- This is not full MD throughput: PBC, Verlet rebuild amortization, virial, integrator, output, and matched community engines remain unaligned.
- The result supports the execution topology, not final production performance.

### low_rank_chemistry_front_reduces_fixed_descriptor_bias

Question: Does moving learnable capacity into the species/local L0 front help more than keeping a fixed tiny descriptor and only training a head?

Evidence:
- Stage178 L0 local/species is best by current DFT F RMSE, E RMSE, force max, and throughput among Stage178 variants.
- Stage178 L0 improves substantially over Stage176 local_l0_rank3 on energy and force errors under the available audit context.

Evidence limits:
- This does not prove a unique optimal chemistry basis; it only supports trainable low-rank chemistry as a necessary direction.
- The evidence is on OC20NEB-style data, not broad MD datasets.

### edge_relational_rtece_advantage

Question: Do TECE-specific edge-relational scalar sketches add a useful intermediate layer beyond NEP/MTP-like atomic scalar endpoints?

Evidence:
- Stage178 T3 minimal cavity/direct worsens current F RMSE (121.277 vs 116.428) and E RMSE (165.541 vs 116.118) relative to the L0 local/species endpoint.
- Stage178 T3 is also slower at limit1024 (782853 vs 1790966 atoms/s).

Evidence limits:
- This does not falsify edge-relational rTECE in general; it falsifies direct addition under the current training recipe.
- The current T3 path lacks teacher-projected initialization and systematic path selection.

### operator_space_renormalization_beats_from_scratch

Question: Does Schur/Gauss-Newton/operator-space downfolding outperform from-scratch training for the same retained TECE path set?

Evidence:
- TECE_design_space.md defines renormalization as retained coefficients plus a Gram/Schur correction from deleted paths.
- rTECE_review.md states the current branch has scalar students and benchmarking infrastructure but not true renormalization/distillation.

Evidence limits:
- No current stage proves pruning/downfolded initialization beats from-scratch for a fixed student architecture.
- Teacher semantic path projections, Gram blocks, and GN updates are not yet a production training path.

### sobolev_physics_metric_is_needed

Question: Are E/F RMSE alone insufficient, requiring relative NEB/barrier, force max tail, dimer smoothness, rattle-relax RMSD, and eventually virial/HVP?

Evidence:
- Stage178 records E/F RMSE, E/F max, relative image RMSE, barrier RMSE, dimer, rattle-relax RMSD, throughput, and memory.
- The rattle-relax policy is now continuous RMSD/force-tail evidence, not a binary gate.

Evidence limits:
- Virial/stress and Hessian-vector evidence are still missing.
- Rattle-relax thresholds remain deployment-task dependent rather than universal.

### matched_nep_dpa_protocol

Question: Can rTECE be placed on a fair Pareto curve against NEP/DPA1-0-layer with matched data, hardware scope, metrics, physical tests, and throughput taxonomy?

Evidence:
- Stage177 explicitly marks non-matching throughput protocols and incomplete physical triage as caveats.
- Stage178 repeats that NEP/DPA rows are smoke baselines and cannot support a superiority claim.

Evidence limits:
- Current NEP/DPA rows are useful context, not rankable production baselines.
- The available atoms/s numbers mix community engine wall time, prebuilt graph model-only, and ASE/autograd protocols.

### user_ready_production_model

Question: Is the current rTECE implementation ready for normal users as a reliable potential with training, inference, ASE, PBC, virial, and full MD semantics?

Evidence:
- rTECE_review.md scores PBC/virial/long-time MD completeness and true distillation as early-stage.
- Current Stage178 evidence is useful research evidence but not full MD validation.

Evidence limits:
- There is a production training entrypoint and ASE-oriented path, but production reliability requires geometry, stress, and long-run tests.

## Nice-To-Have Optimizations

| id | status | reason |
|---|---|---|
| `triton_or_nvalchemi_kernel_fusion` | `nice_to_have_after_science_closure` | Kernel fusion can improve throughput, but it does not prove TECE operator-space degradation or renormalized distillation. |
| `multi_gpu_full_md_scaling` | `nice_to_have_after_single_gpu_protocol_alignment` | Multi-GPU scaling matters for deployment, but only after single-GPU semantics and matched baselines are scientifically clean. |
| `zbl_dispersion_electrostatic_baselines` | `useful_physics_prior_not_core_proof` | Physical baselines can reduce residual burden, but the central claim is systematic TECE path downfolding. |

## User Readiness

Status: `research_prototype_not_user_ready`.

Usable for:
- research experiments on scalar-sketched TECE student design
- controlled training/benchmark/physical-diagnostic runs by developers

Not yet usable for:
- ordinary user deployment as a reliable production potential
- claims of superiority over production NEP/DPA1-0-layer
- long-time periodic MD with validated virial/stress semantics

## Recommended Next Stage

`stage180_minimal_renormalization_proof`: Prove or falsify that TECE-style renormalized initialization/distillation improves a fixed student path set over from-scratch training.

Must compare:
- same student architecture from scratch
- teacher/truncated or ordinary supervised initialization
- GN or Schur-style renormalized initialization
- renormalized initialization plus teacher E/F residual distillation

Must report:
- E RMSE/MAE/max
- F RMSE/MAE/max
- relative image and barrier RMSE
- dimer smoothness
- rattle-relax RMSD and force tail
- atoms/s and peak allocated/reserved memory under the same protocol
