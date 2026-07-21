# Stage178 Representation Upgrade Plan

- stage: `stage178_representation_upgrade_after_local_l0_endpoint`
- train file: `runs/oc20neb_tace_mace/tece-distill-20260717/mixed_train_tw0.75.extxyz`
- valid file: `/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz`
- max steps: `20000`
- primary metric: `dft_f_rmse_mev_a`

## Question

After Stage176 proved a very fast but inaccurate local-L0 endpoint, which front-loaded TECE/rTECE representation increments recover E/F RMSE and physical behavior at tolerable throughput cost?

## Candidate Ladder

| candidate | tier | step | marginal paths | allocation |
|---|---|---|---|---|
| stage178_l0_local_species | T1_scalar_endpoint_with_trainable_local_chemistry | l0_local_lowrank_species_front | atomic.radial_density, atomic.species_basis_density, atomic.local_l0_lowrank_density | trainable_species_and_local_l0_front_before_head |
| stage178_l1_atomic_cross | T2_atomic_scalarized_cross_radial | l1_sparse_cross_radial_invariants | atomic.vector_norm, atomic.vector_cross_radial_dot | learnable_radial_projection_and_sparse_l1_power_spectrum |
| stage178_t3_minimal_cavity_direct | T3_minimal_rtece_edge_relational | minimal_cavity_edge_relational_sketch | edge.cavity.vector_dot, edge.direct.radial | minimal_edge_relational_sketch_after_atomic_cross_radial_front |

## Review Basis

- TECE_design_space.md: rank candidates by retained/deleted TECE path sets, Sobolev E/F evidence, and hardware cost rather than MAE alone.
- rTECE_review.md §P1: prioritize radial POD/low-rank sketches, sparse cross-radial invariants, explicit path registry, and cavity edge kernels.
- rTECE_review.md: chemical collisions should be addressed by low-rank species bases, not by non-deployable C/N-specialized corrections.
- Stage177: Stage176 reaches high ASE/autograd atoms/s but fails E RMSE and rattle physical triage, so the next move is representation capacity before final-head width.
