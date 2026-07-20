# Stage135 L2 Atomic Projection Plan

- stage: `stage135_l2_atomic_projection`
- row set: `stage135-l2-atomic-projection`
- semantics: `fixed_stage132_broad_teacher_rattle_train_l2_atomic_projection`
- train file: `runs/oc20neb_tace_mace/rtece-stage132-broad-teacher-distill/augmented_train_base2048_plus_teacher_rattle512.extxyz`
- train limit: `2560` configs

## Question

With Stage132 broad-teacher-rattle data, fixed ZBL, fixed 64,64 head, and no edge sketch, does controlled L_A=2 atomic quadrupole projection reduce projection error, DFT E/F RMSE/max tails, and physical rattle failure enough to justify its throughput cost relative to the Stage132 L_A=1 atomic anchor?

## Rows

| variant | params | representation params | scalar paths | role |
|---|---:|---:|---|---|
| l2_active_nrad12_species24_radial_species8_cross3_h64 | 29885 | 4412 | atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius | controlled_L_A_2_atomic_front_not_wider_head |

## Review Basis

- TECE_design_space: prioritize a fused moment pass with L_A=2 or 3, immediate scalarization, sparse high-nu scalar paths, no persistent high-l node state, and a small scalar head.
- rTECE_review: treat current scalar rTECE as a promising endpoint, but separate projection/representation error from label and weighting heuristics before further kernel work.
- Stage132: current L_A=1 atomic front is the throughput/RMSE anchor but still fails the rattle physical gate.
- Stage133/134: teacher-relax trajectory and source balancing did not recover a clean benchmark/physical Pareto row, so the next isolated variable is representation bandwidth.
