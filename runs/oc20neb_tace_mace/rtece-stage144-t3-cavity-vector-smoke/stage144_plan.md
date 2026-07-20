# Stage144 T3 Cavity Vector Plan

- stage: `stage144_t3_cavity_vector`
- row set: `stage144-t3-cavity-vector`
- semantics: `stage143_active_set_guided_minimal_t3_edge_relational_training`
- train file: `runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/weighted_train_base2048_plus_teacher_relax640_forceonly_eanchor.extxyz`
- train limit: `512` configs
- Stage143 source: `runs/oc20neb_tace_mace/rtece-stage143-semantic-active-set/stage143_results_summary.json`

## Question

With the Stage140/141 L2 conditioned atomic anchor and Stage142 force-only teacher-relax training measure fixed, does adding exactly one edge.cavity.vector_dot architecture increment create a real RMSE/max-error/physical-tail Pareto movement that justifies the first true rTECE edge-relational cost?

## Rows

| variant | params | representation params | readout params | isolated increment | scalar paths |
|---|---:|---:|---:|---|---|
| t3_l2_cavity_vector_cond32_h64 | 51496 | 25959 | 25537 | edge.cavity.vector_dot | atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot |

## Review Basis

- TECE_design_space: rTECE is the middle tier with sparse edge-relational scalar ECE sketches after immediate scalarization.
- TECE_design_space: test one controlled semantic path increment and charge real hardware throughput, not just parameter count.
- rTECE_review: report E/F RMSE and max errors separately, then dimer/rattle physical probes; do not use MAE alone.
- Stage142: more same-window teacher-relax coverage did not close the tail, so do not add more data before checking operator-basis value.
- Stage143: edge.cavity.vector_dot is the minimal T3 active-set step; cavity quadrupole/direct radial remain deferred.
