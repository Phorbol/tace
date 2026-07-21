# Stage164 Force-Protected UV Training Plan

- stage: `stage164_force_protected_uv_training`
- row set: `stage164-force-protected-uv`
- semantics: `stage163_force_protected_uv_training`
- train file: `runs/oc20neb_tace_mace/tece-distill-20260717/mixed_train_tw0.75.extxyz`
- Stage163 source: `runs/oc20neb_tace_mace/rtece-stage163-force-protected-active-set/stage163_force_protected_active_set.json`
- force-protected gate: sampled force regression <= `0.1` and positive energy gain
- relative energy weight: `0.25`

## Question

Does the Stage163 force-protected uv edge-frame scalar increment repair the Stage162 raw-energy case/slab/adsorbate baseline without losing the Stage157 force RMSE/max and relative NEB metrics?

## Rows

| variant | params | gate | isolated increment | scalar paths |
|---|---:|---|---|---|
| stage164_uv_force_gate_rel0p25_b32 | 55144 | force_regression<=0.10_and_energy_gain | edge.cavity.target/source_vector_projection | atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot,edge.direct.radial,edge.cavity.target_vector_projection,edge.cavity.source_vector_projection |

## Review Basis

- TECE_design_space: promote semantic TECE paths only by physical error gain under hardware cost, not by descriptor residual alone.
- rTECE_review: keep E/F RMSE and max errors, energy gauge, and physical probes explicit.
- Stage162: raw E RMSE is mostly case/slab/adsorbate offset, not a simple global E0 bug.
- Stage163: uv is the only candidate passing a 10% sampled-force regression gate with positive energy gain.
