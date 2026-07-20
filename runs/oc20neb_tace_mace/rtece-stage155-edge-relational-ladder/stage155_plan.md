# Stage155 Edge-Relational Ladder Plan

Stage155 follows Stage154, where `edge.cavity.vector_dot` was the first clear architecture-side raw-energy win while generic L2 capacity failed.

## Question

Which low-cost edge-relational semantic increment explains the remaining case-level energy offset: direct radial edge baseline, cavity quadrupole relation, or their combination?

## Rows

| row | params | rep params | readout params | axis | scalar paths |
|---|---:|---:|---:|---|---|
| `stage155_t3_cavity_vector_direct_mixed2048` | 54886 | 27685 | 27201 | add direct edge radial baseline to Stage154 cavity-vector T3 | atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot,edge.direct.radial |
| `stage155_t3_cavity_vector_quad_mixed2048` | 54757 | 27620 | 27137 | add cavity quadrupole relation to Stage154 cavity-vector T3 | atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot,edge.cavity.quadrupole_frobenius |
| `stage155_t3_cavity_vector_quad_direct_mixed2048` | 55015 | 27750 | 27265 | test combined full low-cost cavity/direct edge semantic set | atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot,edge.cavity.quadrupole_frobenius,edge.direct.radial |

## Document Basis

- `TECE_design_space.md`: rTECE should keep a sparse edge-relational scalar middle tier, selected by projection/error benefit and hardware cost.
- `rTECE_review.md`: avoid another ad hoc variant sweep; move through semantic operator basis, cavity relational paths, teacher projection, Schur/GN, active-set cost.
- Stage154: do not widen generic L2 capacity; isolate marginal edge path value under the same training contract.

## Promotion Rule

Promote only edge increments that reduce raw/case-offset E RMSE or F RMSE/max enough to justify measured atoms/s and memory cost relative to Stage154 cavity-vector; do not select on MAE alone.
