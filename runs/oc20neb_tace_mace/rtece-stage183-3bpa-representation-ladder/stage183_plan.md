# Stage183 3BPA Representation Ladder

Goal: test whether TECE semantic representation increments close the Stage182 dihedral PES gap before kernel work or head widening.

## Document Alignment

- TECE_design_space.md §5: rTECE should keep one fused L_A=2/3 moment pass, sparse atomic scalar paths, and sparse edge-relational m_total=0 sketches.
- TECE_design_space.md §11: compare a tiered T1/T2/T3 path ladder by physical error and hardware cost, not parameter count alone.
- rTECE_review.md §4: replace variant switches with semantic path IDs and fix edge sketch duplication by explicit path registry.
- rTECE_review.md: increase representation capacity in front-loaded chemistry/angular/edge paths before widening the final MLP head.
- Stage182: 3BPA gives matched NEP/DPA/rTECE numeric closure and exposes a dihedral PES gap in the current L1/local-L0 student.

## Ladder

- `stage183_l0_local_species`: l0_local_species_basis; marginal paths `atomic.radial_density, atomic.species_basis_density, atomic.local_l0_lowrank_density`
- `stage183_l1_cross`: l1_sparse_cross_radial_power_spectrum; marginal paths `atomic.vector_norm, atomic.vector_cross_radial_dot`
- `stage183_l2_atomic_quadrupole`: l2_atomic_quadrupole_without_edge_cost; marginal paths `atomic.quadrupole_norm, atomic.quadrupole_cross_radial_frobenius`
- `stage183_t3_cavity_vecq`: cavity_vector_quadrupole_edge_relational_sketch; marginal paths `edge.cavity.vector_dot, edge.cavity.quadrupole_frobenius`

## Metrics

- Primary: force RMSE in meV/A across test_300K, test_600K, test_1200K, and test_dih.
- Required: energy RMSE/max, force max, atoms/s, and peak memory.
- Physical followups after numeric ladder: dimer scan and rattle-relax RMSD, treated as continuous diagnostics.

## Decision Rule

- If L2 improves test_dih energy/force RMSE without a large throughput penalty, keep L_A=2 in the active rTECE family.
- If T3 improves test_dih beyond L2, edge-relational sketches are a real rTECE advantage over pure NEP-like scalar endpoints.
- If neither improves test_dih, prioritize teacher trajectory/distillation coverage rather than adding more hand-designed paths.
