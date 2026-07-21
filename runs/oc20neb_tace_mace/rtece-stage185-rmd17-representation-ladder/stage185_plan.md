# Stage185 rMD17 Representation Ladder

Goal: promote rMD17 from smoke data prep to an independent conventional MD closure for the Stage183 TECE semantic ladder.

## Dataset Contract

- molecule: `ethanol`
- source labels are kcal/mol and kcal/mol/A; converted extxyz labels are eV and eV/A.
- train/valid/test are contiguous time-ordered blocks, with train count capped at 1000 for the first closure.

## Document Alignment

- TECE_design_space.md §11/§12: compare a tiered student family by physical error and hardware cost across deployment distributions.
- TECE_design_space.md §5: keep one low-order moment pass and scalarize sparse high-value paths before expensive persistent equivariant state.
- rTECE_review.md P1: path IDs, held-out deployment splits, and comparable supervised/distilled students are prerequisites for real renormalization.
- Stage184 result: L2 is the current 3BPA rTECE Pareto candidate; rMD17 checks whether that conclusion transfers to conventional molecular MD.
- PyG MD17 documentation: original/revised MD17 energy and force labels are kcal/mol and kcal/mol/A, and highly correlated trajectories should use no more than about 1000 training samples.

## Ladder

- `stage185_rmd17_l0_local_species`: l0_local_species_basis; marginal paths `atomic.radial_density, atomic.species_basis_density, atomic.local_l0_lowrank_density`
- `stage185_rmd17_l1_cross`: l1_sparse_cross_radial_power_spectrum; marginal paths `atomic.vector_norm, atomic.vector_cross_radial_dot`
- `stage185_rmd17_l2_atomic_quadrupole`: l2_atomic_quadrupole_without_edge_cost; marginal paths `atomic.quadrupole_norm, atomic.quadrupole_cross_radial_frobenius`
- `stage185_rmd17_t3_cavity_vecq`: cavity_vector_quadrupole_edge_relational_sketch; marginal paths `edge.cavity.vector_dot, edge.cavity.quadrupole_frobenius`

## Decision Rule

- If L2 remains the best accuracy/throughput tradeoff, the Stage184 conclusion transfers beyond 3BPA.
- If T3 wins rMD17 but not 3BPA, edge-relational sketches are dataset/task dependent rather than uniformly Pareto-optimal.
- If all tiers are close on rMD17, rMD17 is a weaker representation stress test and should mainly validate data/engine correctness.
