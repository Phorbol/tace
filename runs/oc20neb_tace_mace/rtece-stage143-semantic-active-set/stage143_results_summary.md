# Stage143 Semantic Active-Set Projection Results

Stage143 moves away from more same-window teacher-relax coverage and instead evaluates an explicit TECE semantic path lattice. The 128-config run completed but was underdetermined, so it is superseded by the 512-config run.

## Jobs

| job | role | state | exit | elapsed | status |
|---:|---|---|---|---|---|
| 685659 | projection limit128 | COMPLETED | 0:0 | 00:00:12 | superseded: underdetermined |
| 685668 | projection limit512 | COMPLETED | 0:0 | 00:00:13 | primary |

Force-Jacobian projection is explicitly deferred because the first smoke showed `_force_descriptor_matrix` was dominated by autograd descriptor Jacobians even at limit 8.

## Train Projection, Limit 512

| candidate | dim | descriptor residual | E residual | E/atom RMSE | E/atom max | dof |
|---|---:|---:|---:|---:|---:|---:|
| t3_l2_cavity_vector_quadrupole_direct | 334 | 3.04453e-08 | 0.0250064 | 0.116526 | 0.44274 | 12 |
| t3_l2_cavity_vector | 331 | 0.00325855 | 0.0291479 | 0.128489 | 0.445395 | 15 |
| t2_l2_atomic_cross | 330 | 0.0061234 | 0.0271027 | 0.122459 | 0.483348 | 16 |
| t2_l1_atomic_cross | 315 | 0.0417389 | 0.029629 | 0.141204 | 0.526329 | 31 |
| t2_l0_species_radial | 300 | 0.117803 | 0.0254517 | 0.11862 | 0.391524 | 46 |

## Valid Projection, Limit 512

| candidate | dim | descriptor residual | E residual | E/atom RMSE | E/atom max | dof |
|---|---:|---:|---:|---:|---:|---:|
| t3_l2_cavity_vector_quadrupole_direct | 334 | 4.41228e-09 | 0.00101831 | 0.00667047 | 0.0518684 | 30 |
| t3_l2_cavity_vector | 331 | 0.00172272 | 0.00204591 | 0.00959632 | 0.0675401 | 33 |
| t2_l2_atomic_cross | 330 | 0.00300329 | 0.00107345 | 0.0071293 | 0.0533993 | 34 |
| t2_l1_atomic_cross | 315 | 0.0223636 | 0.00115953 | 0.00771872 | 0.0503663 | 49 |
| t2_l0_species_radial | 300 | 0.100409 | 0.00129272 | 0.00753335 | 0.0363425 | 64 |

## Interpretation

- Descriptor projection now gives a clean active-set signal: `t2_l0_species_radial` is much more lossy than L1/L2; `t3_l2_cavity_vector` reduces the remaining L2 atomic descriptor residual; the full cavity+direct reference is the zero-loss endpoint by construction.
- Energy-only linear projection is not a sufficient final selection metric. It is useful as a sanity check, but several candidates are close and this does not include force/Jacobian information.
- The next training stage should not add another data multiplier. It should train the minimal T3 edge-relational increment `t3_l2_cavity_vector` against the Stage140/141 L2 atomic incumbent, then run the same DFT RMSE, max error, dimer, and rattle-relax physical triage.
