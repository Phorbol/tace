# Stage151 Energy Projection Diagnostic Summary

Stage151 was triggered by the persistent Stage149 absolute energy error. Stage149 reduced the old energy-loss normalization failure but still had about `92.7` to `107.3` meV/atom DFT E RMSE. Stage150 then showed that post-hoc global or residual per-element gauge calibration only reduced bias modestly and did not remove the E RMSE. Stage151 therefore tests whether the remaining energy error is a simple gauge issue, a mixed teacher/DFT label issue, or a representation/projection issue.

## Jobs

| job | role | state | exit | elapsed |
|---:|---|---|---|---|
| 687445 | mixed train `energy` projection, limit1024 | COMPLETED | 0:0 | 00:00:59 |
| 687446 | DFT valid E/F projection smoke, limit128 | COMPLETED | 0:0 | 00:04:17 |
| 687447 | DFT valid `energy` projection, limit1024 | COMPLETED | 0:0 | 00:00:57 |
| 687448 | mixed train `dft_energy`/`teacher_energy` projection, limit1024 | COMPLETED | 0:0 | 00:00:39 |

## Core Energy Projection Results

All values below are held-out projection errors in eV/atom; multiply by 1000 for meV/atom. The reference basis is the Stage143 semantic supernet:

`atomic.radial_density, atomic.species_basis_density, atomic.vector_norm, atomic.vector_cross_radial_dot, atomic.quadrupole_norm, atomic.quadrupole_cross_radial_frobenius, edge.cavity.vector_dot, edge.cavity.quadrupole_frobenius, edge.direct.radial`

| target | split/file | best candidate | best E/atom RMSE | full-reference E/atom RMSE | interpretation |
|---|---|---|---:|---:|---|
| `energy` | DFT valid, limit1024 | `t3_l2_cavity_vector` / `prefix_007` | 0.00696 | 0.00713 | valid absolute energy is easy for this basis |
| `energy` | mixed train, limit1024 | `t3_full_reference` | 0.09522 | 0.09522 | train-side absolute target is not cleanly recoverable |
| `dft_energy` | same mixed-train structures, limit1024 | `t3_full_reference` | 0.10202 | 0.10202 | poor recoverability is not caused only by teacher mixing |
| `teacher_energy` | same mixed-train structures, limit1024 | `t3_full_reference` | 0.09346 | 0.09346 | teacher target is similarly difficult on train-side structures |

## Manual Candidate Details

| target | candidate | dim | E/atom RMSE |
|---|---|---:|---:|
| DFT valid `energy` | `t2_l0_species_radial` | 300 | 0.02650 |
| DFT valid `energy` | `t2_l1_atomic_cross` | 315 | 0.00835 |
| DFT valid `energy` | `t2_l2_atomic_cross` | 330 | 0.00702 |
| DFT valid `energy` | `t3_l2_cavity_vector` | 331 | 0.00696 |
| DFT valid `energy` | `t3_full_reference` | 334 | 0.00713 |
| mixed train `energy` | `t2_l0_species_radial` | 300 | 1.44646 |
| mixed train `energy` | `t2_l1_atomic_cross` | 315 | 0.19081 |
| mixed train `energy` | `t2_l2_atomic_cross` | 330 | 0.16958 |
| mixed train `energy` | `t3_l2_cavity_vector` | 331 | 0.29134 |
| mixed train `energy` | `t3_full_reference` | 334 | 0.09522 |

## E/F Smoke

The small DFT-valid E/F projection smoke used 128 configs and ranked by held-out E, held-out force, and descriptor dimension. Force errors are in eV/A.

| rank | candidate | dim | E/atom RMSE | F RMSE | F max |
|---:|---|---:|---:|---:|---:|
| 1 | `prefix_005` | 327 | 0.00714 | 0.10551 | 1.42781 |
| 2 | `prefix_004` | 315 | 0.00611 | 0.10697 | 1.46403 |
| 3 | `t2_l1_atomic_cross` | 315 | 0.00611 | 0.10697 | 1.46403 |
| 4 | `prefix_006` | 330 | 0.00779 | 0.10556 | 1.41631 |
| 5 | `t2_l2_atomic_cross` | 330 | 0.00779 | 0.10556 | 1.41631 |

This smoke does not justify the Stage149 T3 cavity-vector training result as a Pareto improvement. It instead says the next active-set step should treat L1/L2 atomic cross paths as the current low-cost front, and only add edge-relational paths when a held-out E/F or physical test pays for the hardware cost.

## Interpretation

- The poor Stage149 energy RMSE is real, not just a metric artifact.
- Stage150 already ruled out a simple missing residual E0/gauge correction.
- Stage151 rules out the narrower hypothesis that the 0.75 teacher / 0.25 DFT mixed target alone is the dominant energy problem: on the same 1024 train structures, `dft_energy`, `teacher_energy`, and mixed `energy` all have about 93-102 meV/atom full-reference projection RMSE.
- The sharp contrast between DFT-valid projection around 7 meV/atom and train-side projection around 100 meV/atom means current rTECE semantic descriptors are fitting an easy valid manifold but do not span the train-side absolute energy structure. This is an architecture/projection-distribution issue, not a final-head widening issue.
- The model can still capture relative NEB shape much better than absolute offsets, as Stage149 relative image/barrier RMSE was far smaller than absolute E RMSE. Absolute case-level energy remains the weak axis.

## Next Priority

1. Add a train-side active-set/projection stage that uses Schur-style held-out ranking on `dft_energy`, `teacher_energy`, and forces, not only DFT valid energy.
2. Do not widen only the final MLP head. If capacity is added, add front-loaded representation capacity tied to TECE path semantics: learnable radial/channel projection, better species-conditioned radial adapters, or selected scalar path interactions.
3. Run a clean DFT-only supervised control on the same architecture and training framework. If DFT-only energy improves strongly while force/throughput remains comparable, the mixed distillation objective needs to be redesigned.
4. Keep physical tests separate from energy RMSE: dimer smoothness, rattle-relax RMSD/fmax, and C/N-stratified relax behavior remain required external probes, but they should not block this energy root-cause path.
