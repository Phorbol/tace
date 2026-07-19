# Stage119 front-loaded representation ladder

## Why this stage exists

The previous 577-1000 parameter scalar endpoints are useful as T4 throughput limits, but they are too aggressive as the main rTECE student family. They use mostly fixed radial/moment/scalar descriptors, so the learnable part quickly becomes a small interpolation head over a fixed feature space. That explains the fast overfitting/early plateau observed in Stage92-118 and matches the review critique that fixed species/Z-power chemistry is a P0 representation issue.

Stage119 moves capacity forward into the representation while keeping the final readout head fixed at `64,64`. This follows `TECE_design_space.md`: delete persistent high-l edge/node state and dense per-edge channel mixing, but retain a controlled subset of TECE semantic axes: low-rank species basis, learnable radial mixing, low-order L1/L2 scalarized moments, cross-radial invariants, and cavity edge relational scalar sketches. It also follows `rTECE_review.md`: repair the fixed chemistry representation before spending more queue time on pure kernel optimization.

The external-paper alignment is: TACE uses learned element/node embeddings and learnable radial/path weights as part of its controlled ACE-style representation; TECE/ECE keeps high-order edge interactions and attention as a costly but expressive end; MACE shows higher-body equivariant messages improve expressivity but are not the target ultra-throughput endpoint. Stage119 is therefore a middle route: keep early scalarization and no persistent equivariant state, but add small learnable front-end basis modules.

## Code changes

- Added `species_basis_mode` to `RTECEScalarConfig`, manifests, route contracts, training summaries, CLI, Lightning builder, matrix sbatch, and wrapper generator.
- Added `species_basis_mode=learnable_embedding`, initialized exactly as the existing fixed Z-power basis in fp64. At initialization, descriptors are numerically equivalent to the old fixed basis; after training, the basis can learn element relationships from DFT/teacher labels.
- Learnable species basis is only supported by the autograd force path for now. Analytic/Triton force backends reject it explicitly because their descriptor derivative chains have not been extended.
- Added Stage119 row set `frontloaded-representation-stage119` with fixed `hidden_channels=64,64` and varying front representation axes.
- Wrapper generation still uses self-contained exports inside the generated script and does not use `sbatch --export`, `--mem`, or `--cpus-per-task`.

## Stage119 rows

| row | TECE front axes | species basis | params | representation params | readout params |
|---|---|---:|---:|---:|---:|
| `l0_species8_learnembed_h64` | L0 learnable species density | 8 | 9833 | 872 | 8961 |
| `l1_species16_cross_learnembed_h64` | L1 + cross-radial + species | 16 | 15465 | 1704 | 13761 |
| `l2_species16_atomic_cross_learnembed_h64` | L2 atomic + cross-radial + species | 16 | 16169 | 1704 | 14465 |
| `l2_species16_cavity_edge_learnembed_h64` | L2 cavity edge + species | 16 | 14993 | 1680 | 13313 |
| `l2_species32_cavity_edge_learnembed_h64` | L2 cavity edge + wider species | 32 | 24801 | 3296 | 21505 |
| `l2_species32_cavity_atomic_cross_learnembed_h64` | L2 cavity + atomic cross-radial + wider species | 32 | 26233 | 3320 | 22913 |

Important caveat: although the head width is fixed, descriptor dimensionality still increases the first readout layer size. So Stage119 is better than Stage118's pure head-capacity ladder, but it is not the final clean capacity allocation. The next architecture change should add a front-end low-rank/bottleneck projection for species-radial/cavity descriptors so total parameter growth is less dominated by readout input dimensionality.

## Generated artifacts

- Wrapper index: `runs/oc20neb_tace_mace/rtece-stage119-frontloaded-representation-wrappers/rtece_pareto_sweep_index.json`
- Wrapper audit JSON: `runs/oc20neb_tace_mace/rtece-stage119-frontloaded-representation-design/stage119_wrapper_contract_audit.json`
- Wrapper audit Markdown: `runs/oc20neb_tace_mace/rtece-stage119-frontloaded-representation-design/stage119_wrapper_contract_audit.md`

The extxyz preflight passed for the Stage119 windows: 2048 train frames, 256 train-valid frames, 1024 DFT-valid frames, and 1024 teacher-valid frames. The contract audit passed after including `SPECIES_BASIS_MODE=learnable_embedding` in expected exports.

## Tests run

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py -q -k 'learnable_species_basis or species_basis_mode or stage119_frontloaded or matrix_sbatch_forwards_species_basis_mode'
# 4 passed
```

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py -q -k 'species_basis_variant or species_cavity_path or builds_species_cavity or training_config_builders_forward_species_basis_mode or learnable_species_basis or stage118_representation_ladder or stage119_frontloaded or matrix_submit_helper_generates_wrapper or matrix_sbatch_forwards_species_basis_mode or stage_sweep_summary_preserves_design_metadata'
# 11 passed
```

## Next priority

1. Do not treat Stage118 as the main architecture direction; keep it as a head-capacity control.
2. Before submitting Stage119, add one more small architecture patch: a learnable low-rank descriptor bottleneck/projection for wide species/cavity descriptors, so the representation/readout split is cleaner.
3. Then run a small Stage119 smoke subset, compare E/F RMSE plus max errors, and only after that spend full 20k-batch queue time.
4. Physical gates remain dimer scan and rattle+relax, but they should diagnose architecture/generalization after RMSE and benchmark contracts are comparable, not block every coding step.
