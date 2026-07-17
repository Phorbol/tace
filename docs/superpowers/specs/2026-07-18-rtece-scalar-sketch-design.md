# rTECE Scalar-Sketched Student Design

## Purpose

The next stage is not another compact-TACE width or radial sweep. Stage 1-4 already show that reduced TACE variants form useful T1/T2 Pareto points, but their throughput remains in the 4e4-7e4 atoms/s class on the OC20NEB benchmark. The missing high-throughput branch must test the TECE design-space claim that early scalarization and streaming edge scalar sketches can approach NEP/MTP/DPA1-0-style execution while preserving a small amount of TECE edge-relational information.

This spec defines the first T3 prototype: `rtece_scalar`, an independent scalar-sketched rTECE model and benchmark path. Its job is to produce a high-throughput Pareto anchor, not to match the full TACE/TECE teacher.

## Source Alignment

`TECE_design_space.md` frames TECE as the mother operator space and NEP, MTP, DPA1-0, PaiNN, compact TACE, and full TECE as progressively richer projections. The document identifies the true cost drivers as persistent wide equivariant state, dense per-edge channel mixing, tensor-product path density, repeated gather/scatter, attention, and edge nonlinearities. It recommends a student with:

- one fused low-order moment pass;
- no persistent high-rank equivariant edge/node state;
- selected atomic scalar contractions;
- selected edge-relational scalar sketches;
- small scalar energy head;
- conservative forces from energy gradients;
- measured hardware Pareto evaluation.

The current TACE experiments validate this ordering. `scalar_fast` is cheap but too lossy, `edge_min` and `edge_radial_mixed` recover accuracy but remain too slow for NEP/DPA1-0-like throughput, and `edge_radial_active6_mixed` proves true active-set projection is possible but not sufficient.

## Design Choice

Use an independent prototype and training/benchmark scripts first, not a new Hydra-native TACE model class.

Rationale:

- It directly tests the missing T3 execution model without inheriting the full TACE representation lifecycle.
- It minimizes changes to the existing TACE training stack while the architecture is still experimental.
- It gives a clean throughput lower/upper bound for scalar-sketch descriptors before investing in fused kernels or full integration.

The prototype should still use the same OC20NEB data split, label convention, accuracy metrics, and benchmark reporting as the existing TECE distillation matrix.

## Model Definition

### Inputs

The first implementation consumes a batched graph object with:

- atomic numbers `z`;
- positions `pos`;
- edge index `(source, target)`;
- edge vectors and distances;
- graph batch indices.

The graph can come from a simple ASE neighbor-list path in the prototype. Full TACE datamodule integration is deliberately deferred.

### Descriptor Stages

1. Pair/radial basis
   - Compute radial features `R_n(r_ij)` with cutoff.
   - Include element-pair embeddings or element-conditioned radial weights.
   - This is the T4/pair endpoint.

2. Low-order atomic moments
   - Build moments once per atom:
     - scalar density `A_i^0 = sum_j f_n(r_ij)`;
     - vector moment `A_i^1 = sum_j f_n(r_ij) rhat_ij`;
     - quadrupole-like moment `A_i^2 = sum_j f_n(r_ij) (rhat_ij rhat_ij^T - I/3)`.
   - Do not store or propagate multi-layer irreps.
   - Use Cartesian tensors for the first prototype because the document explicitly allows Cartesian `l<=2` moments for high-throughput implementation.

3. Atomic scalar contractions
   - Produce scalar descriptors from the moments:
     - radial densities;
     - `|A_i^1|^2`;
     - `trace(A_i^2 A_i^2)`;
     - optional low-order cross-radial contractions.
   - These are NEP/MTP-like atomic invariant endpoints inside the TECE projection hierarchy.

4. Edge-relational scalar sketches
   - For each edge, compute a small static instruction set and scatter back to the target atom:
     - `A_i^1 · A_j^1`;
     - `trace(A_i^2 A_j^2)`;
     - `rhat_ij · A_i^1`;
     - `rhat_ij · A_j^1`;
     - `rhat_ij^T A_i^2 rhat_ij`;
     - `rhat_ij^T A_j^2 rhat_ij`.
   - Gate each sketch by radial and element-pair features.
   - These sketches are the first rTECE-specific branch: they keep source-target environment relation without retaining a full edge equivariant state.

5. Scalar head
   - Concatenate element embedding, pair/radial density, atomic contractions, and edge-relational sketch sums.
   - Use a small element-aware scalar MLP to predict atomic energy.
   - Sum atomic energies to graph energy.
   - Compute forces as `-grad(E, pos)` to preserve conservative forces.

## First Variants

| variant | descriptor set | purpose |
|---|---|---|
| `rtece_pair` | radial/pair + element head | T4 maximum-throughput lower-accuracy baseline |
| `rtece_atomic_moments` | pair + atomic scalar contractions | NEP/MTP-like endpoint inside this code path |
| `rtece_edge_sketch8` | atomic moments + 8 edge-relational sketch channels | first true scalar-sketched rTECE point |
| `rtece_edge_sketch16` | atomic moments + 16 sketch channels | projection-error versus cost test |

The variants must be generated by a config builder, not hand-edited YAML, so the Pareto path is reproducible.

## Training And Distillation

Use the same data policy as Stage 2-4:

- train target: `runs/oc20neb_tace_mace/tece-distill-20260717/mixed_train_tw0.75.extxyz`;
- validation target: original DFT validation file;
- teacher benchmark target: `teacher_valid.extxyz`;
- DFT labels remain the primary external accuracy axis;
- teacher labels quantify distillation/projection error.

Loss:

- energy per atom loss;
- force loss from conservative autograd forces;
- optional teacher/DFT mixed labels already encoded in the train file.

Do not add Hessian, relational latent loss, or virial loss in the first prototype. Those are later distillation stages after the high-throughput architecture has a measurable Pareto position.

## Benchmark Contract

The prototype benchmark must report the same core fields as `benchmark_models.py`:

- `atoms_per_second`;
- `configs_per_second`;
- `seconds_per_pass`;
- `peak_allocated_mb`;
- `mae_e_mev_atom`;
- `rmse_e_mev_atom`;
- `mae_f_mev_a`;
- `rmse_f_mev_a`;
- model class and variant name.

The first benchmark can include graph construction if that is the only implemented path, but it must explicitly mark whether timing includes graph construction. A second model-only benchmark should follow once graphs can be cached.

## Success Criteria

The T3 branch is promising if:

- `rtece_edge_sketch8` is more accurate than `scalar_fast` at comparable or better throughput;
- `rtece_edge_sketch8` is substantially faster than compact TACE variants, ideally with a step change beyond the 4e4-7e4 atoms/s range;
- edge sketches improve force MAE over `rtece_atomic_moments`, proving the TECE edge-relational idea adds value beyond NEP-like atomic moments;
- memory is clearly below compact TACE.

The T3 branch should be redirected if:

- Python/PyTorch scatter overhead prevents any throughput improvement over compact TACE;
- edge sketches do not improve accuracy over atomic moments;
- force autograd dominates runtime so much that conservative scalar sketches cannot be tested without a fused kernel.

In those cases, the next action is not descriptor proliferation. It is either a fused/Triton edge-sketch kernel or a narrower T4 scalar endpoint for maximum-throughput anchoring.

## Non-Goals

- Do not match full TACE teacher accuracy in the first prototype.
- Do not implement full RRA or SO(2) edge tensor state.
- Do not add dense per-edge channel mixing.
- Do not add softmax attention.
- Do not integrate into the full Hydra/TensorModel path until the prototype shows a meaningful Pareto position.
- Do not use parameter count as a success metric.

## Test Plan

Initial unit tests:

1. Moment descriptors are invariant under global rotation for scalar outputs.
2. Edge-relational sketch descriptors are invariant under global rotation.
3. The model energy is permutation-invariant over atom ordering for an isomorphic graph.
4. Conservative forces have shape `(num_atoms, 3)` and are gradients of the scalar energy.
5. Variant builder produces the four named variants with monotonic descriptor dimensions.
6. Benchmark summary accepts rTECE result JSON and ranks it with existing TACE/MACE points.

Initial integration tests:

1. Train on a tiny synthetic or small OC20NEB subset for a few steps without NaNs.
2. Produce a benchmark JSON with required fields.
3. Run one DFT and one teacher benchmark on a small fixed config subset.

## Implementation Order

1. Add descriptor helper tests for moment/sketch invariance and variant dimensions.
2. Implement `benchmarks/oc20neb_tace_mace/rtece_scalar_model.py`.
3. Implement a small training script for extxyz data and conservative forces.
4. Implement a benchmark path that emits the existing summary JSON schema.
5. Add config generation for the four variants.
6. Run tiny tests and a small-data smoke training.
7. Train the first full OC20NEB variants only after the smoke path is stable.
8. Update the Pareto summary and Stage-5 design record.

## Stage-5 Decision Rule

After the first full benchmark, update the design record with:

- where each rTECE variant sits relative to `scalar_fast`, `edge_min_mixed`, `edge_radial_mixed_20k`, MACE hybrid, and TACE OEq teacher;
- whether edge-relational sketches improve accuracy per cost over atomic moments;
- whether the prototype is bottlenecked by descriptor math, scatter, autograd forces, or graph construction;
- whether the next stage should be fused kernel work, teacher-conditioned sketch selection, or Hydra/TensorModel integration.
