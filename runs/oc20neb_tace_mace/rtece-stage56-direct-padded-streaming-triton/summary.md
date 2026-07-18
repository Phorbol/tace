# Stage 56: Direct-Padded Streaming Triton Descriptor+Force Backend

Stage 56 adds the first GPU-kernel lowering of the Stage55 idea that `edge_index` is not part of the scalar rTECE semantics. It is not yet the final cell-list backend: it still scans the padded per-configuration pair space. The change is nevertheless a real runtime renormalization step because descriptor and force kernels can now consume `(pos, batch counts, starts, z)` directly without materializing a public edge list.

## What changed

- Added `element_density_direct_padded_descriptors_triton(...)` and `element_density_direct_padded_forces_triton(...)` under `tace.models.rtece_triton_kernels`.
- Added `RTECEScalarModel.forward_element_density_direct_padded_triton_descriptor_force_analytic_forces(...)`. The method validates the scalar element-density endpoint, derives `counts/starts` from sorted `batch`, ignores `edge_index`, computes descriptors with the direct-padded Triton kernel, then applies the descriptor-gradient force kernel over the same direct-padded candidate stream.
- Added workflow and benchmark force mode `analytic_element_direct_padded_descriptor_force`.
- Extended `rtece_route_contract(...)` so this mode reports `triton_direct_padded_descriptor`, `triton_direct_padded_descriptor_force`, `direct_active_nopbc`, and `streaming_padded_candidates`.

## Verification

- TDD red check: Stage56 target tests failed before implementation because the direct-padded kernel API, route contract, and model method did not exist.
- Targeted Stage56 tests: `4 passed, 69 deselected, 1 warning`.
- Full rTECE test file: `73 passed, 1 warning`.
- CLI exposure check: `benchmark_rtece_scalar.py --help` lists `analytic_element_direct_padded_descriptor_force`.
- Current login node CUDA availability: `False`, device count `0`, so GPU numeric and throughput validation could not run in this turn. A CUDA-available job should run the existing optional numeric test and the OC20NEB benchmark path before treating this as a Pareto point.

## Interpretation

This is a useful intermediate lowering point, not the final backend. It deletes materialized edge-list lifetime from the element-density descriptor/force path at the GPU kernel interface, but it still pays padded candidate scanning. Against the TECE design-space document, this isolates two runtime axes that were previously entangled: persistent edge-state lifetime versus candidate-space size.

The next priority is now concrete: submit/run a GPU benchmark for `analytic_element_direct_padded_descriptor_force` against `analytic_element_triton_descriptor_force`; if it is semantically correct but not throughput-dominant, use the same kernel interface shape to replace padded pair offsets with true cell-list candidate streaming.
