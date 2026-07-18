# Stage 52: Cell-List Fused Descriptor Oracle

Question: can the Stage51 cell-list candidate representation be fused directly into the scalar rTECE element-density descriptor, without materializing or consuming an intermediate `edge_index`, while preserving the exact direct-active descriptor semantics?

Setup:

- Same direct-distance rTECE semantics as Stages 37-51.
- Descriptor family: `rtece_element_density`, cutoff 5.0 A, `num_radial=8`, descriptor dim 16.
- Same OC20NEB valid geometry source and synthetic trajectory step 19 displacement used in Stage51.
- CPU oracle/prototype only: cell size = cutoff, scan 27 neighboring cells, filter by true distance, and accumulate `[rho, rho_z]` descriptors directly into atoms. The implementation deliberately ignores `graph.edge_index`.

| limit configs | atoms | active edges | cell candidates | candidate/padded | candidate/active | max abs descriptor diff | direct graph CPU s | edge-list descriptor CPU s | cell-list fused descriptor CPU s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 5,298 | 69,804 | 207,504 | 0.372159 | 2.972666 | 6.77e-15 | 0.013755 | 0.480110 | 2.010375 |
| 512 | 31,831 | 472,830 | 1,661,434 | 0.428721 | 3.513808 | 9.16e-15 | 0.125035 | 0.993020 | 4.695996 |

Interpretation:

- The semantic bridge is now closed for the current direct-active element-density endpoint. A descriptor accumulated directly from cell-list candidates matches `packed_element_density_descriptors(torch_radius_nopbc_graph(...))` to numerical roundoff, with max absolute differences below `1e-14`.
- This is an algorithm-level validation, not a performance claim. The Python/Torch cell-list prototype is slower than the existing edge-list descriptor path, because it is a clarity-first oracle with per-configuration Python loops.
- The TECE/TACE meaning is clean: after semantic projection into scalar element-density descriptors and hardware downfolding into fused descriptor/force kernels, the next representation to renormalize is local topology lifetime. The edge list does not have to be a persistent public object for this endpoint; it can be lowered into candidate generation plus descriptor/force accumulation.
- This supports the next production direction: a low-level fused cell-list descriptor/force provider. NVIDIA `nvalchemi-toolkit-ops` or DeepMD-style edge force/virial kernels may be useful implementation references, but the acceptance criterion remains the TECE route: exact descriptor/force semantics first, then measured Pareto movement in throughput, memory, and error.

Verification:

- TDD red: the new tests first failed on missing `cell_list_packed_element_density_descriptors`.
- Target tests: `python -m pytest test/test_rtece_scalar.py -q -k cell_list_packed_element_density_descriptors` -> 2 passed.
- Full rTECE tests: `python -m pytest test/test_rtece_scalar.py -q` -> 62 passed.
- `run_descriptor_oracle.py` generated `descriptor_oracle_summary.json` plus per-limit JSON outputs.
