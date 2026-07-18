# Stage 40: Graph-Semantics-Aware Pareto Summary

Stage 40 updated the TECE/TACE Pareto summary tooling so direct-active graph semantics cannot be silently mixed with ASE/PBC graph semantics. This is a methodology/compiler step, not a model-architecture change.

Implementation gate:

- `summarize_tece_distill.py` now preserves `graph_construction_backend` and `graph_update_backend` in student rows.
- Markdown student and Pareto-front tables now include a `graph backend` column.
- `test_rtece_summary_preserves_graph_construction_backend` covers both JSON row construction and Markdown output.
- The test was first run red and failed with `KeyError: graph_construction_backend`, then passed after the summary change.

Verification with the Stage-39 direct-active JSON files produced summary rows like:

| variant | graph backend | force mode | atoms/s | DFT F MAE | teacher F MAE | params |
|---|---|---|---:|---:|---:|---:|
| radial4h16 | torch_radius_nopbc | analytic_element_triton_descriptor_force | 87762046 | 43.85 | 47.71 | 449 |
| radial8h24 | torch_radius_nopbc | analytic_element_triton_descriptor_force | 64507178 | 30.20 | 35.84 | 1057 |
| radial8h32 | torch_radius_nopbc | analytic_element_triton_descriptor_force | 48192947 | 27.87 | 33.73 | 1665 |

Stage-40 interpretation against TECE/TACE:

- This closes a benchmark-methodology gap exposed by Stages 37-39. Graph semantics are now explicit in generated Pareto artifacts, so direct-distance rTECE rows and future PBC-correct rows can be compared or separated deliberately.
- The change supports the source-document requirement that hardware cost be reported as a coupled model-system quantity. The graph construction backend is part of the deployed operator realization, not incidental metadata.
- No model parameters, descriptors, force kernels, or graph construction behavior changed in this stage. It only makes the existing direct-active front auditable.
- Next priority: use this graph-semantics-aware summary while measuring a consistent direct-active trajectory/update path. The clean experiment is direct-active initial graph construction plus a matching fast update/provider backend; the larger alternative is a PBC-correct branch that adds periodic displacement/shift state to `RTECEGraph` and fused evaluators.
