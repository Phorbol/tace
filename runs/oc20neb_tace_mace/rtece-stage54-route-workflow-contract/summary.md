# Stage 54: rTECE Route And Workflow Contract

Stage 54 is an engineering-boundary stage, not a new accuracy/throughput Pareto point. It promotes the current scalar rTECE route from benchmark-local scripts into formal package APIs that can train, save, load, infer, and report the TECE/TACE degradation route in a machine-readable way.

## What changed

- Added `rtece_route_contract(...)` to `tace.models.rtece_scalar`. A config plus runtime force/graph backend now maps to a structured contract containing semantic tier, descriptor family, retained TECE groups, deleted TECE groups, descriptor width, force realization, descriptor realization, graph semantics, graph construction/update backend, edge-state lifetime, and Pareto axes.
- Added `tace.models.rtece_workflow` with formal `save_checkpoint`, `load_checkpoint`, `predict`, `loss_for_batch`, `evaluate_loss`, and `train_steps` APIs. Checkpoints now carry `tece_route` metadata alongside config and weights.
- Exported rTECE workflow APIs from `tace.models`, so the model repository itself has normal training/inference entrypoints instead of requiring benchmark-local helpers.
- Refactored `benchmarks/oc20neb_tace_mace/train_rtece_scalar.py` to reuse the core workflow functions. The benchmark keeps its historical two-value `load_checkpoint` wrapper for old benchmark/profile scripts, while `load_checkpoint_with_metadata` exposes the core metadata-aware loader.
- Updated the TECE distillation summary table to include the TECE route column, so Pareto rows carry semantic degradation information instead of only names, speed, and errors.

## Verification

- Targeted workflow/contract tests: `5 passed, 63 deselected, 1 warning`.
- Full rTECE test file: `68 passed, 1 warning`.

## Interpretation

This closes a structural gap raised during review: rTECE is now a real model-family endpoint inside the TACE package, with repo-body training and inference APIs. It does not yet add a new fused cell-list runtime point or improve MAE/RMSE. Its value is that every future Pareto row can be audited against the TECE/TACE route: what semantic groups were retained, what equivariant state was deleted, how forces were realized, and how long edge state lived in memory.

The next priority remains the Stage52/53 runtime target: implement and benchmark a fused cell-list descriptor/force backend for the element-density scalar endpoint. The contract added here gives that backend a clean place in the same route table rather than treating it as an isolated kernel experiment.
