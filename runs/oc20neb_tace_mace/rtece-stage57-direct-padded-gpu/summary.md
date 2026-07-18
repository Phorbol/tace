# Stage 57: GPU Validation Harness For Direct-Padded Streaming

Stage 57 prepared the GPU benchmark path for the Stage56 direct-padded streaming backend. The code change is small but important: the reusable `rtece_scalar_benchmark.sbatch` script now forwards graph construction and graph update backend controls, so direct-active `torch_radius_nopbc` semantics can be forced from Slurm jobs instead of relying on benchmark defaults.

## Harness change

- Added `GRAPH_CONSTRUCTION_BACKEND=${GRAPH_CONSTRUCTION_BACKEND:-torch_radius_nopbc}` to `rtece_scalar_benchmark.sbatch`.
- Added `GRAPH_UPDATE_BACKEND=${GRAPH_UPDATE_BACKEND:-ase_neighborlist}` to the same script.
- Forwarded both values to `benchmark_rtece_scalar.py`.
- Added a regression test that locks the sbatch contract.

## Verification

- Targeted sbatch test: `1 passed, 73 deselected, 1 warning`.
- Full rTECE test file after the harness change: `74 passed, 1 warning`.

## Submitted GPU jobs

All jobs were submitted from commit `3be8ebd` or its immediate local state with the Stage56 backend present. No benchmark JSON was produced because the jobs were cancelled by Slurm before stdout/stderr were created.

| job | mode | requested shape | state | evidence |
|---:|---|---|---|---|
| 679358 | `analytic_element_triton_descriptor_force` | 1x V100, `flood-1o2gpu`, 1024 configs, 5 passes | `CANCELLED by 0` after 1 s | `sacct`, no log/output files |
| 679359 | `analytic_element_direct_padded_descriptor_force` | 1x V100, `flood-1o2gpu`, 1024 configs, 5 passes | `CANCELLED by 0` after 1 s | `sacct`, no log/output files |
| 679362 | direct-padded retry | 1x V100, `rush-1o2gpu` | cancelled manually | pending reason: QOS not permitted on 16V100 |
| 679363 | edge-index Triton retry | 4x V100, `rush-gpu`, 256 configs, 2 passes | `CANCELLED by 0` after 2 s | `sacct`, no log/output files |
| 679364 | direct-padded retry | 4x V100, `rush-gpu`, 256 configs, 2 passes | `CANCELLED by 0` after 2 s | `sacct`, no log/output files |
| 679366 | direct-padded shape retry | 4x V100, `rush-gpu`, `ntasks=4`, 64 configs, 1 pass | `CANCELLED by 0` after 2 s | `sacct`, no log/output files |

## Interpretation

Stage57 did not produce a GPU Pareto point. The failed jobs did not execute Python, so they say nothing about correctness or throughput of the Stage56 kernel. They do prove that the benchmark harness now has the needed graph backend controls, and they identify the current external blocker for GPU validation: Slurm cancels these jobs immediately after allocation with `Reason=None`.

The next useful action is either to run the benchmark through an interactive allocation on a compute node, or to use a known-good project sbatch template from this cluster and transplant only the rTECE benchmark command. Do not interpret the cancelled jobs as an algorithmic negative result.
