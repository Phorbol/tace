# Stage144 Submission Diagnostic

Date: 2026-07-20

Stage144 plan commit: `2fdd807` (`Plan rTECE stage144 cavity vector`).

## Submitted Jobs

| job | role | state | note |
|---:|---|---|---|
| 685722 | training | cancelled | Cancelled after about 19 minutes because the Python process never entered GPU training. |
| 685724 | physical triage | cancelled | Dependency job cancelled together with the stuck training job. |

## Observed Failure Mode

The training wrapper launched correctly and did not hit the previous SAI sbatch failure mode. The wrapper contained no `--export`, `--mem`, or `--cpus-per-task` flags, and stdout/stderr recorded the expected Stage144 configuration:

- train file: `runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/weighted_train_base2048_plus_teacher_relax640_forceonly_eanchor.extxyz`
- scalar paths: `atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot`
- `moment_l_max=2`, `species_basis_channels=24`, `radial_species_adapter_channels=8`, `descriptor_conditioner=residual_mlp:32`

On compute node `16v100n01`, the launched Python process stayed at about 0.1 percent CPU with no GPU compute application visible in `nvidia-smi`. `/proc/<pid>/wchan` reported `wait_on_page_bit_common` and later `cl_sync_io_wait`, with no training summary or benchmark JSON produced.

A local follow-up probe also timed out:

```bash
timeout 60 /home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -c "import torch"
```

This means the current failure is an environment/shared-filesystem or Python/Torch import page-wait problem, not a Stage144 model result. It should not be interpreted as evidence against `edge.cavity.vector_dot`.

## Next Action

After the Python/Torch import path is responsive again, rerun either:

1. a small Stage144 smoke with `limit_configs=512`, `valid_limit_configs=64`, `bench_limit_configs=128`, `max_steps=2000`; or
2. the full Stage144 training wrapper already committed under `runs/oc20neb_tace_mace/rtece-stage144-t3-cavity-vector/`.

Do not change the architecture before obtaining a clean Stage144 training/benchmark/physical result.
