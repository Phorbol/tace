# Stage145 Submission Log

## Final effective submissions

- NEP job: `687379`
- DeepMD job: `687381`
- sacct command: `sacct -j 687379,687381 --format=JobID,JobName,State,ExitCode,Elapsed,NodeList%30,MaxRSS`
- NEP state at record time: `RUNNING` on `16v100n01`; GPUMD `nep` reached training step 1000 with RMSE-F-Train `0.44271`.
- DeepMD state: `COMPLETED 0:0` on `16v100n01`; elapsed `00:03:48`; MaxRSS `8044684K`; trained 20000 steps and produced checkpoints.
- Interpretation: `deepmd completed; nep running training, not a launcher failure`.

## Launcher/debug history

- Jobs `687350`/`687351`: submitted to `4V100` and remained pending because this user is not permitted to use that partition; canceled and wrapper moved to `16V100`.
- Direct `16V100` + `rush-gpu` submission attempt: rejected by Slurm with `QOSMinGRES`; wrapper moved to `flood-1o2gpu` for 1-GPU jobs.
- Jobs `687366`/`687367`: launched on `16V100/flood-1o2gpu` but failed before training because wrappers used module `python`; NEP module lacked `numpy`, DeepMD module lacked `ase`. Wrapper conversion step now uses `/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python`.
- Job `687380`: DeepMD reached GPU training setup but failed in output-stat because split systems lacked per-system global `type_map.raw`; converter now writes global `type_map.raw` inside every split system directory.

## Current wrapper contract

- Partition/QOS: `16V100` / `flood-1o2gpu`.
- Sbatch forbidden options are absent from wrapper scripts; see wrapper audit in the run log.
- Shell flags: `set -eo pipefail` to avoid Lmod failure under `set -u`.
- Conversion Python: `/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python`.
