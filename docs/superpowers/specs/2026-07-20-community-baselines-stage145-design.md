# Community Baselines Stage145 Design

## Purpose

Stage145 adds community high-throughput endpoints to the rTECE Pareto map. The goal is not to tune NEP or DeepMD to exhaustion. The goal is to place NEP and a DPA-like DeepMD baseline on the same data, metric, and throughput axes used by the current rTECE stage140/stage142/stage144 experiments.

This is required by `TECE_design_space.md`: NEP and DPA1-0 are not just narrative comparisons. They are concrete scalarized, high-throughput endpoints in the unified TECE degradation space. A serious rTECE Pareto claim needs measured distance to those endpoints.

## Approved Approach

Use Approach A:

- Train community baselines on the same stage142 weighted mixed-label training contract used by recent rTECE experiments.
- Benchmark each baseline against DFT labels, teacher labels, and the mixed training label where available.
- Report the same primary axes as rTECE: E/F MAE, E/F RMSE, E/F max error, E bias, atoms/s, dimer scan, and rattle-relax behavior.

The first implementation may mark DeepMD DPA1-0 as "DPA-like" if the installed DeepMD-kit 3.1.2 schema does not expose an exact DPA1-0 attention-depth-zero descriptor. The exact DeepMD descriptor config must be recorded in the manifest and summary.

## Inputs

Primary train source:

```text
runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/weighted_train_base2048_plus_teacher_relax640_forceonly_eanchor.extxyz
```

The file contains `energy`/`forces` mixed labels, `dft_energy`/`dft_forces`, `teacher_energy`/`teacher_forces`, and sample weights `energy_weight`/`forces_weight`.

Validation and benchmark sources should reuse the current rTECE benchmark files and limits used by stage140/stage142/stage144 whenever possible. If a format-specific converter cannot preserve a field, the converter must write a manifest field explaining the lost field and why it is not used by that baseline.

## Baseline Rows

### NEP

Use SAI module:

```bash
module load gpumd/4.8-cuda12.4
```

The module provides `nep`, `gpumd`, and `gnep`. Training should use GPUMD NEP input files derived from the installed `examples/nep_train` format. The converted `train.xyz` must use the mixed `energy` and `forces` fields for training in the first row.

The first NEP row should be a practical NEP4-style baseline with moderate generation count suitable for a stage smoke/full workflow. The exact `nep.in` content is part of the output artifact.

### DeepMD

Use SAI module:

```bash
module load deepmd-kit/3.1.2
```

The module provides `dp train`, `dp freeze`, and `dp test`. The preferred row is a DPA1-0 or attention-depth-zero DPA-like descriptor if the local schema supports it. If not, the first row uses the closest low-attention DeepMD descriptor exposed by `dp doc-train-input`, clearly labeled as an approximation rather than a true DPA1-0 result.

DeepMD data conversion must use explicit type maps and preserve train/valid split identity in a manifest.

## Metrics

Every completed row must produce one machine-readable summary with:

- train contract: source file, label target, weights used or ignored, train count, valid count;
- model contract: module version, input config, type map, cutoff, descriptor family, train steps/generations;
- DFT errors: E MAE/RMSE/max/bias and F MAE/RMSE/max;
- teacher/mixed errors when evaluable from extxyz fields;
- throughput: atoms/s, configs/s, device, warmup/measure pass counts, graph or neighbor-list inclusion level;
- physical checks: dimer scan and rattle-relax outputs or explicit reason if a row cannot run those checks yet;
- comparison anchors: stage140 ew2, stage142, and stage144 when available.

RMSE is the first ranking metric. MAE is retained but cannot be the only reported accuracy measure. Max error and physical probes are mandatory for interpreting tails.

## Throughput Contract

Stage145 must not compare unlike timing layers without labels. Timing outputs must say whether they are:

- model-only fixed graph;
- graph-amortized inference;
- full force loop;
- full MD loop.

The first acceptable baseline is graph-amortized or full force loop on a single GPU, plus a note on whether the community code includes neighbor rebuild cost. Atom-count scaling is a follow-up after at least one NEP and one DeepMD row produce valid accuracy summaries.

## Sbatch Contract

Wrappers must comply with the local SAI constraints:

- no `--export`;
- no `--mem`;
- no `--cpus-per-task`;
- use `--nodes`, `--ntasks`, `--gpus-per-node`;
- set environment variables inside the wrapper body with ordinary `export VAR=value` if needed.

The NEP executable must be run on a GPU compute node because running it on the login node can fail with CUDA driver/runtime mismatch.

## Outputs

Stage145 writes under:

```text
runs/oc20neb_tace_mace/community-baselines-stage145/
```

Expected committed artifacts:

- a stage145 manifest JSON;
- converter scripts or benchmark scripts under `benchmarks/oc20neb_tace_mace/`;
- no-export sbatch wrappers;
- tests covering manifest materialization and no forbidden sbatch flags;
- a results summary after jobs complete.

Large generated training datasets, model checkpoints, frozen models, and raw outputs are not committed unless already covered by repository policy.

## Non-Goals

Stage145 does not replace stage144. It gives the rTECE Pareto curve external endpoints. The stage144 `edge.cavity.vector_dot` experiment remains the immediate rTECE architecture question.

Stage145 does not claim a final NEP or DPA optimum. It creates a reproducible first comparison. Later stages can tune NEP/DPA more aggressively after the initial coordinates are known.

## Validation

Before implementation is considered ready:

- materialization script compiles;
- generated wrappers contain no forbidden sbatch flags;
- unit tests confirm manifest rows include NEP and DeepMD contracts;
- at least one smoke conversion for a small extxyz subset succeeds;
- all reported success claims are backed by fresh command output.

## Priority After Spec

1. Implement data conversion and manifest materialization.
2. Generate NEP and DeepMD smoke wrappers.
3. Run smoke jobs on GPU nodes.
4. If smoke succeeds, submit full stage145 jobs.
5. Summarize against stage140/stage142 and the latest stage144 result if available.
