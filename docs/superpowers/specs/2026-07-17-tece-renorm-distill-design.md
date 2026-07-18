# TECE Renormalized Distillation Design

## Acceptance Standard: TECE/TACE Pareto Compiler

The final target is not a single student checkpoint that is as accurate as the largest teacher. The target is a clean, explainable model-compiler route that maps the TECE/TACE operator space to a sequence of nested, deployable student architectures and produces a measured Pareto curve between error and hardware cost.

A valid route must satisfy all of the following:

1. It is unified with the TECE/TACE theory: each simplification corresponds to deleting, projecting, factorizing, scalarizing, or downfolding explicit TECE semantic groups such as radial mode, angular bandwidth, irrep channel, correlation order, contraction path, edge placement, depth, and edge state lifetime.
2. It produces real compact dense models, not only masked full models. Any retained subspace must reduce actual downstream tensor dimensions, edge buffers, tensor-product instructions, or scalar-kernel work.
3. It separates projection error from training/distillation error. A bad architecture cannot be rescued by teacher labels alone; the route must expose where projection error enters.
4. It reports a Pareto table over DFT and teacher MAE/RMSE, throughput, peak memory, and latency, not parameter count.
5. It includes the high-throughput end of the curve, even when that endpoint is much less accurate than TACE/MACE. For NEP/DPA1-0-like throughput, the student must remove persistent wide equivariant edge/node state and move toward early scalarization or streaming scalar sketches.

The compiler route is therefore:

| tier | TECE/TACE interpretation | expected cost behavior | role in Pareto curve |
|---|---|---|---|
| `T0 full teacher` | wide edge/node equivariant state, dense/rich paths, full radial response | best accuracy, worst memory/throughput | upper-accuracy anchor |
| `T1 compact equivariant TACE` | reduce L, channel, depth, radial rank, edge update while preserving exact equivariance and conservative energy | moderate acceleration, still tensor-product dominated | current `edge_min`/`edge_radial` family |
| `T2 active-set compact TACE` | select radial/channel/path groups as true compact dimensions, with refit/distillation renormalizing deleted groups | builds a structured local Pareto front | current Stage-4 radial active-set experiment |
| `T3 scalar-sketched rTECE` | build low-order moments once, keep only selected scalar atomic and edge-relational contractions, stream edge sketches | should approach NEP/MTP/DPA1-0 style throughput class | required high-throughput branch |
| `T4 scalar endpoint` | pure radial/pair, NEP-like fixed scalar contractions, or DPA1-0-like low-rank geometric Gram descriptors | maximum throughput, lower accuracy | lower-accuracy/high-speed anchor |

The near-term implementation must not stop at `T1/T2`. Radial active-set is useful because it tests the compiler machinery inside existing TACE, but the route toward 1e7 atoms-step/s class throughput requires the `T3/T4` early-scalarization branch.

## Goal

Build the shortest closed-loop experiment that tests whether the TECE design-space idea can yield a reasonable low-cost, high-throughput student model on the existing OC20NEB fullcase-200 workflow.

## Source-Document Constraints

- `TECE_design_space.md` says the student must be selected as a low-cost projection of the TECE operator space, not as arbitrary channel shrinkage.
- `TECE_design_space.md` says deleted paths should be compensated through teacher/student projection or distillation, not copied with missing weights set to zero.
- `TECE_design_space.md` says hardware cost must be measured as throughput, memory, and latency, not parameter count.
- TACE model docs currently recommend `CgtpInteraction` because it supports OpenEquivariance/CuEquivariance fusion; `SO2Interaction` is not the first experiment because it lacks fusion-library support.
- TACE edge-update docs rank `IdentityEdgeUpdate` as the conservative low-cost choice and `Element2EdgeUpdate` as the richer but costlier choice.

## Experiment

Use an existing trained TACE checkpoint as the teacher. Generate teacher-labeled `extxyz` train and validation files by replacing the training target keys with teacher energy and forces while preserving original DFT labels under separate keys. Train a small matrix of reduced TACE students on those teacher labels. Benchmark each student against both the original DFT validation file and the teacher-labeled validation file.

## Student Matrix

The first matrix tests three projections:

- `scalar_fast`: persistent node state is scalar-only or nearly scalar-only; this tests early scalarization and maximum throughput.
- `edge_min`: keeps minimal edge-relational capability with `Element2EdgeUpdate`, but reduces angular bandwidth, depth, and channel count.
- `path_scalar`: keeps low persistent angular bandwidth but allows a small higher-correlation product basis to test whether selected scalar paths are cheaper than wide persistent tensor state.

## Success Criteria

The idea remains promising if at least one student reaches:

- lower memory than the current TACE OEq benchmark,
- higher throughput than the current TACE OEq benchmark,
- teacher-validation force MAE clearly below a same-size-from-DFT baseline or DFT-validation force MAE close enough to justify a second round.

The idea should be deprioritized if all students lose accuracy badly while only improving parameter count, or if throughput does not improve after reducing persistent angular state.

## Stage 1 Results: OC20NEB Reduced TACE Students

Measured on 2026-07-17 with OEq acceleration and the existing OC20NEB fullcase-200 workflow.

| variant | atoms/s | peak alloc MB | teacher F MAE | DFT F MAE | interpretation |
|---|---:|---:|---:|---:|---|
| scalar_fast | 74026.7 | 153.6 | 45.26 | 44.70 | Confirms early scalarization is cheap, but projection error is too high. |
| edge_min | 45726.5 | 504.8 | 37.25 | 37.52 | Best first-round compact TECE point; retaining minimal edge-relational state recovers much of the deleted accuracy at still-low hardware cost. |
| edge_min_mixed | 45256.3 | 504.8 | 37.32 | 37.50 | Label-level 0.75 teacher / 0.25 DFT anchor is slightly better on DFT and slightly worse on teacher; useful, but not the main bottleneck. |

Baselines: MACE hybrid is 38309.7 atoms/s, 609.8 MB, 32.40 DFT force MAE; TACE OEq teacher is 23632.2 atoms/s, 2429.7 MB, 23.84 DFT force MAE.

Stage interpretation against `TECE_design_space.md`:

- Hardware-cost claim is validated: reducing persistent angular state gives real throughput and memory improvements, not just fewer parameters.
- Pure scalar projection is too lossy; the retained edge-relational subspace is the first viable compact point.
- Simple label-level DFT anchoring helps only marginally, so the dominant error is still architecture/projection error rather than only optimization/distillation error.
- `path_scalar` under the current TACE config is not a true scalar-sketched rTECE test and should not be prioritized over edge-retained variants.

Next priority:

1. Stay on the `edge_min` family and test architecture projection before more loss variants.
2. Add one small capacity step that preserves the same cost class, such as modest radial/channel increase or selective edge relation capacity, and benchmark against `edge_min_mixed`.
3. In parallel, design a true scalar-sketched rTECE prototype only if it compiles to streaming scalar edge sketches; do not treat the current `path_scalar` config as sufficient evidence for or against rTECE.

## Stage 2 Results: `edge_radial_mixed`

The first architecture-projection follow-up keeps the `edge_min_mixed` angular and edge-relational structure fixed, but raises radial/readout capacity (`radial_MLP [64, 64]`, readout hidden `[16]`). This directly tests the source-document claim that the current bottleneck is projection capacity inside a low-cost TECE subspace, not more label mixing.

Measured checkpoints on 2026-07-17:

| checkpoint | atoms/s | peak alloc MB | teacher F MAE | DFT F MAE | validation DFT F MAE | interpretation |
|---|---:|---:|---:|---:|---:|---|
| 6672 | 45886.0 | 528.7 | 42.58 | 42.46 | 40.27 | Too early; validation improved over `edge_min_mixed`, but benchmark accuracy was not useful yet. |
| 10000 | 45398.3 | 528.7 | 40.15 | 40.34 | 38.69 | Positive architecture signal versus `edge_min_mixed` at the same checkpoint depth. |
| 15000 | 46396.2 | 528.7 | 37.76 | 38.00 | 37.30 | Continued improvement; close to but not yet better than `edge_min_mixed` final DFT accuracy. |
| 20000 | 43971.4 | 528.7 | 35.99 | 36.51 | 36.13 | Best compact student so far; beats `edge_min_mixed` accuracy with modest memory increase and still high throughput. |

Stage-2 interpretation against `TECE_design_space.md`:

- This validates radial/readout capacity as a useful low-cost projection axis inside the retained edge-relational TECE subspace.
- The improvement from `edge_min_mixed` to `edge_radial_mixed_20k` is larger than the label-mixing improvement, so the dominant Stage-1 bottleneck was architecture/projection capacity rather than only distillation loss.
- Hardware cost remains in the desired class: `edge_radial_mixed_20k` is about 1.86x faster and 4.6x lower peak allocation than the TACE OEq teacher, while also faster and lower-memory than the MACE hybrid baseline. It is less accurate than MACE hybrid, so the current result is a TECE-compiler feasibility signal, not final deployment parity.
- The tradeoff versus `edge_min_mixed` is acceptable for the next round: DFT force MAE improves from 37.50 to 36.51 meV/A, teacher force MAE from 37.32 to 35.99 meV/A, peak allocation rises from 504.8 to 528.7 MB, and atoms/s drops from 45.3k to 44.0k.

Next priority:

1. Promote `edge_radial_mixed` as the current compact base.
2. Explore finer radial-shell or radial-rank active-set variants before adding dense angular state; this follows the document recommendation to select basis/path/rank before changing loss.
3. Keep mixed teacher/DFT labels as the default training target for this family, but do not spend more time on teacher-weight sweeps until projection variants plateau.
4. Treat true scalar-sketched rTECE as a separate prototype requiring streaming scalar edge sketches; the current TACE configs are not enough evidence for or against it.

## Stage 3 Plan: Radial Active-Set Decomposition

Based on Stage 2, the next experiment should decompose the `edge_radial_mixed` gain without widening persistent angular state:

| variant | change versus `edge_min_mixed` | question |
|---|---|---|
| `edge_readout_mixed` | keep radial `[48, 48]`, add readout hidden `[16]` | Was the Stage-2 gain mostly readout capacity? |
| `edge_radial_slim_mixed` | use radial `[64, 32]`, readout hidden `[16]` | Can a slimmer radial rank retain most of the gain at lower cost than `[64, 64]`? |

Interim result on 2026-07-17:

| variant | epoch 7 DFT val F MAE | epoch 15 DFT val F MAE | decision |
|---|---:|---:|---|
| `edge_readout_mixed` | 46.42 | 43.05 | Stopped; readout-only does not explain the Stage-2 gain. |
| `edge_radial_slim_mixed` | 47.76 | 42.87 | Stopped; `[64, 32]` is too slim to retain the `[64, 64]` gain. |
| `edge_radial_mixed` reference | 43.85 | 40.27 | Keep as current compact base. |

Stage-3 decision:

1. Do not spend 20k benchmark budget on `edge_readout_mixed` or `edge_radial_slim_mixed`; both were cancelled after epoch 15.
2. The Stage-2 gain likely needs the fuller `[64, 64]` radial response, not only the final readout hidden layer.
3. The next projection experiment should be radial-shell or path active-set selection around the `edge_radial_mixed` base, not uniform radial shrinking and not dense angular expansion.

## Stage 4 Plan: Real Radial Active-Set Architecture

After re-reading `TECE_design_space.md`, the next priority is not another hidden-size sweep. The document's relevant ordering is:

1. preserve symmetry and conservative energy,
2. reduce representation-space cost through selected basis/path/rank groups,
3. measure actual throughput and memory,
4. only then refine distillation losses.

The current TACE implementation now has a concrete radial active-set hook: `radial_basis.active_indices`. `Representation` still computes the configured full radial basis, but only the selected columns are passed to node embedding, edge embedding, edge update, and interaction construction. This changes the actual downstream module dimensions, so it is a model-architecture projection rather than a YAML-only hyperparameter sweep.

New variants generated around the Stage-2 base:

| variant | change versus `edge_radial_mixed` | question |
|---|---|---|
| `edge_radial_active6_mixed` | keep radial MLP `[64, 64]`, readout `[16]`, pass only radial basis indices `[0,1,2,3,4,5]` downstream | Can a 6/8 radial subspace retain the Stage-2 gain with lower downstream edge cost? |
| `edge_radial_active4_mixed` | same but pass only indices `[0,1,2,3]` | Where does projection error become too large? |

Important limitation: this first version is a static prefix active-set, not yet teacher-POD/SVD or shell-wise sensitivity selection. It deliberately tests the cheap closed-loop question first: whether radial-rank projection around the best compact TACE point is viable enough to justify more expensive sensitivity/POD machinery.

Verification so far:

- `test/test_oc20neb_distill_tools.py`: 19 passed.
- Generated configs in `runs/oc20neb_tace_mace/tece-distill-20260717/configs/` for `edge_radial_active6_mixed` and `edge_radial_active4_mixed`.

Decision rule for this stage:

1. Train both active-set variants only to early checkpoints first.
2. Compare epoch-7/15 DFT validation force MAE against `edge_radial_mixed` reference (43.85 / 40.27 meV/A).
3. If `active6` is close, continue it to 20k and benchmark throughput/memory.
4. If both lag badly, do not spend more GPU on prefix radial masks; switch to teacher-conditioned radial POD/SVD or scalar-sketched rTECE.

Interim result on 2026-07-18:

| variant | job | epoch 7 DFT val F MAE | epoch 15 DFT val F MAE | decision |
|---|---:|---:|---:|---|
| `edge_radial_active6_mixed` | 678439 | 46.72 | 42.37 | Completed 20k; epoch 39 / step 16680 DFT val F MAE reached 36.54 meV/A, close to `edge_radial_mixed_20k` validation/reference accuracy. Benchmark job 678478 submitted for the step-20000 checkpoint. |
| `edge_radial_active4_mixed` | 678440 | 48.88 | 44.40 | Cancelled after epoch 15; 4/8 prefix radial mask creates too much projection error. |
| `edge_radial_mixed` reference | - | 43.85 | 40.27 | Current compact base; step-10000 DFT val F MAE was 38.69. |

Stage-4 benchmark result against `TECE_design_space.md`:

| variant | atoms/s | peak alloc MB | teacher F MAE | DFT F MAE | interpretation |
|---|---:|---:|---:|---:|---|
| `edge_radial_active6_mixed_20k` | 45289.3 | 528.7 | 37.46 | 37.44 | True radial active-set architecture point; slightly faster than full-radial, but accuracy falls back near `edge_min_mixed`. |
| `edge_radial_mixed_20k` | 43971.4 | 528.7 | 35.99 | 36.51 | Better T2 accuracy point; still far from NEP/DPA1-0 throughput class. |
| `edge_min_mixed_20k` | 45256.3 | 504.8 | 37.32 | 37.50 | Similar cost/accuracy to active6 with lower memory; remains a competitive compact TACE point. |

Stage-4 interpretation against `TECE_design_space.md`:

- This is now a true architecture projection: selected radial columns reduce downstream edge/node/update/interaction dimensions, not only train hyperparameters.
- Prefix 6/8 radial active-set is viable but not Pareto-dominant: it gives about 3% speedup over `edge_radial_mixed_20k` but loses about 0.93 meV/A DFT force MAE and does not reduce peak allocation under the current benchmark.
- Prefix 4/8 radial active-set is too lossy and was cancelled.
- The result supports the document warning that basis/path selection should be data-conditioned; naive prefix truncation is only a quick feasibility probe.
- More importantly, compact TACE variants remain in the 4e4-7e4 atoms/s range on this benchmark. Reaching NEP/DPA1-0-like throughput requires the T3/T4 branch: early scalarization and streaming scalar edge sketches, not more small TACE hyperparameter sweeps.


## Stage 5 Smoke: rTECE Scalar-Sketched Prototype

The independent T3 prototype has passed the first tiny training and benchmark smoke tests. This stage is not a Pareto claim yet; it proves that the conservative scalar-sketch path can train, save/load, emit forces from energy gradients, and write the same benchmark schema used by the TACE/MACE distillation matrix.

Smoke setup on 2026-07-18:

| item | value |
|---|---|
| variant | `rtece_pair` |
| train labels | `mixed_train_tw0.75.extxyz` |
| train configs / steps | 8 configs / 4 steps |
| benchmark configs | first 8 DFT valid configs, 400 atoms total |
| device | CPU fallback; CUDA was not available in this shell |
| checkpoint | `runs/oc20neb_tace_mace/rtece-scalar-smoke/rtece_scalar.pt` |

Smoke metrics:

| metric | value | interpretation |
|---|---:|---|
| final train loss | 3.6135 | finite conservative energy/force loss; smoke only |
| DFT F MAE | 55.24 meV/A | finite but not trained enough for accuracy conclusion |
| DFT F RMSE | 103.44 meV/A | smoke only |
| DFT E MAE | 5689.69 meV/atom | expected poor value after 4 steps without energy offset handling |
| CPU atoms/s, graph construction included | 375.03 | includes ASE neighbor-list graph construction; not comparable to GPU TACE/MACE Pareto rows |
| CPU atoms/s, prebuilt batched graph | 1694.93 | still CPU smoke, but closer to model-forward benchmark semantics |
| parameters | 4865 | small scalar head; parameter count is diagnostic only, not the success metric |

Stage-5 interpretation against `TECE_design_space.md`:

- This is the first true T3 branch: it removes persistent high-rank equivariant state and retains only low-order scalar contractions/sketches before the MLP head.
- The force path is conservative because forces are `-grad(total_energy, positions)`, so the route remains a usable potential-energy model rather than an unconstrained force regressor.
- The benchmark JSON is schema-compatible with the existing Pareto summary. The original smoke row used `includes_graph_construction: true`; a follow-up benchmark now supports prebuilt batched graphs with `includes_graph_construction: false` and `prebuilt_batched_graph: true`, which is the correct method for GPU Pareto comparison.
- The next priority is a GPU smoke/full run for `rtece_pair`, `rtece_atomic_moments`, and `rtece_edge_sketch8` under the prebuilt batched benchmark, plus energy normalization/offset handling before interpreting energy MAE. Do not return to T2 radial/hidden-size sweeps unless T3 fails on GPU throughput.


## Stage 6 Smoke: GPU rTECE Pareto Methodology And Energy Shift

The first GPU rTECE matrix used prebuilt batched graphs, so the timed loop measured model forward/force autograd rather than ASE neighbor-list construction. This is the correct benchmark semantics for comparing against the existing TACE/MACE Pareto rows.

Setup:

| item | value |
|---|---|
| jobs | 678575 without energy shift; 678579 with fitted per-atom energy shift |
| train labels | `mixed_train_tw0.75.extxyz` |
| train configs / steps | 64 configs / 100 steps |
| benchmark configs | first 256 DFT valid and first 256 teacher valid configs |
| graph timing | prebuilt batched graph, `includes_graph_construction=false` |
| GPU | single V100 |

GPU matrix result:

| variant | shift | atoms/s | peak alloc MB | params | DFT F MAE | teacher F MAE | interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| `rtece_pair` | 0.0000 | 4530206 | 178.6 | 4865 | 343.01 | 343.59 | throughput signal exists, but unshifted residual target is poorly conditioned |
| `rtece_atomic_moments` | 0.0000 | 2582370 | 434.0 | 5889 | 319.68 | 319.31 | slower, still poorly conditioned |
| `rtece_edge_sketch8` | 0.0000 | 648335 | 687.0 | 6401 | 323.28 | 322.94 | large throughput cost with no early accuracy gain |
| `rtece_pair` | -4.8954 | 4487036 | 178.6 | 4865 | 64.91 | 66.70 | current best T3 smoke point; about 100x compact-TACE throughput with worse force MAE |
| `rtece_atomic_moments` | -4.8954 | 2555088 | 434.0 | 5889 | 66.13 | 66.62 | not Pareto-dominant at 100 steps |
| `rtece_edge_sketch8` | -4.8954 | 643255 | 687.0 | 6401 | 67.27 | 67.73 | dominated by pair at this stage |

Stage-6 interpretation against `TECE_design_space.md`:

- The fitted per-atom energy shift is a zeroth-order renormalization baseline: it removes the coarse extensive energy scale and lets the scalar sketch learn residual structure. This is theoretically cleaner than asking a tiny scalar head to relearn the dominant extensive offset.
- Force MAE improved from about 320-343 meV/A to about 65-67 meV/A after the shift, while throughput stayed essentially unchanged. This validates energy normalization as the right next priority.
- `rtece_pair` is currently the cleanest throughput-first point: 4.49M atoms/s on one V100 with 178.6 MB peak allocation and 64.91 meV/A DFT force MAE after only 100 steps. It is not NEP/DPA-class yet, but it is already two orders of magnitude faster than the compact TACE benchmarks in this repository.
- `rtece_edge_sketch8` is not worth longer training until the edge sketch implementation is fused or otherwise made cheaper; it loses too much throughput without early accuracy benefit.
- Next priority: longer training for `rtece_pair`, with `rtece_atomic_moments` as a secondary control. Do not spend more GPU on T2 radial masks unless the T3 pair point fails to improve with training.


## Stage 7 Smoke: rTECE Pair Learning-Rate And Duration Sweep

After Stage 6, the best architecture was `rtece_pair`; `atomic_moments` and `edge_sketch8` were not Pareto-dominant in the current implementation. Stage 7 therefore tested whether the `rtece_pair` point improves by longer residual training.

Setup: 512 mixed-label train configs, 1024 DFT/teacher valid benchmark configs, prebuilt batched graph timing, single V100.

| job | LR | steps | atoms/s | peak alloc MB | DFT F MAE | teacher F MAE | DFT E MAE | decision |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 678579 | 1e-3 | 100 | 4487036 | 178.6 | 64.91 | 66.70 | 1425.0 | short-run baseline; good force but smaller 256-config benchmark batch |
| 678582 | 1e-3 | 2000 | 6810119 | 718.8 | 114.98 | 116.58 | 1597.0 | too unstable at longer duration |
| 678585 | 3e-4 | 2000 | 6818215 | 718.8 | 95.30 | 97.13 | 1646.1 | better than 1e-3 but still unstable |
| 678586 | 1e-4 | 2000 | 6784287 | 718.8 | 60.79 | 63.42 | 1645.9 | current best rTECE throughput-first point |
| 678588 | 1e-4 | 5000 | 6777603 | 718.8 | 81.78 | 84.03 | 1659.5 | overtraining/last-checkpoint degradation |

Stage-7 interpretation:

- `rtece_pair` can reach about 6.8M atoms/s on one V100 for 1024-config prebuilt batches. This is still below the desired 1e7 atoms step/s class, but it is already far beyond compact TACE throughput in this repo.
- The best force MAE so far is 60.79 meV/A DFT / 63.42 meV/A teacher, which is worse than compact TACE but in the intended low-precision/high-throughput regime.
- Longer training without checkpoint selection is not reliable: 5000 steps is worse than 2000, and high LR degrades badly.
- Next priority is not more blind training. Add validation/best-checkpoint support and then rerun `rtece_pair` with LR=1e-4, saving the best model by validation force loss.


## Stage 8 Smoke: rTECE Pair Best-Validation Checkpoint

Stage 7 showed that the final checkpoint can be worse than an earlier model. Stage 8 added validation-loss checkpoint selection and reran `rtece_pair` with LR=1e-4, 512 train configs, 64 DFT validation configs, 5000 max steps, and evaluation every 100 steps.

| job | checkpoint rule | best step | atoms/s | peak alloc MB | DFT F MAE | teacher F MAE | DFT E MAE | interpretation |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 678588 | last step | 5000 | 6777603 | 718.8 | 81.78 | 84.03 | 1659.5 | last checkpoint degrades |
| 678597 | best validation loss | 200 | 6814191 | 718.8 | 56.59 | 60.12 | 1612.1 | current best rTECE point |

Stage-8 interpretation:

- Best-validation checkpointing is necessary for this low-capacity residual scalar model. The useful model appears early; later optimization can reduce the instantaneous training sample loss while degrading validation force MAE.
- The current best point is `rtece_pair`, best-step 200: 6.81M atoms/s, 718.8 MB peak allocation on 1024-config prebuilt V100 benchmark, 56.59 meV/A DFT force MAE.
- This is still not the final target of >1e7 atoms step/s, but it establishes a quantitative T3 Pareto anchor that compact TACE could not reach.
- Next priority: refine early checkpoint resolution (`eval_interval` 20-50) and benchmark batch scaling; do not add heavier edge sketches until pair has been fully characterized.


## Stage 9 Smoke: rTECE Validation-Subset Selection

Stage 8 established best-checkpoint selection, but the validation subset matters. Stage 9 compared a finer early checkpoint interval against a larger validation subset.

| job | validation configs | eval interval | max steps | best step | atoms/s | peak alloc MB | DFT F MAE | teacher F MAE | interpretation |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 678597 | 64 | 100 | 5000 | 200 | 6814191 | 718.8 | 56.59 | 60.12 | useful best-checkpoint baseline |
| 678600 | 64 | 20 | 1000 | 140 | 6829702 | 718.8 | 71.82 | 74.39 | finer interval overfits noisy 64-config validation |
| 678603 | 256 | 100 | 1000 | 1000 | 6742026 | 718.8 | 50.91 | 54.56 | current best rTECE point |
| 678606 | 256 | 100 | 2000 | 1700 | 6827578 | 718.8 | 66.50 | 69.13 | longer run degrades despite validation selection |

Stage-9 interpretation:

- The current best point is `rtece_pair` from job 678603: 6.74M atoms/s, 50.91 meV/A DFT force MAE, 54.56 meV/A teacher force MAE.
- A larger validation subset is more important than a finer interval; 64-config validation picked a worse checkpoint when evaluated on 1024 configs.
- The useful low-capacity pair model is still early-training limited. Longer runs can degrade even with validation selection, so future training should emphasize validation design, LR schedule, and possibly force/energy loss weighting before adding architecture complexity.
- Next throughput priority is benchmark batch scaling for the 678603 checkpoint to check whether the same model can approach the 1e7 atoms step/s class at larger prebuilt batches.


## Stage 10 Smoke: rTECE Pair Batch-Scaling Throughput

Stage 10 benchmarked the current best checkpoint from job 678603 at larger prebuilt batches. This isolates model forward/force-autograd throughput from ASE graph construction and tests whether the T3 pair point can approach the >1e7 atoms step/s target by batching alone.

| job | limit configs | atoms | atoms/s | configs/s | peak alloc MB | DFT F MAE | interpretation |
|---:|---:|---:|---:|---:|---:|---:|---|
| 678615 | 1024 | 59193 | 6809129 | 117793 | 718.8 | 50.91 | baseline 1024-config batch |
| 678615 | 2048 | 119271 | 7094841 | 121825 | 1406.3 | 49.64 | modest throughput gain |
| 678615 | 4096 | 250355 | 7313717 | 119658 | 2963.7 | 47.93 | best accuracy estimate on larger subset |
| 678618 | 8192 | 525769 | 7442352 | 115959 | 6193.9 | 49.33 | throughput nearly plateaus before V100 memory limit |

Stage-10 interpretation:

- Batching alone raises the current best rTECE pair point from 6.8M to 7.44M atoms/s, but it does not reach 1e7. The curve is flattening by 4096-8192 configs.
- The remaining gap is likely implementation/kernel overhead in PyTorch scatter/autograd force evaluation, not insufficient batch size.
- The current quantitative Pareto anchor is therefore: `rtece_pair`, best-validation checkpoint from job 678603, 47.9-50.9 meV/A DFT force MAE depending on benchmark subset, 6.8-7.4M atoms/s on one V100, prebuilt graph, 0.7-6.2GB peak depending on batch.
- Next architecture/compiler priority: optimize the rTECE pair force path, or export/fuse scalar descriptor and analytic force kernels. Adding edge sketches is lower priority because `edge_sketch8` was throughput-dominated and not more accurate in early tests.


## Stage 11 Smoke: Analytic rTECE Pair Force Path

Stage 10 showed that batching the autograd force path plateaued around 7.4M atoms/s. Stage 11 replaced only the `rtece_pair` inference force path with a semi-analytic chain rule: the MLP gradient with respect to scalar pair densities is still computed by autograd, but the radial-density derivative with respect to distances and positions is evaluated explicitly. Training still uses the original conservative autograd path.

Correctness gate:

- `test_pair_analytic_forces_match_autograd_forces` verifies that analytic-pair energies and forces match the autograd path in double precision.
- Full rTECE test file after the change: 15 passed.

Throughput comparison for the same job-678603 best checkpoint:

| force mode | configs | atoms | atoms/s | peak alloc MB | DFT F MAE | interpretation |
|---|---:|---:|---:|---:|---:|---|
| autograd | 1024 | 59193 | 6809129 | 718.8 | 50.9084 | Stage-10 baseline |
| analytic_pair | 1024 | 59193 | 14401868 | 350.6 | 50.9084 | >2.1x faster, same forces within benchmark precision |
| autograd | 4096 | 250355 | 7313717 | 2963.7 | 47.9323 | batch scaling plateau |
| analytic_pair | 4096 | 250355 | 16091763 | 1405.4 | 47.9323 | reaches NEP/DPA-like throughput class |
| autograd | 8192 | 525769 | 7442352 | 6193.9 | 49.3271 | memory-heavy plateau |
| analytic_pair | 8192 | 525769 | 16538878 | 2918.6 | 49.3271 | current best throughput point |

Stage-11 interpretation:

- This is the first closed-loop rTECE point that satisfies the original throughput target: a TECE-grounded scalar-renormalized model reaches >1e7 atoms/s on one V100 while preserving conservative forces.
- The current Pareto anchor is `rtece_pair` with fitted per-atom energy shift, best-validation checkpoint from job 678603, analytic pair force path, and prebuilt graph benchmark: 14.4-16.5M atoms/s with about 48-51 meV/A DFT force MAE depending on benchmark subset.
- A force-mode-aware summary row now records the 4096-config analytic benchmark as 16.09M atoms/s, 47.93 meV/A DFT force MAE, and 51.22 meV/A teacher force MAE.
- The result supports the document hypothesis that the key bottleneck was not only representation dimension, but the lifetime of high-rank/autograd state in the force path. Early scalarization plus analytic/fused force propagation is the clean route toward NEP/DPA-class throughput inside the TECE framework.
- Next priority: characterize robustness/transfer of the analytic pair point, then explore analytic/fused extensions only after pair is fully characterized. Edge sketches remain lower priority because they were accuracy-neutral and throughput-dominated before force-path optimization.

## Stage 12 Smoke: Analytic rTECE Pair Offset-Window Robustness

Stage 11 used prefix slices of the valid extxyz file. Stage 12 added `--start-config` support to the rTECE benchmark and reran the same job-678603 best checkpoint on non-prefix windows. This checks whether the current >1e7 atoms/s endpoint is a stable low-precision model point rather than an artifact of the first valid configurations.

Implementation gate:

- `benchmark_rtece_scalar.py` now exposes `--start-config`, records `start_config` and `extxyz_index` in JSON, and uses a local ASE extxyz window loader.
- `rtece_scalar_benchmark.sbatch` forwards `START_CONFIG`.
- Full rTECE test file after the change: 18 passed.

DFT valid-window comparison for the same checkpoint and analytic-pair force mode:

| job | extxyz index | configs | atoms | atoms/s | peak alloc MB | DFT F MAE | DFT F RMSE | DFT E MAE | interpretation |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| 678636 | `:4096` | 4096 | 250355 | 16091763 | 1405.4 | 47.93 | 118.66 | 1408.55 | prefix baseline from Stage 11 |
| 678654 | `4096:8192` | 4096 | 275414 | 16188377 | 1534.0 | 50.59 | 128.06 | 1494.59 | same throughput class and same force-error regime |
| 678655 | `8192:12288` | 1808 | 120704 | 15241383 | 686.6 | 50.23 | 126.36 | 1345.99 | tail window is shorter but still stable |

Stage-12 interpretation:

- The analytic `rtece_pair` point is not merely a prefix-window artifact. Across the valid file, DFT force MAE stays around 48-51 meV/A while single-V100 prebuilt-graph throughput stays around 15.2-16.2M atoms/s.
- The model should be treated as the current high-throughput/low-precision Pareto endpoint, not as a high-accuracy replacement for TACE/MACE. The large energy MAE remains a known limitation of this aggressively scalarized residual model.
- The result strengthens the TECE/TACE document hypothesis: the decisive simplification is progressive deletion of persistent high-rank equivariant state plus analytic/fused force propagation, not another small parameter sweep of the original architecture.
- Next priority: produce a more systematic Pareto surface around this endpoint by varying only theoretically ordered rTECE axes: validation/training label mix and capacity at fixed scalarized pair physics first, then analytic/fused low-order descriptors if they provide measured force-error reduction without destroying the >1e7 atoms/s regime.

## Stage 13 Smoke: rTECE Pair Training-Axis Pareto Sweep

Stage 12 established that the analytic `rtece_pair` endpoint is robust across valid-file windows. Stage 13 therefore stayed on the same TECE/TACE projection, rather than adding richer descriptors, and varied only ordered training/capacity axes inside the scalarized pair model. This separates projection error from train/distillation error while preserving the high-throughput execution path.

Implementation gate:

- `train_rtece_scalar.py` now exposes `--hidden-channels`, `--energy-weight`, and `--force-weight`, and records them in `train_summary.json`.
- `rtece_scalar_matrix.sbatch` forwards these axes and `FORCE_MODE`, and separates `TRAIN_VALID_FILE` from DFT/teacher benchmark validation files. This allows teacher-only training while still evaluating against DFT and teacher validation.
- Full rTECE test file after the change: 22 passed.

4096-config prefix benchmark comparison, all using `rtece_pair` and analytic-pair force mode:

| job | hidden | train labels | force weight | params | atoms/s | DFT F MAE | teacher F MAE | DFT E MAE | interpretation |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 678603/678636 | 64x64 | mixed tw0.75 | 10 | 4865 | 16091763 | 47.93 | 51.22 | 1408.5 | Stage-11 baseline |
| 678668 | 128x64 | mixed tw0.75 | 10 | 9601 | 15370207 | 54.22 | 57.01 | 1410.0 | larger head is slower and less accurate in this run |
| 678669 | 64x64 | teacher-only | 10 | 4865 | 16064967 | 43.89 | 47.37 | 1404.5 | teacher labels help versus mixed baseline |
| 678670 | 32x32 | mixed tw0.75 | 10 | 1409 | 18628337 | 44.30 | 47.49 | 1408.5 | new maximum-throughput point with modest error cost |
| 678671 | 64x64 | mixed tw0.75 | 30 | 4865 | 16230938 | 36.13 | 40.69 | 1410.6 | new best error/throughput point in the T3/T4 branch |

Offset-window robustness for the new force-weight-30 checkpoint:

| job | extxyz index | configs | atoms/s | DFT F MAE | DFT F RMSE | DFT E MAE | interpretation |
|---:|---|---:|---:|---:|---:|---:|---|
| 678671 | `:4096` | 4096 | 16230938 | 36.13 | 117.60 | 1410.6 | prefix benchmark |
| 678673 | `4096:8192` | 4096 | 16158452 | 40.83 | 129.30 | 1492.4 | same high-throughput class, still better than Stage-11 baseline |
| 678675 | `8192:12288` | 1808 | 15423347 | 38.75 | 126.48 | 1352.6 | shorter tail window but stable force-error regime |

Stage-13 interpretation:

- The most important result is methodological: within the same scalarized TECE projection, changing the force loss weight gives a large force-error improvement without sacrificing the >1e7 atoms/s regime. This means the previous 48-51 meV/A point was not purely projection-limited.
- The current best balanced endpoint is `rtece_pair`, 64x64 scalar head, mixed labels, `force_weight=30`, analytic-pair force path: about 15.4-16.2M atoms/s and 36-41 meV/A DFT force MAE across valid windows.
- The current maximum-throughput endpoint is `rtece_pair`, 32x32 scalar head, mixed labels, `force_weight=10`: 18.63M atoms/s with 44.30 meV/A DFT force MAE on the 4096-config prefix benchmark. This is a genuine second Pareto point, not only a parameter-count reduction.
- Teacher-only labels improve over the mixed baseline at the same architecture, so the next label-axis test should compare teacher-only plus `force_weight=30` and possibly mixed tw values after regenerating label mixes.
- Larger head capacity is not automatically useful; 128x64 is slower and worse here. The next architecture expansion should not be a blind MLP width increase. It should be a TECE-ordered descriptor addition with analytic/fused force path, after the pair branch's label/loss axes plateau.

## Stage 14 Smoke: rTECE Pair Combined Loss/Label Axes

Stage 13 showed two useful axes: `force_weight=30` improves the balanced 64x64 point, while a 32x32 scalar head improves throughput with only moderate force-error cost. Stage 14 combined these axes before adding any richer descriptors. This follows the TECE review rule: exhaust train/distillation axes inside the current projection before changing the architecture projection.

4096-config prefix comparison:

| job | hidden | train labels | force weight | params | atoms/s | DFT F MAE | teacher F MAE | DFT E MAE | interpretation |
|---:|---|---|---:|---:|---:|---:|---:|---:|---|
| 678671 | 64x64 | mixed tw0.75 | 30 | 4865 | 16230938 | 36.13 | 40.69 | 1410.6 | best balanced Stage-13 point |
| 678681 | 64x64 | teacher-only | 30 | 4865 | 16152730 | 36.30 | 40.80 | 1414.7 | teacher-only does not improve over mixed when force weight is already high |
| 678670 | 32x32 | mixed tw0.75 | 10 | 1409 | 18628337 | 44.30 | 47.49 | 1408.5 | Stage-13 maximum-throughput point |
| 678682 | 32x32 | mixed tw0.75 | 30 | 1409 | 18623067 | 39.42 | 43.14 | 1407.1 | new maximum-throughput Pareto point |

Offset-window robustness for the new 32x32 force-weight-30 point:

| job | extxyz index | configs | atoms/s | DFT F MAE | DFT F RMSE | DFT E MAE | interpretation |
|---:|---|---:|---:|---:|---:|---:|---|
| 678682 | `:4096` | 4096 | 18623067 | 39.42 | 114.51 | 1407.1 | prefix benchmark |
| 678685 | `4096:8192` | 4096 | 18642930 | 42.27 | 123.65 | 1491.4 | stable high-throughput offset window |
| 678686 | `8192:12288` | 1808 | 17649690 | 41.53 | 121.27 | 1351.4 | shorter tail window but still stable |

Stage-14 interpretation:

- The current scalarized-pair Pareto front now has two clean points: 64x64/force30 for lower error at about 15.4-16.2M atoms/s, and 32x32/force30 for higher throughput at about 17.6-18.6M atoms/s. Both remain conservative energy models with analytic pair forces.
- Teacher-only labels do not beat mixed labels once force loss is properly weighted, so the immediate label-axis priority drops. If more label work is needed, regenerate mixed train files at different teacher weights rather than only teacher-only.
- The next architecture step should be TECE-ordered and analytic/fused from the start: either add the cheapest scalar descriptor that can receive an analytic force path, or formalize the pair model as the T4 endpoint and stop expanding it until the Pareto table needs a middle point between compact TACE and scalar pair.

## Stage 15 Smoke: Density-Quadratic Scalar Contraction

Stage 14 left one open architecture question: whether the next useful point between pure pair and heavier moments could be a very cheap scalar contraction with the same analytic force path. Stage 15 added `rtece_density_quadratic`, which augments pair density descriptors `rho_n` with `rho_n^2`. This is a TECE-ordered scalar contraction: it increases local scalar correlation order without introducing vector/quadrupole moments, edge sketches, persistent high-rank state, or autograd force propagation.

Implementation gate:

- `RTECEScalarConfig` now has `use_density_quadratic`.
- `build_rtece_config("rtece_density_quadratic")` creates the new variant.
- `forward_density_analytic_forces` matches autograd forces for density-only descriptors, including `rho^2`.
- `benchmark_rtece_scalar.py` exposes `--force-mode analytic_density`.
- Full rTECE test file after the change: 24 passed.

4096-config prefix comparison against the Stage-14 pair front, all with force weight 30:

| job | variant | hidden | params | atoms/s | DFT F MAE | teacher F MAE | DFT E MAE | force mode | interpretation |
|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 678682 | `rtece_pair` | 32x32 | 1409 | 18623067 | 39.42 | 43.14 | 1407.1 | analytic_pair | current high-throughput point |
| 678671 | `rtece_pair` | 64x64 | 4865 | 16230938 | 36.13 | 40.69 | 1410.6 | analytic_pair | current lower-error point |
| 678709 | `rtece_density_quadratic` | 32x32 | 1665 | 12099993 | 46.39 | 49.33 | 1395.7 | analytic_density | dominated: slower and worse force MAE |
| 678708 | `rtece_density_quadratic` | 64x64 | 5377 | 11041542 | 50.56 | 53.14 | 1395.2 | analytic_density | dominated: slower and worse force MAE |

Stage-15 interpretation:

- This is a useful negative result. A TECE-ordered scalar contraction is not automatically Pareto-improving; `rho^2` increases descriptor/head work enough to lose throughput and does not improve force error under the current training setup.
- The pure pair model remains the current T4 endpoint and is not improved by naive scalar polynomial enrichment.
- The next architecture step should not be another density polynomial. If an intermediate point is needed, it should add genuinely new geometric information with a planned analytic/fused force path, such as a carefully selected low-order moment norm or a teacher-conditioned scalar sketch, and must be tested first as a small isolated variant.
- Until that exists, the cleanest deliverable is the pair Pareto front: 32x32/force30 for maximum throughput and 64x64/force30 for lower force MAE.

## Stage 16 Smoke: Element-Conditioned Radial Density

Stage 15 showed that simply increasing scalar polynomial order with `rho_n^2` is dominated. Stage 16 therefore tested a more TECE-semantic descriptor: `rtece_element_density` augments the pure pair density `sum_j R_n(r_ij)` with neighbor-element-conditioned radial density `sum_j (z_j / z_max) R_n(r_ij)`. This restores one deleted chemistry/radial semantic group while still deleting persistent angular state, edge sketches, vector/quadrupole moments, and full force autograd. The force path remains analytic-density: the extra chain-rule term is the same radial derivative weighted by neighbor element.

Implementation gate:

- `RTECEScalarConfig` now has `use_element_density`.
- `build_rtece_config("rtece_element_density")` creates the new ordered variant.
- Scalar descriptors are rotation invariant but sensitive to neighbor element changes.
- `forward_density_analytic_forces` matches full autograd forces for element-density descriptors.
- Full rTECE test file after the change: 27 passed.

4096-config prefix comparison against the Stage-14/15 front, all with mixed labels and force weight 30:

| job | variant | hidden | params | atoms/s | DFT F MAE | teacher F MAE | DFT E MAE | force mode | interpretation |
|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 678682 | `rtece_pair` | 32x32 | 1409 | 18622904 | 39.42 | 43.14 | 1407.1 | analytic_pair | high-throughput pair point |
| 678671 | `rtece_pair` | 64x64 | 4865 | 16230621 | 36.13 | 40.69 | 1410.6 | analytic_pair | lower-error pair point |
| 678749 | `rtece_element_density` | 32x32 | 1665 | 14696599 | 28.14 | 33.93 | 1409.4 | analytic_density | new lower-error rTECE Pareto point |
| 678750 | `rtece_element_density` | 64x64 | 5377 | 13057248 | 35.11 | 39.86 | 1408.6 | analytic_density | dominated by element-density 32x32 |
| 678709 | `rtece_density_quadratic` | 32x32 | 1665 | 12100417 | 46.39 | 49.33 | 1395.7 | analytic_density | dominated negative result |

Offset-window robustness for the new 32x32 element-density point:

| job | extxyz index | configs | atoms/s | DFT F MAE | DFT F RMSE | DFT E MAE | interpretation |
|---:|---|---:|---:|---:|---:|---:|---|
| 678749 | `:4096` | 4096 | 14696599 | 28.14 | 109.93 | 1409.4 | prefix benchmark |
| 678757 | `4096:8192` | 4096 | 14729932 | 30.75 | 117.55 | 1495.2 | same throughput class and stable force-error regime |
| 678759 | `8192:12288` | 1808 | 13884471 | 31.18 | 116.16 | 1353.0 | shorter tail window but still stable |

Stage-16 interpretation:

- This is the first positive architecture expansion beyond pure pair. It is not a generic parameter tweak: it restores neighbor chemistry in the radial density while staying inside the early-scalarized TECE/TACE degradation route.
- The new Pareto front now has three rTECE scalar points: pair-32 for maximum throughput, pair-64 for intermediate error/throughput, and element-density-32 for substantially lower force error at still >1e7 atoms/s.
- The contrast with Stage 15 is important. A naive scalar polynomial with the same descriptor dimensionality as element-density was slower and worse; the useful axis is TECE semantic content, not descriptor count.
- The 64x64 element-density result is dominated, so the next architecture step should use 32x32/force30 first. Wider MLPs are lower priority unless a descriptor proves bottlenecked by readout capacity.
- Next priority: add the cheapest geometric scalar carrying genuinely new information with an analytic force path, most likely vector moment norm before quadrupole norm or edge sketches. If that is dominated, freeze the rTECE scalar Pareto front and move to kernel fusion/export around pair and element-density.


## Stage 17 Smoke: Vector Moment Norm

Stage 16 showed that restoring neighbor-element-conditioned radial density is Pareto-positive. Stage 17 tested the next geometric semantic block, `rtece_vector_moments`, which augments density descriptors with the rotational invariant `||sum_j R_n(r_ij) u_ij||^2`. This is theoretically cleaner than jumping to full atomic moments because it adds only vector-norm information and still omits quadrupole norms and edge sketches. The implementation uses an analytic chain-rule force path: density and MLP gradients are still differentiated with respect to scalar descriptors, while the radial/unit-vector derivative of the vector moment is evaluated explicitly.

Implementation gate:

- `RTECEScalarConfig` now has `use_vector_moments`.
- `build_rtece_config("rtece_vector_moments")` creates the new ordered variant.
- Vector-moment descriptors are rotation invariant and geometry sensitive.
- `forward_density_analytic_forces` matches full autograd forces for vector-moment descriptors.
- Full rTECE test file after the change: 29 passed.

4096-config prefix comparison, all with mixed labels and force weight 30:

| job | variant | hidden | params | atoms/s | peak alloc MB | DFT F MAE | teacher F MAE | force mode | interpretation |
|---:|---|---|---:|---:|---:|---:|---:|---|---|
| 678682 | `rtece_pair` | 32x32 | 1409 | 18622904 | 1405.4 | 39.42 | 43.14 | analytic_pair | maximum-throughput front point |
| 678671 | `rtece_pair` | 64x64 | 4865 | 16230621 | 1405.4 | 36.13 | 40.69 | analytic_pair | intermediate pair front point |
| 678749 | `rtece_element_density` | 32x32 | 1665 | 14696599 | 1404.9 | 28.14 | 33.93 | analytic_density | current lower-error rTECE front point |
| 678770 | `rtece_vector_moments` | 32x32 | 1665 | 6673068 | 3401.9 | 34.41 | 38.83 | analytic_density | dominated; too slow and less accurate than element-density |

Stage-17 interpretation:

- This is a useful negative result. The vector-moment semantic block is theoretically meaningful, but the current unfused PyTorch scatter/analytic-force implementation makes it too expensive.
- The comparison isolates hardware cost from parameter count: vector moments and element density both have 1665 parameters, but vector moments are about 2.2x slower and use about 2.4x more peak allocation.
- Because vector moments are dominated by element-density 32x32, do not run vector 64x64, quadrupole-only, or full atomic moments as the next priority in this implementation.
- The clean current rTECE Pareto front remains pair-32, pair-64, and element-density-32. The next priority shifts from adding descriptor semantics to reducing implementation cost: fusion/export of pair and element-density analytic kernels, or a cheaper element-conditioned scalar formulation.


## Stage 18 Smoke: Element-Density Head-Capacity Pareto Refinement

Stage 17 showed that adding vector moments is dominated by element-conditioned density under the current unfused implementation. Stage 18 therefore did not add another descriptor. It kept the successful Stage-16 semantic projection fixed and varied only the scalar head capacity. This is a parameter-axis refinement around a validated TECE semantic block, not a return to arbitrary architecture search.

4096-config prefix comparison, all with `rtece_element_density`, mixed labels, force weight 30, and analytic-density forces:

| job | hidden | params | atoms/s | DFT F MAE | teacher F MAE | DFT E MAE | interpretation |
|---:|---|---:|---:|---:|---:|---:|---|
| 678774 | 16x16 | 577 | 16019208 | 38.69 | 42.98 | 1412.5 | dominated by pair-64 |
| 678773 | 24x24 | 1057 | 15920358 | 30.19 | 35.83 | 1402.9 | new intermediate rTECE front point |
| 678749 | 32x32 | 1665 | 14696599 | 28.14 | 33.93 | 1409.4 | lower-error element-density front point |

Offset-window robustness for the new 24x24 point:

| job | extxyz index | configs | atoms/s | DFT F MAE | DFT F RMSE | DFT E MAE | interpretation |
|---:|---|---:|---:|---:|---:|---:|---|
| 678773 | `:4096` | 4096 | 15920358 | 30.19 | 111.50 | 1402.9 | prefix benchmark |
| 678777 | `4096:8192` | 4096 | 15990835 | 33.42 | 120.71 | 1486.8 | same throughput class and stable force-error regime |
| 678778 | `8192:12288` | 1808 | 15018695 | 32.85 | 118.97 | 1371.8 | shorter tail window but still stable |

Stage-18 interpretation:

- The current rTECE scalar Pareto front has four useful points: pair-32 for maximum throughput, pair-64, element-density-24, and element-density-32 for the lowest force MAE in this branch.
- Element-density 24x24 is the best new compromise: it recovers most of the 32x32 element-density accuracy gain while regaining about 1.2M atoms/s.
- Element-density 16x16 crosses the projection/capacity limit: it loses the chemistry benefit and is dominated by pair-64.
- The next priority is no longer adding semantic descriptors in Python. The front is now good enough to justify implementation work: fuse/export pair and element-density analytic kernels, or otherwise reduce scatter/MLP overhead while preserving the same descriptors and conservative force path.


## Stage 19 Smoke: Packed Element-Density Force Path

Stage 18 identified implementation cost as the next priority. Stage 19 tested a narrow PyTorch-level optimization for `rtece_element_density`: compute `[rho, rho_z]` in one packed scatter, take one gradient with respect to the packed descriptor tensor, and combine the edge force scale as `(grad_rho + z_j grad_rho_z) dR/dr`. This preserves the same model architecture, checkpoint, descriptors, and conservative force path.

Implementation gate:

- `RTECEScalarModel.forward_element_density_packed_analytic_forces` matches full autograd forces for element-density descriptors.
- `benchmark_rtece_scalar.py` exposes `--force-mode analytic_element_packed`, so old and new force paths can be A/B benchmarked on the same checkpoint.
- Full rTECE test file after the change: 30 passed.

4096-config prefix A/B benchmark:

| checkpoint | force mode | atoms/s | seconds/pass | peak reserved MB | DFT F MAE | interpretation |
|---|---|---:|---:|---:|---:|---|
| element-density 24x24 | analytic_density | 15920358 | 0.015725 | 1716 | 30.193 | Stage-18 front point |
| element-density 24x24 | analytic_element_packed | 15416268 | 0.016240 | 2032 | slower; no accuracy change |
| element-density 32x32 | analytic_density | 14696599 | 0.017035 | 1716 | 28.142 | Stage-16 lower-error point |
| element-density 32x32 | analytic_element_packed | 14325009 | 0.017477 | 2032 | slower; no accuracy change |

Stage-19 interpretation:

- The packed force path is a useful negative implementation result. It proves numerical equivalence but does not improve throughput or memory behavior.
- The bottleneck is not merely the number of Python-level scatter calls; packing creates a wider temporary descriptor tensor and increases reserved memory.
- The current Pareto front remains the Stage-18 front using `analytic_density`. Do not spend more time on PyTorch-level tensor packing.
- The next implementation priority should be a real fused/exported evaluator for pair and element-density descriptors, or a lower-level custom kernel that computes density, element-density, MLP input, and edge force scale without materializing avoidable intermediate tensors.


## Stage 20 Smoke: rTECE Scalar Profiler

Stage 19 rejected PyTorch-level descriptor packing. Stage 20 added a reusable `profile_rtece_scalar.py` profiler and profiled representative front points: pair-32 with `analytic_pair` and element-density-24 with `analytic_density`, each on the same 4096-config prebuilt V100 benchmark window.

Implementation gate:

- `profile_rtece_scalar.py` loads an rTECE checkpoint, builds a prebuilt batched graph, profiles selected force mode passes with `torch.profiler`, exports JSON top ops, and can optionally export a Chrome trace.
- `rtece_scalar_profile.sbatch` runs the profiler on the cluster.
- Full rTECE test file after the change: 31 passed.

Profiler evidence, inclusive device time over 5 profiled passes:

| point | leading ops | interpretation |
|---|---|---|
| pair-32 analytic pair | `aten::mul` 16.75 ms, `aten::linear/addmm` 7.49 ms, `index_add_` 6.23 ms, `div` 5.96 ms, `index` 4.95 ms | radial/force elementwise and scatter/gather are at least as important as MLP work |
| element-density-24 analytic density | `aten::mul` 24.01 ms, `index_add_` 9.32 ms, `index` 7.51 ms, `div` 5.96 ms, `sum` 5.18 ms | extra chemistry descriptor mainly adds elementwise/scatter/gather cost |

Stage-20 interpretation:

- The fusion target is now concrete: radial basis plus cutoff, density/element-density accumulation, descriptor-gradient edge scale, and force accumulation should be fused or exported together.
- MLP-only fusion is not enough, because GEMM is visible but not the dominant exclusive story; the scalar physics path launches many small elementwise, gather, and scatter kernels.
- This matches the TECE/TACE document logic: the useful model degradation has already removed persistent equivariant state; the remaining bottleneck is execution of the scalarized renormalized operator, so the next work should be a fused low-level evaluator for the current Pareto front rather than adding more semantic descriptors.


## Stage 21 Smoke: Triton Pair Force Evaluator

Stage 20 made the next priority explicit: after the TECE/TACE projection removes persistent equivariant state, the remaining high-throughput bottleneck is the scalarized radial/scatter/force evaluator. Stage 21 therefore did not change the trained model architecture or checkpoint. It added an optional Triton force-accumulation path for the pure pair rTECE endpoint, fusing radial derivatives and edge force accumulation while keeping the same density descriptor, scalar MLP head, conservative chain rule, and labels.

Implementation gate:

- `RTECEScalarModel.forward_pair_triton_force_analytic_forces` supports only pure `rtece_pair` descriptors and rejects CPU graphs before importing Triton.
- `benchmark_rtece_scalar.py` and `profile_rtece_scalar.py` expose `--force-mode analytic_pair_triton_force`.
- The Triton path keeps descriptor/MLP gradients in PyTorch and replaces only the edge force kernel, making it an evaluator optimization rather than an architecture change.
- Full rTECE scalar test file after the change: 32 passed.

4096-config prefix A/B benchmark on the same pair-32 checkpoint:

| checkpoint | force mode | params | atoms/s | seconds/pass | peak alloc MB | peak reserved MB | DFT F MAE | DFT F RMSE | interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| pair-32 | analytic_pair | 1409 | 18622904 | 0.013443 | 1405.4 | 1716 | 39.42031 | 114.50856 | previous maximum-throughput front point |
| pair-32 | analytic_pair_triton_force | 1409 | 31097432 | 0.008051 | 675.2 | 1084 | 39.42031 | 114.50856 | same error, 1.67x throughput, lower memory |

Profiler evidence, inclusive device time over 5 profiled passes:

| point | leading ops | interpretation |
|---|---|---|
| pair-32 analytic pair | `aten::mul` 16.75 ms, `aten::linear/addmm` 7.49 ms, `index_add_` 6.23 ms, `div` 5.96 ms | PyTorch radial/force elementwise and scatter dominate the old high-throughput endpoint |
| pair-32 Triton force | `aten::linear/addmm` 7.50 ms, `_pair_force_kernel` 4.28 ms, `aten::mul` 4.95 ms, `index_add_` 3.33 ms | fused force accumulation removes much of the scalar physics overhead; MLP/density construction is now a leading bottleneck |

Stage-21 interpretation:

- This is the first positive evaluator-level result. It validates the Stage-20 priority: after the TECE/TACE renormalized projection to scalar pair density, low-level force evaluation can move the same model into the NEP/DPA-like high-throughput regime without changing the learned parameters.
- The current Pareto front changes: pair-32 Triton becomes the maximum-throughput point at about 31.1M atoms/s with the same 39.42 meV/A force MAE; element-density-24 and element-density-32 remain the lower-error scalar rTECE front points.
- The remaining gap is now architectural and evaluator-coupled. Pair-32 is fast but less accurate; element-density is more accurate but still uses unfused PyTorch scalar density and force paths.
- Next priority: extend the same fused-evaluator idea to the TECE-positive element-density descriptor, but do it as a clean scalar renormalized operator: compute `rho` and `rho_z`, descriptor gradients, radial derivative, and force accumulation without materializing avoidable edge temporaries. In parallel, test whether smaller readout heads on the now-fast pair endpoint create a useful ultra-fast sub-front, because profiler shows GEMM/MLP is now the largest remaining pair cost.


## Stage 22 Smoke: Triton Element-Density Force Evaluator

Stage 21 proved that a fused scalar force evaluator can move the pure pair rTECE endpoint into a NEP/DPA-like throughput regime without changing model parameters. Stage 22 applied the same evaluator discipline to the TECE-positive element-density descriptor, which had been the best lower-error scalar rTECE front point. The model architecture, checkpoints, labels, and conservative chain rule are unchanged; only the edge force-scale evaluation and force accumulation are moved into Triton.

Implementation gate:

- `RTECEScalarModel.forward_element_density_triton_force_analytic_forces` supports only density plus element-density descriptors and rejects CPU graphs before importing Triton.
- `benchmark_rtece_scalar.py` and `profile_rtece_scalar.py` expose `--force-mode analytic_element_triton_force`.
- The Triton kernel evaluates `(grad_rho + z_j grad_rho_z) dR/dr` and atomic force scatter directly from node-level scaled atomic numbers.
- Full rTECE scalar test file after the change: 33 passed.

4096-config prefix A/B benchmark on the same element-density checkpoints:

| checkpoint | force mode | params | atoms/s | seconds/pass | peak alloc MB | peak reserved MB | DFT F MAE | DFT F RMSE | interpretation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| element-density 24x24 | analytic_density | 1057 | 15920358 | 0.015725 | 1404.9 | 1716 | 30.19297 | 111.49715 | previous intermediate front point |
| element-density 24x24 | analytic_element_triton_force | 1057 | 30655057 | 0.008167 | 674.7 | 1084 | 30.19297 | 111.49715 | same error, 1.93x throughput |
| element-density 32x32 | analytic_density | 1665 | 14696599 | 0.017035 | 1404.9 | 1716 | 28.14186 | 109.92687 | previous lower-error front point |
| element-density 32x32 | analytic_element_triton_force | 1665 | 26033459 | 0.009617 | 674.7 | 1084 | 28.14186 | 109.92687 | same error, 1.77x throughput |

Profiler evidence for element-density 24x24, inclusive device time over 5 profiled passes:

| point | leading ops | interpretation |
|---|---|---|
| element-density 24x24 analytic density | `aten::mul` 24.01 ms, `index_add_` 9.32 ms, `index` 7.51 ms, `div` 5.96 ms | PyTorch scalar descriptor and force path launch many elementwise/gather/scatter kernels |
| element-density 24x24 Triton force | `aten::mul` 7.10 ms, `index_add_` 6.46 ms, `_element_density_force_kernel` 4.55 ms, `sub/pow/index/div` 2.3-3.1 ms | force accumulation is largely fused; descriptor density construction scatter is now the remaining physics bottleneck |

Stage-22 interpretation:

- This is a second positive evaluator-level result and the strongest evidence so far that the TECE/TACE degradation route is practical: the same scalar renormalized descriptor hierarchy now spans about 31.1M atoms/s at 39.42 meV/A and 26.0M atoms/s at 28.14 meV/A on the 4096-config V100 window.
- The Pareto front is now mostly evaluator-shifted: pair-32 Triton is the maximum-throughput endpoint, element-density 24x24 Triton is the near-maximum-throughput lower-error point, and element-density 32x32 Triton is the current lower-error scalar point. Pair-64 and all unfused versions are dominated on this benchmark window.
- The remaining bottleneck is no longer the old analytic force scatter alone. For element-density, PyTorch still materializes radial features, neighbor-weighted radial features, and two density scatters before the MLP gradient. A next clean implementation step is a fused descriptor-construction plus force evaluator, but it must preserve conservative forces and exact descriptor semantics.
- The next architecture question should be narrower: after fusion, do smaller readout heads or fewer radial channels give a useful ultra-fast sub-front without losing the element-density chemistry benefit? This is now a systematic Pareto refinement within the TECE scalar projection, not random hyperparameter search.


## Stage 23 Smoke: Radial/Head Capacity Under the Triton Element-Density Evaluator

Stage 22 moved the TECE-positive element-density descriptor into the high-throughput regime. Stage 23 tested whether the next Pareto axis should be a controlled scalar-projection capacity reduction rather than adding new moment descriptors. The experiment fixed the descriptor semantics to `rtece_element_density`, fixed the conservative Triton force evaluator, and swept only radial resolution and scalar readout width.

Implementation gate:

- `train_rtece_scalar.py` now exposes `--num-radial`, records `num_radial` in `train_summary.json`, and saves it in the checkpoint config.
- `rtece_scalar_matrix.sbatch` forwards `NUM_RADIAL`, making radial-channel sweeps reproducible from Slurm logs.
- `benchmark_rtece_scalar.py` now records `hidden_channels` and `num_radial`; `summarize_tece_distill.py` preserves them in rows.
- Full rTECE scalar test file after the change: 35 passed.

4096-config prefix sweep, all `rtece_element_density`, mixed labels, force weight 30, and `analytic_element_triton_force`:

| num radial | hidden | params | atoms/s | peak alloc MB | DFT F MAE | teacher F MAE | interpretation |
|---:|---|---:|---:|---:|---:|---:|---|
| 4 | 16x16 | 449 | 42305218 | 436.4 | 43.91 | 47.77 | new maximum-throughput endpoint; large accuracy penalty |
| 4 | 24x24 | 865 | 40459620 | 436.4 | 48.65 | 52.13 | dominated by radial4/16x16 |
| 6 | 16x16 | 513 | 35899727 | 554.9 | 45.42 | 49.18 | dominated by radial4/16x16 |
| 6 | 24x24 | 961 | 34333375 | 554.9 | 42.87 | 46.67 | high-throughput intermediate point, still much worse than pair-32 |
| 8 | 24x24 | 1057 | 30655057 | 674.7 | 30.19 | 35.83 | Stage-22 near-maximum-throughput lower-error point |
| 8 | 32x32 | 1665 | 26033459 | 674.7 | 28.14 | 33.93 | current lower-error scalar point |

Offset-window robustness for the new radial4/16x16 maximum-throughput endpoint:

| extxyz index | configs | atoms/s | DFT F MAE | DFT F RMSE | DFT E MAE | interpretation |
|---|---:|---:|---:|---:|---:|---|
| `:4096` | 4096 | 42954558 | 43.91 | 114.91 | 1430.0 | prefix rebenchmark with architecture metadata |
| `4096:8192` | 4096 | 43229041 | 43.93 | 122.23 | 1516.6 | stable force MAE, higher RMSE on harder window |
| `8192:12288` | 1808 | 38643787 | 42.99 | 120.32 | 1349.6 | shorter tail window, same force-error regime |

Stage-23 interpretation:

- Reducing radial channels is a real TECE scalar-projection capacity knob, not arbitrary parameter tuning. It coarsens the learned radial density basis while preserving element-conditioned scalar density and conservative forces.
- `num_radial=4, hidden=16x16` creates a new ultra-fast endpoint around 43M atoms/s with stable force MAE around 43-44 meV/A. This is faster than the pair-32 Triton endpoint but less accurate, so it belongs only to the extreme-throughput end of the Pareto curve.
- The sweep shows a non-monotonic capacity effect: radial4/24x24 is worse than radial4/16x16, and radial6/16x16 is dominated. Wider heads do not automatically recover accuracy when radial resolution is too coarse; the descriptor bottleneck is semantic/radial, not just MLP capacity.
- The current scalar rTECE Pareto front is now: radial4 element-density 16x16 for maximum throughput, radial6 element-density 24x24 as a high-throughput intermediate, pair-32 Triton, element-density 24x24, and element-density 32x32. The next clean step is either (1) radial5/hidden16-24 to fill the large error gap between radial4 and radial8, or (2) fused descriptor construction to reduce the remaining `index_add_` cost without further semantic loss.


## Stage 24 Smoke: Radial5 Gap-Fill Under the Triton Element-Density Evaluator

Stage 23 established two separated regimes: radial4 gives an extreme-throughput endpoint with about 44 meV/A force MAE, while radial8 recovers much better accuracy near 30 meV/A at lower throughput. Stage 24 tested the missing middle point, `num_radial=5`, under the same TECE-positive element-density descriptor and `analytic_element_triton_force` evaluator. This keeps the degradation axis clean: only radial density resolution and scalar head width vary.

4096-config prefix sweep, all `rtece_element_density`, mixed labels, force weight 30, and `analytic_element_triton_force`:

| num radial | hidden | params | atoms/s | peak alloc MB | DFT F MAE | teacher F MAE | interpretation |
|---:|---|---:|---:|---:|---:|---:|---|
| 4 | 16x16 | 449 | 42954558 | 436.4 | 43.91 | 47.77 | Stage-23 maximum-throughput endpoint |
| 5 | 16x16 | 481 | 38825023 | 495.4 | 39.02 | 43.07 | new useful middle point; faster and slightly more accurate than pair32 |
| 5 | 20x20 | 681 | 38384295 | 495.4 | 49.64 | 53.20 | dominated; widening head did not help |
| 5 | 24x24 | 913 | 37893470 | 495.4 | 46.57 | 50.23 | dominated by radial5/16x16 |
| 6 | 24x24 | 961 | 34333375 | 554.9 | 42.87 | 46.67 | dominated by radial5/16x16 on prefix window |
| 8 | 24x24 | 1057 | 30655057 | 674.7 | 30.19 | 35.83 | lower-error element-density point |

Offset-window robustness for radial5/16x16:

| extxyz index | configs | atoms/s | DFT F MAE | DFT F RMSE | DFT E MAE | interpretation |
|---|---:|---:|---:|---:|---:|---|
| `:4096` | 4096 | 38423189 | 39.02 | 114.24 | 1449.6 | prefix rebenchmark |
| `4096:8192` | 4096 | 39634440 | 41.61 | 123.82 | 1533.4 | stable error band, harder window |
| `8192:12288` | 1808 | 33595106 | 40.55 | 122.25 | 1358.6 | shorter tail window, same force-error regime |

Stage-24 interpretation:

- Radial5/16x16 is the first clean gap-fill point between the radial4 ultra-fast endpoint and the radial8 lower-error element-density points. It keeps element-conditioned radial density, conservative forces, and the Triton evaluator, but uses only 481 parameters and 5 radial channels.
- This point weakly dominates pair32 on the prefix benchmark: higher throughput, fewer parameters, lower memory, and slightly lower DFT force MAE. It also has a clear TECE explanation: adding one radial channel beyond radial4 recovers enough radial resolution to reduce force error without returning to the full radial8 cost.
- Wider heads at radial5 are negative results. The `20x20` and `24x24` heads are slower and much less accurate, reinforcing the Stage-23 conclusion that readout width does not compensate for coarse/noisy radial projection under this small-data distillation setup.
- The current scalar rTECE Pareto front is now cleaner: radial4/16x16 for maximum throughput, radial5/16x16 as the useful fast middle point near pair-level accuracy, radial8/24x24 for much lower force error, and radial8/32x32 for the current lowest scalar force error. The next priority should shift from head-width sweeps to either fused descriptor construction or a radial5 training robustness repeat with a different seed/longer training, because the non-monotonic head behavior suggests training variance or optimization stability matters.


## Stage 25 Smoke: Radial5/16x16 Seed Robustness

Stage 24 found radial5/16x16 as a promising middle Pareto point. Stage 25 tested whether that conclusion is stable to initialization. The experiment added explicit seed plumbing and repeated the same architecture with `SEED=1` and `SEED=2`, keeping `rtece_element_density`, `num_radial=5`, `hidden=16x16`, train=512, valid=256, force weight 30, and `analytic_element_triton_force` fixed.

Implementation gate:

- `train_rtece_scalar.py` now exposes `--seed`, sets NumPy/Torch/CUDA seeds before model initialization, and records `seed` in `train_summary.json`.
- `rtece_scalar_matrix.sbatch` forwards and logs `SEED`.
- Full rTECE scalar test file after the change: 36 passed.

4096-config prefix seed comparison:

| seed | best valid loss | atoms/s | DFT F MAE | teacher F MAE | DFT E MAE | interpretation |
|---:|---:|---:|---:|---:|---:|---|
| pre-seed plumbing | 3.4773 | 38825023 | 39.02 | 43.07 | 1449.6 | optimistic Stage-24 point |
| 1 | 3.4156 | 38876690 | 42.48 | 46.52 | 1430.1 | same throughput, worse force MAE |
| 2 | 3.4345 | 38774313 | 45.78 | 49.21 | 1429.4 | same throughput, much worse force MAE |

Stage-25 interpretation:

- Throughput is robust because it is determined by architecture and evaluator: all radial5/16x16 seeds stay near 38.8M atoms/s with 481 parameters and the same memory class.
- Accuracy is not robust under the current 512-config, 1000-step distillation setup. The seed0/pre-plumbing point weakly dominates pair32 on the prefix window, but seed1 and seed2 do not. Therefore radial5/16x16 should be treated as a promising candidate band, not a proven stable Pareto point.
- The best-valid loss does not rank the DFT force MAE reliably across seeds: seed1/2 have better validation loss than seed0 but worse DFT force MAE. This implies the small train/valid proxy is not sufficient for selecting a robust low-precision endpoint.
- The next priority should not be another head-width sweep. It should either improve selection robustness, such as longer training or a larger validation/selection window for radial5/16x16, or shift to fused descriptor construction where the architecture is already reliable. For the Pareto curve, report radial5 as a seed-sensitive band until repeated/longer runs narrow the error range.


## Stage 26: Radial5/16x16 Longer-Training Robustness

Stage 26 tested the first Stage-25 follow-up directly: keep the candidate architecture fixed but increase training duration and the validation-selection window. The experiment used `rtece_element_density`, `num_radial=5`, `hidden=16x16`, 481 parameters, train=512, valid=512, force weight 30, energy weight 1, `analytic_element_triton_force`, and `MAX_STEPS=3000` for seeds 0, 1, and 2.

4096-config prefix comparison:

| seed | best step | best valid loss | atoms/s | DFT F MAE | DFT F RMSE | teacher F MAE | DFT E MAE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2100 | 4.4110 | 38606435 | 56.03 | 124.13 | 58.99 | 1409.5 |
| 1 | 600 | 4.4038 | 38660247 | 44.59 | 115.36 | 48.41 | 1403.4 |
| 2 | 900 | 4.4078 | 38870724 | 38.28 | 112.60 | 42.24 | 1428.1 |

Stage-26 interpretation:

- Longer training plus a 512-config valid selector does not stabilize the radial5/16x16 force-error band. Throughput remains robust at about 38.6-38.9M atoms/s, but DFT force MAE spans 38.28-56.03 meV/A.
- The best-valid losses are almost tied while DFT force MAE differs by nearly 18 meV/A. This confirms the Stage-25 warning: the current mixed-label train/valid proxy is not a reliable checkpoint selector for the DFT-force Pareto objective.
- Radial5/16x16 remains TECE-interpretable as a minimal radial-resolution recovery beyond radial4, but it should not be presented as a single deterministic Pareto-front point. It is a seed-sensitive candidate band unless the protocol explicitly uses best-of-N seed selection and reports that cost.
- The current clean Pareto statement is therefore conservative: radial4/16x16 is the maximum-throughput endpoint; radial5/16x16 is a stochastic candidate band around 38.6-38.9M atoms/s; pair32 remains a more stable comparable-error point; radial8/24x24 and radial8/32x32 remain the lower-error scalar TECE points.
- The next priority should move away from more radial/head sweeps. Either improve the selection protocol by validating/checkpointing directly on the DFT-force target over a larger slice, or implement fused descriptor construction for the stable element-density family so the lower-error radial8 points can move toward the radial4/radial5 throughput regime without adding semantic complexity.


## Stage 27: Triton Fused Element-Density Descriptor

Stage 27 tested the second Stage-26 priority: keep the stable scalar TECE element-density architecture fixed, but reduce edge/state lifetime by moving descriptor construction from PyTorch radial tensors and `index_add_` into a fused Triton edge kernel. The new `analytic_element_triton_descriptor_force` path builds packed `[rho, rho_z]` descriptors with Triton, then reuses the existing Triton conservative force kernel.

Implementation gate:

- Added a CPU/PyTorch equivalence anchor, `packed_element_density_descriptors`, for packed `[rho, rho_z]` descriptors.
- Added `element_density_descriptors_triton` and `RTECEScalarModel.forward_element_density_triton_descriptor_force_analytic_forces`.
- Exposed `analytic_element_triton_descriptor_force` in benchmark and profiler force-mode choices.
- Full rTECE scalar test file after the change: 38 passed.

4096-config prefix benchmark on the same radial8/24x24 checkpoint (`rtece-scalar-678773`, 1057 parameters):

| force mode | atoms/s | seconds/pass | peak alloc MB | DFT F MAE | DFT F RMSE | DFT E MAE |
|---|---:|---:|---:|---:|---:|---:|
| `analytic_element_triton_force` | 29413444 | 0.008512 | 674.7 | 30.19297 | 111.49715 | 1402.9 |
| `analytic_element_triton_descriptor_force` | 50975964 | 0.004911 | 212.1 | 30.19297 | 111.49715 | 1402.9 |

Stage-27 interpretation:

- This is a clean positive result for the TECE/TACE renormalization route because it changes hardware realization, not model semantics: same checkpoint, same scalar element-density descriptor, same conservative analytic force, same force error within float noise.
- The bottleneck diagnosis from the TECE design document is supported. Removing long-lived edge radial tensors and PyTorch `index_add_` descriptor construction raises radial8/24x24 throughput from about 29.4M to 51.0M atoms/s and cuts peak allocation from 674.7MB to 212.1MB on the 4096-config prefix.
- The Pareto front must be recomputed. The lower-error radial8/24x24 point now exceeds the previous radial4 maximum-throughput endpoint measured with the older evaluator, so architecture simplification and hardware renormalization are not independent axes.
- The next priority is not more training. It is to rebenchmark radial4/16x16, radial5/16x16, radial8/24x24, and radial8/32x32 under `analytic_element_triton_descriptor_force`, then report a new Pareto curve with the same DFT/teacher windows.


## Stage 28: Fused Descriptor rTECE Pareto Rebenchmark

Stage 28 rebenchmarked the current element-density scalar rTECE front with the Stage-27 fused evaluator. The checkpoints were unchanged; the benchmark fixed `analytic_element_triton_descriptor_force`, DFT/teacher valid prefix `:4096`, 250355 atoms, prebuilt batched graph, float32, and one V100.

DFT/teacher prefix comparison:

| point | num radial | hidden | params | atoms/s | peak alloc MB | DFT F MAE | DFT F RMSE | teacher F MAE | status |
|---|---:|---|---:|---:|---:|---:|---:|---:|---|
| radial4h16 | 4 | 16x16 | 449 | 72644836 | 176.6 | 43.91 | 114.91 | 47.77 | max-throughput endpoint |
| radial5h16 seed2-long | 5 | 16x16 | 481 | 64515626 | 178.0 | 38.28 | 112.60 | 42.24 | seed-sensitive middle point |
| radial5h16 stage24 | 5 | 16x16 | 481 | 64463762 | 178.0 | 39.02 | 114.24 | 43.07 | dominated by seed2-long under best-seed selection |
| radial8h24 | 8 | 24x24 | 1057 | 50929958 | 212.1 | 30.19 | 111.50 | 35.83 | lower-error front point |
| radial8h32 | 8 | 32x32 | 1665 | 40750024 | 242.7 | 28.14 | 109.93 | 33.93 | lowest-error scalar point |

Stage-28 interpretation:

- The fused evaluator produces a new Pareto curve rather than a minor implementation optimization. The maximum-throughput scalar TECE endpoint is now radial4/16x16 at 72.6M atoms/s, while radial8/24x24 reaches 50.9M atoms/s at 30.19 meV/A.
- This supports the document thesis that model degradation and hardware renormalization are coupled. A less degraded radial8 descriptor can remain high-throughput once descriptor construction is fused and edge-state lifetime is shortened.
- The current clean front is radial4/16x16, radial5/16x16 as a seed-sensitive band, radial8/24x24, and radial8/32x32. The radial5 seed2-long point dominates the Stage-24 radial5 checkpoint, but Stage 25-26 require reporting radial5 as stochastic unless the protocol explicitly uses best-of-N seeds.
- The next training-method priority is direct DFT-force validation selection or a reported best-of-N seed protocol for radial5. The next implementation priority is to make the fused descriptor path the default for eligible element-density benchmarks, then profile graph construction/neighbor-list overhead because the model force pass is now extremely fast.


## Stage 29: Default Auto Evaluator And Graph-Construction Bottleneck

Stage 29 integrated the fused element-density evaluator into the benchmark/profiler default path. The new `auto` force-mode selector resolves eligible CUDA/float32 scalar element-density models to `analytic_element_triton_descriptor_force`; other variants or devices remain on autograd unless a force mode is explicitly requested. The full rTECE scalar test file after the selector change: 39 passed.

The same radial8/24x24 checkpoint was then benchmarked on the DFT valid `:1024` prefix, 59193 atoms, one V100. Both rows requested `--force-mode auto` and resolved to `analytic_element_triton_descriptor_force`.

| mode | prebuilt graph | includes graph construction | atoms/s | configs/s | seconds/pass | DFT F MAE |
|---|---:|---:|---:|---:|---:|---:|
| auto fused evaluator | yes | no | 34203147 | 591692 | 0.001731 | 33.17 |
| auto fused evaluator | no | yes | 8567 | 148 | 6.909510 | 33.17 |

Profiler top CUDA events for the prebuilt graph path were `_element_density_descriptor_kernel` at 2747.9 us over five passes, `_element_density_force_kernel` at 1084.9 us, then the MLP/autograd head around 436.8 us for `aten::linear`/`addmm` and 382.7 us for `AddmmBackward0`/`aten::mm`.

Stage-29 interpretation:

- The fused evaluator is now the default implementation for the eligible element-density scalar TECE front, so future Pareto measurements no longer require hand-passing the new force mode.
- Once the batched graph exists, the model pass is extremely fast and still TECE-clean: same descriptors, same scalar head, same conservative force, but hardware-realized with short edge-state lifetime.
- End-to-end throughput through the current ASE-style path is dominated by graph construction and per-configuration execution. Including graph construction drops the same checkpoint from 34.2M atoms/s to 8.6k atoms/s on the 1024-config window.
- The next clean priority is therefore not another radial/head sweep. It is batched graph construction, graph/neighbor cache reuse, or direct MD-runtime integration that keeps neighbor lists and edge buffers alive across steps. This is the next system-level renormalization axis required before claiming NEP/DPA-style throughput in an end-to-end setting.


## Stage 30: Graph Runtime Cost Split

Stage 30 split the Stage-29 end-to-end bottleneck into three runtime modes for the same radial8/24x24 checkpoint. All rows use DFT valid `:1024`, 59193 atoms, float32, one V100, and `--force-mode auto`, which resolves to `analytic_element_triton_descriptor_force`.

| runtime mode | prebuilt graph | graph construction timed | batched model pass | atoms/s | configs/s | seconds/pass | DFT F MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| prebuilt batched graph | yes | no | yes | 36564104 | 632535 | 0.001619 | 33.17 |
| rebuild graphs, then batch | no | yes | yes | 9879 | 171 | 5.992034 | 33.17 |
| rebuild graph per config | no | yes | no | 8599 | 149 | 6.883463 | 33.17 |

Stage-30 interpretation:

- Batching the model pass after rebuilding all graphs improves the current ASE-style end-to-end path by only about 15% over rebuilding and evaluating per configuration. That is far too small to explain the gap to the prebuilt batched graph throughput.
- The dominant runtime bottleneck is therefore ASE neighbor-list and graph construction, not the fused rTECE force pass and not primarily per-config Python model dispatch.
- This sharpens the TECE system-renormalization route: the local graph/edge buffer must become a persistent coarse-grained state across MD steps, with skin/cache updates, rather than a transient object reconstructed from scratch each step.
- The next implementation target should be a graph-cache benchmark that builds a graph once, reuses it across repeated force passes, and models amortized rebuild intervals. The longer-term target is an MD-runtime neighbor-list provider feeding the fused descriptor/force kernels directly.


## Stage 31: Cached Graph Replay And Edge-State Lifetime

Stage 31 implemented the Stage-30 graph-cache benchmark. The test keeps the same radial8/24x24 `rtece_element_density` checkpoint, DFT valid `:1024` prefix, 59193 atoms, float32, one V100, and `--force-mode auto`, which resolves to `analytic_element_triton_descriptor_force`. The new replay mode builds one graph template, reuses `z`, `edge_index`, and `batch`, refreshes positions, and runs one batched fused conservative force pass.

| runtime mode | prebuilt graph | graph construction timed | replay cached graph | atoms/s | configs/s | seconds/pass | peak alloc MB | DFT F MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prebuilt batched graph | yes | no | no | 36790610 | 636453 | 0.001609 | 63.2 | 33.17 |
| cached graph replay | no | no | yes | 36793984 | 636512 | 0.001609 | 64.6 | 33.17 |
| rebuild graphs, then batch | no | yes | no | 9885 | 171 | 5.987913 | 78.2 | 33.17 |

Amortized throughput under the measured cost model, using `seconds_per_step = cached_model_pass + graph_rebuild / K = 0.001608768 + 5.987913 / K`:

| graph rebuild interval K | amortized atoms/s | interpretation |
|---:|---:|---|
| 1 | 9883 | current rebuild-every-step ASE-style path |
| 10 | 98589 | still graph-bound |
| 100 | 962677 | below target throughput class |
| 1000 | 7791955 | near target, still below 1e7 atoms-step/s |
| 2000 | 12860426 | crosses 1e7 under this model |
| 5000 | 21092515 | graph cost no longer dominant |
| 10000 | 26813771 | approaches cached model-pass ceiling |

Stage-31 interpretation against TECE/TACE:

- This confirms that the high-throughput rTECE scalar force pass is not the current limiting factor once descriptor construction and conservative force accumulation are fused. Cached graph replay matches the prebuilt graph path within measurement noise while preserving identical force error.
- The missing system-level renormalization axis is edge-state lifetime. In TECE/TACE terms, the local neighbor graph and edge buffers must be treated as a persistent coarse-grained state with controlled refresh, not as transient Python/ASE objects reconstructed every force call.
- This is consistent with the TECE paper's emphasis on edge many-body construction and the TACE paper's controlled ACE hierarchy: the projection route cannot stop at deleting channels/radial modes. It must also downfold the runtime representation so retained edge relations are short-lived in kernels but long-lived as reusable topology state.
- The measured end-to-end target is now quantitative. With the current ASE rebuild cost, a rebuild interval around 2000 force steps is needed to exceed 1e7 atoms-step/s. A production MD neighbor provider could lower the rebuild term, so the next experiment should measure or prototype that provider rather than run another radial/head sweep.

Next priority:

1. Implement a minimal MD-runtime neighbor-list interface or synthetic provider that updates positions on device and keeps edge buffers persistent across force calls.
2. Measure graph rebuild/update amortization on trajectory-like displacement rather than static extxyz replay.
3. Keep the current scalar Pareto front fixed for now: radial4/16x16, radial5 band, radial8/24x24, radial8/32x32 under the fused evaluator. Reopen architecture sweeps only after the graph-lifetime bottleneck is reduced or quantified in an MD-style loop.


## Stage 32: Synthetic Trajectory Replay Benchmark

Stage 32 turned the Stage-31 amortized cached-graph estimate into a measured MD-style trajectory benchmark. The code adds `--trajectory-replay-steps`, `--trajectory-rebuild-interval`, and `--trajectory-displacement-std` to the rTECE scalar benchmark. This mode keeps graph topology and edge buffers persistent, updates positions with a deterministic small displacement, and optionally rebuilds the batched ASE graph every `K` force steps. In this mode, `atoms_per_second` is atom-step/s.

The benchmark keeps the same radial8/24x24 `rtece_element_density` checkpoint, DFT valid `:1024` prefix, 59193 atoms, float32, one V100, and `--force-mode auto`, resolving to `analytic_element_triton_descriptor_force`.

| mode | force steps/pass | graph rebuild interval K | atom-step/s | seconds/pass | peak alloc MB | DFT F MAE at first frame |
|---|---:|---:|---:|---:|---:|---:|
| cached trajectory | 2000 | 0 | 39909204 | 2.966383 | 69.1 | 33.17 |
| trajectory + rebuild | 2001 | 2000 | 13329198 | 8.886145 | 105.0 | 33.17 |
| trajectory + rebuild | 1001 | 1000 | 7906961 | 7.493674 | 105.0 | 33.17 |
| trajectory + rebuild | 1001 | 500 | 4444052 | 13.332921 | 125.2 | 33.17 |
| trajectory + rebuild | 501 | 100 | 977274 | 30.345312 | 125.2 | 33.17 |

Stage-32 interpretation against TECE/TACE:

- The measured trajectory rows validate the Stage-31 cost model. The cached model force step is about 1.48 ms for 59193 atoms, while one full ASE graph rebuild for the 1024-config window costs about 5.92-6.01 s.
- The 1e7 atom-step/s target is now quantitative rather than aspirational: K=1000 is insufficient at 7.91M atom-step/s, while K=2000 crosses the target at 13.33M atom-step/s under this synthetic displacement benchmark.
- This does not mean a production MD loop should blindly rebuild every 2000 steps. It means the next layer must expose a displacement/skin-validity criterion and a faster neighbor update provider so edge-state lifetime becomes controlled, not guessed.
- In TECE/TACE compiler terms, the current high-throughput branch has three distinct axes: semantic projection into scalar element-density descriptors, hardware downfolding into fused descriptor/force kernels, and system-level renormalization of graph/edge-buffer lifetime. The third axis is now the limiting one for end-to-end throughput.

Next priority:

1. Add a displacement-aware graph-cache validity probe: track max displacement since rebuild, skin margin, and estimated rebuild necessity for trajectory replay.
2. Prototype a faster neighbor-provider boundary that can feed persistent `edge_index`/edge buffers without going through full ASE reconstruction every rebuild.
3. Defer more scalar architecture sweeps until graph lifetime/update cost is either reduced or built into the Pareto table as a first-class deployment parameter.


## Stage 33: Skin-Aware Graph Cache Validity

Stage 33 added a displacement/skin validity probe to the trajectory replay benchmark. This converts the fixed `K` from Stage 32 into a physically interpretable edge-state lifetime rule: rebuild when `max_displacement_since_rebuild > 0.5 * skin_margin`. The benchmark records skin margin, half-skin threshold, max displacement, rebuild count, rebuild steps, and rebuild causes.

Verification and implementation gate:

- Added `graph_cache_displacement_probe(reference_positions, positions, skin_margin)`.
- Added `--trajectory-skin-margin` to `benchmark_rtece_scalar.py`.
- `test/test_rtece_scalar.py`: 42 passed after the change.

GPU probe setup: radial8/24x24 `rtece_element_density`, DFT valid `:256`, 12770 atoms, float32, one V100, `--force-mode auto`, `trajectory_displacement_std=0.001`. This is a small-window validity probe; absolute atom-step/s should not be mixed directly with the 1024-config Stage-32 throughput rows because the smaller batch underutilizes the GPU.

| skin margin A | half-skin threshold A | force steps | rebuild count | rebuild steps | atom-step/s | DFT F MAE |
|---:|---:|---:|---:|---|---:|---:|
| 0.000 | n/a | 200 | 0 | [] | 14880257 | 34.14 |
| 0.008 | 0.004 | 200 | 0 | [] | 14623702 | 34.14 |
| 0.004 | 0.002 | 100 | 0 | [] | 14750307 | 34.14 |
| 0.002 | 0.001 | 20 | 10 | [1,3,5,7,9,11,13,15,17,19] | 15839 | 34.14 |

Stage-33 interpretation against TECE/TACE:

- Edge-state lifetime is now a controlled deployment parameter, not a guessed rebuild interval. This directly supports the TECE/TACE route where retained edge relations become persistent coarse-grained topology state while descriptor/force work is short-lived and fused in kernels.
- The skin criterion sharply separates valid cache reuse from graph-bound execution. In this synthetic trajectory, the observed no-rebuild max displacement is about 0.00173 A, so skin margins 0.004 A and 0.008 A preserve cached throughput; skin margin 0.002 A rebuilds every two steps and collapses throughput.
- Stage 32 showed that K around 2000 is needed to cross 1e7 atom-step/s on the 1024-config window. Stage 33 clarifies the missing physical condition: such a lifetime is only legitimate if displacement remains below half the selected skin margin, or if a faster provider can update neighbor topology cheaply.

Next priority:

1. Prototype a neighbor-provider boundary with explicit `valid_until_rebuild`/`needs_rebuild` state, so MD loops can query cache validity instead of hard-coding K.
2. Measure the cost of graph-validity checks separately from full ASE rebuilds. The validity check should be device-side and O(N), otherwise it can become another hidden bottleneck.
3. Only after this provider layer exists should we revisit scalar architecture variants, because the current limiting axis is graph lifetime/update cost rather than descriptor semantics.


## Stage 34: Graph-Cache Provider Boundary And Validity Timing

Stage 34 added a reusable trajectory graph-cache provider boundary and separated O(N) cache-validity timing from fused model timing. This follows Stage 33 directly: the MD loop should query `needs_rebuild` from state, not hard-code a rebuild interval.

Verification and implementation gate:

- Added `TrajectoryGraphCacheProvider(reference_positions, skin_margin)` with `check(positions)` and `mark_rebuilt(positions, step, cause)`.
- Added `--trajectory-validity-only` to time position generation plus provider validity checks without model evaluation or graph rebuild.
- Validity-only JSON rows now set `prediction_errors_available=false` and leave MAE/RMSE fields as `null`, so they cannot be mistaken for model Pareto rows.
- `test/test_rtece_scalar.py`: 45 passed after the change.

GPU setup: radial8/24x24 `rtece_element_density`, DFT valid `:1024`, 59193 atoms, float32, one V100, `--force-mode auto`, `trajectory_displacement_std=0.001`, `trajectory_skin_margin=0.004`, 2000 force steps/pass.

| mode | validity only | force steps | rebuild count | max displacement A | atom-step/s | seconds/pass | ms/step | peak alloc MB | DFT F MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| provider validity check | yes | 2000 | 0 | 0.001733 | 356832007 | 0.331770 | 0.166 | 24.5 | n/a |
| cached fused model + validity | no | 2000 | 0 | 0.001733 | 38889577 | 3.044158 | 1.522 | 69.8 | 33.17 |

Stage-34 interpretation against TECE/TACE:

- The graph/edge state lifetime axis is now represented as an explicit provider interface, matching the TECE/TACE compiler interpretation: retained edge topology is persistent coarse-grained state, while descriptor/force work remains short-lived fused kernel work.
- The O(N) validity probe is not the dominant bottleneck on this benchmark window. It costs about 0.166 ms/step, versus about 1.522 ms/step for cached fused model plus validity. However, this is still about 11% of the cached model step, so production code should keep the check device-side and avoid host synchronization.
- With skin margin 0.004 A, the synthetic trajectory stays valid for 2000 force steps because max displacement is about 0.00173 A, below the half-skin threshold 0.002 A. This gives a physically interpretable condition under which the Stage-32 K=2000 throughput claim is legitimate.
- The remaining end-to-end gap is no longer deciding when a graph is valid. It is what happens when `needs_rebuild=True`: the current fallback is still full ASE reconstruction. The next clean target is a fast neighbor update/rebuild provider and a benchmark that separates provider update cost from fused model cost.

Next priority:

1. Prototype the provider update path: keep the provider interface, but add a `rebuild_graph_from_positions` backend abstraction so ASE reconstruction is only one backend.
2. Measure three costs separately on the same window: validity check, provider update/rebuild, fused model force pass.
3. Fold these deployment costs into the Pareto table as first-class axes alongside architecture error, because graph lifetime/update cost now determines whether scalar rTECE reaches NEP/DPA-style throughput end-to-end.


## Stage 35: ASE Graph Update Backend Timing

Stage 35 added a `GraphUpdateBackend` boundary and timed graph update/rebuild separately from validity checks and fused model evaluation. The current backend is still `ase_neighborlist`; the purpose of this stage is to quantify the remaining deployment gap and make the backend replaceable.

Verification and implementation gate:

- Added `GraphUpdateBackend(name, rebuild_fn)` with rebuild count, per-rebuild times, and total rebuild time.
- Added `--trajectory-update-only` to run position generation, cache validity checks, and provider graph updates/rebuilds without model evaluation.
- Normal trajectory replay now routes rebuilds through the same backend and records the same update timing fields.
- `test/test_rtece_scalar.py`: 46 passed after the change.

GPU setup: radial8/24x24 `rtece_element_density`, DFT valid `:1024`, 59193 atoms, float32, one V100, `--force-mode auto`, `trajectory_displacement_std=0.001`.

| mode | skin margin A | force steps | rebuilds | graph update total s | mean update s | seconds/pass | atom-step/s | DFT F MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| update-only valid cache | 0.004 | 2000 | 0 | 0.000 | n/a | 0.325606 | 363586905 | n/a |
| update-only invalid cache | 0.002 | 20 | 10 | 53.943 | 5.394 | 53.948049 | 21944 | n/a |
| fused model + invalid updates | 0.002 | 20 | 10 | 54.377 | 5.438 | 54.416800 | 21755 | 33.17 |

Stage-35 interpretation against TECE/TACE:

- The deployment cost is now separated into three measured pieces: validity check, graph update/rebuild, and fused rTECE force pass. This is the system-level counterpart of the TECE/TACE operator projection route.
- When the cache remains valid, update-only timing is essentially the Stage-34 O(N) validity cost: about 0.326 s for 2000 steps on the 59193-atom window.
- When the cache is invalid every two steps, the current ASE backend dominates end-to-end cost. Ten graph rebuilds take about 54 s, roughly 5.4 s per 1024-config rebuild. Adding 20 fused force evaluations increases total time by only about 0.47 s.
- Therefore invalid-cache execution is graph-update-bound, not model-bound. This cleanly answers the current priority question: replacing or accelerating the provider backend is necessary before further scalar architecture sweeps can move end-to-end throughput.

Next priority:

1. Implement a faster provider backend candidate, even if approximate at first: reuse existing edge topology with skin, or build a cell-list neighbor update path outside ASE for the batched graph.
2. Keep reporting Pareto rows as `(architecture error, cached model throughput, validity cost, update/rebuild cost)`, because the deployment front is now a coupled model-system curve.
3. Revisit scalar descriptor variants only after the provider backend no longer dominates invalid-cache throughput.


## Stage 36: Cached-Topology Update Backend

Stage 36 added the first non-ASE graph update backend candidate to the trajectory provider boundary. The new `cached_topology` backend reuses the cached `z`, `edge_index`, and `batch` tensors and only refreshes positions through `replay_graph_positions`. This is deliberately an upper-bound backend: it measures the cost of position refresh under a fixed topology assumption, not a real neighbor-list rebuild.

Implementation gate:

- `benchmark_rtece_scalar.py` now exposes `--graph-update-backend {ase_neighborlist,cached_topology}`.
- `make_graph_update_backend(...)` constructs either the existing ASE rebuild backend or the cached-topology position-refresh backend.
- `test/test_rtece_scalar.py`: 47 passed after the change.
- A CPU smoke with forced skin invalidation used `cached_topology`, recorded 4 graph updates, and spent 13.7 us total in graph update timing.

GPU setup: radial8/24x24 `rtece_element_density`, DFT valid `:1024`, 59193 atoms, float32, one V100, `--force-mode auto`, `trajectory_displacement_std=0.001`, `trajectory_skin_margin=0.002`.

| backend | mode | force steps | rebuild/update events | graph update total s | mean update s | seconds/pass | atom-step/s | DFT F MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `cached_topology` | update-only invalid cache | 20 | 10 | 0.0000446 | 0.00000446 | 0.003787 | 312592485 | n/a |
| `cached_topology` | fused model + invalid updates | 20 | 10 | 0.0000529 | 0.00000529 | 0.032912 | 35970261 | 33.17 |
| `cached_topology` | fused model + invalid updates | 2000 | 1039 | 0.005264 | 0.00000507 | 3.196984 | 37030530 | 33.17 |
| `ase_neighborlist` | update-only invalid cache | 20 | 10 | 53.943 | 5.394 | 53.948049 | 21944 | n/a |
| `ase_neighborlist` | fused model + invalid updates | 20 | 10 | 54.377 | 5.438 | 54.416800 | 21755 | 33.17 |

Stage-36 interpretation against TECE/TACE:

- This validates the provider/backend abstraction introduced in Stage 35. The same invalid-cache benchmark can now expose graph-update cost as a backend-dependent deployment axis rather than hiding it inside model timing.
- Cached-topology update reduces the measured update event from about 5.39 s for ASE rebuild to about 5 us for topology reuse plus position refresh. With forced invalidation every two steps, atom-step/s rises from about 2.18e4 with ASE updates to about 3.60e7 with cached-topology updates.
- The result should not be interpreted as a correct invalid-neighbor solution. It assumes topology can be reused even when the skin criterion reports invalid. Its value is to quantify the upper bound of the TECE system-renormalized runtime when edge topology is persistent and only positions are refreshed.
- The clean theoretical route is now sharper: semantic projection and fused scalar descriptors already deliver high model throughput; graph lifetime and safe topology update determine end-to-end throughput. The next stage should implement or benchmark a real fast neighbor provider, such as a cell-list or MD-runtime neighbor-list interface, rather than return to scalar head or radial sweeps.


## Stage 37: Torch Radius Non-PBC Update Backend

Stage 37 added and benchmarked `torch_radius_nopbc`, the first backend that actually recomputes `edge_index` without ASE. It rebuilds directed cutoff edges independently inside each batched configuration using torch tensor distance checks. It deliberately does not implement periodic boundary conditions, so it is a provider-boundary and performance probe rather than a production-correct neighbor backend for the current PBC OC20NEB slabs.

Implementation gate:

- `benchmark_rtece_scalar.py` now exposes `--graph-update-backend {ase_neighborlist,cached_topology,torch_radius_nopbc}`.
- `torch_radius_nopbc_graph(template, positions, cutoff=...)` reuses `z` and `batch`, refreshes `pos`, and rebuilds per-config directed edges under the cutoff.
- `test/test_rtece_scalar.py`: 48 passed after the change.
- A CPU smoke with forced skin invalidation used `torch_radius_nopbc`, recorded 4 graph updates, and spent 0.898 ms total in graph update timing on 2 configs / 100 atoms.

GPU setup: radial8/24x24 `rtece_element_density`, DFT valid `:1024`, 59193 atoms, float32, one V100, `--force-mode auto`, `trajectory_displacement_std=0.001`, `trajectory_skin_margin=0.002`.

| backend | mode | force steps | rebuild/update events | graph update total s | mean update s | seconds/pass | atom-step/s | DFT F MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `cached_topology` | update-only invalid cache | 20 | 10 | 0.0000446 | 0.00000446 | 0.003787 | 312592485 | n/a |
| `torch_radius_nopbc` | update-only invalid cache | 20 | 10 | 1.716636 | 0.171664 | 1.721270 | 687783 | n/a |
| `torch_radius_nopbc` | fused model + invalid updates | 20 | 10 | 1.729607 | 0.172961 | 1.762445 | 671714 | 33.17* |
| `ase_neighborlist` | update-only invalid cache | 20 | 10 | 53.943 | 5.394 | 53.948049 | 21944 | n/a |

`*` The trajectory benchmark collects prediction errors at step 0 before the first forced rebuild, so this MAE is not evidence that the non-PBC rebuilt graph preserves model accuracy.

Topology comparison on the same `:1024` window:

| metric | value |
|---|---:|
| ASE edge entries | 1,221,178 |
| ASE unique edges | 1,193,050 |
| torch non-PBC unique edges | 825,510 |
| overlap with ASE unique edges | 825,510 |
| missing ASE/PBC unique edges | 367,540 |
| extra torch unique edges | 0 |
| raw ASE unique-edge recall | 0.6919 |
| active ASE direct-distance unique edges | 825,510 |
| active unique-edge recall | 1.0000 |
| inactive ASE direct-distance entries | 368,880 |
| duplicate ASE edge entries | 28,128 |
| mismatch configs | 1024 / 1024 |

Stage-37 interpretation against TECE/TACE:

- This is the first measured non-ASE topology rebuild backend. It improves invalid-cache update-only throughput from 2.19e4 atom-step/s with ASE to 6.88e5 atom-step/s, about a 31x speedup.
- The backend is still far from the cached-topology upper bound: 0.172 s per update versus about 5 us for fixed-topology position refresh. The remaining cost is Python per-config looping and many small dense radius kernels, not scalar rTECE model evaluation.
- The raw topology comparison shows that `torch_radius_nopbc` recovers only 69.2% of ASE unique edges because the OC20NEB configs have full PBC. However, the current rTECE graph does not store periodic shift vectors, so model geometry uses direct coordinate differences. Under that implemented direct-distance semantics, `torch_radius_nopbc` recovers 100% of active unique edges inside the cutoff and drops 368,880 inactive ASE edge entries.
- This exposes a code/theory mismatch that is now a priority: either formalize the current rTECE endpoint as a direct-distance graph and remove ASE PBC overhead, or extend `RTECEGraph` and the fused evaluator to carry PBC shift/displacement state. A production PBC neighbor provider still needs a batched cell-list or MD-runtime neighbor interface; the current non-PBC torch backend is useful evidence, not the final backend.


## Stage 38: Direct Active Graph Construction Backend

Stage 38 formalized the Stage-37 active-edge finding as a benchmark switch. The new `--graph-construction-backend torch_radius_nopbc` builds the initial and timed atom graphs with direct-distance cutoff edges, rather than ASE/PBC neighbor entries. It does not change the rTECE checkpoint, scalar descriptors, force mode, or model parameters.

Implementation gate:

- `benchmark_rtece_scalar.py` now exposes `--graph-construction-backend {ase_neighborlist,torch_radius_nopbc}`.
- `atoms_to_torch_radius_nopbc_graph(...)` builds a single-configuration direct-distance graph.
- All benchmark graph construction paths route through `build_atom_graph(...)`.
- `test/test_rtece_scalar.py`: 49 passed after the change.
- A CPU smoke on 4 configs ran with `graph_construction_backend=torch_radius_nopbc` and wrote the backend into JSON.

GPU setup: radial8/24x24 `rtece_element_density`, DFT valid `:1024`, 59193 atoms, float32, one V100, `--force-mode auto`.

| graph construction backend | runtime mode | seconds/pass | atoms/s or atom-step/s | peak alloc MB | DFT F MAE | DFT F RMSE |
|---|---|---:|---:|---:|---:|---:|
| `ase_neighborlist` | prebuilt model pass | 0.001784 | 33186591 | 63.2 | 33.1687 | 109.3660 |
| `torch_radius_nopbc` | prebuilt model pass | 0.001416 | 41816583 | 57.2 | 33.1837 | 109.3663 |
| `ase_neighborlist` | rebuild graphs, then batch | 5.963333 | 9926 | 78.2 | 33.1687 | 109.3660 |
| `torch_radius_nopbc` | rebuild graphs, then batch | 0.429820 | 137716 | 60.1 | 33.1837 | 109.3663 |
| `ase_neighborlist` | cached trajectory, 2000 steps | 3.183696 | 37185080 | 69.8 | 33.1687 | 109.3660 |
| `torch_radius_nopbc` | cached trajectory, 2000 steps | 2.733743 | 43305459 | 63.7 | 33.1837 | 109.3663 |

Stage-38 interpretation against TECE/TACE:

- Direct active graph construction is a real hardware-cost improvement under the current rTECE graph semantics. Prebuilt model throughput rises by about 1.26x, cached trajectory throughput by about 1.16x, and rebuild+batch throughput by about 13.9x.
- The force-error change is negligible on this window: +0.015 meV/A force MAE and essentially unchanged RMSE. This supports the Stage-37 active-edge probe: the dropped ASE/PBC entries are inactive under the current direct-coordinate geometry, while duplicate ASE entries create only a tiny descriptor perturbation.
- This is a system-level TECE renormalization step, not a scalar architecture change. It shortens edge-state lifetime and removes edge entries outside the implemented operator semantics.
- The route now has a cleaner fork. For the current high-throughput rTECE endpoint, direct active graphs should be the default benchmark semantics. For a PBC-correct endpoint, `RTECEGraph` and the fused descriptor/force kernels must carry periodic shift/displacement state before further neighbor-provider work can be considered physically complete.


## Stage 39: Direct-Active Scalar rTECE Pareto Front

Stage 39 rebenchmarked the Stage-28 scalar rTECE front under the Stage-38 direct-active graph semantics. The checkpoints, scalar descriptors, force mode, labels, and validation windows are unchanged; only graph construction changes from ASE/PBC neighbor entries to direct-distance active edges via `--graph-construction-backend torch_radius_nopbc`.

GPU setup: DFT and teacher valid `:4096`, 250355 atoms, float32, one V100, `--force-mode auto` resolving to `analytic_element_triton_descriptor_force`, `measure_passes=5`.

| point | radial | hidden | params | direct-active atoms/s | DFT F MAE | teacher F MAE | peak alloc MB |
|---|---:|---|---:|---:|---:|---:|---:|
| radial4h16 | 4 | 16x16 | 449 | 87762046 | 43.85 | 47.71 | 150.8 |
| radial5h16_stage24 | 5 | 16x16 | 481 | 86308965 | 38.72 | 42.80 | 152.3 |
| radial5h16_seed2_long | 5 | 16x16 | 481 | 82331741 | 38.26 | 42.22 | 152.3 |
| radial8h24 | 8 | 24x24 | 1057 | 64507178 | 30.20 | 35.84 | 186.4 |
| radial8h32 | 8 | 32x32 | 1665 | 48192947 | 27.87 | 33.73 | 217.0 |

Comparison against Stage-28 ASE/PBC graph front on DFT valid `:4096`:

| point | ASE/PBC atoms/s | direct-active atoms/s | speedup | ASE/PBC DFT F MAE | direct-active DFT F MAE | delta MAE |
|---|---:|---:|---:|---:|---:|---:|
| radial4h16 | 72644836 | 87762046 | 1.21x | 43.91 | 43.85 | -0.06 |
| radial5h16_stage24 | 64463762 | 86308965 | 1.34x | 39.02 | 38.72 | -0.30 |
| radial5h16_seed2_long | 64515626 | 82331741 | 1.28x | 38.28 | 38.26 | -0.02 |
| radial8h24 | 50929958 | 64507178 | 1.27x | 30.19 | 30.20 | +0.01 |
| radial8h32 | 40750024 | 48192947 | 1.18x | 28.14 | 27.87 | -0.27 |

Stage-39 interpretation against TECE/TACE:

- Direct-active graph construction improves every point on the scalar rTECE front by about 1.18-1.34x without a meaningful force-error penalty. This confirms Stage 38 was not a radial8h24-specific artifact.
- The current high-throughput endpoint is radial4h16 at 87.8M atoms/s and 43.85 meV/A DFT force MAE. The current lower-error scalar endpoint is radial8h32 at 48.2M atoms/s and 27.87 meV/A DFT force MAE. The best balanced direct-active point is radial8h24 at 64.5M atoms/s and 30.20 meV/A DFT force MAE.
- This is now the clean current direct-distance rTECE Pareto front: radial resolution and scalar head capacity define the semantic degradation axis, while direct-active graph construction removes edge entries outside the implemented operator semantics.
- A PBC-correct branch remains separate. It must add periodic displacement/shift state to `RTECEGraph` and the fused descriptor/force kernels before it can be compared fairly with the direct-distance branch.


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

# Stage 41: Direct-Active Update-Backend Consistency

Stage 41 checked whether the Stage-37 update-backend conclusion still holds when the initial graph construction also uses the Stage-38/39 direct-active semantics. The checkpoint, scalar descriptor, force mode, validation window, and trajectory displacement match the previous radial8h24 trajectory probes; the key change is `--graph-construction-backend torch_radius_nopbc` for every row.

GPU setup: radial8h24 `rtece_element_density`, DFT valid `:1024`, 59193 atoms, one V100, float32, `--force-mode auto` resolving to `analytic_element_triton_descriptor_force`, `trajectory_displacement_std=0.001`.

| graph backend | update backend | mode | force steps | update events | update total s | seconds/pass | atom-step/s | DFT F MAE |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `torch_radius_nopbc` | default/no rebuild | valid cached trajectory, skin 0.004 | 2000 | 0 | 0.000000 | 2.674001 | 44272977 | 33.18 |
| `torch_radius_nopbc` | `cached_topology` | invalid update-only, skin 0.002 | 20 | 10 | 0.000046 | 0.003753 | 315412557 | n/a |
| `torch_radius_nopbc` | `cached_topology` | invalid model+updates, skin 0.002 | 20 | 10 | 0.000052 | 0.027125 | 43645228 | 33.18 |
| `torch_radius_nopbc` | `torch_radius_nopbc` | invalid update-only, skin 0.002 | 20 | 10 | 1.723663 | 1.728869 | 684760 | n/a |
| `torch_radius_nopbc` | `torch_radius_nopbc` | invalid model+updates, skin 0.002 | 20 | 10 | 1.729914 | 1.762232 | 671796 | 33.18 |

Stage-41 interpretation against TECE/TACE:

- Direct-active initial graph semantics are consistent with the Stage-36/37 provider conclusion. Cached topology remains an upper bound near the fused model runtime, while real `torch_radius_nopbc` topology rebuild is still two orders of magnitude too slow for end-to-end high-throughput invalid-cache execution.
- The direct-active cached trajectory improves over the earlier ASE/PBC initial-graph trajectory because it carries fewer active edges and less memory, but it does not solve topology update when the skin validity criterion fails.
- This strengthens the current route: the clean rTECE Pareto front is now model-semantics complete for the direct-distance branch, and the limiting missing component is a batched/fused neighbor provider or MD-runtime neighbor-list interface.
- Next priority should be implementation work on a batched direct-active provider/cell-list boundary, not more radial/head sweeps. The separate PBC-correct branch remains valid but larger: it requires periodic shift/displacement state in `RTECEGraph` and the fused descriptor/force kernels.

# Stage 42: Grouped Direct-Radius Update Backend

Stage 42 implemented and benchmarked `torch_radius_nopbc_grouped`, a grouped direct-radius update backend for the current direct-distance rTECE branch. It preserves the same direct active-edge semantics as `torch_radius_nopbc`, but replaces the Python loop over many per-config radius checks with one padded batched radius computation over `(num_configs, max_atoms, max_atoms)`.

Implementation gate:

- Added `torch_radius_nopbc_grouped_graph(...)` and exposed `--graph-update-backend torch_radius_nopbc_grouped`.
- Added tests that the grouped graph matches the existing loop backend, that the update backend rebuilds the same directed edges, and that CLI help exposes the new backend.
- Full rTECE test file after the change: 52 passed.

GPU setup: radial8h24 `rtece_element_density`, direct-active initial graph, DFT valid `:1024`, 59193 atoms, one V100, float32, `--force-mode auto`, forced invalid cache with `trajectory_skin_margin=0.002`, `trajectory_displacement_std=0.001`, 20 force steps and 10 update events.

| update backend | mode | update total s | mean update s | seconds/pass | atom-step/s | peak alloc MB | DFT F MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| `torch_radius_nopbc` | update-only | 1.772007 | 0.177201 | 1.776811 | 666283 | 55.7 | n/a |
| `torch_radius_nopbc_grouped` | update-only | 0.012035 | 0.001204 | 0.017411 | 67994130 | 241.3 | n/a |
| `torch_radius_nopbc` | model+updates | 1.719516 | 0.171952 | 1.752224 | 675633 | 77.6 | 33.18 |
| `torch_radius_nopbc_grouped` | model+updates | 0.012803 | 0.001280 | 0.043798 | 27029982 | 263.1 | 33.18 |

Stage-42 interpretation against TECE/TACE:

- This is a positive system-renormalization result. It does not change descriptors, learned parameters, graph semantics, or conservative force evaluation. It only changes the realization of the retained direct active-edge topology update.
- Grouping removes the dominant Python/per-config small-kernel overhead exposed in Stage 41. Update-only cost improves by about 147x, and invalid-cache model+updates throughput improves by about 40x.
- The grouped backend still does not reach the cached-topology upper bound from Stage 41: 27.0M atom-step/s versus about 43.6M atom-step/s for model+cached-topology invalid updates on the same window. The remaining gap is now dense padded radius work plus allocation/materialization overhead, not ASE or per-config Python dispatch.
- Memory rises from about 78MB to 263MB for model+updates because the grouped path materializes padded distance/mask tensors. This is acceptable as a provider prototype but not the final implementation.
- Next priority: turn this into a cleaner provider by avoiding padded all-pairs materialization, either with a fused cell-list/radius kernel or a grouped-by-size backend. The current result is strong enough to deprioritize the old loop backend and confirms that provider realization is a first-class TECE/TACE deployment axis.

# Stage 43: Grouped-By-Size Direct-Radius Backend

Stage 43 tested the Stage-42 follow-up idea: reduce the dense padded all-pairs memory of `torch_radius_nopbc_grouped` by grouping configurations with the same atom count, then running a smaller batched radius calculation per size group. This preserves the direct-distance rTECE graph semantics and conservative force path, but changes provider realization.

Implementation gate:

- Added `torch_radius_nopbc_grouped_by_size_graph(...)` and exposed `--graph-update-backend torch_radius_nopbc_grouped_by_size`.
- Added tests that grouped-by-size matches the loop backend edges, that the backend factory works, and that CLI help exposes the backend.
- Full rTECE test file after the change: 53 passed.

GPU setup: radial8h24 `rtece_element_density`, direct-active initial graph, DFT valid `:1024`, 59193 atoms, one V100, float32, `--force-mode auto`, forced invalid cache with `trajectory_skin_margin=0.002`, `trajectory_displacement_std=0.001`, 20 force steps and 10 update events.

| update backend | mode | update total s | mean update s | seconds/pass | atom-step/s | peak alloc MB | DFT F MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| `torch_radius_nopbc_grouped` | update-only | 0.012194 | 0.001219 | 0.017505 | 67628193 | 241.3 | n/a |
| `torch_radius_nopbc_grouped_by_size` | update-only | 1.046924 | 0.104692 | 1.052414 | 1124900 | 61.1 | n/a |
| `torch_radius_nopbc_grouped` | model+updates | 0.012321 | 0.001232 | 0.045059 | 26273400 | 263.1 | 33.18 |
| `torch_radius_nopbc_grouped_by_size` | model+updates | 1.163704 | 0.116370 | 1.195055 | 990632 | 82.6 | 33.18 |

Stage-43 interpretation against TECE/TACE:

- This is a negative provider result. Grouping by atom count reduces peak allocation by about 3.2x for model+updates, but it gives up the main Stage-42 win: update cost rises from about 1.2 ms/update to 116 ms/update.
- The result is still useful theoretically because it separates two system-renormalization costs: padded dense all-pairs memory versus Python-level grouping/edge-materialization overhead. In this implementation, launch and edge-splitting overhead dominate the saved pair work.
- The grouped-by-size backend should not replace `torch_radius_nopbc_grouped` for the current direct-active deployment front. It is a diagnostic branch showing that lowering memory with Python control flow is the wrong priority.
- Next priority: a fused direct-radius/cell-list provider that avoids both all-config max padding and Python per-size edge splitting. The provider should produce `edge_index` or direct edge buffers in one low-level path, closer to the fused descriptor/force kernels already validated for scalar rTECE.

# Stage 44: Chunked Grouped Direct-Radius Backend

Stage 44 tested `torch_radius_nopbc_grouped_chunked`, a provider variant between Stage-42 all-config grouping and Stage-43 by-size grouping. It preserves the direct-distance active-edge semantics, keeps config order, and chunks consecutive configurations into padded radius batches. The default chunk size is 128 configs, which reduces peak padded tensor size without introducing by-size Python edge splitting.

Implementation gate:

- Added `torch_radius_nopbc_grouped_chunked_graph(...)` and exposed `--graph-update-backend torch_radius_nopbc_grouped_chunked`.
- Added tests that chunked grouped radius matches the loop backend, that the backend factory works, and that CLI help exposes the backend.
- Full rTECE test file after the change: 54 passed.

GPU setup: radial8h24 `rtece_element_density`, direct-active initial graph, DFT valid `:1024`, 59193 atoms, one V100, float32, `--force-mode auto`, forced invalid cache with `trajectory_skin_margin=0.002`, `trajectory_displacement_std=0.001`, 20 force steps and 10 update events.

| update backend | mode | update total s | mean update s | seconds/pass | atom-step/s | peak alloc MB | DFT F MAE |
|---|---|---:|---:|---:|---:|---:|---:|
| `torch_radius_nopbc_grouped` | update-only | 0.012132 | 0.001213 | 0.017416 | 67977303 | 241.3 | n/a |
| `torch_radius_nopbc_grouped_chunked` | update-only | 0.033966 | 0.003397 | 0.037933 | 31208976 | 73.4 | n/a |
| `torch_radius_nopbc_grouped_by_size` | update-only | 1.040244 | 0.104024 | 1.045440 | 1132404 | 61.1 | n/a |
| `torch_radius_nopbc_grouped` | model+updates | 0.012259 | 0.001226 | 0.043711 | 27083689 | 263.1 | 33.18 |
| `torch_radius_nopbc_grouped_chunked` | model+updates | 0.035949 | 0.003595 | 0.065868 | 17973177 | 95.0 | 33.18 |
| `torch_radius_nopbc_grouped_by_size` | model+updates | 1.135829 | 0.113583 | 1.167079 | 1014379 | 82.6 | 33.18 |

Stage-44 interpretation against TECE/TACE:

- Chunked grouping is a useful provider Pareto point. It keeps the same rTECE direct active graph semantics and conservative scalar force path, but trades about 1.5x lower model+update throughput for about 2.8x lower peak allocation compared with all-config grouped radius.
- This resolves the Stage-43 ambiguity: reducing memory is possible without falling back to Python by-size edge splitting. The right intermediate path is coarse chunking or a fused provider, not fine Python grouping.
- The current direct-active invalid-cache provider front is now: `cached_topology` as a topology-reuse upper bound, `torch_radius_nopbc_grouped` for maximum update throughput, `torch_radius_nopbc_grouped_chunked` for lower memory, and `torch_radius_nopbc_grouped_by_size` as a diagnostic negative result.
- The next clean implementation target remains a fused direct-radius/cell-list provider. Stage 44 suggests that chunk granularity should be a tunable deployment parameter until that lower-level provider exists.


# Stage 45: Chunk-Size Sweep For Chunked Direct-Radius Backend

Stage 45 made the Stage-44 chunk granularity explicit through `--graph-update-chunk-configs` and swept chunk sizes for the chunked grouped direct-radius provider. This is not a model-architecture change: the checkpoint, scalar element-density descriptor, direct-active graph semantics, and conservative fused force path stay fixed. The stage tests whether provider chunking is a reproducible deployment axis in the TECE/TACE Pareto table.

Implementation gate:

- `benchmark_rtece_scalar.py` now exposes `--graph-update-chunk-configs`, forwards it into `torch_radius_nopbc_grouped_chunked`, and records it in benchmark JSON.
- `test_chunked_torch_radius_update_backend_honors_chunk_size` verifies the backend factory passes the chunk-size setting through to the chunked radius updater.
- The CLI help test now verifies that the chunk-size option is exposed.
- Full rTECE scalar test file after the change: 55 passed.

GPU setup: radial8h24 `rtece_element_density`, direct-active initial graph, DFT valid `:1024`, 59193 atoms, one V100, float32, `--force-mode auto`, forced invalid cache with `trajectory_skin_margin=0.002`, `trajectory_displacement_std=0.001`, 20 force steps and 10 update events.

| update backend | chunk configs | mode | update total s | seconds/pass | atom-step/s | peak alloc MB | DFT F MAE |
|---|---:|---|---:|---:|---:|---:|---:|
| `torch_radius_nopbc_grouped` | all | update-only | 0.011970 | 0.017392 | 68,070,707 | 241.3 | n/a |
| `torch_radius_nopbc_grouped_chunked` | 64 | update-only | 0.069001 | 0.073236 | 16,165,038 | 61.3 | n/a |
| `torch_radius_nopbc_grouped_chunked` | 128 | update-only | 0.035014 | 0.039086 | 30,288,403 | 73.4 | n/a |
| `torch_radius_nopbc_grouped_chunked` | 256 | update-only | 0.021553 | 0.025552 | 46,331,185 | 110.0 | n/a |
| `torch_radius_nopbc_grouped_chunked` | 512 | update-only | 0.015826 | 0.019991 | 59,220,668 | 174.8 | n/a |
| `torch_radius_nopbc_grouped` | all | model+updates | 0.012118 | 0.043540 | 27,189,963 | 263.1 | 33.18 |
| `torch_radius_nopbc_grouped_chunked` | 64 | model+updates | 0.068641 | 0.099996 | 11,839,051 | 83.0 | 33.18 |
| `torch_radius_nopbc_grouped_chunked` | 128 | model+updates | 0.037028 | 0.066623 | 17,769,557 | 95.0 | 33.18 |
| `torch_radius_nopbc_grouped_chunked` | 256 | model+updates | 0.021643 | 0.051295 | 23,079,365 | 130.5 | 33.18 |
| `torch_radius_nopbc_grouped_chunked` | 512 | model+updates | 0.016391 | 0.046616 | 25,395,950 | 197.7 | 33.18 |

Stage-45 interpretation against TECE/TACE:

- Chunk size is a real provider Pareto axis. Larger chunks reduce update time but increase peak allocation; smaller chunks reduce memory while preserving the same graph semantics and force error.
- Chunk64 is already above the 1e7 atom-step/s target in the forced-invalid model+update benchmark with about 83MB peak allocation. Chunk256 is the best balanced provider point in this sweep: about 85% of all-config grouped model+update throughput with about half the peak allocation.
- This supports the TECE/TACE system-renormalization route. Once the model has been degraded to scalar descriptors and fused conservative force kernels, topology-update realization and edge-buffer lifetime are first-class deployment axes, not incidental implementation details.
- The next clean target remains a fused direct-radius/cell-list provider. Until that exists, `--graph-update-chunk-configs` should be reported in Pareto artifacts together with graph construction and graph update backend metadata.


# Stage 46: Direct-Radius Provider Work Metadata

Stage 46 added provider-work metadata to `GraphUpdateBackend` and reran the Stage-45 forced-invalid direct-radius sweep. This is a benchmark/compiler instrumentation step: it does not change rTECE descriptors, learned parameters, direct-active graph semantics, or conservative force evaluation. Its purpose is to quantify the all-pairs work and chunking overhead that a fused direct-radius/cell-list provider must remove.

Implementation gate:

- `GraphUpdateBackend` now carries `static_metadata` and `last_metadata`.
- Direct-radius update backends record `num_configs`, `num_atoms`, `max_atoms_per_config`, `num_chunks`, `max_chunk_configs`, exact per-config pair slots, padded pair slots, padding overhead ratio, and last directed-edge count.
- Benchmark JSON now records `graph_update_backend_metadata`.
- The test was first run red and failed on missing `static_metadata`; after implementation, the new metadata test passed and the full rTECE scalar test file passed with 56 tests.

GPU setup: radial8h24 `rtece_element_density`, direct-active initial graph, DFT valid `:1024`, 59193 atoms, one V100, float32, `--force-mode auto`, forced invalid cache with `trajectory_skin_margin=0.002`, `trajectory_displacement_std=0.001`, 20 force steps and 10 update events.

| update backend | chunk configs | mode | atom-step/s | update total s | peak alloc MB | chunks | padded pair slots | padding overhead | directed edges | DFT F MAE |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `torch_radius_nopbc_grouped` | all | update-only | 68,543,495 | 0.011981 | 241.3 | 1 | 7,750,656 | 2.095 | 825,584 | n/a |
| `torch_radius_nopbc_grouped_chunked` | 64 | update-only | 16,279,118 | 0.068561 | 61.3 | 16 | 4,956,992 | 1.340 | 825,584 | n/a |
| `torch_radius_nopbc_grouped_chunked` | 128 | update-only | 29,808,822 | 0.035617 | 73.4 | 8 | 5,365,888 | 1.450 | 825,584 | n/a |
| `torch_radius_nopbc_grouped_chunked` | 256 | update-only | 46,420,113 | 0.021265 | 110.0 | 4 | 6,211,328 | 1.679 | 825,584 | n/a |
| `torch_radius_nopbc_grouped_chunked` | 512 | update-only | 59,142,031 | 0.015863 | 174.8 | 2 | 7,574,528 | 2.047 | 825,584 | n/a |
| `torch_radius_nopbc_grouped` | all | model+updates | 27,338,063 | 0.012368 | 263.1 | 1 | 7,750,656 | 2.095 | 825,584 | 33.18 |
| `torch_radius_nopbc_grouped_chunked` | 64 | model+updates | 12,175,411 | 0.067447 | 83.0 | 16 | 4,956,992 | 1.340 | 825,584 | 33.18 |
| `torch_radius_nopbc_grouped_chunked` | 128 | model+updates | 17,863,813 | 0.035686 | 95.0 | 8 | 5,365,888 | 1.450 | 825,584 | 33.18 |
| `torch_radius_nopbc_grouped_chunked` | 256 | model+updates | 22,841,929 | 0.022921 | 130.5 | 4 | 6,211,328 | 1.679 | 825,584 | 33.18 |
| `torch_radius_nopbc_grouped_chunked` | 512 | model+updates | 24,931,106 | 0.017448 | 197.7 | 2 | 7,574,528 | 2.047 | 825,584 | 33.18 |

Stage-46 interpretation against TECE/TACE:

- The direct-radius provider bottleneck is now quantitatively decomposed. The exact per-config dense pair work for this window is 3,699,489 pair slots, while all-config grouped radius evaluates 7,750,656 padded slots, about 2.10x exact.
- Chunk64 reduces padded work to 4,956,992 slots, about 1.34x exact, but pays 16 chunk launches/materializations and is much slower. Chunk512 approaches grouped throughput because it uses only two chunks, but its padded work is almost as high as all-config grouping.
- This explains the Stage-45 Pareto curve: the current PyTorch provider trades launch count against padded tensor size. The next provider should not be another Python grouping variant; it should approach exact pair/cell-list work while keeping launch count close to one.
- The fused/cell-list target is concrete: generate about 0.826M directed active edges from about 3.70M exact per-config pair slots without materializing long-lived padded distance/mask tensors. This is the next system-level TECE/TACE renormalization step before more model architecture changes.


# Stage 47: Ragged Exact-Pair Direct-Radius Provider

Stage 47 tested the Stage-46 hypothesis that a provider with exact per-config pair slots and low launch count might dominate padded grouped/chunked radius updates. The new `torch_radius_nopbc_ragged` backend preserves the same direct-distance graph semantics and conservative rTECE force path, but constructs exact ragged all-pair candidates instead of padded all-config or chunked masks.

Implementation gate:

- Added `torch_radius_nopbc_ragged_graph(...)` and exposed `--graph-update-backend torch_radius_nopbc_ragged`.
- Added tests that ragged edges match the loop backend, that the update backend records exact work metadata, and that CLI help exposes the new backend.
- The tests were first run red and failed on missing function/factory/CLI choice; after implementation, the targeted tests passed and the full rTECE scalar test file passed with 57 tests.

GPU setup: radial8h24 `rtece_element_density`, direct-active initial graph, DFT valid `:1024`, 59193 atoms, one V100, float32, `--force-mode auto`, forced invalid cache with `trajectory_skin_margin=0.002`, `trajectory_displacement_std=0.001`, 20 force steps and 10 update events.

| update backend | mode | atom-step/s | update total s | peak alloc MB | chunks | padded pair slots | padding overhead | directed edges | DFT F MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `torch_radius_nopbc_grouped` | update-only | 66,746,689 | 0.012180 | 241.3 | 1 | 7,750,656 | 2.095 | 825,584 | n/a |
| `torch_radius_nopbc_grouped_chunked` chunk256 | update-only | 46,507,195 | 0.021406 | 110.0 | 4 | 6,211,328 | 1.679 | 825,584 | n/a |
| `torch_radius_nopbc_ragged` | update-only | 39,074,531 | 0.025768 | 383.9 | 1 | 3,699,489 | 1.000 | 825,584 | n/a |
| `torch_radius_nopbc_grouped` | model+updates | 26,766,448 | 0.012598 | 263.1 | 1 | 7,750,656 | 2.095 | 825,584 | 33.18 |
| `torch_radius_nopbc_grouped_chunked` chunk256 | model+updates | 22,949,920 | 0.021976 | 130.5 | 4 | 6,211,328 | 1.679 | 825,584 | 33.18 |
| `torch_radius_nopbc_ragged` | model+updates | 20,805,343 | 0.025858 | 405.9 | 1 | 3,699,489 | 1.000 | 825,584 | 33.18 |

Stage-47 interpretation against TECE/TACE:

- Ragged exact-pair construction is a useful negative result. It removes padded pair-slot overhead completely, but in PyTorch it materializes long integer tensors for graph ids, pair offsets, and source/destination indices. That memory traffic dominates the saved distance work.
- The result sharpens the provider route: the next front point will not come from another Python/Torch tensor reshaping strategy. Chunk256 remains the best practical memory/throughput compromise in the current provider family, while all-config grouped remains the fastest high-memory update backend.
- The fused/cell-list provider requirement is now stronger and more specific. It must generate active edges or feed descriptor/force kernels without materializing either padded distance/mask tensors or ragged all-pair index tensors.
- This fits the TECE/TACE system-renormalization thesis: after semantic projection and fused scalar descriptors, the remaining deployment bottleneck is not parameter count or model architecture. It is the runtime representation of local edge topology and its lifetime in memory.


# Stage 48: Triton Padded Direct-Radius Provider

Stage 48 tested the Stage-47 conclusion that the next useful provider step must move below Python/Torch tensor reshaping. The new `torch_radius_nopbc_triton_padded` backend preserves direct-active semantics, but scans padded candidate slots inside a Triton kernel and writes active directed edges with an atomic counter. It deliberately keeps the padded candidate space for this first lower-level test; the question is whether avoiding long-lived PyTorch distance/mask/ragged-index tensors is enough to improve the provider front.

Implementation gate:

- Added `direct_radius_padded_edges_triton(...)` and a Triton `_direct_radius_padded_edges_kernel`.
- Added `torch_radius_nopbc_triton_padded_graph(...)` with CPU/non-float32 fallback to the grouped backend, and exposed `--graph-update-backend torch_radius_nopbc_triton_padded`.
- Added tests that the backend matches loop edges through the CPU fallback, that factory metadata is recorded, and that CLI help exposes the backend.
- The tests were first run red on missing function/factory/CLI choice; after implementation and one kernel compile fix, the full rTECE scalar test file passed with 58 tests.

GPU setup: radial8h24 `rtece_element_density`, direct-active initial graph, DFT valid `:1024`, 59193 atoms, one V100, float32, `--force-mode auto`, forced invalid cache with `trajectory_skin_margin=0.002`, `trajectory_displacement_std=0.001`, 20 force steps and 10 update events.

| update backend | mode | atom-step/s | update total s | peak alloc MB | padded pair slots | padding overhead | directed edges | DFT F MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `torch_radius_nopbc_grouped` | update-only | 68,486,684 | 0.012020 | 241.3 | 7,750,656 | 2.095 | 825,584 | n/a |
| `torch_radius_nopbc_grouped_chunked` chunk256 | update-only | 48,274,213 | 0.020571 | 110.0 | 6,211,328 | 1.679 | 825,584 | n/a |
| `torch_radius_nopbc_ragged` | update-only | 39,868,611 | 0.025378 | 383.9 | 3,699,489 | 1.000 | 825,584 | n/a |
| `torch_radius_nopbc_triton_padded` | update-only | 141,308,212 | 0.004430 | 255.0 | 7,750,656 | 2.095 | 825,584 | n/a |
| `torch_radius_nopbc_grouped` | model+updates | 26,536,430 | 0.012534 | 263.1 | 7,750,656 | 2.095 | 825,584 | 33.18 |
| `torch_radius_nopbc_grouped_chunked` chunk256 | model+updates | 23,294,254 | 0.021821 | 130.5 | 6,211,328 | 1.679 | 825,584 | 33.18 |
| `torch_radius_nopbc_ragged` | model+updates | 21,688,771 | 0.025748 | 405.9 | 3,699,489 | 1.000 | 825,584 | 33.18 |
| `torch_radius_nopbc_triton_padded` | model+updates | 37,057,631 | 0.003897 | 276.8 | 7,750,656 | 2.095 | 825,584 | 33.18 |

Stage-48 interpretation against TECE/TACE:

- This is a positive provider-renormalization result. It does not change the scalar rTECE model, descriptor, checkpoint, graph semantics, or force path. It changes only the runtime realization of direct-radius topology update.
- The result validates the Stage-47 diagnosis: the dominant problem was not only pair-slot count. PyTorch tensor materialization for masks and ragged indices was a first-class bottleneck. Moving candidate filtering and active-edge writing into Triton raises update-only throughput to 141M atom-step/s and model+updates to 37.1M atom-step/s.
- The provider front changes again. `torch_radius_nopbc_triton_padded` is now the high-throughput invalid-cache update backend. `chunk256` remains a lower-memory fallback, but no longer the balanced throughput point when memory permits.
- The next clean target is not another Torch backend. It is a counted/two-pass Triton provider or cell-list provider that keeps the lower-level kernel execution path while avoiding full preallocation of a `2 x padded_pair_slots` int64 edge buffer. That would combine Stage48 speed with Stage46/47 memory discipline.

# Stage 49: Triton Counted Direct-Radius Provider

Stage 49 tested the Stage-48 next step directly: keep the lower-level Triton candidate scan and active-edge writer, but remove the full `2 x padded_pair_slots` int64 edge-buffer allocation. The new `torch_radius_nopbc_triton_counted` backend uses two passes over the same padded candidate space. Pass 1 counts active directed edges only; pass 2 allocates an exact `2 x num_edges` edge buffer and writes the active edges with the existing atomic writer kernel generalized by output stride.

Implementation gate:

- Added `_direct_radius_count_edges_kernel(...)` and `direct_radius_counted_edges_triton(...)`.
- Generalized `_direct_radius_padded_edges_kernel(...)` with an `edge_stride` constexpr so padded and counted providers share the active-edge writer.
- Added `torch_radius_nopbc_triton_counted_graph(...)` with the same CPU/non-float32 fallback as the padded backend.
- Exposed `--graph-update-backend torch_radius_nopbc_triton_counted`, factory wiring, static provider metadata, and tests.
- The Stage49 tests were first run red on missing import/factory/CLI choice. After implementation, `test/test_rtece_scalar.py` passed with 59 tests.

GPU setup: radial8h24 `rtece_element_density`, direct-active initial graph, DFT valid `:1024`, 59193 atoms, one V100, float32, `--force-mode auto`, forced invalid cache with `trajectory_skin_margin=0.002`, `trajectory_displacement_std=0.001`, 20 force steps and 10 update events.

| update backend | mode | atom-step/s | update total s | peak alloc MB | peak reserved MB | padded pair slots | padding overhead | directed edges | DFT F MAE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `torch_radius_nopbc_grouped` | update-only | 66,748,236 | 0.012397 | 241.3 | 346.0 | 7,750,656 | 2.095 | 825,584 | n/a |
| `torch_radius_nopbc_grouped_chunked` chunk256 | update-only | 47,059,209 | 0.021184 | 110.0 | 174.0 | 6,211,328 | 1.679 | 825,584 | n/a |
| `torch_radius_nopbc_ragged` | update-only | 39,427,477 | 0.025690 | 383.9 | 492.0 | 3,699,489 | 1.000 | 825,584 | n/a |
| `torch_radius_nopbc_triton_padded` | update-only | 154,149,942 | 0.003715 | 255.0 | 286.0 | 7,750,656 | 2.095 | 825,584 | n/a |
| `torch_radius_nopbc_triton_counted` | update-only | 146,335,457 | 0.003444 | 43.7 | 74.0 | 7,750,656 | 2.095 | 825,584 | n/a |
| `torch_radius_nopbc_grouped` | model+updates | 27,909,688 | 0.012639 | 263.1 | 406.0 | 7,750,656 | 2.095 | 825,584 | 33.18 |
| `torch_radius_nopbc_grouped_chunked` chunk256 | model+updates | 23,534,235 | 0.022151 | 130.5 | 214.0 | 6,211,328 | 1.679 | 825,584 | 33.18 |
| `torch_radius_nopbc_ragged` | model+updates | 21,476,197 | 0.026212 | 405.9 | 552.0 | 3,699,489 | 1.000 | 825,584 | 33.18 |
| `torch_radius_nopbc_triton_padded` | model+updates | 37,468,188 | 0.003911 | 276.8 | 346.0 | 7,750,656 | 2.095 | 825,584 | 33.18 |
| `torch_radius_nopbc_triton_counted` | model+updates | 36,141,905 | 0.003524 | 77.2 | 126.0 | 7,750,656 | 2.095 | 825,584 | 33.18 |

Stage-49 interpretation against TECE/TACE:

- This is a positive memory-renormalization result. It does not change the scalar rTECE model, descriptor, checkpoint, graph semantics, or force path, so the DFT force MAE/RMSE is unchanged. It changes only the lifetime and size of the runtime edge representation.
- Counted keeps most of the Stage48 throughput: 94.9% of Triton padded update-only atom-step/s and 96.5% of Triton padded model+updates atom-step/s. In return, peak allocated memory drops from 255.0 MB to 43.7 MB in update-only and from 276.8 MB to 77.2 MB in model+updates.
- This cleanly separates two TECE deployment axes: Stage48 removed Python/Torch materialization from active-edge filtering; Stage49 removes the padded edge-buffer allocation. The remaining padded work is candidate scanning, not stored edge state.
- The current provider Pareto front is now two Triton points: `torch_radius_nopbc_triton_padded` for maximum single-batch throughput, and `torch_radius_nopbc_triton_counted` for nearly the same throughput with far lower memory pressure and better scaling headroom.
- The next highest-value test is not another Torch provider. It is either a larger-system/bigger-batch counted-vs-padded stress test, or a cell-list/fused descriptor provider that removes the remaining padded candidate scan and moves closer to the NEP/DPA-style throughput target while preserving the TECE/TACE direct-active semantics.


# Stage 50: Triton Counted Batch-Size Stress

Stage 50 stress-tested the Stage49 counted provider against Stage48 padded over increasing real-data batch sizes. The aim was to decide whether counted should replace padded as the large-batch provider front, or whether it is specifically a memory-headroom point. The OC20NEB valid extxyz contains 10000 configs, so the stress range was 1024, 2048, 4096, 8192, and the real-data upper bound 10000.

Setup: radial8h24 `rtece_element_density`, direct-active initial graph, one V100, float32, update-only trajectory replay, 20 replay steps, forced invalid cache, comparing only `torch_radius_nopbc_triton_padded` and `torch_radius_nopbc_triton_counted`.

| limit configs | backend | atoms | atom-step/s | seconds/pass | update enqueue s | peak alloc MB | peak reserved MB | directed edges | padded pair slots | counted/padded throughput | counted/padded alloc |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | `triton_padded` | 59,193 | 155,998,009 | 0.007589 | 0.003706 | 255.0 | 286.0 | 825,584 | 7,750,656 | n/a | n/a |
| 1024 | `triton_counted` | 59,193 | 146,506,776 | 0.008081 | 0.003414 | 43.7 | 74.0 | 825,584 | 7,750,656 | 0.939 | 0.171 |
| 2048 | `triton_padded` | 119,271 | 180,892,419 | 0.013187 | 0.008160 | 510.1 | 582.0 | 1,634,364 | 15,501,312 | n/a | n/a |
| 2048 | `triton_counted` | 119,271 | 172,418,404 | 0.013835 | 0.006429 | 86.7 | 158.0 | 1,634,364 | 15,501,312 | 0.953 | 0.170 |
| 4096 | `triton_padded` | 250,355 | 257,656,655 | 0.019433 | 0.014725 | 1,327.7 | 1,466.0 | 3,459,786 | 40,960,000 | n/a | n/a |
| 4096 | `triton_counted` | 250,355 | 207,028,037 | 0.024186 | 0.010465 | 181.3 | 322.0 | 3,459,786 | 40,960,000 | 0.804 | 0.137 |
| 8192 | `triton_padded` | 525,769 | 325,610,466 | 0.032294 | 0.026098 | 2,658.2 | 2,926.0 | 7,303,632 | 81,920,000 | n/a | n/a |
| 8192 | `triton_counted` | 525,769 | 249,657,496 | 0.042119 | 0.015261 | 382.2 | 650.0 | 7,303,632 | 81,920,000 | 0.767 | 0.144 |
| 10000 | `triton_padded` | 646,473 | 341,899,920 | 0.037817 | 0.031106 | 3,247.3 | 3,582.0 | 9,006,668 | 100,000,000 | n/a | n/a |
| 10000 | `triton_counted` | 646,473 | 258,546,783 | 0.050008 | 0.017702 | 471.3 | 806.0 | 9,006,668 | 100,000,000 | 0.756 | 0.145 |

Stage-50 interpretation against TECE/TACE:

- Counted should not replace padded as the maximum-throughput provider in the current real-data regime. Padded remains faster at every tested batch size and reaches 342M atom-step/s at 10000 configs.
- Counted is still a clean memory-renormalized provider. It uses only 13.7-17.1% of padded peak allocated memory across the stress range, and 471 MB vs 3247 MB at the 10000-config upper bound.
- The timing interpretation must use `seconds_per_pass`, not only `graph_update_total_s`. The latter is CPU-side enqueue timing and can make counted look faster even when end-to-end GPU elapsed time is slower. This matters for future Pareto tables.
- The current provider Pareto front should be stated conditionally: `torch_radius_nopbc_triton_padded` is the peak-throughput point when memory permits; `torch_radius_nopbc_triton_counted` is the memory-headroom point for larger systems or constrained devices.
- The next highest-priority architecture step is now sharper. Another counted/two-pass variant is unlikely to close the NEP/DPA-style throughput gap. The clean TECE/TACE route is a cell-list or fused descriptor provider that removes the remaining padded candidate scan and avoids the extra count pass, while preserving direct-active semantics and the scalar rTECE force path.


# Stage 51: Cell-List Oracle Work Analysis

Stage 51 tested the Stage50 priority before writing a complex runtime kernel: how much padded candidate scan would a direct-active cell-list provider remove on the same real OC20NEB trajectory geometry? This is an oracle/work-analysis stage, not a new model or production provider. It preserves the rTECE checkpoint, scalar descriptor, cutoff, direct-active semantics, and Stage50 trajectory replay geometry.

Implementation gate:

- Added `cell_list_oracle_work_metadata(...)`, a CPU oracle that counts padded slots, all-pair nonself slots, 27-neighbor-cell directed candidate pairs, active directed edges, and candidate ratios.
- Added a unit test that fixes cell-list oracle semantics on a two-config toy geometry.
- Added `runs/oc20neb_tace_mace/rtece-stage51-cell-list-oracle-work/run_oracle.py` and JSON outputs for 1024, 2048, 4096, 8192, and 10000 configs.
- The oracle uses the same cutoff 5.0 A and synthetic trajectory step 19 geometry as the final Stage50 invalid-cache rebuild; active edge counts match the Stage50 provider metadata.

| limit configs | atoms | padded slots | all-pair nonself | cell candidates | active edges | candidate/padded | candidate/active | max cell occupancy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 59,193 | 7,750,656 | 3,640,296 | 2,850,522 | 825,584 | 0.367778 | 3.452734 | 18 |
| 2048 | 119,271 | 15,501,312 | 7,386,312 | 5,754,520 | 1,634,364 | 0.371228 | 3.520954 | 18 |
| 4096 | 250,355 | 40,960,000 | 16,402,870 | 12,406,382 | 3,459,786 | 0.302890 | 3.585881 | 18 |
| 8192 | 525,769 | 81,920,000 | 35,936,006 | 26,219,846 | 7,303,632 | 0.320066 | 3.589974 | 20 |
| 10000 | 646,473 | 100,000,000 | 44,242,578 | 32,313,616 | 9,006,668 | 0.323136 | 3.587744 | 20 |

Stage-51 interpretation against TECE/TACE:

- The cell-list direction is justified, but the expected gain is bounded. On real OC20NEB batches, cell-list candidates are about 30-37% of padded slots, so it can remove roughly 63-70% of candidate distance checks.
- Cell-list alone is not a 10x route. The candidate set remains about 3.5x the active directed edge count, so a runtime provider that only replaces padded scanning with cell-list edge materialization may improve throughput but probably will not close the NEP/DPA-style gap.
- The next clean architecture step is therefore a fused cell-list descriptor/force provider: generate cell-list candidates and accumulate scalar rTECE radial/element-density descriptors or force contributions directly, minimizing intermediate edge lifetime. This follows the TECE/TACE renormalization logic more closely than adding another graph-update backend.
- The current Pareto interpretation becomes three-tiered: `triton_padded` is the peak-throughput edge-list provider when memory permits; `triton_counted` is the low-memory edge-list provider; cell-list/fused descriptor is the next candidate to move the frontier because it attacks padded candidate work rather than only edge-list allocation.


# Stage 52: Cell-List Fused Descriptor Oracle

Stage 52 implemented the first semantics-level bridge from Stage51's cell-list work analysis to a fused scalar rTECE descriptor. The new `cell_list_packed_element_density_descriptors(...)` path preserves the same direct-active, no-PBC graph semantics as the current high-throughput branch, but it does not consume `edge_index`. It partitions atoms by batch, builds cutoff-sized cells, scans the 27-cell stencil, filters true cutoff distances, and directly accumulates the packed `[rho, rho_z]` element-density descriptor into atoms.

Implementation gate:

- Added `cell_list_packed_element_density_descriptors(...)` with the same narrow eligibility as the existing packed element-density descriptor: scalar density plus element density only, no quadratic/vector/atomic moments and no edge sketches.
- Added tests that first failed on the missing function, then verified exact agreement with `packed_element_density_descriptors(torch_radius_nopbc_graph(...))` and verified that bogus cross-batch `edge_index` input is ignored.
- Added `runs/oc20neb_tace_mace/rtece-stage52-cell-list-fused-descriptor/run_descriptor_oracle.py` to check real OC20NEB geometry windows and emit JSON evidence.

| limit configs | atoms | active edges | cell candidates | candidate/padded | candidate/active | max abs descriptor diff | direct graph CPU s | edge-list descriptor CPU s | cell-list fused descriptor CPU s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 5,298 | 69,804 | 207,504 | 0.372159 | 2.972666 | 6.77e-15 | 0.013755 | 0.480110 | 2.010375 |
| 512 | 31,831 | 472,830 | 1,661,434 | 0.428721 | 3.513808 | 9.16e-15 | 0.125035 | 0.993020 | 4.695996 |

Stage-52 interpretation against TECE/TACE:

- This closes the current semantic gap between Stage51's cell-list candidate-space argument and the actual scalar rTECE descriptor. The element-density descriptor does not require persistent public edge-list state; the retained TECE scalar statistics can be accumulated directly from local cell candidates.
- The result is deliberately not a new throughput point. The CPU prototype is slower than the existing edge-list descriptor path, so it should be treated as an oracle and executable specification for a lower-level fused kernel.
- The next priority remains algorithm-first but implementation-aware: produce a fused cell-list descriptor/force provider that preserves this exact descriptor semantics and conservative force chain rule, then benchmark it against `triton_padded` and `triton_counted`. NVIDIA nvalchemi toolkit ops and DeepMD edge-force/edge-virial implementations can be studied as backend references, but they should serve the TECE compiler route rather than define it.
- The Pareto-route narrative is now cleaner: model semantics are downfolded to scalar element-density descriptors; descriptor/force edge lifetime is already shortened by Triton fused evaluators; topology representation can now be lowered further from materialized edge-list to cell-list candidate streaming.


# Stage 53: Formal rTECE Scalar Model Entrypoint

Stage 53 responded to a structural issue in the previous route: most of the high-throughput rTECE model code lived under `benchmarks/oc20neb_tace_mace/`, so it was easy to mistake the work for benchmark-only parameter tuning rather than a real model endpoint. The implementation now promotes the semantic rTECE scalar model into `tace.models.rtece_scalar` while keeping the old benchmark path as a compatibility shim.

Implementation gate:

- Added `tace/models/rtece_scalar.py` as the canonical implementation of `RTECEScalarConfig`, `RTECEGraph`, `RTECEScalarModel`, scalar descriptors, packed element-density descriptors, and the Stage52 cell-list descriptor oracle.
- Added `tace/models/rtece_triton_kernels.py` as the canonical implementation of the current rTECE Triton pair force, element-density descriptor/force, and direct-radius provider kernels.
- Replaced `benchmarks/oc20neb_tace_mace/rtece_scalar_model.py` and `benchmarks/oc20neb_tace_mace/rtece_triton_kernels.py` with re-export shims so historical training and benchmark scripts keep the same imports.
- Exported the rTECE scalar API from `tace.models`. Heavy full-TACE exports are now optional at import time, so a lightweight scalar endpoint can be imported even when the full e3nn/TensorModel stack is unavailable or incompatible in the local environment.
- Added tests that fix this boundary: `tace.models` and the old benchmark modules must resolve to the same rTECE classes/functions and Triton kernel functions.

Stage-53 interpretation against TECE/TACE:

- This is not a new accuracy/throughput Pareto point; it is an architecture-boundary correction. The scalar endpoint is now represented as a formal model in the TACE package, not only as an OC20NEB benchmark artifact.
- The theory boundary is cleaner: `tace.models.rtece_scalar` owns the semantic projection/downfolding from TECE/TACE into scalar sufficient statistics; `tace.models.rtece_triton_kernels` owns reusable low-level kernels; benchmark scripts own dataset-specific distillation, provider stress tests, and Pareto measurement.
- This makes the next fused cell-list descriptor/force work easier to judge. A stabilized kernel should land under the formal rTECE backend instead of remaining purely benchmark-local.
- Next priority: implement the Stage52 cell-list fused descriptor/force contract as a real backend and benchmark it against `triton_padded` and `triton_counted`. The model semantics and existing Triton kernels have now moved to core; the remaining gap is the fused cell-list runtime.


# Stage 54: rTECE Route And Workflow Contract

Stage 54 addresses the model-repository structure problem exposed after Stage53. Moving `RTECEScalarModel` into `tace.models` made rTECE importable as a formal endpoint, but training, inference, checkpoint metadata, and Pareto reporting were still partly benchmark-shaped. This made the TECE/TACE degradation route implicit in code names rather than explicit in the model artifact.

Implementation gate:

- Added `rtece_route_contract(...)` under `tace.models.rtece_scalar`. It maps `(RTECEScalarConfig, force_mode, graph backend)` to a machine-readable TECE route: semantic tier, descriptor family, retained and deleted TECE groups, descriptor width, force realization, descriptor realization, graph semantics, graph construction/update backend, edge-state lifetime, and Pareto axes.
- Added `tace.models.rtece_workflow` with `save_checkpoint`, `load_checkpoint`, `predict`, `loss_for_batch`, `evaluate_loss`, and `train_steps`. Checkpoints now store `tece_route` metadata with config and weights.
- Exported core workflow functions from `tace.models` as `save_rtece_checkpoint`, `load_rtece_checkpoint`, `predict_rtece`, `rtece_loss_for_batch`, `evaluate_rtece_loss`, and `train_rtece_steps`.
- Refactored `benchmarks/oc20neb_tace_mace/train_rtece_scalar.py` to reuse the core workflow. Historical benchmark/profile scripts keep the two-value `load_checkpoint(...)` wrapper, while `load_checkpoint_with_metadata` points to the core metadata-aware API.
- Updated the distillation matrix summary to include the TECE route column, so Pareto rows report semantic degradation and runtime realization together with error and throughput.

Verification:

- Targeted workflow/contract tests: `5 passed, 63 deselected, 1 warning`.
- Full rTECE test file: `68 passed, 1 warning`.

Stage-54 interpretation against TECE/TACE:

- This is not a new MAE/RMSE or throughput point. It is a route-contract stage that makes the model compiler hypothesis auditable in normal repository code.
- The original TECE design-space note says the relevant hardware variable is not parameter count but edge bytes, kernel work, scatter, communication, latency, and especially how long equivariant edge/node state survives. The new route contract records those choices explicitly: retained scalar density groups, deleted persistent equivariant state groups, force realization, graph backend, and edge-state lifetime.
- This answers the structural concern that rTECE was only a benchmark artifact. The current scalar endpoint can now be trained, checkpointed, loaded, and used for inference through `tace.models`, and its artifacts carry the semantic degradation route needed for systematic Pareto comparison.
- The priority after this stage should not drift back to hidden-size tuning. The next clean experiment remains the Stage52 fused cell-list descriptor/force backend for the element-density scalar endpoint, benchmarked against `torch_radius_nopbc_triton_padded` and `torch_radius_nopbc_triton_counted` with the same route metadata attached.


# Stage 55: Cell-List Descriptor+Force Oracle Entry

Stage 55 converts the Stage52 cell-list descriptor oracle into a real, explicit rTECE force mode. It is still a PyTorch/CPU-oriented oracle rather than a high-throughput Triton or nvalchemi-style kernel, but it is now reachable from the formal model, workflow API, and benchmark CLI.

Implementation gate:

- Added `RTECEScalarModel.forward_element_density_cell_list_packed_analytic_forces(...)`. The method ignores caller-supplied `edge_index`, constructs direct-active no-PBC pairs from batch-sorted positions using the Stage52 cell-list semantics, accumulates packed `[rho, rho_z]` descriptors, and applies the same conservative descriptor-gradient force chain rule as `forward_element_density_packed_analytic_forces(...)`.
- Added workflow dispatch for `analytic_element_cell_list_descriptor_force` and benchmark CLI support for the same force mode.
- Extended `rtece_route_contract(...)` so this mode reports `cell_list_fused_descriptor_oracle`, `cell_list_analytic_descriptor_force`, `direct_active_nopbc`, and `streaming_cell_candidates_oracle`.
- Added tests that first failed on the missing method/force mode, then verified that the cell-list force path matches packed edge-list energy, atomic energy, and forces while ignoring a bogus input edge list.

Verification:

- Targeted Stage55 tests: `2 passed, 68 deselected, 1 warning`.
- Full rTECE test file: `70 passed, 1 warning`.

CPU oracle smoke, using the existing `rtece-stage23-radial4-hidden16` checkpoint and first 8 OC20NEB teacher-valid configs:

| mode | atoms/s | seconds/pass | DFT F MAE meV/A | DFT F RMSE meV/A |
|---|---:|---:|---:|---:|
| `analytic_element_cell_list_descriptor_force` | 4028.8 | 0.099286 | 52.099328 | 96.659877 |
| `analytic_element_packed` | 2093.3 | 0.191085 | 52.099328 | 96.659877 |

Stage-55 interpretation against TECE/TACE:

- This is a semantic/backend closure stage, not a new GPU Pareto point. The smoke confirms identical force error to the materialized packed edge-list path on the tested window.
- The TECE design-space claim is sharpened: for the T3 element-density scalar endpoint, retained information is the scalar radial and neighbor-element density statistics; the public edge list is not part of the model semantics. It is only one runtime realization.
- Therefore the next high-value implementation target is a fused GPU cell-list descriptor+force backend that preserves this exact route contract while removing materialized edge-list lifetime and padded candidate scans. NVIDIA nvalchemi-toolkit-ops or DeepMD edge-force/edge-virial code can be backend references, but the algorithmic target is now fixed by the route contract and tests.


# Stage 56: Direct-Padded Streaming Triton Descriptor+Force Backend

Stage 56 adds the first GPU-kernel lowering of the Stage55 observation that the T3 element-density scalar endpoint does not semantically require a public materialized `edge_index`. This is not yet the final cell-list backend because it still scans padded per-configuration pair slots, but descriptor and force kernels can now consume `(pos, batch counts, starts, z)` directly and keep edge state implicit inside the kernel.

Implementation gate:

- Added `element_density_direct_padded_descriptors_triton(...)` and `element_density_direct_padded_forces_triton(...)` under `tace.models.rtece_triton_kernels`. These scan direct padded candidate slots and atomically accumulate packed element-density descriptors or conservative force contributions without outputting an edge list.
- Added `RTECEScalarModel.forward_element_density_direct_padded_triton_descriptor_force_analytic_forces(...)`. It validates the scalar element-density endpoint, derives `counts/starts` from sorted `batch`, ignores `edge_index`, computes descriptors with the direct-padded Triton descriptor kernel, differentiates the scalar head with respect to descriptors, and applies the direct-padded force kernel.
- Added workflow and benchmark force mode `analytic_element_direct_padded_descriptor_force`.
- Extended `rtece_route_contract(...)` so this mode reports `triton_direct_padded_descriptor`, `triton_direct_padded_descriptor_force`, `direct_active_nopbc`, and `streaming_padded_candidates`.
- Added tests that first failed on the missing kernel API, route contract, and model method; a CUDA-conditional numerical test compares direct-padded streaming against the existing edge-index Triton path when a GPU is available.

Verification:

- Targeted Stage56 tests: `4 passed, 69 deselected, 1 warning`.
- Full rTECE test file: `73 passed, 1 warning`.
- Benchmark CLI help lists `analytic_element_direct_padded_descriptor_force`.
- The current login node reports `torch.cuda.is_available() == False` and device count `0`; therefore Stage56 is not yet a measured GPU Pareto point.

Stage-56 interpretation against TECE/TACE:

- This separates two hardware-cost axes that the TECE design-space document treats as first-class: edge-state lifetime and candidate-space work. Stage56 removes public edge-list lifetime at the GPU descriptor/force interface, but still keeps padded candidate scanning.
- The route is cleaner than another graph-update backend because the descriptor and force semantics now flow from the scalar TECE statistics directly, not through a persistent edge tensor.
- The next priority is to run this backend on a CUDA node against `analytic_element_triton_descriptor_force`, then replace the padded pair-slot iterator with real cell-list candidate streaming if the result confirms that padded candidate work remains the bottleneck.


# Stage 57: GPU Validation Harness And Slurm Cancellation

Stage 57 prepared the benchmark path needed to validate Stage56 on a GPU. The reusable `rtece_scalar_benchmark.sbatch` script now forwards `GRAPH_CONSTRUCTION_BACKEND` and `GRAPH_UPDATE_BACKEND` to `benchmark_rtece_scalar.py`, with direct-active `torch_radius_nopbc` as the default graph construction backend. This matters because the direct-padded streaming backend must be compared against the existing edge-index Triton descriptor/force path under the same no-PBC direct-active semantics.

Implementation gate:

- Added `GRAPH_CONSTRUCTION_BACKEND=${GRAPH_CONSTRUCTION_BACKEND:-torch_radius_nopbc}` and `GRAPH_UPDATE_BACKEND=${GRAPH_UPDATE_BACKEND:-ase_neighborlist}` to `rtece_scalar_benchmark.sbatch`.
- Forwarded both values into the benchmark CLI.
- Added a test that locks the sbatch graph-backend contract.

Verification:

- Targeted sbatch test: `1 passed, 73 deselected, 1 warning`.
- Full rTECE test file after the harness change: `74 passed, 1 warning`.

GPU validation attempts:

| job | mode | requested shape | state |
|---:|---|---|---|
| 679358 | `analytic_element_triton_descriptor_force` | 1x V100, `flood-1o2gpu`, 1024 configs, 5 passes | `CANCELLED by 0` after 1 s |
| 679359 | `analytic_element_direct_padded_descriptor_force` | 1x V100, `flood-1o2gpu`, 1024 configs, 5 passes | `CANCELLED by 0` after 1 s |
| 679362 | direct-padded retry | 1x V100, `rush-1o2gpu` | cancelled manually after Slurm reported QOS not permitted on `16V100` |
| 679363 | edge-index Triton retry | 4x V100, `rush-gpu`, 256 configs, 2 passes | `CANCELLED by 0` after 2 s |
| 679364 | direct-padded retry | 4x V100, `rush-gpu`, 256 configs, 2 passes | `CANCELLED by 0` after 2 s |
| 679366 | direct-padded shape retry | 4x V100, `rush-gpu`, `ntasks=4`, 64 configs, 1 pass | `CANCELLED by 0` after 2 s |

Stage-57 interpretation against TECE/TACE:

- This is not a model or backend negative result. The jobs did not create stdout/stderr and did not produce benchmark JSON, so Python did not reach the rTECE benchmark path.
- The TECE route priority remains unchanged: direct-padded streaming needs CUDA numeric/throughput validation against edge-index Triton; if padded work remains the bottleneck, the next runtime renormalization is true cell-list candidate streaming.
- The practical next step is to use a known-good SAI sbatch template or an interactive compute allocation to run the exact benchmark commands, instead of continuing blind Slurm submissions from the cancelled script.


# Stage 58: TACE-Style E0 Reference And SAI-Safe Benchmark Submission

Stage 58 closes two blocking issues raised by the code review and the failed GPU validation attempts. The Slurm cancellation was traced to a submission-method constraint rather than an rTECE model failure: on SAI, rTECE benchmark jobs must not be submitted with `sbatch --export=ALL,...` for parameter passing. A minimal `16V100`/`flood-1o2gpu` probe with `hostname` and `nvidia-smi -L` completed, so the partition/QOS/single-GPU shape itself is valid. Future rTECE benchmark submissions should use a generated wrapper sbatch whose body exports variables, then submit the wrapper with plain `sbatch wrapper.sbatch`.

The more important model-side correction is energy reference handling. The review correctly identified the old scalar `energy_per_atom_shift = sum(E_s)/sum(N_s)` as incompatible with TACE/MACE-style energy comparability on composition-varying data. rTECE now supports a TACE-style per-element reference map:

```text
E_ref(s) = sum_Z N_{sZ} E0_Z
E_model(s) = sum_i MLP_i + E_ref(s)
```

When E0s are not provided, the training script now fits average per-element E0s from the training set by least squares over the composition matrix `N_{sZ}` against total energies `E_s`, matching the standard TACE/MACE practice more closely than a single global per-atom shift. The legacy `energy_per_atom_shift` remains as a checkpoint-compatible fallback, but it is no longer the default rTECE training reference.

Implementation gate:

- Added `atomic_energies` to `RTECEScalarConfig` and a shared `atomic_reference_energy(...)` path used by autograd and analytic/Triton inference forwards.
- Added `fit_atomic_energies(...)` in `train_rtece_scalar.py`; default training now stores least-squares per-element E0s in the config and `train_summary.json`.
- Preserved old checkpoints by keeping `energy_per_atom_shift` and normalizing serialized `atomic_energies` keys back to integer atomic numbers on load.
- Added an inference-only guard for analytic/Triton force backends. These paths detach descriptor geometry or take `create_graph=False`; they are valid benchmark/inference force evaluators but not force-training backends.
- Added `submit_rtece_scalar_benchmark.py`, which generates an SAI-safe wrapper with GPU `#SBATCH` resource lines and no Slurm `--export` option.
- Recorded the SAI `sbatch --export` constraint in the local `sai-user-guide` skill and Codex memory.

Verification completed in this stage:

- Per-element E0 model and least-squares tests: passed.
- CPU CLI smoke confirmed default training writes least-squares `atomic_energies` and leaves legacy `energy_per_atom_shift` at zero.
- Workflow/checkpoint/no-export helper tests: passed.

Stage-58 interpretation against TECE/TACE:

- This is a scientific-closure fix, not a throughput optimization. It makes rTECE energy residuals comparable to TACE/MACE residual training by removing a composition-dependent baseline error before judging the reduced scalar operator family.
- The next Pareto experiments should be rerun for the active radial/hidden front because old energy MAE values were contaminated by the global-shift baseline. Force MAE rankings may remain informative, but energy MAE and teacher/DFT energy comparability must be regenerated.
- Review-priority order after this stage: first PBC graph ABI with edge shifts/cell, then chemical low-rank species basis and cavity/radial edge sketches, then the renormalization/distillation manifest that maps deleted TECE paths to student descriptors. Kernel-level cell-list fusion remains useful only after these semantic contracts are stable.
- After the review P0 correctness issues are closed, add physics validation beyond aggregate MAE/RMSE: element-pair dimer scans from roughly 0.5x to 5x covalent-radius scale to inspect energy/force smoothness and repulsion behavior, plus train/test structure rattle-and-relax comparisons against original TACE using final energy, force stability, and relaxed RMSD. These tests should be used to judge rTECE generalization and physical usability after the semantic bugs are fixed, not as a substitute for fixing the bugs.
- Early rattle-and-relax observations suggest rTECE relax RMSD is especially large for non-metal adsorbates such as C/N-containing species. Treat this as an algorithmic diagnostic: bucket relax failures by adsorbate chemistry and local coordination, then test whether low-rank species bases and cavity/radial angular sketches repair those basins. This is consistent with the review concern that a single `Z` moment causes chemical collisions and that current scalar sketches delete too much angular environment for adsorbate bonding.


# Stage 59: PBC Graph ABI And TACE Neighbor Provider Alignment

Stage 59 addresses the review P0 that rTECE used periodic neighbor indices without periodic image shifts. This was a physical correctness bug: for PBC structures, `edge_index` alone does not define the displacement. The rTECE graph ABI now carries the same information class as TACE graph data: `cell`, integer `edge_shifts`, and `edge_batch`. The unique displacement convention is now:

```text
d_e = r_dst(e) - r_src(e) + S_e H_batch(e)
```

Implementation gate:

- Extended `RTECEGraph` with optional `cell`, `edge_shifts`, and `edge_batch` fields while preserving old no-PBC call sites.
- Updated `compute_pair_geometry(...)` so every descriptor and analytic force path uses shifted PBC displacements when shift metadata is present.
- Updated `collate_graphs(...)` and cached position replay to preserve cell/shift metadata across batched graphs.
- Split training graph construction into `atoms_to_rtece_graph(...)` and `atoms_to_graph(...)`; the former is graph-only and the latter attaches energy/force labels.
- Replaced rTECE training graph construction with TACE's `tace.dataset.neighbour_list.get_neighborhood`, defaulting to `matscipy` for real periodic data and falling back to `ase` only for zero-cell nonperiodic molecule tests.
- Added benchmark graph construction backend `matscipy_neighborlist`; `ase_neighborlist` remains a reference/correctness backend and `torch_radius_nopbc` remains the direct-active high-throughput no-PBC backend.

Verification:

- PBC two-atom boundary test confirms shifted distances are 0.2 A rather than the unshifted 4.8 A.
- Benchmark `matscipy_neighborlist` construction test confirms the same shifted geometry.
- Full rTECE test file: `82 passed, 1 warning`.

Stage-59 interpretation against TECE/TACE:

- This keeps the task aligned with review P0 #2, not with a side optimization. ASE is no longer the implied main graph route for periodic rTECE; it is a reference backend.
- The graph-provider design now matches TACE's existing matscipy/PyG data contract closely enough that later high-throughput providers can be swapped under the same ABI.
- nvalchemi-toolkit-ops remains a candidate GPU neighbor/edge-force provider, but only after this cell/shift ABI is stable. The next model-side review priorities are still chemical low-rank species basis and cavity/radial angular sketches for the non-metal adsorbate failures.


# Stage 60: Low-Rank Species Basis For Chemical-Collision Repair

Stage 60 addresses review P0 #6: the old `rtece_element_density` descriptor restored only the first neighbor atomic-number moment,

```text
rho_Z,in = sum_j R_n(r_ij) (Z_j / Z_max),
```

so two equal-geometry environments with the same neighbor count and the same summed atomic number can collide. The review's example is `C + C` versus `B + N`: both have `Z` sum 12, but they are chemically different. This failure mode is consistent with the user's early rattle-and-relax observation that rTECE can produce much larger relaxed RMSD on non-metal adsorbates such as C/N-containing adsorbates.

The new `rtece_species_basis4` variant adds a fixed low-rank species basis while preserving the T3 scalarized execution principle:

```text
rho_a,in = sum_j R_n(r_ij) (Z_j / Z_max)^a,  a = 1..4.
```

This is a conservative first chemistry-channel repair, not the final learned species representation. It keeps the model in the scalar-density family and makes the retained TECE group explicit as `low_rank_neighbor_species_basis`. The immediate purpose is to test whether a tiny chemistry basis can reduce composition/adsorbate collisions before adding angular or edge-relational state.

Implementation gate:

- Added `species_basis_channels` to `RTECEScalarConfig` and `build_rtece_config("rtece_species_basis4")`.
- Extended scalar descriptor construction with low-rank neighbor species densities.
- Added a route-contract semantic tier, `T3_low_rank_species_density`, so summary artifacts can distinguish this model from the old first-moment `element_density` route.
- Exposed the variant in the rTECE training CLI and distillation summarizer.
- Added a regression test where old element-density descriptors intentionally collide for equal-distance `C+C` and `B+N` neighbors, while the species-basis descriptors separate them.

Stage-60 interpretation against TECE/TACE:

- This change is not a blind width or parameter sweep. It restores a deleted TECE chemistry-channel axis in a compressed scalar form.
- It is still a fixed polynomial basis over `Z`; the cleaner v2 route should replace this with a learned or teacher-POD/SVD species basis once the descriptor manifest/compiler exists.
- It does not address edge self-leakage or radial-channel averaging. The next review-aligned architecture tasks remain cavity edge moments and low-rank radial/cross-radial sketches.
- After the remaining review P0 bugs are fixed, the dimer-scan and rattle/relax validation suite should bucket errors by element pair and adsorbate chemistry to test whether this species-basis repair specifically improves C/N and other non-metal adsorbate basins.


# Stage 61: Cavity Edge Sketches For Self-Edge Leakage Repair

Stage 61 addresses review P0 #4: the original `rtece_edge_sketch8/16` relational descriptors used full atomic moments on both sides of an edge. For an edge `j -> i`, the target moment contains the edge's own direct contribution:

```text
A_i = A_{i\j} + a_ij.
```

Therefore terms intended to describe edge/environment relations, such as `u_ij dot v_i` or `u_ij^T Q_i u_ij`, also contain a direct radial pair component from the same edge. This is not a symmetry violation, but it breaks clean TECE body-order and source-target relational bookkeeping.

The new `rtece_cavity_edge_sketch8` variant keeps the old edge sketch variants for checkpoint compatibility and adds a cavity version of the same eight scalar slots. For each edge it uses

```text
A_{i cavity} = A_i - a_ij,
A_{j cavity} = A_j - a_ji,
```

for vector and quadrupole relational terms, while the last two direct radial slots remain explicit pair/radial paths. This follows the review recommendation to separate `direct_edge` from `cavity_i/cavity_j` semantics rather than letting them leak into one feature.

Implementation gate:

- Added `use_cavity_edge_sketches` to `RTECEScalarConfig` and `build_rtece_config("rtece_cavity_edge_sketch8")`.
- Updated `edge_relational_sketches(...)` so the cavity variant subtracts the current directed edge from target-side moments and the reciprocal edge contribution from source-side moments when a reverse edge exists.
- Added route-contract metadata: `T3_cavity_edge_scalar_sketch`, `cavity_atomic_moment_sketch`, `cavity_edge_relational_scalar_sketches`, and `direct_edge_radial_path`.
- Exposed the variant in the training CLI and summary inference path.
- Fixed summary config reconstruction so `species_basis_channels` and `use_cavity_edge_sketches` are preserved when rebuilding route contracts.
- Added a regression test where an isolated two-atom reciprocal pair has nonzero full-moment edge sketches but zero cavity environment terms, with direct radial terms unchanged.

Stage-61 interpretation against TECE/TACE:

- This is a model-semantics repair, not a throughput optimization. It makes the edge path closer to a true rTECE relational scalar projection with auditable retained/deleted operator groups.
- The old full-moment edge sketch remains available as a compatibility/control variant. New experiments should prefer the cavity variant when testing edge-relational semantics.
- This stage does not fix review #5: vector/quadrupole radial channels are still averaged before edge contraction. The next architecture priority should be low-rank radial/cross-radial sketches, ideally in a form that can later be selected by teacher covariance or force-weighted sensitivity.


# Stage 62: Low-Rank Radial Edge Sketches

Stage 62 addresses review P0 #5: even after the Stage 61 cavity repair, edge sketches still collapsed vector and quadrupole radial channels by `mean(dim=1)` before forming edge-relational scalars. That made near-shell and mid/far-shell orientation information indistinguishable whenever they shared the same radial average.

The new `rtece_cavity_radial_edge_sketch14` variant keeps the Stage 61 cavity semantics and replaces the single radial mean with two fixed low-rank radial shell projections. For a radial-channel moment `A_{in}` it now forms

```text
A_{ik} = mean_{n in shell k} A_{in},  k = 1,2,
```

then keeps sparse same-shell and cross-shell edge invariants for vector and quadrupole moments, plus direct radial slots. This is the fixed-projection version of the review's recommended

```text
A_tilde_{iklm} = sum_n U_kn A_{inlm},  K << N_r.
```

Implementation gate:

- Added `radial_edge_sketch_channels` to `RTECEScalarConfig`.
- Added `build_rtece_config("rtece_cavity_radial_edge_sketch14")` with two radial sketch channels and fourteen edge scalar slots.
- Added `_project_radial_edge_channels(...)`, which preserves shell information that a full radial mean collapses.
- Added sparse cross-radial vector and quadrupole edge terms while retaining explicit direct radial slots.
- Added route-contract metadata: `T3_cavity_radial_edge_scalar_sketch`, `cavity_radial_atomic_moment_sketch`, `low_rank_radial_edge_moment_sketches`, and `cross_radial_edge_invariants`.
- Exposed the variant in training CLI and summary inference, and preserved the new radial sketch field when summary rebuilds route configs.

Verification gate:

- A regression test constructs two artificial radial-channel tensors with identical full radial mean but different two-shell projections.
- Existing edge-sketch rotation invariance now covers `rtece_cavity_radial_edge_sketch14` as well as the full and cavity controls.
- Summary route reconstruction now confirms the radial-cavity variant is not misclassified as historical `element_density` just because the variant name contains `radial`.

Stage-62 interpretation against TECE/TACE:

- This is still a hand-fixed projection, not the final teacher-SVD/POD radial compiler. Its value is to make the missing TECE radial-rank axis explicit in the executable model and route manifest.
- The theoretical ordering is now cleaner: direct scalar pair density, element/species chemistry density, cavity edge relations, and low-rank radial edge relations are separate retained TECE groups rather than hidden inside one sketch name.
- The next decision should compare two priorities from the documents: either formalize the route manifest/path-spec compiler so these variants stop being ad hoc Boolean combinations, or run a small training/benchmark slice for `rtece_species_basis4` and `rtece_cavity_radial_edge_sketch14` after E0/PBC fixes to see whether the semantic repairs actually improve the non-metal adsorbate failure buckets.


# Stage 63: Semantic-Repair Smoke Matrix

Stage 63 tests whether the recent review-driven semantic repairs can pass the normal rTECE train/checkpoint/benchmark path and whether their early signal justifies a larger Pareto run. This is deliberately a small smoke matrix, not a final accuracy or throughput claim.

Setup:

| item | value |
|---|---|
| Slurm job | `679597` |
| submission method | generated no-export wrapper, plain `sbatch wrapper` |
| variants | `rtece_species_basis4`, `rtece_cavity_radial_edge_sketch14` |
| train labels | `mixed_train_tw0.75.extxyz` |
| train configs / steps | 16 configs / 4 steps |
| validation configs | 8 mixed-valid configs |
| benchmark configs | 32 DFT-valid + 32 teacher-valid configs |
| hidden / radial | `16,16` / `num_radial=4` |
| force mode | `autograd` |
| graph timing | prebuilt batched graph, `includes_graph_construction=false` |
| GPU | single V100 |

Implementation gate:

- Added `submit_rtece_scalar_matrix.py`, a SAI-safe matrix wrapper generator mirroring the benchmark submit helper.
- The helper writes parameter overrides as body `export VAR=...` lines and submits only `sbatch wrapper`; it does not use `sbatch --export=ALL,...`.
- The wrapper also avoids `--mem` and `--cpus-per-task`, matching the SAI constraint recorded in the local skill/memory.

Smoke results:

| variant | DFT F MAE | teacher F MAE | DFT E MAE | atoms/s | peak alloc MB | params | interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| `rtece_species_basis4` | 43.97 | 43.91 | 284.92 | 437666 | 36.7 | 641 | Fast autograd scalar-density chemistry repair, but weak early force/energy signal in this tiny run. |
| `rtece_cavity_radial_edge_sketch14` | 38.61 | 37.83 | 51.10 | 174361 | 101.0 | 737 | Better early force/energy signal, but much slower because edge/radial sketches still use full autograd and materialized edges. |

Stage-63 interpretation against TECE/TACE:

- The no-export wrapper solved the previous SAI submission failure mode: the job ran on `16v100n08`, entered Python, trained both variants, and produced all benchmark JSON files.
- The semantic repairs are executable through the normal repository path, not only descriptor unit tests.
- The early result supports the review logic qualitatively: adding cavity/radial edge information improves force error versus the pure low-rank species-density repair in this smoke setting.
- It also confirms the hardware warning from the TECE design document: edge/radial relational semantics are not useful for the high-throughput Pareto front unless they are paired with an analytic/fused force path or a route compiler that can keep edge state short-lived. Autograd edge sketches are not a high-throughput endpoint.
- Next priority should not be a large autograd `cavity_radial_edge_sketch14` sweep. The cleaner next step is either (1) formalize a route/path manifest so these retained groups become explicit compiler objects, or (2) implement an analytic/fused force path for the cheapest chemistry/radial scalar descriptors before scaling benchmark size.


# Stage 64: Path Manifest Artifact For Auditable rTECE Routes

Stage 64 addresses the review concern that current rTECE variants are still mostly Boolean feature switches and benchmark labels, not a true renormalization/distillation compiler. The next clean step is to make every student route produce an explicit path manifest that records which TECE groups are retained, how they are projected, where the scalar path is placed, and what implementation cost group it belongs to.

Implementation gate:

- Added `rtece_path_manifest(config, ...)` with schema `rtece_path_manifest.v1`.
- The manifest records a stable `manifest_hash`, config payload, route contract, moment specs, scalar path specs, retained/deleted TECE groups, and `compiler_status`.
- Moment specs include `ell`, chemistry basis, and radial projection, for example `moment.l1.vector` with `fixed_two_shell_mean` for `rtece_cavity_radial_edge_sketch14`.
- Scalar path specs include stable path ids, placement (`atomic` or `edge`), inputs, contraction type, radial projection, cavity flag, and cost group.
- Checkpoints now save `tece_path_manifest` alongside `tece_route`; load metadata reconstructs it for older checkpoints.
- Pareto summary rows now include the full manifest plus `tece_path_manifest_hash`; markdown tables print the manifest hash next to the TECE route.
- `tace.models` formally exports `rtece_path_manifest`, so this is part of the normal model package API rather than a benchmark-local helper.

Stage-64 interpretation against TECE/TACE:

- This is not yet a full compiler: it does not infer teacher path sensitivities, solve projection Gram systems, perform Schur-complement downfolding, or search the route space.
- It is the necessary compiler substrate. From this point, two students with the same human-readable variant but different radial projection, chemistry basis, force backend, graph semantics, or edge-state lifetime can be distinguished by manifest hash and path ids.
- The route is now closer to the TECE/TACE design requirement that simplifications be expressed as deleting, projecting, scalarizing, or downfolding explicit semantic groups rather than as loose hyperparameter names.
- Next priority should use this manifest in one of two concrete ways: (1) a small route registry / path-spec compiler that instantiates configs from manifest-like specs, or (2) a benchmark summarizer that groups and compares Pareto points by manifest hash and retained/deleted path groups, so architecture search is no longer variant-string driven.

# Stage 65: Manifest-Grouped Pareto Summary

Stage 65 follows the Stage 64 path-manifest substrate and addresses the same review concern from the analysis side: benchmark comparison must stop treating a human-readable variant string as the architecture identity. A route is now grouped by `tece_path_manifest_hash`, with retained TECE groups and scalar path ids displayed next to the best measured throughput and force errors.

Implementation gate:

- Added `manifest_group_rows(rows)` to `summarize_tece_distill.py`. It groups benchmark rows by manifest hash, records the semantic tier, descriptor family, variants sharing the manifest, retained/deleted TECE groups, scalar path ids, force modes, graph backends, best atoms/s, and best DFT/teacher force MAE.
- Markdown summaries now include a `Manifest Groups` section. The main student and Pareto-front rows still exist, but the report now also exposes the TECE path object being compared.
- JSON summaries now include `manifest_groups`, so later route-search tooling can consume the same architecture-level rows without re-parsing markdown.
- Regression tests cover grouping multiple benchmark rows under one radial-cavity manifest and rendering the manifest-group table for a species-basis route.

Stage-65 interpretation against TECE/TACE:

- This converts the experimental comparison unit from `variant` to a stable TECE path manifest. That is required for a systematic Pareto curve because the same route can appear under multiple labels, seeds, graph backends, or force backends.
- It does not yet implement teacher path sensitivity, POD/SVD radial basis selection, Schur-complement downfolding, or an executable path-spec compiler. It makes those next steps auditable by giving every measured point a stable semantic identity.
- This directly supports the acceptance standard: future Pareto rows can now be filtered or grouped by deleted/retained semantic groups, scalar path families, graph semantics, and force realization, instead of by loose naming conventions.

Next priority:

1. Promote the manifest schema into a small route registry/path-spec constructor, so selected scalar paths instantiate model configs rather than being inferred from variant names.
2. Use the manifest groups in the next benchmark report to compare actual architecture-level Pareto points; do not scale another autograd cavity-edge sweep unless its path group has an analytic/fused force route.
3. Return to algorithmic closure after the registry: teacher-conditioned radial/species path selection or a lightweight projection-error diagnostic, because those are the next missing pieces of true renormalization/distillation.

# Stage 66: Route Registry And Manifest Reconstruction

Stage 66 takes the next small compiler-substrate step after manifest-grouped summaries. The code no longer treats `build_rtece_config()` as a long variant-name branch. Instead, the ordered variant set is represented by a registry-backed config table, and the public API can expose route/path metadata for each deployable rTECE architecture.

Implementation gate:

- Added `available_rtece_variants()` as the stable ordered list of registered rTECE scalar architectures.
- Reworked `build_rtece_config(variant)` to instantiate from `_RTECE_VARIANT_CONFIG_KWARGS` rather than from hardcoded `if` branches. This keeps current behavior but makes the design space explicitly enumerable.
- Added `build_rtece_config_from_manifest(manifest)`, which reconstructs the architecture-level `RTECEScalarConfig` from a `rtece_path_manifest.v1` payload and preserves the manifest hash for unshifted/no-E0 architecture configs.
- Added `rtece_variant_registry()`, returning per-variant semantic tier, descriptor family, retained/deleted TECE groups, moment ids, scalar path ids, cost groups, config payload, and manifest hash.
- Exported the registry APIs from both `tace.models.rtece_scalar` and the formal `tace.models` package entrypoint; the historical benchmark shim receives them through the existing wildcard import.

Stage-66 interpretation against TECE/TACE:

- This is still not a teacher-aware renormalization compiler: it does not choose paths from teacher covariance, solve projection Gram systems, or downfold deleted groups.
- It removes another layer of naming-driven architecture logic. The current T3/T4 design space is now enumerable as registered path-manifest architectures, which is the minimum structure needed before adding path-level selection, sensitivity ranking, or hardware-cost-conditioned search.
- The registry also makes benchmark reports auditable: a manifest group can now be traced back to an exported architecture spec and rebuilt into the same config, rather than existing only as a row label.

Next priority:

1. Introduce a real path-spec constructor for the scalar endpoint, where selected `ScalarPathSpec` ids drive descriptor dimensions and feature assembly directly instead of being collapsed back into Boolean flags.
2. After that, add the first projection-error diagnostic: compare a route's retained descriptor covariance/teacher-force sensitivity against deleted path groups on a bounded OC20NEB subset.
3. Keep GPU benchmark scaling focused on registered routes with analytic/fused force backends; avoid large autograd edge-sketch sweeps until the path group has a viable inference backend.

# Stage 67: Atomic Scalar Path-Id Constructor

Stage 67 implements the first executable path-spec constructor rather than another variant-string alias. The scope is deliberately atomic scalar paths only: these paths already have exact descriptor dimensions and feature-order semantics in the current rTECE scalar endpoint, while edge/cavity paths still need their internal basis functions split before they can honestly be selected one-by-one.

Implementation gate:

- Added `RTECEScalarConfig.scalar_path_ids`, used only when a route is constructed from explicit scalar path ids. Legacy registered variants keep `scalar_path_ids=None` and preserve their existing behavior.
- Added `build_rtece_config_from_path_ids(...)`, currently accepting atomic scalar path ids: `atomic.radial_density`, `atomic.element_density`, `atomic.species_basis_density`, `atomic.density_square`, `atomic.vector_norm`, and `atomic.quadrupole_norm`.
- `descriptor_dim(config)` now sums selected path dimensions when `scalar_path_ids` is present.
- `atomic_scalar_descriptors()` and `density_scalar_descriptors()` now assemble descriptors in the selected path-id order, so path ids drive actual feature layout, not only metadata.
- `rtece_path_manifest()` filters scalar path specs to the selected path list and stores `scalar_path_ids` in the manifest config payload; `build_rtece_config_from_manifest()` reconstructs this selected-path config.
- Packed/fused element-density descriptor paths now reject custom path-id orders unless they match the canonical `(atomic.radial_density, atomic.element_density)` layout, preventing silent descriptor-order mismatches.
- The constructor is exported from the formal `tace.models` entrypoint and the benchmark shim.

Stage-67 interpretation against TECE/TACE:

- This is the first point where selected TECE scalar path ids control the real deployed descriptor dimension and feature assembly order. That moves the code beyond Boolean feature switches for the atomic scalar endpoint.
- It is still not the full route compiler. Edge relational paths are not yet individually selectable because `edge_relational_sketches()` still emits a bundled sketch vector, and teacher-conditioned path selection/downfolding is not implemented.
- The clean next step is to split edge/cavity sketch construction into named basis functions so edge scalar path ids can drive descriptor assembly the same way atomic path ids now do. After that, projection-error diagnostics can compare retained/deleted path groups directly.

Next priority:

1. Factor `edge_relational_sketches()` into stable named edge basis functions and make selected edge scalar path ids control their columns.
2. Add a bounded projection-error diagnostic over registered/path-id routes before scaling more training, because the acceptance criterion requires separating projection error from distillation/optimization error.
3. Keep backend guards explicit: any fused or analytic backend with fixed descriptor layout must reject incompatible path-id order.

# Stage 68: Named Non-Radial Edge Path Selection

Stage 68 extends the Stage 67 path-id constructor from atomic scalar paths to the first executable edge scalar paths. The goal is still compiler substrate, not a new training claim: selected edge path ids now control real descriptor columns for the non-radial cavity/full-moment edge sketch family.

Implementation gate:

- Added supported edge path dimensions for `edge.full_moment.vector_dot`, `edge.cavity.vector_dot`, `edge.cavity.quadrupole_frobenius`, and `edge.direct.radial`.
- `build_rtece_config_from_path_ids(...)` now accepts these non-radial edge path ids, sets `use_atomic_moments`, `use_cavity_edge_sketches`, and the true `num_edge_sketches` from the selected path dimensions, and rejects unsupported radial-cross edge ids.
- `descriptor_dim(config)` counts selected edge paths directly; `edge.direct.radial` contributes two scalar columns while vector/quadrupole edge contractions contribute one column each.
- `edge_relational_sketches()` now has a selected-path branch that builds named edge values and concatenates only the requested edge path columns, while preserving the legacy bundled 8-column sketch output for registered variants.
- Regression tests verify that a selected `edge.cavity.vector_dot` route produces one edge descriptor column equal to the first column of legacy `rtece_cavity_edge_sketch8`, and that selected edge manifests reconstruct to the same config/hash.

Stage-68 interpretation against TECE/TACE:

- This is a real architecture-projection step: deleting edge paths now reduces the actual deployed descriptor dimension and head input, rather than only masking a label or changing metadata.
- The edge path semantics are still limited to non-radial mean-projected sketches. The Stage 62 radial/cross-radial cavity paths remain bundled and are intentionally rejected by the path-id constructor until their basis functions are split with clear dimensions.
- The result closes another review gap: edge relational scalar paths can now be compared as explicit retained/deleted TECE groups in manifest/Pareto reports.

Next priority:

1. Add a bounded projection-error diagnostic over selected path-id routes, because the compiler now has enough atomic and non-radial edge path granularity to measure what is lost when a path group is deleted.
2. Separately split radial-cavity cross paths into named basis functions before allowing `edge.cavity.vector_cross_radial_dot` or related ids in `build_rtece_config_from_path_ids(...)`.
3. Do not spend GPU on large autograd edge sweeps until a route's selected paths have either a compact autograd cost profile or an analytic/fused inference path.

# Stage 69: Descriptor Projection-Error Diagnostic

Stage 69 adds the first bounded diagnostic for separating architecture projection error from later distillation or optimization error. It does not train a student. It asks whether the descriptors retained by a candidate path-id route can linearly reconstruct the descriptors of a richer reference route on a bounded set of structures.

Implementation gate:

- Added `benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py`.
- Added `projection_residual_metrics(source, target)`, which solves a ridge-regularized least-squares projection from candidate descriptors to reference descriptors and reports Frobenius residual, target norm, relative residual, source/target dimension, and sample count.
- Added `make_projection_diagnostic_row(...)`, which computes rTECE descriptors for candidate/reference configs over supplied graphs and records manifest hashes, selected path ids, deleted path ids, dimensions, graph count, and projection residual metrics.
- Added a CLI skeleton that reads a bounded extxyz subset through the existing `atoms_to_rtece_graph()` path, accepts a reference path-id list and multiple `name:path_id,path_id` candidates, and writes JSON with schema `rtece_projection_diagnostic.v1`.
- Regression tests cover an exactly spanned synthetic descriptor target and a real rTECE graph comparison where deleting `edge.direct.radial` is reported as a deleted scalar path.

Stage-69 interpretation against TECE/TACE:

- This begins the required projection/distillation separation: before spending training budget, a candidate route can be checked for descriptor-space information loss relative to a richer path set.
- The current metric is descriptor-space only. It is not yet teacher-force sensitivity, Hessian-vector distillation, or Schur-complement downfolding. Those require teacher/model gradients or cached teacher path activations.
- The diagnostic is still useful for path search because the route compiler now has atomic and non-radial edge path granularity; deleted path ids can be measured instead of guessed from variant names.

Next priority:

1. Run this diagnostic on a small OC20NEB subset for candidate path-id routes such as `radial_density`, `radial_density+edge.cavity.vector_dot`, and `radial_density+edge.cavity.vector_dot+edge.direct.radial`.
2. Add optional teacher-force or teacher-energy sensitivity weighting to the projection residual once a teacher checkpoint/path cache is available in the same workflow.
3. Feed the diagnostic rows into the manifest-group Pareto summary so projection residual, MAE/RMSE, and atoms/s appear together for the same route hash.

# Stage 70: Label-Independent Manifest Hash And First Projection Run

Stage 70 fixes a manifest identity issue exposed by the first Stage 69 diagnostic run. Two routes with identical selected path ids and architecture parameters but different human-readable `variant` labels produced different `manifest_hash` values. That contradicted the Stage 65 requirement that manifest grouping compare architecture routes rather than labels.

Implementation gate:

- `rtece_path_manifest()` now preserves `config.variant` in the manifest payload for provenance, but removes it from the hash input. The `manifest_hash` is therefore an architecture/path identity rather than a run label identity.
- Added a regression test showing that two path-id configs with different variant labels but identical architecture/path specs share the same manifest hash.
- Re-ran the Stage 69 projection diagnostic on 8 structures from `mixed_train_tw0.75.extxyz` with reference path ids `atomic.radial_density, edge.cavity.vector_dot, edge.direct.radial`.

First bounded diagnostic result on 2026-07-18:

| candidate | dim | deleted paths | relative descriptor residual | interpretation |
|---|---:|---|---:|---|
| `radial` | 3 | `edge.cavity.vector_dot`, `edge.direct.radial` | 0.1273 | Deleting both non-radial edge paths leaves a visible descriptor-space projection gap. |
| `radial_cavity_vec` | 4 | `edge.direct.radial` | 0.00542 | Most of this reference descriptor space is linearly recoverable once the cavity vector-dot path is retained. |
| `radial_cavity_vec_direct` | 6 | none | 2.1e-13 | Sanity check; identical path route reconstructs itself and now shares the reference manifest hash. |

Stage-70 interpretation against TECE/TACE:

- This directly improves the Pareto/compiler substrate: route grouping is now stable under label changes, which is required for systematic architecture search.
- The first projection diagnostic suggests that `edge.cavity.vector_dot` captures much more of the selected reference descriptor subspace than direct radial edge channels on this tiny subset. This is not yet an accuracy or force-sensitivity claim, but it gives a rational next candidate order for cheap path-id training/benchmark runs.
- Because the diagnostic is descriptor-space only, it should be used to prioritize candidates, not to replace DFT/teacher MAE/RMSE or physical relax/dimer tests.

Next priority:

1. Feed projection residual into the summary JSON/Markdown next to manifest groups so descriptor projection, MAE/RMSE, and atoms/s can be read together.
2. Run a tiny train/benchmark smoke for `radial` vs `radial_cavity_vec` path-id routes only if the training CLI can instantiate path-id configs without reverting to variant names.
3. Add teacher-force sensitivity weighting after the unweighted descriptor residual is integrated into the route report.

# Stage 71: Projection Residuals In Pareto Summary

Stage 71 closes the Stage-70 reporting loop by attaching bounded descriptor projection diagnostics to the same manifest groups used by the TECE/TACE Pareto summary. This is a compiler-methodology change: it does not train a new model or change a checkpoint, but it makes architecture projection error visible next to MAE/RMSE and throughput for the same path-manifest hash.

Implementation gate:

- `summarize_tece_distill.py` now accepts repeated `--projection-diagnostic` JSON files emitted by `analyze_rtece_projection_error.py`.
- Summary JSON now preserves raw `projection_diagnostics` rows and enriches each `manifest_groups` row with `projection_relative_residual`, `projection_deleted_scalar_path_ids`, `projection_num_samples`, and `projection_candidate` when the diagnostic `candidate_manifest_hash` matches the group hash.
- Markdown manifest-group tables now include projection residual, deleted projection paths, and projection sample count next to best atoms/s, DFT force MAE, and teacher force MAE.
- If multiple projection rows target the same manifest hash, the summary keeps the lowest relative residual and correctly treats `0.0` as a valid best residual, not as a missing value.
- Direct execution of `summarize_tece_distill.py --help` now follows the same repository-root import path convention as the other rTECE benchmark scripts.

Verification on 2026-07-18:

- `python -m py_compile benchmarks/oc20neb_tace_mace/summarize_tece_distill.py` passed.
- `python benchmarks/oc20neb_tace_mace/summarize_tece_distill.py --help` passed and exposes `--projection-diagnostic`.
- Focused summary/projection tests passed: 13 passed, 94 deselected.
- Generated `runs/oc20neb_tace_mace/rtece-stage71-combined-summary/direct_active_projection_summary.{json,md}` from the Stage-39 direct-active front and Stage-69 projection diagnostic.

Real-artifact check:

- The generated Stage-71 manifest groups all report `projection_relative_residual = null` because the Stage-39 direct-active Pareto rows are element-density routes, while the Stage-69/70 diagnostic candidates are path-id routes over `atomic.radial_density`, `edge.cavity.vector_dot`, and `edge.direct.radial`.
- This is a useful negative integration result: the summary join works by manifest hash, but the current projection diagnostic has not yet been run for the already trained element-density front, and the diagnostic path-id candidates have not yet been trained/benchmarked.

Stage-71 interpretation against TECE/TACE:

- This implements the acceptance-standard requirement to separate projection error from training/distillation error in the report layer. A candidate route can now be read as: retained scalar paths, deleted path groups, descriptor-space residual, best observed DFT/teacher force error, and hardware throughput.
- The join key is the label-independent path manifest hash from Stage 70, so the report compares architecture routes rather than arbitrary run names.
- The residual is still unweighted descriptor-space reconstruction error. It is useful for path prioritization, but it is not yet teacher-force sensitivity, Hessian-aware Sobolev error, or Schur-complement downfolding.

Next priority:

1. Make the training CLI instantiate selected path-id configs, then run tiny `radial` versus `radial_cavity_vec` train/benchmark smoke tests so the Stage-69 projection residuals and observed MAE/throughput share manifest hashes.
2. If path-id training is blocked by analytic/fused backend layout constraints, first implement an explicit autograd-only path-id training/evaluation contract and keep fused backends guarded.
3. Add teacher-force sensitivity weighting to the projection diagnostic only after the unweighted path-id route report joins cleanly with trained benchmark rows.

# Stage 72: Path-Id Route Training And Projection/Benchmark Join

Stage 72 implements the next Stage-71 priority: selected scalar path-id routes can now be trained and benchmarked as real rTECE models, then joined with descriptor projection diagnostics in the summary. This is still a tiny CPU smoke, not a production Pareto claim, but it closes the route identity loop that was missing after Stage 70.

Implementation gate:

- `train_rtece_scalar.py` now accepts `--scalar-path-ids`. When present, `--variant` is treated as a route label and the config is built through `build_rtece_config_from_path_ids(...)`; otherwise registered variants keep the old path.
- Training summaries now write `scalar_path_ids`, and checkpoints already preserve the selected path-id config.
- `rtece_scalar_matrix.sbatch` forwards optional `SCALAR_PATH_IDS` through a bash array without using forbidden `sbatch --export`.
- `benchmark_rtece_scalar.py` now writes both a pure architecture/path manifest (`tece_architecture_path_manifest`) and a runtime manifest (`tece_path_manifest`). This separates descriptor projection identity from force/backend provenance.
- `summarize_tece_distill.py` groups benchmark rows by architecture/path manifest and falls back to exact `candidate_scalar_path_ids` matching for projection diagnostics whose descriptor-only hash differs because head capacity or fitted atomic energies changed.

Verification on 2026-07-18:

- Path-id training/benchmark/summary closure tests passed: 6 passed, 107 deselected.
- Tiny real-data CPU training smoke completed for `radial` with `atomic.radial_density` and for `radial_cavity_vec` with `atomic.radial_density,edge.cavity.vector_dot`, using 2 train configs, 2 steps, `num_radial=3`, hidden `[4]`, and fitted per-element atomic energies.
- Generated `runs/oc20neb_tace_mace/rtece-stage72-path-id-smoke/path_id_projection_summary.{json,md}` joining projection residuals to measured DFT/teacher benchmark rows.

Tiny CPU smoke result on 8 validation configs:

| route | scalar paths | projection residual | deleted projection paths | CPU atoms/s | DFT F MAE | teacher F MAE | interpretation |
|---|---|---:|---|---:|---:|---:|---|
| `radial` | `atomic.radial_density` | 0.1273 | `edge.cavity.vector_dot`, `edge.direct.radial` | 285.3 | 118.71 | 119.08 | Very lossy descriptor projection and poor tiny-smoke force error. |
| `radial_cavity_vec` | `atomic.radial_density`, `edge.cavity.vector_dot` | 0.00542 | `edge.direct.radial` | 133.1 | 63.67 | 60.30 | Much lower descriptor residual and better tiny-smoke force error, at higher CPU autograd cost. |

Stage-72 interpretation against TECE/TACE:

- This is the first working path-id route closure: selected TECE scalar paths drive training config, checkpoint metadata, benchmark artifact identity, projection residual join, and summary tables.
- The direction of the tiny smoke is consistent with Stage 70: retaining `edge.cavity.vector_dot` greatly lowers descriptor projection residual and also improves DFT/teacher force MAE in a minimal training run. This is not yet statistical proof because the run is tiny and CPU/autograd-only.
- The lower throughput of `radial_cavity_vec` is expected: this path currently uses unfused cavity moment/edge autograd machinery. It should not be compared with the fused element-density front until it has an analytic or fused evaluator, or until the goal is explicitly projection-quality rather than high-throughput.
- The manifest lesson is important: full deployable model hash and descriptor projection identity are related but not identical. Hidden head capacity, energy reference fitting, force backend, and graph backend must remain explicit rather than silently changing the projection join.

Next priority:

1. Run a small GPU path-id training/benchmark matrix for `radial`, `radial_cavity_vec`, and `radial_cavity_vec_direct` if the autograd edge path fits, using the same projection diagnostic reference so the joined summary becomes a real early Pareto slice.
2. If `radial_cavity_vec` is too slow under autograd, do not widen it blindly; first decide whether to implement a cheap analytic/fused evaluator for that selected edge path or keep it as a projection diagnostic candidate only.
3. Add teacher-force sensitivity weighting after the unweighted path-id route report is reproduced at non-tiny scale.

# Stage 73: Small GPU Path-Id Route Matrix And Negative Projection Check

Stage 73 runs the first non-tiny GPU path-id route matrix after Stage 72 closed the training/benchmark/projection identity loop. The goal was not to maximize accuracy against the largest teacher, but to test whether the TECE path-retention order suggested by the unweighted projection diagnostic becomes a reasonable early Pareto slice when trained and benchmarked as deployable rTECE models.

Implementation and submission gate:

- `submit_rtece_scalar_matrix.py` now accepts `--scalar-path-ids` and writes `SCALAR_PATH_IDS=...` inside the generated wrapper, so path-id route jobs can be submitted without changing the canonical matrix sbatch script.
- The generated rTECE benchmark and matrix wrappers now describe the SAI rule as self-contained wrapper parameter passing. Wrapper bodies use ordinary `export VAR=value`, and the submission command remains plain `sbatch wrapper.sbatch`.
- Regression tests check that generated wrapper text contains no command-line Slurm environment export flag, no `--mem`, and no `--cpus-per-task`, while preserving required GPU/QOS lines.
- Stage-73 wrapper grep over `runs/oc20neb_tace_mace/rtece-stage73-path-id-gpu/submit` found no forbidden Slurm export, memory, or CPU-per-task directives before submission.

Stage-73 jobs on 2026-07-18:

| route | job id | scalar paths | train/valid configs | benchmark configs | max steps | force mode |
|---|---:|---|---:|---:|---:|---|
| `radial` | 679965 | `atomic.radial_density` | 64/64 | 256 | 200 | autograd |
| `radial_cavity_vec` | 679966 | `atomic.radial_density`, `edge.cavity.vector_dot` | 64/64 | 256 | 200 | autograd |
| `radial_cavity_vec_direct` | 679967 | `atomic.radial_density`, `edge.cavity.vector_dot`, `edge.direct.radial` | 64/64 | 256 | 200 | autograd |

Small GPU matrix result, joined with the Stage-69/70 projection diagnostic:

| route | projection residual | deleted projection paths | GPU atoms/s | DFT F MAE | DFT F RMSE | teacher F MAE | teacher F RMSE | best step | best valid loss |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| `radial` | 0.1273 | `edge.cavity.vector_dot`, `edge.direct.radial` | 2.96M | 29.45 | 120.02 | 34.25 | 119.43 | 200 | 0.7752 |
| `radial_cavity_vec` | 0.00542 | `edge.direct.radial` | 1.08M | 35.16 | 122.58 | 38.53 | 122.24 | 200 | 0.6983 |
| `radial_cavity_vec_direct` | 2.1e-13 | none | 1.08M | 34.57 | 121.61 | 39.20 | 121.38 | 100 | 0.7391 |

Stage-73 interpretation against TECE/TACE and the review document:

- The unweighted descriptor projection residual is not yet a reliable force-accuracy proxy. It correctly measures descriptor subspace deletion, but on this 64-config/200-step GPU slice the route with the largest residual (`radial`) is both fastest and best on DFT/teacher force MAE. This is negative evidence against using raw descriptor reconstruction as the only renormalization criterion.
- Adding `edge.cavity.vector_dot` lowers descriptor residual by about 23.5x relative to radial-only, but current unfused autograd edge evaluation costs about 2.7x throughput and does not improve force MAE in this run. The next algorithmic question is whether this path needs teacher-force/Sobolev weighting, better distillation, or dataset stratification before its extra descriptor information becomes useful.
- Adding `edge.direct.radial` reduces the residual to numerical zero relative to the reference descriptor set, but gives almost no throughput difference and no clear force gain over `radial_cavity_vec`. This supports the Stage-70 priority order: do not expand direct edge paths blindly.
- This result reinforces the review conclusion that the branch is still a scalar student and performance testbed, not yet a full TECE renormalization/distillation compiler. The next proof step must connect deleted paths to energy/force/virial sensitivity, not just to descriptor least-squares residual.
- The observed path-id autograd throughput is useful but not the final high-throughput target. Review-aligned backend work should eventually replace slow graph/edge autograd pieces with a graph ABI carrying PBC edge vectors and fused edge-gradient force/virial, but kernel work is lower priority than making the path deletion metric physically meaningful.

Next priority:

1. Upgrade the projection diagnostic from unweighted descriptor residual to teacher-force or energy/force Sobolev-weighted residual on the same path-id routes, so deleted TECE paths are ranked by physical sensitivity rather than raw feature reconstruction.
2. Keep `radial` as the current high-throughput endpoint baseline and `radial_cavity_vec` as the first semantic edge-path candidate, but do not widen edge autograd routes until weighted projection or distillation shows force-error benefit.
3. Start a bounded physical-generalization harness after the review-critical semantic bugs remain under control: dimer scans for smooth E/F and rattle+relax RMSD stratified by element/adsorbate class, with special attention to non-metal C/N adsorbates observed by the user.
4. Plan the graph/geometry ABI change from ASE-style graph construction toward TACE-compatible PyG/matscipy/NVIDIA-op paths only as part of the PBC edge-vector/force/virial closure, not as an isolated graph-build micro-optimization.

# Stage 74: Sample-Weighted Projection Diagnostic Substrate

Stage 74 implements the first weighted projection substrate requested by the Stage-73 negative check. The goal is to move from raw descriptor least-squares residual toward a physically meaningful TECE deletion metric. This stage still does not implement full teacher-force sensitivity, Hessian-vector distillation, or virial-aware Sobolev projection; it adds the required sample-weighted Gram machinery and runs a bounded force-norm weighted probe.

Implementation gate:

- `projection_residual_metrics(...)` now accepts optional per-sample weights and computes weighted least-squares by scaling source and target descriptors with `sqrt(weight)`. The reported residual and target norms use the same weighted metric.
- The projection diagnostic row builder accepts `sample_weights`, and the CLI accepts `--sample-weight-json` containing either a list of weights or `{"weight_source": ..., "sample_weights": [...]}`.
- Projection diagnostic payloads record `weighted`, `sample_weight_json`, and `sample_weight_source`. The Pareto summary loader preserves that metadata, and manifest groups expose `projection_weighted` plus `projection_sample_weight_source` in summary JSON.
- Regression tests cover weighted residual behavior, JSON weight loading, row-level weight passthrough, and summary metadata propagation.

Bounded force-norm weighted probe on the same 8-config/419-atom Stage-69 subset:

| weighting | route | projection residual | interpretation |
|---|---|---:|---|
| raw descriptor | `radial` | 0.1273 | Stage-69 baseline. |
| DFT force-norm mean1 | `radial` | 0.1660 | Force-magnitude weighting makes radial-only deletion look worse. |
| teacher force-norm mean1 | `radial` | 0.1622 | Same direction as DFT force-norm weighting. |
| raw descriptor | `radial_cavity_vec` | 0.00542 | Stage-69 baseline. |
| DFT force-norm mean1 | `radial_cavity_vec` | 0.00582 | Slightly larger, same qualitative rank. |
| teacher force-norm mean1 | `radial_cavity_vec` | 0.00578 | Slightly larger, same qualitative rank. |
| teacher force-norm mean1 | `radial_cavity_vec_direct` | 2.85e-13 | Full reference route remains numerical zero. |

Joined Stage-74 teacher-force-norm summary with Stage-73 GPU benchmarks:

| route | weighted residual | weight source | GPU atoms/s | DFT F MAE | teacher F MAE |
|---|---:|---|---:|---:|---:|
| `radial` | 0.1622 | `teacher_force_norm_per_atom_mean1` | 2.96M | 29.45 | 34.25 |
| `radial_cavity_vec` | 0.00578 | `teacher_force_norm_per_atom_mean1` | 1.08M | 35.16 | 38.53 |
| `radial_cavity_vec_direct` | 2.85e-13 | `teacher_force_norm_per_atom_mean1` | 1.08M | 34.57 | 39.20 |

Stage-74 interpretation against TECE/TACE and the review document:

- The weighting substrate is necessary and now exists, but simple per-atom force magnitude is not sufficient. It preserves the descriptor residual ordering and even increases the gap between `radial` and edge-retaining routes, while the Stage-73 force MAE still favors `radial` on the small GPU slice.
- Therefore the missing ingredient is not merely sample importance by force size. The next TECE-renormalization metric must estimate teacher/model sensitivity of deleted paths to E/F/V outputs, for example Jacobian-weighted projection, Sobolev Gram blocks, finite-difference teacher response, HVP, or cached teacher semantic path activations.
- This result is useful negative evidence: it prevents us from declaring `edge.cavity.vector_dot` Pareto-useful solely because raw or force-magnitude-weighted descriptor residual is small. Any edge path retained for the high-throughput front must also show force-error or physical-generalization benefit under the same training and benchmark protocol.
- The graph-build question should be handled inside the review-mandated PBC/edge-vector ABI. ASE graph construction is too slow for final high-throughput MD, but changing graph builders alone will not fix the Stage-73 algorithmic mismatch between descriptor residual and force MAE.

Next priority:

1. Add a real teacher-sensitivity weight generator for projection diagnostics. Minimum viable version: compute per-atom or per-descriptor sensitivity from teacher force residual/Jacobian probes on the same bounded subset, then re-run the path-id route residual table.
2. Keep the current `radial` route as the endpoint baseline for throughput/accuracy. Treat `radial_cavity_vec` as a semantic candidate requiring proof through weighted sensitivity and physical tests, not as an automatic Pareto improvement.
3. After the sensitivity metric is available, run dimer scan and rattle+relax tests on representative metal and non-metal adsorbates to check whether the metric predicts the C/N relaxation RMSD failure mode observed by the user.

# Stage 75: rTECE Head-Jacobian Projection Weights

Stage 75 adds the first Jacobian-style projection weight generator. This is a bridge toward the review document Sobolev/Gram/downfolding requirement, not the final teacher-TECE sensitivity metric. Because no original TACE/TECE teacher semantic path cache is available in the current workflow, the bounded implementation uses a trained rTECE reference route as a surrogate: it measures the L2 norm of the rTECE energy head gradient with respect to each atom descriptor row.

Implementation gate:

- Added `make_rtece_projection_weights.py`.
- The script loads an rTECE checkpoint, rebuilds bounded rTECE graphs from extxyz, computes `rtece_descriptors(...)`, evaluates the checkpoint energy head, and writes mean-normalized `rtece_head_jacobian_l2` sample weights.
- The emitted JSON uses the Stage-74 `rtece_projection_sample_weights.v1` schema and is directly consumable by `analyze_rtece_projection_error.py --sample-weight-json`.
- Tests verify that a linear energy head with descriptor weights `[3, 4]` emits per-atom raw Jacobian weights of `5`, and that JSON writing preserves the sample-weight schema.

Bounded run on the same 8-config/419-atom Stage-69 subset:

| source checkpoint | route | raw sensitivity range | normalized weight range |
|---|---|---:|---:|
| Stage-73 `radial_cavity_vec_direct` best checkpoint | full path-id reference | 0.0349-0.0484 | 0.799-1.107 |

Head-Jacobian weighted projection result:

| weighting | route | projection residual | Stage-73 GPU atoms/s | DFT F MAE | teacher F MAE |
|---|---|---:|---:|---:|---:|
| raw descriptor | `radial` | 0.1273 | 2.96M | 29.45 | 34.25 |
| rTECE head Jacobian | `radial` | 0.1257 | 2.96M | 29.45 | 34.25 |
| raw descriptor | `radial_cavity_vec` | 0.00542 | 1.08M | 35.16 | 38.53 |
| rTECE head Jacobian | `radial_cavity_vec` | 0.00531 | 1.08M | 35.16 | 38.53 |
| rTECE head Jacobian | `radial_cavity_vec_direct` | 1.94e-13 | 1.08M | 34.57 | 39.20 |

Stage-75 interpretation against TECE/TACE and the review document:

- The code path for Jacobian-weighted projection is now real and reproducible, but the surrogate rTECE head is too low-capacity or too smooth on this subset to provide strong sample discrimination. Its weights span only about 0.80-1.11 after normalization, so the resulting residuals are nearly identical to the raw descriptor residuals.
- This is a useful negative result. It shows that merely using the current student/reference rTECE energy head as a sensitivity proxy does not resolve the Stage-73 mismatch where `radial` has worse descriptor residual but better force MAE and throughput.
- The next sensitivity metric must be closer to the original teacher and to force/Sobolev behavior: teacher force residual stratification, finite-difference teacher response, teacher-force Jacobian probes, HVP, or cached TACE/TECE semantic path activations. A student-head Jacobian is acceptable as plumbing, not as the final renormalization criterion.
- Algorithmically, the current Pareto baseline remains `radial` for the path-id slice. `edge.cavity.vector_dot` remains a semantic candidate, but it still lacks evidence that its descriptor advantage becomes force accuracy or physical-generalization advantage under the current training setup.

Next priority:

1. Add a teacher/label-side sensitivity generator that uses existing `teacher_forces`/`dft_forces` and model errors to emphasize atoms/configurations where the low-rank route actually fails, instead of relying on the smooth rTECE head alone.
2. Use that residual-aware weighting to rerun the same path-id projection table and check whether it correlates with C/N adsorbate relax RMSD failures.
3. Only after this sensitivity metric starts predicting accuracy/generalization should we spend GPU on larger edge-path training or graph-backend acceleration.

# Stage 76: Force-Residual Weighted Projection Probe

Stage 76 extends the Stage-75 weight generator from surrogate head sensitivity to label/model-error-side sensitivity. The purpose is to emphasize atoms where the current high-throughput low-rank endpoint actually fails against teacher forces, then ask whether deleted TECE paths explain those failures in the projection metric.

Implementation gate:

- `make_rtece_projection_weights.py` now supports `--weight-mode rtece_force_residual_l2` in addition to `rtece_head_jacobian_l2`.
- The residual mode loads a target force array from extxyz, currently `teacher_forces` or `dft_forces`, predicts forces with an rTECE checkpoint, computes per-atom `||F_model - F_target||_2`, and emits mean-normalized projection sample weights.
- Tests cover the pure force-residual payload contract and mean-normalization behavior.

Bounded run on the same 8-config/419-atom Stage-69 subset:

| source checkpoint | target force array | raw residual range | normalized weight range |
|---|---|---:|---:|
| Stage-73 `radial` best checkpoint | `teacher_forces` | 0.00193-7.23 | 0.0120-45.08 |

Force-residual weighted projection result joined with Stage-73 GPU benchmarks:

| route | weighted residual | weight source | Stage-73 GPU atoms/s | DFT F MAE | teacher F MAE |
|---|---:|---|---:|---:|---:|
| `radial` | 0.1622 | `rtece_force_residual_l2:teacher_forces` | 2.96M | 29.45 | 34.25 |
| `radial_cavity_vec` | 0.00582 | `rtece_force_residual_l2:teacher_forces` | 1.08M | 35.16 | 38.53 |
| `radial_cavity_vec_direct` | 2.64e-13 | `rtece_force_residual_l2:teacher_forces` | 1.08M | 34.57 | 39.20 |

Stage-76 interpretation against TECE/TACE and the review document:

- Residual-aware weighting produces a much sharper sample distribution than the Stage-75 head-Jacobian weights: normalized weights span about 0.012-45.1 instead of 0.80-1.11. This is closer to a useful Sobolev-style failure metric because it focuses on atoms where the endpoint model is wrong.
- Even under this failure-focused metric, `edge.cavity.vector_dot` still greatly reduces projection residual, but the Stage-73 trained `radial_cavity_vec` model is slower and worse in force MAE. Therefore the remaining mismatch is no longer just a raw descriptor metric problem. It points to training/optimization, head capacity, dataset stratification, or physical test distribution as the next bottleneck.
- The clean theoretical route is now sharper: path deletion cost should be measured with weighted projection, but a path is only Pareto-useful after the training procedure can convert that weighted information into lower force/relax error at acceptable throughput. The current evidence says `radial` remains the path-id high-throughput baseline, while `edge.cavity.vector_dot` needs targeted validation rather than blind widening.
- This also connects to the user observation about C/N adsorbate relax RMSD: a residual-aware diagnostic can now produce per-atom weights that can be stratified by element or adsorbate class. That is the next practical way to check whether non-metal adsorbates are driving the edge-path need.

Next priority:

1. Add element/adsorbate-class stratification for projection weights and benchmark errors, starting with C/N-containing adsorbates in the OC20NEB subset.
2. Compare radial vs radial-cavity routes on the high residual-weight atoms rather than only aggregate MAE; if cavity helps there, rerun training with stratified validation/checkpoint selection.
3. If cavity still fails on the residual-focused subset, prioritize distillation loss/head capacity or physical tests before any graph-backend acceleration.

# Stage 77: Element-Stratified Force-Residual Weights

Stage 77 implements the first element/focus-group stratification layer for the weighted projection diagnostics. This follows the Stage-76 conclusion and the user observation that C/N adsorbate relax RMSD may be disproportionately bad. The implementation is deliberately conservative: `C_or_N`, `CHNO`, and `not_CHNO` are lightweight element proxies, not a true adsorbate/site annotation.

Implementation gate:

- Added `stratify_rtece_projection_weights.py`.
- The script loads a projection sample-weight JSON, reloads the matching extxyz symbols, checks weight/atom alignment, and emits JSON plus Markdown grouped by element and by focus groups.
- Focus groups are `C_or_N`, `CHNO`, and `not_CHNO`; per-element rows are also reported.
- Group rows include atom count, sample fraction, total weight fraction, mean/max weight, and each group share of the global top-weight atoms.
- Regression tests cover weight-fraction accounting, top-weight share accounting, and symbol/weight length mismatch rejection.

Bounded Stage-77 run on the same 8-config/419-atom subset, using Stage-76 radial teacher-force residual weights:

| group | atom fraction | residual-weight fraction | mean normalized weight | top-10 percent weight share |
|---|---:|---:|---:|---:|
| `CHNO` | 0.158 | 0.688 | 4.37 | 0.824 |
| `C_or_N` | 0.033 | 0.356 | 10.66 | 0.436 |
| `not_CHNO` | 0.842 | 0.312 | 0.371 | 0.176 |

Leading elements for radial teacher-force residual weights:

| element | atom fraction | residual-weight fraction | mean normalized weight | top-10 percent weight share |
|---|---:|---:|---:|---:|
| C | 0.026 | 0.246 | 9.38 | 0.300 |
| H | 0.117 | 0.213 | 1.82 | 0.238 |
| O | 0.007 | 0.118 | 16.51 | 0.149 |
| N | 0.007 | 0.110 | 15.37 | 0.136 |

Radial versus radial-cavity comparison on the same subset:

| route | residual raw mean | residual raw max | `C_or_N` weight fraction | `CHNO` weight fraction | `C_or_N` top-10 percent weight share |
|---|---:|---:|---:|---:|---:|
| `radial` | 0.160 | 7.23 | 0.356 | 0.688 | 0.436 |
| `radial_cavity_vec` | 0.174 | 7.28 | 0.331 | 0.652 | 0.427 |

Stage-77 interpretation against TECE/TACE and the review document:

- The user-observed C/N problem is supported as a real priority on this bounded slice: C and N are only 3.3 percent of atoms but carry 35.6 percent of radial teacher-force residual weight. CHNO atoms are 15.8 percent of atoms but carry 68.8 percent of residual weight.
- The cavity-vector route slightly reduces C/N and CHNO residual concentration, but its overall residual mean is worse than radial on this subset, consistent with the Stage-73 aggregate force MAE. This means the edge path is not automatically Pareto-useful just because projection residual says it carries information.
- The next clean experiment is not a wider edge-path architecture sweep. It is stratified validation/checkpoint selection or loss weighting: train/evaluate radial and radial-cavity routes while explicitly monitoring C/N or CHNO residual subsets, then see whether the cavity path improves the failure mode without sacrificing too much throughput.
- This stage also clarifies the physical-test plan: dimer scans and rattle+relax should be stratified by these element/focus groups, with C/N adsorbates pulled out instead of hidden in aggregate OC20NEB metrics.

Next priority:

1. Add stratified benchmark summaries for existing Stage-73 checkpoints: force MAE/RMSE by `C_or_N`, `CHNO`, and element, using DFT and teacher forces.
2. Use those summaries to decide whether a stratified validation/checkpoint run is justified for `radial_cavity_vec`.
3. If C/N remains bad under both routes, move to physical dimer/rattle+relax tests before changing graph backends or adding more edge paths.

# Stage 78: Element-Stratified Force Error Benchmark

Stage 78 closes the Stage-77 diagnostic loop by moving from residual-weight concentration to actual force-error concentration. This directly follows the Stage-77 next priority: summarize existing Stage-73 checkpoints by `C_or_N`, `CHNO`, and element under the same MAE/RMSE convention used by `benchmark_rtece_scalar.py`.

Implementation gate:

- Added `stratify_rtece_force_errors.py`.
- The script loads an rTECE checkpoint and extxyz configs, rebuilds the same rTECE graphs, predicts forces, loads a chosen reference force array, and emits JSON plus Markdown grouped by element and focus group.
- The reported force MAE/RMSE is component-wise in meV/A, matching the existing benchmark definition.
- Focus groups are kept identical to Stage 77: `C_or_N`, `CHNO`, and `not_CHNO`.
- Regression coverage checks the pure force-error stratification contract for element rows, focus rows, component counts, and MAE/RMSE units.

Bounded Stage-78 run on the same 8-config/419-atom subset used in Stages 76-77:

| route | target | all-atom F MAE | all-atom F RMSE | `C_or_N` F MAE | `CHNO` F MAE | `not_CHNO` F MAE |
|---|---|---:|---:|---:|---:|---:|
| `radial` | `teacher_forces` | 79.79 | 369.90 | 899.37 | 350.42 | 29.18 |
| `radial_cavity_vec` | `teacher_forces` | 86.32 | 373.03 | 904.60 | 358.12 | 35.51 |
| `radial` | `dft_forces` | 77.95 | 370.61 | 892.03 | 344.77 | 28.06 |
| `radial_cavity_vec` | `dft_forces` | 86.29 | 373.84 | 903.79 | 356.68 | 35.73 |

Leading teacher-force element errors:

| route | C F MAE | N F MAE | O F MAE | H F MAE | best non-CHNO reference |
|---|---:|---:|---:|---:|---:|
| `radial` | 764.74 | 1393.01 | 1244.17 | 138.86 | Ga 43.11 / Cd 62.75 |
| `radial_cavity_vec` | 772.48 | 1389.01 | 1264.47 | 146.49 | Ga 47.95 / Cd 63.60 |

Stage-78 interpretation against TECE/TACE and the review document:

- The C/N failure mode is now supported by actual force errors, not only by projection weights. On this bounded slice, `C_or_N` component MAE is about 899 meV/A for `radial` versus about 29 meV/A for `not_CHNO`; this is roughly a 31x separation.
- The `edge.cavity.vector_dot` path does not repair the failure. It slightly lowers the relative C/N concentration in Stage 77 only because its non-CHNO error also rises; in absolute component MAE it is worse or essentially tied for C/N and worse overall.
- Therefore the next clean TECE step is not adding more cavity-vector capacity or changing low-level graph backends. The deleted-path metric has found a chemically concentrated failure mode, but the current candidate path does not convert that information into force accuracy.
- This agrees with the review document: before kernel acceleration or wider route sweeps, the branch needs physical and semantic closure tests. The immediate target should be dimer scans and rattle+relax RMSD stratified around C/N/CHNO adsorbates, plus checking whether the error comes from dataset scarcity, missing angular chemistry, or force/energy normalization.

Next priority:

1. Build the bounded physical-generalization harness promised in the review path: C/N/O/H dimer scans over 0.5-5 covalent-radius scale, reporting E/F smoothness and force spikes.
2. Add rattle+relax comparison grouped by `C_or_N`/`CHNO` versus metal-only or `not_CHNO` structures, using final RMSD and convergence failures, not only single-step force MAE.
3. Only after the physical tests identify the missing semantic block should we consider a new TECE path beyond radial or `edge.cavity.vector_dot`.

# Stage 79: CHNO Dimer Scan Physical Probe

Stage 79 starts the physical-generalization harness requested by the review path and by the user observation about C/N adsorbate relax failures. This is not a throughput benchmark and not an absolute energy comparison because the rTECE energy reference is still a fitted atomic-energy/offset convention. The useful signal is the shape of the energy/force curve under an extreme geometry scan: finite values, smooth adjacent steps, and whether the shortest separation has a repulsive force direction.

Implementation gate:

- Added `dimer_scan_rtece.py`.
- The script constructs isolated dimers with distances from `min_scale` to `max_scale` times the ASE covalent-radius sum, defaulting to the requested 0.5-5 range.
- It loads an rTECE checkpoint through `tace.models.rtece_workflow.load_checkpoint`, builds graphs through the existing `atoms_to_rtece_graph`, predicts through `tace.models.rtece_workflow.predict`, and emits JSON plus Markdown.
- Summary rows report finite/nonfinite counts, energy range, maximum force, adjacent energy/force jumps, shortest-distance minus longest-distance energy, shortest-distance force sign, and a `short_force_repulsive` boolean. For the x-axis dimer convention, atom 0 repulsion means negative x force.
- Regression tests cover the covalent-radius distance grid and the dimer smoothness/short-range summary contract.

Bounded Stage-79 run on Stage-73 checkpoints, CPU autograd, four CHNO dimers, 12 points each:

| route | pair | short distance A | short-long dE eV | short force eV/A | short repulsive | max abs force eV/A | max dE step eV | max dF step eV/A |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `radial` | C-N | 0.735 | -0.0207 | 0.0204 | 0 | 0.0246 | 0.0140 | 0.0142 |
| `radial` | C-O | 0.710 | -0.0212 | 0.0200 | 0 | 0.0247 | 0.0134 | 0.0134 |
| `radial` | N-H | 0.510 | -0.0253 | 0.0169 | 0 | 0.0247 | 0.0101 | 0.0102 |
| `radial` | O-H | 0.485 | -0.0257 | 0.0165 | 0 | 0.0248 | 0.00950 | 0.00960 |
| `radial_cavity_vec` | C-N | 0.735 | 0.00110 | -0.00575 | 1 | 0.00879 | 0.00472 | 0.00695 |
| `radial_cavity_vec` | C-O | 0.710 | 0.00126 | -0.00552 | 1 | 0.00872 | 0.00470 | 0.00654 |
| `radial_cavity_vec` | N-H | 0.510 | 0.00215 | -0.00406 | 1 | 0.00890 | 0.00351 | 0.00501 |
| `radial_cavity_vec` | O-H | 0.485 | 0.00226 | -0.00382 | 1 | 0.00881 | 0.00343 | 0.00478 |

Stage-79 interpretation against TECE/TACE and the review document:

- Both routes are numerically finite and smooth on this bounded dimer scan. There are no NaN/Inf failures and adjacent energy/force jumps are small.
- The `radial` route has the wrong short-range sign on all four CHNO pairs: the closest distance is lower in energy than the far distance and atom 0 force points toward atom 1. This is a direct physical-generalization failure, consistent with the Stage-78 C/N/CHNO force-error concentration.
- The `radial_cavity_vec` route flips the short-range force sign to repulsive for all four pairs and reduces the energy/force jump metrics, so the cavity vector path contains some useful geometric information. However, the repulsion is extremely weak: only about 0.004-0.006 eV/A at 0.5 covalent-radius scale and about 0.001-0.002 eV short-range energy lift. This is not a physically adequate close-contact prior.
- Therefore Stage 79 partially reopens the cavity-vector route, but only as a physical-shape signal, not as a Pareto candidate. Stage 78 still says it does not improve aggregate or C/N force MAE in the trained checkpoint. The correct next step is to test whether the same sign/weak-repulsion issue appears in rattle+relax and then decide whether the missing TECE block is a short-range radial repulsive core, data augmentation, or a stratified loss, instead of blindly widening edge paths.
- This also sharpens the theoretical route: for high-throughput T3/T4 endpoints, a scalarized TECE model may need an explicit low-cost short-range two-body core as part of the renormalized retained operator basis. That is a clean TECE degradation axis because it preserves scalar streaming cost while restoring a physically necessary deleted high-curvature response.

Next priority:

1. Implement the bounded rattle+relax harness using the same Stage-73 checkpoints and `C_or_N`/`CHNO` grouping, reporting convergence, final RMSD, and force spikes.
2. If rattle+relax confirms short-range collapse or weak CHNO restoration, add a controlled candidate T4 operator: a cheap radial repulsive-core path or dimer-augmented loss, then measure whether it fixes dimer/rattle behavior without destroying the high-throughput endpoint.
3. Keep low-level graph/kernel acceleration behind these physical checks; the current bottleneck is physical closure of the scalarized retained operator, not ASE graph construction.

# Stage 80: rTECE Rattle+Relax Physical Probe

Stage 80 adds the second physical-generalization diagnostic requested by Stage 78/79 and by the review document: rattle a structure, relax it with the rTECE checkpoint through an ASE calculator, then report convergence, final RMSD, and force spikes by the same chemically focused groups. This is not a throughput benchmark. It tests whether the scalarized retained TECE operator gives a locally restoring conservative field after small geometry perturbations.

Implementation gate:

- Added `rattle_relax_rtece.py`.
- The script loads Stage-73 rTECE checkpoints with `load_checkpoint`, rebuilds graphs through the existing `atoms_to_rtece_graph`, predicts through `tace.models.rtece_workflow.predict`, and exposes the model as an ASE `Calculator` for LBFGS relax.
- It records initial/final RMSD to the reference structure, initial/final/max `fmax`, convergence, energy change, wall time, and per-structure focus groups.
- Focus groups keep the Stage-77/78 `C_or_N`, `CHNO`, and `not_CHNO` definitions, and add `CHNO_no_CN` because the current 512-config mixed training prefix has `CHNO=512`, `C_or_N=507`, and `not_CHNO=0`; without this extra group there is no non-CHNO control inside this slice.
- Regression tests cover focus-group classification, RMSD calculation, convergence aggregation, and max-force-spike summary.

A first loose run with `start=0`, `limit=2`, `rattle_std=0.03 A`, `fmax=0.05`, and `max_steps=5` showed that both structures were C/N/CHNO and that `radial` could pass without meaningful motion because the initial rTECE forces were already below the loose threshold. This run is retained only as a harness smoke, not as a physical conclusion.

Bounded strict Stage-80 run on Stage-73 checkpoints, CPU autograd, `mixed_train_tw0.75.extxyz`, `start=58`, `limit=4`, `rattle_std=0.03 A`, `fmax=0.01`, `max_steps=10`:

| route | group | count | converged frac | mean initial RMSD A | mean final RMSD A | max final RMSD A | max fmax eV/A |
|---|---|---:|---:|---:|---:|---:|---:|
| `radial` | all | 4 | 0.25 | 0.05246 | 0.2331 | 0.3341 | 0.04728 |
| `radial` | `C_or_N` | 3 | 0.00 | 0.05299 | 0.2578 | 0.3341 | 0.04728 |
| `radial` | `CHNO_no_CN` | 1 | 1.00 | 0.05087 | 0.1591 | 0.1591 | 0.02550 |
| `radial_cavity_vec` | all | 4 | 0.00 | 0.05246 | 0.5308 | 0.7685 | 0.1201 |
| `radial_cavity_vec` | `C_or_N` | 3 | 0.00 | 0.05299 | 0.6075 | 0.7685 | 0.1201 |
| `radial_cavity_vec` | `CHNO_no_CN` | 1 | 0.00 | 0.05087 | 0.3009 | 0.3009 | 0.05542 |

Representative per-structure outcomes:

| route | index | formula | groups | converged | final RMSD A | final fmax eV/A |
|---|---:|---|---|---:|---:|---:|
| `radial` | 58 | `CH2Au24NSc12` | `C_or_N,CHNO` | 0 | 0.2661 | 0.0205 |
| `radial` | 59 | `C2HCa16In16O2Rh16` | `C_or_N,CHNO` | 0 | 0.3341 | 0.0246 |
| `radial` | 60 | `HOSb12Ti52` | `CHNO,CHNO_no_CN` | 1 | 0.1591 | 0.00906 |
| `radial` | 61 | `CH2Hf24NSb48` | `C_or_N,CHNO` | 0 | 0.1731 | 0.0168 |
| `radial_cavity_vec` | 58 | `CH2Au24NSc12` | `C_or_N,CHNO` | 0 | 0.7685 | 0.1201 |
| `radial_cavity_vec` | 59 | `C2HCa16In16O2Rh16` | `C_or_N,CHNO` | 0 | 0.5485 | 0.0506 |
| `radial_cavity_vec` | 60 | `HOSb12Ti52` | `CHNO,CHNO_no_CN` | 0 | 0.3009 | 0.0507 |
| `radial_cavity_vec` | 61 | `CH2Hf24NSb48` | `C_or_N,CHNO` | 0 | 0.5054 | 0.0548 |

Force-sign sanity check:

- For config 58, a finite step of `1e-3` along the predicted force lowered energy for both routes, while the opposite step raised energy.
- `radial`: `dE(+F)=-1.037e-4 eV`, `dE(-F)=+1.037e-4 eV`.
- `radial_cavity_vec`: `dE(+F)=-2.788e-4 eV`, `dE(-F)=+2.787e-4 eV`.
- Therefore the rattle+relax divergence is not explained by an ASE force-sign bug; the rTECE force convention is consistent with `F=-dE/dR`.

Stage-80 interpretation against TECE/TACE and the review document:

- The user-observed C/N relaxation problem is supported by a second physical diagnostic. In the strict window, `radial` C/N structures never reach `fmax=0.01` within 10 LBFGS steps and move from about 0.053 A initial RMSD to about 0.258 A mean final RMSD.
- The `CHNO_no_CN` single structure is not a complete control set, but it behaves less badly under `radial` than the C/N subset. The current distillation slice lacks true `not_CHNO` metal-only structures, so a separate control dataset is required before claiming element-general statistics.
- `radial_cavity_vec` is worse in this relax probe: it has larger initial forces, zero convergence, and much larger final RMSD. This overrides the Stage-79 dimer-only sign benefit for the current checkpoint. The cavity-vector path contains some local geometric information, but in this trained route it does not form a stable low-precision relax model.
- The clean TECE conclusion is now narrower: do not widen edge paths or chase graph builders first. The missing ingredient for a high-throughput scalar endpoint is a controlled local physical prior or loss correction, most likely a cheap short-range radial repulsive core and/or targeted dimer/rattle augmentation, plus stratified validation so C/N failures are visible during checkpoint selection.

Next priority:

1. Add a controlled T4 short-range radial-core candidate that preserves scalar streaming and conservative forces, then rerun Stage-79 dimer and Stage-80 rattle+relax before any throughput claim.
2. Add stratified C/N or CHNO validation/selection for rTECE checkpoints so force/relax failures cannot hide behind aggregate mixed-label validation loss.
3. Build or locate a small metal-only/not-CHNO control set for rattle+relax; the current mixed training prefix cannot answer that comparison.
4. Keep nvalchemi/matscipy graph-builder acceleration behind the physical-closure gates. ASE graph construction is too slow for final deployment, but it is not the current blocker for proving the TECE degradation route works.

# Stage 81: T4 Short-Range Radial-Core Candidate

Stage 81 implements the first controlled T4 operator suggested by Stage 79/80: a conservative short-range two-body radial core. The purpose is not to recover full TACE accuracy. It tests whether a scalar streaming endpoint can regain the missing high-curvature local response that was deleted when persistent equivariant state and multi-layer message passing were renormalized away.

Architecture implementation:

- Added optional `RTECEScalarConfig` fields: `use_short_range_repulsion`, `short_range_repulsion_strength`, `short_range_repulsion_beta`, and `short_range_repulsion_radius_scale`.
- The core energy is added after the learned scalar head and atomic reference energy in `RTECEScalarModel.forward`, so autograd forces remain conservative: `F=-dE/dR`.
- The radial core uses directed graph edges with half weighting to avoid double counting symmetric edge lists. Its threshold is `radius_scale*(r_cov_i+r_cov_j)` from covalent radii, and its smooth overlap is `softplus(beta*(r0-r))/beta`.
- The route manifest records `short_range_repulsion` and adds `short_range_radial_core` to retained TECE groups plus `short_range_physical_prior` to Pareto axes. This makes the change visible to the TECE compiler/reporting layer rather than hiding it as an external post-processing trick.
- Analytic/fused force backends reject configs with `use_short_range_repulsion=True` for now, because those paths do not yet include the radial-core force term. Current Stage-81 evaluation must use `force_mode=autograd`.
- `train_rtece_scalar.py` now exposes CLI arguments for the core, and `rtece_scalar_matrix.sbatch` forwards them through script-local environment variables without using forbidden `sbatch --export`.

Regression coverage:

- Short-distance H-H with zero learned head has higher energy than a long-distance pair, gives repulsive atom-0 force, and conserves net force.
- Checkpoint save/load and route manifest preserve the short-range core parameters.
- Analytic pair force backend rejects radial-core configs until a correct analytic force path is implemented.
- The training config builder and rTECE matrix sbatch forward the new architecture options without `--export`.

Bounded Stage-81 overlay smoke on the Stage-73 `radial` checkpoint, CPU autograd, four Stage-79 CHNO dimers, and the Stage-80 strict rattle window (`start=58`, `limit=4`, `rattle_std=0.03 A`, `fmax=0.01`, `max_steps=10`):

| core strength | dimer short repulsive | C-N short force eV/A | N-H short dE eV | relax converged frac | all mean final RMSD A | C/N mean final RMSD A | max fmax eV/A |
|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline radial | 0/4 | +0.0204 | -0.0253 | 0.25 | 0.2331 | 0.2578 | 0.0473 |
| 0.1 | 4/4 | -0.0531 | -0.0123 | 0.00 | 0.2766 | 0.2357 | 0.0879 |
| 1.0 | 4/4 | -0.7146 | +0.1047 | 0.00 | 0.2346 | 0.1844 | 0.9512 |
| 10.0 | 4/4 | -7.3296 | +1.2752 | 0.00 | 0.3753 | 0.3172 | 9.5842 |

Stage-81 interpretation against TECE/TACE and the review document:

- The short-range radial core is a valid TECE degradation axis: it restores a deleted local high-curvature response with scalar two-body cost and conservative forces, without adding persistent equivariant state.
- The dimer failure from Stage 79 is directly addressed. All tested amplitudes flip the short-distance force sign to repulsive for C-N, C-O, N-H, and O-H. This means the missing operator is at least partly short-range radial physics, not only angular/cavity information.
- The rattle+relax result is mixed. `strength=1.0` improves the C/N mean final RMSD from 0.2578 A to 0.1844 A on the bounded window, but it does not converge within 10 LBFGS steps and raises max force to about 0.95 eV/A. `strength=10.0` is clearly too stiff; `strength=0.1` is too weak to fix N-H/O-H short-distance energy shape and does not improve all-structure RMSD.
- Therefore this stage is an architecture-enabling result, not a Pareto-front success. The clean conclusion is that T4 needs a calibrated radial-core prior and likely training-time exposure, not a blind fixed overlay on an already trained radial checkpoint.

Next priority:

1. Run a small trained matrix for `radial_core` with `strength` around 0.3-1.0 and possibly `radius_scale` below 1.0, using the same mixed-label training protocol plus Stage-79/80 physical probes.
2. Add dimer/rattle augmentation or stratified C/N validation so checkpoint selection sees the physical failure mode instead of optimizing only aggregate mixed-label force loss.
3. If a calibrated core improves dimer and rattle without destroying force MAE, implement the analytic radial-core force contribution and then re-enter throughput/Pareto benchmarking.
4. Do not spend the next stage on nvalchemi or graph-builder acceleration; Stage 81 again shows the main blocker is physical closure of the scalarized operator, not kernel throughput.

# Stage 82: Trained Radial-Core Matrix Result

Stage 82 runs the trained follow-up required by Stage 81. Stage 81 only overlaid a fixed short-range core onto an already trained `radial` checkpoint; that proved the T4 radial-core axis is physically meaningful, but not deployable. Stage 82 trains the same scalar radial endpoint with the short-range core present from the start, then evaluates it through the same benchmark, dimer, and rattle gates used in Stages 79-81.

Implementation/reproducibility gate:

- Updated `submit_rtece_scalar_matrix.py` so Slurm wrappers can set `USE_SHORT_RANGE_REPULSION`, `SHORT_RANGE_REPULSION_STRENGTH`, `SHORT_RANGE_REPULSION_BETA`, and `SHORT_RANGE_REPULSION_RADIUS_SCALE`.
- The wrapper path preserves the SAI constraint: parameters are exported inside the wrapper body and the submitted command remains plain `sbatch wrapper.sbatch`; no command-line `--export` is used.
- Regression coverage checks that the submit helper emits radial-core parameters and still omits `--export`.
- Slurm jobs `680276`, `680278`, and `680279` completed training plus DFT/teacher benchmarks.

Common setup: scalar path `atomic.radial_density`, hidden `16,16`, `num_radial=8`, 512 mixed-label train configs, 128 valid configs, 512 DFT/teacher benchmark configs, force weight 30, `force_mode=autograd`, `beta=20.0`, `radius_scale=1.0`. The benchmark rows are prebuilt-batched-graph model timings, not graph-construction timings.

Training and benchmark rows:

| route | strength | best step | best valid loss | DFT E MAE | DFT F MAE | DFT F RMSE | teacher F MAE | atoms/s | params |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Stage-73 `radial` baseline | 0.0 | 200 | NA | 191.39 | 29.45 | 120.02 | 34.25 | 2.96e6 | 369 |
| `radial_core_s0p3` | 0.3 | 100 | 0.5333 | 167.04 | 34.51 | 105.21 | 39.00 | 3.04e6 | 449 |
| `radial_core_s0p6` | 0.6 | 300 | 0.6201 | 171.40 | 44.38 | 115.82 | 48.22 | 2.98e6 | 449 |
| `radial_core_s1p0` | 1.0 | 500 | 0.8262 | 178.57 | 61.53 | 139.35 | 64.87 | 3.07e6 | 449 |

Stage-79 dimer probe on the trained checkpoints, CPU autograd, C/N/O/H dimers over 0.5-5 covalent-radius scale:

| route | short repulsive pairs | C-N short F | C-N short dE | N-H short F | N-H short dE | max short-range abs F |
|---|---:|---:|---:|---:|---:|---:|
| Stage-73 `radial` baseline | 0/4 | +0.0204 | -0.0207 | +0.0169 | -0.0253 | 0.0248 |
| Stage-81 overlay 0.1 | 4/4 | -0.0531 | NA | NA | -0.0123 | NA |
| Stage-81 overlay 1.0 | 4/4 | -0.7146 | NA | NA | +0.1047 | NA |
| `radial_core_s0p3` | 4/4 | -0.2555 | +0.0419 | -0.2045 | +0.00953 | 0.2555 |
| `radial_core_s0p6` | 4/4 | -0.4941 | +0.1049 | -0.3769 | +0.0348 | 0.4941 |
| `radial_core_s1p0` | 4/4 | -0.8052 | +0.1857 | -0.6026 | +0.0641 | 0.8052 |

Stage-80 strict rattle+relax probe on `mixed_train_tw0.75.extxyz`, `start=58`, `limit=4`, `rattle_std=0.03 A`, `fmax=0.01`, `max_steps=10`:

| route | DFT F MAE | converged frac | all mean final RMSD A | C/N mean final RMSD A | CHNO-no-CN final RMSD A | max fmax eV/A |
|---|---:|---:|---:|---:|---:|---:|
| Stage-73 `radial` baseline | 29.45 | 0.25 | 0.2331 | 0.2578 | 0.1591 | 0.0473 |
| Stage-73 `radial_cavity_vec` | 35.16 | 0.00 | 0.5308 | 0.6075 | 0.3009 | 0.1201 |
| Stage-81 overlay 0.1 | NA | 0.00 | 0.2766 | 0.2357 | NA | 0.0879 |
| Stage-81 overlay 1.0 | NA | 0.00 | 0.2346 | 0.1844 | NA | 0.9512 |
| `radial_core_s0p3` | 34.51 | 0.00 | 0.3476 | 0.2896 | 0.5217 | 0.2785 |
| `radial_core_s0p6` | 44.38 | 0.00 | 0.1730 | 0.1283 | 0.3070 | 0.5309 |
| `radial_core_s1p0` | 61.53 | 0.00 | 0.1456 | 0.1199 | 0.2228 | 0.8797 |

Stage-82 interpretation against TECE/TACE and the review document:

- The trained short-range radial core is still a clean T4 TECE degradation axis: it preserves scalar streaming, adds no persistent equivariant state, is visible in the path manifest as `short_range_radial_core`, and fixes the dimer short-range sign for all tested CHNO pairs.
- Training-time exposure changes the rattle behavior relative to a post-hoc overlay. `s0p6` and `s1p0` improve the C/N mean final RMSD from the Stage-73 `radial` baseline 0.2578 A to 0.1283 A and 0.1199 A. This supports the document's hypothesis that the scalarized endpoint needs a retained low-cost local high-curvature operator for non-metal adsorbates.
- The same rows are not Pareto-front successes yet. Force MAE worsens monotonically with core strength: DFT force MAE goes from 29.45 meV/A for baseline `radial` to 34.51, 44.38, and 61.53 meV/A. The strict relax windows do not converge within 10 LBFGS steps, and max force spikes rise to 0.53-0.88 eV/A for the routes that best lower C/N RMSD.
- `s0p3` is not useful: it fixes dimer sign but worsens all-structure and C/N relax RMSD relative to baseline. `s0p6` is the most informative candidate because it halves C/N RMSD while not being as stiff as `s1p0`, but it needs force-spike control before it can be a Pareto point.
- This result answers the graph-builder question for the current priority order: ASE graph construction is too slow for final end-to-end MD, and earlier stages already showed provider lifetime dominates full rebuild execution. However, Stage 82 is an algorithmic physical-closure gate. Moving now to nvalchemi/matscipy/DeepMD edge-force kernels would optimize a model whose C/N relax and force-spike tradeoff is not yet closed.

Next priority after Stage 82:

1. Calibrate the radial-core axis rather than widen the model: run a small `radius_scale`/strength grid around `strength=0.4-0.8` and `radius_scale=0.75-0.95`, because current `radius_scale=1.0` makes the useful C/N fix too stiff.
2. Add a stratified validation/checkpoint score that includes C/N force or rattle proxy terms. Current best-validation loss selected checkpoints that can improve aggregate loss while leaving non-metal adsorbate relax hidden.
3. Add targeted dimer or close-contact augmentation only after the radius/strength grid shows whether the fixed prior can be calibrated without new data.
4. Implement analytic/fused radial-core forces only after a calibrated core survives the dimer + rattle + MAE gates. The graph/provider path should then be addressed inside the review-mandated PBC edge-vector/force/virial ABI, not as an isolated ASE replacement.

# Stage 83: Radial-Core Radius Calibration

Stage 83 follows directly from Stage 82 rather than opening a new architecture branch. Stage 82 showed that `radius_scale=1.0` can strongly improve C/N rattle RMSD but makes the core too stiff and worsens force MAE. The review-aligned question for this stage is whether the same T4 short-range retained operator can be calibrated by shortening its radial support, keeping the TECE scalar endpoint and training workflow unchanged.

Submitted and completed jobs:

| job id | route | strength | radius scale | beta | scalar paths | train/valid/bench configs | force mode |
|---:|---|---:|---:|---:|---|---|---|
| 680322 | `radial_core_s0p4_r0p85` | 0.4 | 0.85 | 20.0 | `atomic.radial_density` | 512/128/512 | `autograd` |
| 680326 | `radial_core_s0p6_r0p85` | 0.6 | 0.85 | 20.0 | `atomic.radial_density` | 512/128/512 | `autograd` |
| 680323 | `radial_core_s0p8_r0p85` | 0.8 | 0.85 | 20.0 | `atomic.radial_density` | 512/128/512 | `autograd` |
| 680327 | `radial_core_s0p6_r0p75` | 0.6 | 0.75 | 20.0 | `atomic.radial_density` | 512/128/512 | `autograd` |

All jobs were submitted through self-contained wrappers with parameters exported inside the wrapper body; no command-line `sbatch --export` was used.

Training and benchmark rows:

| route | best step | best valid loss | DFT E MAE | DFT F MAE | DFT F RMSE | teacher F MAE | atoms/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage-73 `radial` baseline | 200 | NA | 191.39 | 29.45 | 120.02 | 34.25 | 2.96e6 |
| Stage-82 `radial_core_s0p6` r=1.0 | 300 | 0.6201 | 171.40 | 44.38 | 115.82 | 48.22 | 2.98e6 |
| Stage-83 `s0p4_r0p85` | 100 | 0.5390 | 168.03 | 29.00 | 101.07 | 34.15 | 3.07e6 |
| Stage-83 `s0p6_r0p85` | 100 | 0.5388 | 168.04 | 29.00 | 101.06 | 34.15 | 3.08e6 |
| Stage-83 `s0p8_r0p85` | 100 | 0.5389 | 168.04 | 29.02 | 101.06 | 34.17 | 2.99e6 |
| Stage-83 `s0p6_r0p75` | 100 | 0.5399 | 168.02 | 29.08 | 101.13 | 34.21 | 3.07e6 |

Stage-79 dimer probe on trained checkpoints:

| route | short repulsive pairs | C-N short F | C-N short dE | N-H short F | N-H short dE | max short-range abs F |
|---|---:|---:|---:|---:|---:|---:|
| Stage-73 `radial` baseline | 0/4 | +0.0204 | -0.0207 | +0.0169 | -0.0253 | 0.0248 |
| Stage-83 `s0p4_r0p85` | 4/4 | -0.2396 | +0.0159 | -0.1928 | -0.00221 | 0.2396 |
| Stage-83 `s0p6_r0p85` | 4/4 | -0.3425 | +0.0423 | -0.2642 | +0.0105 | 0.3425 |
| Stage-83 `s0p8_r0p85` | 4/4 | -0.4454 | +0.0687 | -0.3356 | +0.0232 | 0.4454 |
| Stage-83 `s0p6_r0p75` | 4/4 | -0.2541 | +0.00357 | -0.2023 | -0.00804 | 0.2541 |

Stage-80 strict rattle+relax probe on `mixed_train_tw0.75.extxyz`, `start=58`, `limit=4`, `rattle_std=0.03 A`, `fmax=0.01`, `max_steps=10`:

| route | DFT F MAE | converged frac | all mean final RMSD A | C/N mean final RMSD A | CHNO-no-CN final RMSD A | max fmax eV/A |
|---|---:|---:|---:|---:|---:|---:|
| Stage-73 `radial` baseline | 29.45 | 0.25 | 0.2331 | 0.2578 | 0.1591 | 0.0473 |
| Stage-82 `radial_core_s0p6` r=1.0 | 44.38 | 0.00 | 0.1730 | 0.1283 | 0.3070 | 0.5309 |
| Stage-82 `radial_core_s1p0` r=1.0 | 61.53 | 0.00 | 0.1456 | 0.1199 | 0.2228 | 0.8797 |
| Stage-83 `s0p4_r0p85` | 29.00 | 0.00 | 0.1825 | 0.2219 | 0.0642 | 0.4087 |
| Stage-83 `s0p6_r0p85` | 29.00 | 0.00 | 0.2092 | 0.2572 | 0.0650 | 0.4172 |
| Stage-83 `s0p8_r0p85` | 29.02 | 0.00 | 0.1771 | 0.2143 | 0.0654 | 0.3345 |
| Stage-83 `s0p6_r0p75` | 29.08 | 0.00 | 0.1798 | 0.1804 | 0.1778 | 0.3386 |

Stage-83 interpretation against TECE/TACE and the review document:

- Shortening the radial-core support is Pareto-positive for aggregate single-step accuracy. All Stage-83 rows recover Stage-73-level DFT force MAE while keeping the short-distance dimer force repulsive for all four CHNO pairs.
- The force-spike and relax tradeoff is still not closed. Compared with Stage-82 `s0p6` at `radius_scale=1.0`, the best Stage-83 rows reduce max fmax from 0.5309 to about 0.334-0.409 eV/A and recover force MAE, but they lose much of the C/N relax gain. The best current balance is `s0p6_r0p75`: DFT F MAE 29.08 meV/A, all RMSD 0.1798 A, C/N RMSD 0.1804 A, and max fmax 0.3386 eV/A.
- `s0p6_r0p85` is not useful despite good benchmark MAE: C/N RMSD returns to the Stage-73 baseline. This confirms that aggregate MAE/RMSE alone is not sufficient for this problem and supports the review demand for stratified physical diagnostics.
- The clean theoretical conclusion is now stronger: the radial core is a real retained T4 operator in the TECE degradation hierarchy, but the operator must be selected by a multi-objective score that includes dimer sign/shape, C/N relax RMSD, and force-spike control. It cannot be tuned by aggregate supervised loss alone.
- Graph/provider acceleration remains a real deployment axis, but Stage 83 again says the immediate blocker is the calibrated physical closure of the scalar endpoint. Kernel work should wait until a candidate survives these physics gates.

Next priority after Stage 83:

1. Add a stratified checkpoint-selection/evaluation summary that combines DFT force MAE, dimer pass/fail, C/N rattle RMSD, and max fmax into an explicit Pareto score. This is the missing measurement layer between TECE retained paths and architecture choice.
2. Run a narrow follow-up around the current best balance: `strength=0.6-0.9`, `radius_scale=0.70-0.80`, and possibly lower `beta` to smooth force spikes without losing dimer repulsion.
3. Only after this score identifies a stable candidate should we implement analytic radial-core force and re-enter the high-throughput fused/backend track.
4. Do not change low-level graph builders as the next isolated task; when resumed, graph work should be tied to the review P0 PBC edge-vector/force/virial ABI.

# Stage 84: Physical Pareto Scorer

Stage 84 implements the Stage-83 measurement-layer requirement: stop hand-reading separate benchmark, dimer, and rattle tables when deciding the next retained T4 operator. The new scorer combines the existing JSON outputs into explicit physical Pareto rows, while preserving the individual metrics so the score remains auditable.

Implementation gate:

- Added `summarize_rtece_physical_pareto.py`.
- The script consumes one or more cases, each with DFT benchmark JSON, teacher benchmark JSON, dimer JSON, and rattle+relax JSON.
- It emits `rtece_physical_pareto_summary.v1` JSON plus Markdown.
- Each row records DFT/teacher force errors, atoms/s, dimer repulsive fraction, C/N rattle RMSD, rattle max fmax, individual gate booleans, and a lower-is-better `physical_score`.
- Regression coverage checks that benchmark, dimer, and rattle gates are combined correctly and that physical Pareto dominance drops a slower, worse row.

Stage-84 scorer definition used for the current radial-core comparison:

- Benchmark gate: `DFT F MAE <= 35 meV/A`.
- Rattle gate: `C/N mean final RMSD <= 0.20 A` and `max fmax <= 0.40 eV/A`.
- Dimer gate: every dimer pair has finite values and short-distance repulsive force.
- `physical_score = DFT_F_MAE/35 + C/N_RMSD/0.20 + max_fmax/0.40 + 2*dimer_fail_fraction + nonfinite_pair_count`.

Bounded Stage-84 run over Stage-73 baseline plus Stage-82/83 radial-core checkpoints:

| rank | route | gate | score | atoms/s | DFT F MAE | dimer repulsive | C/N RMSD A | max fmax eV/A |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `radial_core_s0p6_r0p75` | 1 | 2.579 | 3.07e6 | 29.08 | 1.00 | 0.180 | 0.339 |
| 2 | `radial_core_s0p8_r0p85` | 0 | 2.737 | 2.99e6 | 29.02 | 1.00 | 0.214 | 0.334 |
| 3 | `radial_core_s0p4_r0p85` | 0 | 2.960 | 3.07e6 | 29.00 | 1.00 | 0.222 | 0.409 |
| 4 | `radial_core_s0p6_r0p85` | 0 | 3.158 | 3.08e6 | 29.00 | 1.00 | 0.257 | 0.417 |
| 5 | Stage-82 `radial_core_s0p6` r=1.0 | 0 | 3.237 | 2.98e6 | 44.38 | 1.00 | 0.128 | 0.531 |
| 6 | Stage-73 `radial` baseline | 0 | 4.248 | 2.96e6 | 29.45 | 0.00 | 0.258 | 0.047 |
| 7 | Stage-82 `radial_core_s1p0` r=1.0 | 0 | 4.557 | 3.07e6 | 61.53 | 1.00 | 0.120 | 0.880 |

Physical Pareto front under score/throughput dominance:

| route | score | atoms/s | gate | interpretation |
|---|---:|---:|---:|---|
| `radial_core_s0p6_r0p85` | 3.158 | 3.08e6 | 0 | fastest row but fails rattle gates; useful as throughput tie point only. |
| `radial_core_s0p6_r0p75` | 2.579 | 3.07e6 | 1 | best current physical candidate; only row passing all configured gates. |

Stage-84 interpretation against TECE/TACE and the review document:

- The scorer makes the current TECE degradation decision explicit: the radial-core axis is not rejected, but only the shorter-support calibrated route currently satisfies all physical gates.
- The original `radial` baseline is not a physical candidate despite good aggregate MAE and low fmax, because it fails the dimer gate and C/N rattle gate. This directly encodes the Stage-79/80 physical-generalization evidence.
- Stage-82 large-support cores are not candidates despite strong C/N RMSD because benchmark MAE and force spikes fail. This prevents selecting a model that looks good only on one physical diagnostic.
- `s0p6_r0p75` is the first clean closed-loop candidate in this branch: it preserves scalar T4 cost, fixes dimer sign, recovers baseline force MAE, improves C/N rattle RMSD, and keeps force spikes under the current 0.40 eV/A threshold. It still needs broader validation before being called a Pareto-front model.

Next priority after Stage 84:

1. Use `s0p6_r0p75` as the local anchor and run a narrow beta/softness grid: keep `radius_scale=0.75`, test `strength=0.6-0.9`, and test lower `beta` values such as 10 and 15 to see whether force spikes smooth without losing dimer repulsion.
2. Add the physical scorer to future matrix collection so every candidate is scored before any graph/backend optimization work.
3. If a beta-smoothed candidate improves the score on this bounded window, broaden the rattle window and add a metal/not-CHNO control set before implementing analytic/fused radial-core forces.
4. Keep graph-provider work tied to the review P0 PBC edge-vector/force/virial ABI; this scorer is now the gate that decides when returning to throughput engineering is justified.


# Stage 85: Radial-Core Beta Softening Calibration

Stage 85 is the direct follow-up to the Stage-84 physical scorer, not a new heuristic sweep. Stage 83 identified `radial_core_s0p6_r0p75` as the first row passing the combined MAE/dimer/CN-rattle/fmax gates, but its strict rattle window still showed a force spike around 0.339 eV/A. The TECE-aligned question for this stage is whether the same retained scalar T4 operator can be softened by the core overlap sharpness `beta`, preserving the scalar endpoint while lowering high-curvature relaxation artifacts.

Submitted and completed jobs:

| job id | route | strength | radius scale | beta | scalar paths | train/valid/bench configs | force mode |
|---:|---|---:|---:|---:|---|---|---|
| 680370 | `radial_core_s0p6_r0p75_b10` | 0.6 | 0.75 | 10.0 | `atomic.radial_density` | 512/128/512 | `autograd` |
| 680371 | `radial_core_s0p6_r0p75_b15` | 0.6 | 0.75 | 15.0 | `atomic.radial_density` | 512/128/512 | `autograd` |
| 680372 | `radial_core_s0p8_r0p75_b10` | 0.8 | 0.75 | 10.0 | `atomic.radial_density` | 512/128/512 | `autograd` |
| 680373 | `radial_core_s0p8_r0p75_b15` | 0.8 | 0.75 | 15.0 | `atomic.radial_density` | 512/128/512 | `autograd` |

All jobs were submitted through self-contained wrappers with parameters exported inside the wrapper body; no command-line `sbatch --export` was used.

Training and benchmark rows:

| route | best step | best valid loss | DFT E MAE | DFT F MAE | DFT F RMSE | teacher F MAE | atoms/s |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage-83 `radial_core_s0p6_r0p75` beta=20 | 100 | 0.5399 | 168.02 | 29.08 | 101.13 | 34.21 | 3.07e6 |
| Stage-85 `s0p6_r0p75_b10` | 100 | 0.5398 | 168.02 | 29.08 | 101.12 | 34.21 | 3.08e6 |
| Stage-85 `s0p6_r0p75_b15` | 100 | 0.5399 | 168.02 | 29.08 | 101.13 | 34.21 | 2.98e6 |
| Stage-85 `s0p8_r0p75_b10` | 100 | 0.5397 | 168.02 | 29.08 | 101.12 | 34.21 | 3.02e6 |
| Stage-85 `s0p8_r0p75_b15` | 100 | 0.5399 | 168.02 | 29.08 | 101.13 | 34.21 | 3.00e6 |

Stage-79 dimer and Stage-80 strict rattle gates were scored with the Stage-84 physical Pareto scorer:

| rank | route | gate | score | atoms/s | DFT F MAE | dimer repulsive | min dimer dE eV | C/N RMSD A | CHNO-no-CN RMSD A | max fmax eV/A |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `radial_core_s0p6_r0p75_b10` | 1 | 2.379 | 3.08e6 | 29.08 | 4/4 | -0.00740 | 0.176 | 0.116 | 0.209 |
| 2 | `radial_core_s0p8_r0p75_b10` | 1 | 2.408 | 3.02e6 | 29.08 | 4/4 | -0.00110 | 0.175 | 0.101 | 0.273 |
| 3 | `radial_core_s0p6_r0p75_b15` | 1 | 2.665 | 2.98e6 | 29.08 | 4/4 | -0.00839 | 0.179 | 0.176 | 0.308 |
| 4 | `radial_core_s0p8_r0p75_b15` | 1 | 2.676 | 3.00e6 | 29.08 | 4/4 | -0.00243 | 0.178 | 0.174 | 0.362 |
| 5 | Stage-83 `radial_core_s0p6_r0p75` beta=20 | 1 | 2.751 | 3.07e6 | 29.08 | 4/4 | -0.00859 | 0.180 | 0.178 | 0.339 |

Physical Pareto front under score/throughput dominance:

| route | score | atoms/s | interpretation |
|---|---:|---:|---|
| `radial_core_s0p6_r0p75_b10` | 2.379 | 3.08e6 | dominates the Stage-83 anchor and all Stage-85 beta rows in this bounded window under the energy-shape-aware score. |

Stage-85 interpretation against TECE/TACE and the review document:

- Beta softening is a real TECE degradation coordinate for the retained short-range radial-core operator. It changes the high-curvature prior without adding persistent equivariant state, extra message passing, or a wider scalar head.
- `s0p6_r0p75_b10` is now the cleanest local candidate: it keeps Stage-73/83-level DFT force MAE, preserves 4/4 short-distance dimer repulsion, reduces C/N strict-rattle RMSD slightly, and cuts the maximum rattle fmax from 0.339 to 0.209 eV/A.
- Increasing strength to 0.8 at beta=10 slightly improves the bounded C/N and CHNO-no-CN RMSDs, but worsens the force spike and loses throughput. Under the current scorer it is dominated by `s0p6_r0p75_b10`.
- The force/energy dimer shape is still not fully solved: some short-distance energy lifts remain slightly negative even when the force sign is repulsive. The scorer now adds a soft penalty `max(0, -min_short_energy_lift)/0.05 eV`, which keeps the current gate force-based but makes this defect visible in the score.
- None of the strict rattle rows converges within 10 LBFGS steps. The comparison is still useful because all candidates share the same bounded protocol, but the next validation stage must broaden the window and report RMSD/fmax trajectories, not just final means.

Priority after Stage 85:

1. Promote `radial_core_s0p6_r0p75_b10` to the local anchor for broader validation.
2. Run a broader stratified rattle+relax validation: more C/N-containing adsorbate structures, a CHNO-no-CN control set, and a metal/non-adsorbate control set.
3. Add the dimer energy-shape term to all future matrix summaries and consider making it a hard gate only after the broader validation shows the tolerance is stable.
4. If the broader validation preserves the Stage-85 advantage, implement analytic/fused radial-core force and then return to graph/provider throughput engineering under the review P0 PBC edge-vector/force/virial ABI.
5. Do not move next to isolated ASE graph-builder replacement. The document-driven blocker remains physical closure of the scalar TECE endpoint; graph work becomes the right priority only after this candidate survives the broader physical gates.


# Stage 86: Broader Rattle Validation of the Stage-85 Anchor

Stage 86 tests whether the Stage-85 local anchor survives a broader physical-generalization window. Stage 85 selected `radial_core_s0p6_r0p75_b10` on a bounded 4-config strict rattle window. The review-aligned concern is that this may still hide the C/N adsorbate relax failure mode seen earlier by the user, so the next check broadened the rattle window before any analytic/fused radial-core force work.

Run setup:

| job id | route | checkpoint | configs | start | limit | rattle std | fmax | max steps | device |
|---:|---|---|---|---:|---:|---:|---:|---:|---|
| 680399 | `radial_core_s0p6_r0p75_b10` | Stage-85 best | `mixed_train_tw0.75.extxyz` | 58 | 48 | 0.03 A | 0.01 eV/A | 10 | cuda |

The selected train window contains 45 C/N-containing configs and 3 CHNO-no-CN controls. A scan of `mixed_train_tw0.75.extxyz` found 4961/5000 configs in `C_or_N`, 39/5000 in `CHNO_no_CN`, and no `not_CHNO` metal-only controls. Attempting to fully read `mixed_valid_tw0.75.extxyz` hit an extxyz frame-length error, so Stage 86 does not claim metal-only coverage.

Broader rattle+relax result:

| group | count | converged frac | mean final RMSD A | max final RMSD A | max fmax eV/A |
|---|---:|---:|---:|---:|---:|
| all | 48 | 0.000 | 0.2365 | 0.4955 | 0.4866 |
| C_or_N | 45 | 0.000 | 0.2444 | 0.4955 | 0.4866 |
| CHNO | 48 | 0.000 | 0.2365 | 0.4955 | 0.4866 |
| CHNO_no_CN | 3 | 0.000 | 0.1183 | 0.1392 | 0.3915 |

The same Stage-84/85 physical scorer, now including the dimer energy-shape penalty, gives:

| variant | gate | score | DFT F MAE | dimer repulsive | dimer energy penalty | C/N RMSD A | max fmax eV/A | atoms/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `radial_core_s0p6_r0p75_b10_broader_rattle` | 0 | 3.417 | 29.08 | 4/4 | 0.00740 | 0.244 | 0.487 | 3.08e6 |

Worst final-RMSD cases in this window are C/N structures, led by config 92 `C2H3OSc40Si24` at 0.496 A final RMSD and config 83 `C2H4Au48K24` at 0.456 A. The highest max fmax is config 82 `CH3Ag12HgNY`-type formula (`CH3Ag12Hg24NY12`) at 0.487 eV/A.

Stage-86 interpretation against TECE/TACE and the review document:

- The Stage-85 local anchor is not yet a robust Pareto-front model. It passes the 4-config local gate but fails the broader 48-config rattle gate: C/N mean RMSD rises from 0.176 A to 0.244 A, and max fmax rises from 0.209 to 0.487 eV/A.
- The failure is concentrated in the C/N-heavy adsorbate population; the small CHNO-no-CN control subset remains much better on RMSD. This matches the user observation that non-metal adsorbates, especially C/N-containing structures, expose the current scalar endpoint weakness.
- The retained radial-core operator remains useful for short-range dimer sign and local force-spike reduction, but it is not sufficient by itself for broader C/N relax stability. This shifts priority from kernel/fusion work back to stratified model selection and possibly adding a low-cost C/N-sensitive scalar operator or training augmentation.
- Since the broader rattle gate fails, analytic/fused radial-core force should not be the next task. It would optimize the throughput of a candidate that is not physically closed under the review-mandated diagnostics.

Priority after Stage 86:

1. Add stratified C/N rattle or force-residual proxy into checkpoint/model selection, so aggregate mixed-label loss cannot select rows that fail C/N relax.
2. Compare two algorithmic fixes before graph/kernel work: targeted C/N close-contact/rattle augmentation versus a low-cost C/N-sensitive scalar retained operator beyond the radial core.
3. Keep `radial_core_s0p6_r0p75_b10` as the local anchor for comparison, but do not promote it to fused-force implementation until it passes broader rattle gates.
4. Repair or avoid the corrupted `mixed_valid_tw0.75.extxyz` path before claiming validation-set stratification; use train-window diagnostics only as bounded evidence.


# Stage 87: C/N Force-Selection Proxy

Stage 87 implements the first Stage-86 priority: add a stratified C/N force-residual proxy so aggregate mixed-label loss cannot hide the non-metal adsorbate failure mode. This is a selection-layer change rather than a new architecture claim. It keeps the TECE route fixed and asks whether the broader rattle failure is already visible in single-step force residuals on the same structures.

Implementation gate:

- Extended `stratify_rtece_force_errors.py` with a C/N-focused selection score.
- The score is `global_force_MAE + focus_excess_weight * max(0, C_or_N_force_MAE - global_force_MAE)` with default `focus_excess_weight=2.0`.
- The JSON and Markdown outputs now report `selection_score_mev_a`, `selection_focus_mae_f_mev_a`, and `selection_focus_excess_mae_f_mev_a`.
- Regression coverage checks that concentrated C/N force error raises the selection score and appears in Markdown.

Stage-87 diagnostic setup:

| route | checkpoint | configs | window | target force arrays | focus label | excess weight |
|---|---|---|---|---|---|---:|
| `radial_core_s0p6_r0p75_b10` | Stage-85 best | `mixed_train_tw0.75.extxyz` sliced to `58:106` | 48 configs | `teacher_forces`, `dft_forces` | `C_or_N` | 2.0 |

The slice is the exact Stage-86 broader-rattle window. It contains 3116 atoms; only 147 atoms are C/N, about 4.7% of atoms.

Force-selection proxy results:

| target | global F MAE | C/N F MAE | C/N excess | selection score | CHNO F MAE | not-CHNO F MAE |
|---|---:|---:|---:|---:|---:|---:|
| teacher | 76.70 | 485.30 | 408.60 | 893.89 | 445.36 | 36.70 |
| DFT | 75.41 | 485.39 | 409.98 | 895.37 | 446.84 | 35.11 |

Top element errors are also non-metal dominated. Against DFT, O is 693.50 meV/A, C is 638.79 meV/A, H is 336.38 meV/A, and N is 275.09 meV/A, while most slab/metal elements are much lower. The teacher-force table gives the same ordering within noise.

Stage-87 interpretation against TECE/TACE and the review document:

- The broader rattle failure is already visible in a cheap single-step force proxy. This means future candidate selection does not need to wait for every row to run expensive LBFGS rattle before rejecting rows with severe C/N residual concentration.
- The issue is not merely teacher mismatch. Teacher and DFT force stratification give almost identical C/N excess: about 409 meV/A above global MAE. This supports the conclusion that the current scalar endpoint has a real C/N local-physics deficit on this OC20NEB window.
- Aggregate force MAE is misleading here because C/N atoms are a small atom fraction. A row can look acceptable globally while failing the exact population that controls adsorbate relax. This validates the Stage-86 priority to add stratified selection before returning to kernels.
- The next algorithmic question is now sharper: does targeted C/N close-contact/rattle weighting fix this residual while preserving scalar T4 cost, or do we need a new low-cost C/N-sensitive retained operator beyond the radial core?

Priority after Stage 87:

1. Use the C/N force-selection proxy as a cheap pre-filter for future radial-core or scalar-operator matrices, then confirm finalists with broader rattle.
2. Run a small algorithmic comparison with the same Stage-87 proxy and Stage-86 rattle window: C/N force-residual weighting or close-contact augmentation versus a low-cost C/N-sensitive scalar retained operator.
3. Keep graph/kernel work deferred. The current evidence says the dominant failure is operator/data selection for C/N local chemistry, not throughput of the existing scalar endpoint.
4. Add `start-config` support to force/projection loaders if we need repeated exact-window stratification without writing sliced extxyz artifacts.

## Stage Review Rule

After each experiment stage, compare results back to the source documents:

1. Did the change test projection, renormalization, distillation, or hardware cost?
2. Did the measured bottleneck match the document's claim about edge/state lifetime and channel mixing?
3. Should the next priority be architecture projection, distillation loss, fusion/export, or benchmark methodology?
