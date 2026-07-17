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

## Stage Review Rule

After each experiment stage, compare results back to the source documents:

1. Did the change test projection, renormalization, distillation, or hardware cost?
2. Did the measured bottleneck match the document's claim about edge/state lifetime and channel mixing?
3. Should the next priority be architecture projection, distillation loss, fusion/export, or benchmark methodology?
