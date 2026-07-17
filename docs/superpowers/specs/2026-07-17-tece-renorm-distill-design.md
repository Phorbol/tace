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


## Stage Review Rule

After each experiment stage, compare results back to the source documents:

1. Did the change test projection, renormalization, distillation, or hardware cost?
2. Did the measured bottleneck match the document's claim about edge/state lifetime and channel mixing?
3. Should the next priority be architecture projection, distillation loss, fusion/export, or benchmark methodology?
