# Stage129 teacher-rattle distillation interpretation

## Question

Stage129 fixes the stage128 physical-Pareto architecture rows and changes only the training distribution: 2048 base mixed-train configs plus 128 TACE-teacher-labeled rattles from the stage128 stress window. The purpose is to test whether adding teacher coverage around a known rattle-relax failure region improves robustness without C/N-specific architecture changes.

This is a direct TECE/TACE renormalization test: do not add an ad hoc element specialization; instead keep the same T3 scalarized low-order moment route and change the projected data coverage.

## Completed sbatch runs

| job | variant | params | steps | best step | DFT E RMSE | DFT F RMSE | DFT F max | atoms/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 685068 | l1_active_nrad12_species20_radial_species8_cross3_h64 | 25449 | 20000 | 18496 | 300.099 | 107.653 | 2387.490 | 1294809 |
| 685070 | l1_active_nrad12_species24_radial_species8_cross3_h64 | 28925 | 20000 | 12240 | 273.849 | 101.557 | 2697.533 | 1224418 |

Both jobs completed through the production `tace.scripts.rtece_train_scalar` path, with Lightning, batch size 8, validation batch size 16, plateau LR scheduler, 500-step warmup, and early-stopping patience 400. Neither job early-stopped; both reached `max_steps=20000`.

## Comparison to stage128

| variant | stage128 DFT E RMSE | stage129 DFT E RMSE | delta | stage128 DFT F RMSE | stage129 DFT F RMSE | delta | stage128 F max | stage129 F max | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| species20/cross3 | 277.925 | 300.099 | +22.173 | 104.793 | 107.653 | +2.861 | 2263.224 | 2387.490 | +124.266 |
| species24/cross3 | 307.299 | 273.849 | -33.450 | 99.480 | 101.557 | +2.077 | 2429.936 | 2697.533 | +267.596 |

The result is mixed, not a clean win. The species24 row improves DFT energy RMSE substantially, but both rows slightly worsen DFT force RMSE and force max error. The species20 row is worse on both energy and force RMSE. Throughput stays effectively unchanged, so the observed changes are from the data/projection test, not a throughput-speed tradeoff.

## Current interpretation

This does not yet prove that the architecture is intrinsically unable to handle the C/N or rattle-relax failure mode. It says that adding 128 teacher-rattle fake-label configs around the stress window is not sufficient to produce a standard-benchmark Pareto improvement. The species24 energy improvement suggests the extra coverage is doing something real, but the force max degradation is a warning: teacher labels on a small local rattle set may improve energy alignment while not controlling the high-force tail.

This keeps the priority aligned with the design documents:

- Continue treating RMSE and max error separately; force max remains a physical risk axis.
- Do the physical external gate next, because the original stage128 failure was rattle force spikes, not ordinary validation F RMSE.
- Do not specialize to C/N yet. The next falsification should ask whether broader teacher trajectory coverage, force/Jacobian-weighted distillation, or additional trainable front-end paths reduce the high-force tail in a systematic route.

## Next step

Run stage129 physical triage with the same dimer scan and C/N rattle-relax settings used in stage128, pointed at the two new `rtece_scalar_best.pt` checkpoints. That decides whether teacher-rattle coverage improves the actual physical failure mode even though standard DFT F RMSE is not improved.
