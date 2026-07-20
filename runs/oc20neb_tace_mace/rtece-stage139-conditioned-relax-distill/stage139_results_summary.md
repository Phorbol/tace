# Stage139 conditioned-relax distillation summary

## Document alignment

Stage139 tests a specific TECE/rTECE hypothesis: keep the low-cost L<=2 scalarized moment front fixed, then change the distillation/deployment measure by adding weighted teacher-relax configurations. This follows the TECE design-space idea that the student should be a controlled projection of the teacher over a target distribution, not a random width sweep.

The rTECE review also says the current prototype must report RMSE/max errors, physical dimer/rattle checks, and production workflow evidence. This stage therefore reports E/F RMSE, E/F max, throughput, C/N rattle-relax RMSD, dimer, rattle, and full sbatch status. The configured short-range potential is `zbl`, but `short_range_repulsion_strength=0.0`, so this is not an active ZBL ablation.

## Sbatch evidence

| task | job | state | exit | elapsed | node | MaxRSS(batch) |
|---|---:|---|---|---:|---|---:|
| train_l2_bneck32 | 685426 | COMPLETED | 0:0 | 00:21:01 | 16v100n01 | 1567524K |
| train_l2_cond32 | 685427 | COMPLETED | 0:0 | 00:20:59 | 16v100n01 | 1804892K |
| physical_l2_bneck32 | 685436 | COMPLETED | 0:0 | 00:00:08 | 16v100n01 | 2372K |
| physical_l2_cond32 | 685437 | COMPLETED | 0:0 | 00:00:06 | 16v100n01 | 824K |

## Training contract

Both Stage139 variants used `weighted_train_base2048_plus_teacher_relax320_eanchor.extxyz` with 2368 train configs, 256 valid configs, batch 8, valid batch 16, max 20000 steps, plateau LR scheduling, 500-step warmup, LR patience 25, and early stopping patience 400. The two jobs reached 20000 steps; best validation steps were 19240 for bneck32 and 16872 for cond32.

## Pareto comparison

All error units are meV/atom for energy and meV/A for forces.

| row | params | atoms/s | E RMSE | E max | F RMSE | F max | C/N RMSD | bench | dimer | rattle | physical |
|---|---|---|---|---|---|---|---|---|---|---|---|
| stage132_l1_atomic | 28925 | 1217882 | 293.391 | 697.301 | 108.355 | 1840.539 | 0.103 | yes | yes | no | no |
| stage137_l2_bneck32 | 21405 | 1061041 | 309.728 | 691.876 | 114.259 | 1861.035 | 0.110 | yes | yes | no | no |
| stage137_l2_cond32 | 51367 | 1023496 | 340.032 | 740.266 | 107.516 | 1914.572 | 0.111 | no | yes | no | no |
| stage138_cond32_edge_residual | 51754 | 518939 | 329.414 | 763.132 | 112.703 | 1834.768 | 0.106 | no | yes | no | no |
| stage139_l2_bneck32_relax_distill | 21405 | 1062072 | 288.657 | 651.674 | 111.387 | 2224.502 | 0.106 | yes | yes | no | no |
| stage139_l2_cond32_relax_distill | 51367 | 1024003 | 388.377 | 794.805 | 103.073 | 1709.661 | 0.101 | no | yes | no | no |

## Interpretation

- Stage139 confirms that the weighted teacher-relax data is not useless: on the L=2 conditioned-front cond32 row, DFT force RMSE improves from Stage137 107.516 meV/A to 103.073 meV/A, force max improves from 1914.572 to 1709.661 meV/A, and C/N rattle RMSD improves from 0.111 A to 0.101 A at roughly the same 1.02M atoms/s throughput.
- The same cond32 row badly damages energy comparability: DFT energy RMSE rises to 388.377 meV/atom with -159.246 meV/atom bias and 794.805 meV/atom max error, so it fails the benchmark gate despite being a force/physical-tail frontier point.
- The bneck32 row moves in the opposite direction: it preserves energy better than Stage137 bneck32 (E RMSE 288.657 vs 309.728 meV/atom, near-zero bias) and passes the benchmark gate, but its force RMSE/max and physical score do not become the frontier.
- Stage138 edge residual remains off the Pareto front for this regime: it costs about half the throughput and does not improve force RMSE or energy max. This supports prioritizing data/loss/renormalization closure over additional edge sketches for the next stage.
- Because short_range_repulsion_potential is set to zbl but short_range_repulsion_strength is 0.0, this stage is not evidence for active ZBL stabilization. Dimer pass here is a diagnostic of the learned/baseline surface under the current zero-strength setting, not a tuned ZBL result.

## Next priority

- Do a source-balanced distillation/calibration stage on the Stage139 cond32 architecture: preserve the force-tail gain from teacher-relax data while correcting the energy gauge/bias with stronger DFT energy anchoring, force-only teacher-relax weighting, or per-source energy normalization.
- Keep the architecture axis fixed during that test. The unresolved variable is loss/data measure, not another head-width or edge-feature change.
- After energy-force balance is recovered, run the same physical triage plus atom-count throughput scaling to update the actual Pareto front.

## Files

- Machine-readable summary: `runs/oc20neb_tace_mace/rtece-stage139-conditioned-relax-distill/stage139_results_summary.json`
- Stage139 training summaries and benchmark JSON files remain under `conditioned_relax_runs/`.
- Stage139 dimer/rattle/physical Pareto JSON and Markdown files remain under `physical_triage/`.
