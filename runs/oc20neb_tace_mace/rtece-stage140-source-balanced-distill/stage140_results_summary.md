# Stage140 source-balanced distillation results

## What changed

Stage140 first fixed weighted E0 fitting (`9eafa37`), then trained two real sbatch rows on the same force-only teacher-relax dataset. Teacher-relax frames keep force/Sobolev weight but have `energy_weight=0`, and weighted least-squares E0s now respect that zero.

## Sbatch evidence

| task | job | state | exit | elapsed | node | MaxRSS(batch) |
|---|---:|---|---|---:|---|---:|
| train_ew1 | 685450 | COMPLETED | 0:0 | 00:09:52 | 16v100n10 | 2005884K |
| train_ew2 | 685449 | COMPLETED | 0:0 | 00:10:24 | 16v100n02 | 2005608K |
| physical_ew1 | 685485 | COMPLETED | 0:0 | 00:00:19 | 16v100n01 | 576828K |
| physical_ew2 | 685486 | COMPLETED | 0:0 | 00:00:15 | 16v100n13 | 567908K |

## Pareto comparison

All error units are meV/atom for energy and meV/A for forces.

| row | params | atoms/s | E RMSE | E max | F RMSE | F max | C/N RMSD | bench | physical score |
|---|---|---|---|---|---|---|---|---|---|
| stage132_l1_atomic | 28925 | 1217882 | 293.391 | 697.301 | 108.355 | 1840.539 | 0.103 | yes | 14.298 |
| stage137_l2_cond32 | 51367 | 1023496 | 340.032 | 740.266 | 107.516 | 1914.572 | 0.111 | no | 14.403 |
| stage139_l2_cond32_relax_distill | 51367 | 1024003 | 388.377 | 794.805 | 103.073 | 1709.661 | 0.101 | no | 13.639 |
| stage140_forceonly_ew1 | 51367 | 1022239 | 324.591 | 679.403 | 104.090 | 1953.044 | 0.106 | no | 13.681 |
| stage140_forceonly_ew2 | 51367 | 1014674 | 306.507 | 677.391 | 96.851 | 2006.939 | 0.095 | yes | 12.426 |

## Interpretation

- Stage140 ew2 is the first L=2 conditioned-front distillation row here that simultaneously passes the E/F benchmark gate and improves the force/RMSD frontier: E RMSE 306.507 meV/atom, F RMSE 96.851 meV/A, C/N RMSD 0.094746 A at 1.015M atoms/s.
- Relative to Stage139 cond32, force-only teacher-relax plus weighted E0s fixes the energy failure: E RMSE improves from 388.377 to 306.507 meV/atom and E max from 794.805 to 677.391 meV/atom. Force RMSE also improves from 103.073 to 96.851 meV/A, although Stage139 still had lower F max (1709.661 vs 2006.939 meV/A).
- Relative to Stage137 cond32, ew2 improves E RMSE, E max, F RMSE, C/N RMSD, benchmark gate, and physical score at essentially unchanged throughput. This supports the document hypothesis that Stage139 was mainly a distillation-measure/E0-source problem, not immediate evidence that the L=2 conditioned scalarized representation is intrinsically invalid.
- ew1 partly fixes Stage139 energy but misses the E RMSE gate at 324.591 meV/atom. Increasing global energy loss weight to 2.0 is a useful Pareto knob for this source-balanced measure rather than just head widening.
- The rattle gate still fails because relaxed fmax remains high, so this is not a final physical-closure result. It is a strong algorithmic validation point for source-balanced renormalized distillation.
- short_range_repulsion_potential is configured as zbl but strength is 0.0, so Stage140 is not evidence for active ZBL stabilization.

## Next priority

- Treat Stage140 ew2 as the current source-balanced L=2 conditioned-front Pareto anchor.
- Run a narrow Stage141 loss-measure frontier around ew2, not a broad architecture sweep: energy_weight 1.5/2.0/3.0 and possibly teacher force multiplier 1.5/2.0 while keeping architecture fixed.
- After selecting the loss point, run atom-count throughput scaling and then decide whether to reintroduce true edge-relational scalar sketches from the TECE document.

## Files

- Machine-readable summary: `runs/oc20neb_tace_mace/rtece-stage140-source-balanced-distill/stage140_results_summary.json`
- Stage140 train summaries and benchmark JSON files are under `source_balanced_runs/`.
- Stage140 dimer/rattle/physical Pareto outputs are under `physical_triage/`.
