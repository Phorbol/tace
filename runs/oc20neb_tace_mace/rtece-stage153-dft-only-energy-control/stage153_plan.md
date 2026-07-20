# Stage153 DFT-Only Energy Control Plan

## Purpose

Stage153 responds to the persistent rTECE absolute-energy error observed after Stage149-152. Stage149 fixed the batched energy-loss normalization bug, but DFT E RMSE stayed around 93-107 meV/atom. Stage150 ruled out a simple residual E0/gauge correction. Stage151/152 showed that train-side absolute energies remain hard even under overdetermined projection fits, while force projection is much less sensitive to the full edge-relational path set.

This stage runs a paired supervised control under the same production rTECE scalar training and benchmark path:

1. `stage153_l2_dft_only_base2048`: standard train labels are remapped to DFT `dft_energy/dft_forces`.
2. `stage153_l2_mixed_base2048`: standard train labels are the existing 0.75 teacher / 0.25 DFT mixed target.

Both use the Stage152 recommended low-cost semantic front: T2 L2 atomic-cross paths.

## Document Alignment

- `TECE_design_space.md`: projection error and training/distillation error must be separated. If DFT-only training is still energy-poor, the next step is representation/path capacity, not another distillation-weight tweak.
- `rTECE_review.md`: energy RMSE/MAE/max, force RMSE/MAE/max, and relative NEB/barrier metrics must remain separate. Global energy shift is insufficient for multi-element data; Stage153 therefore keeps fitted per-element atomic energies and tests the residual structure.
- Stage152: `t2_l2_atomic_cross` / `prefix_006` is the current low-cost E/F compromise; edge-relational paths are optional measured increments, not the default next Pareto point.

## Fixed Training Contract

- Training backend: Lightning rTECE scalar trainer through `benchmarks/oc20neb_tace_mace/rtece_scalar_matrix.sbatch`.
- Steps: 20,000.
- Batch size: 8; valid batch size: 16.
- LR: `1e-3`; warmup: 500; scheduler: plateau; early stopping patience: 400.
- Energy/force weights: 1.0 / 10.0.
- Per-element atomic energies are fitted from the training targets.
- Short-range baseline: ZBL, matching the review correction.
- Benchmark: DFT valid and teacher valid, 1024 configs, with E/F RMSE/MAE/max plus relative energy decomposition.

## Decision Rule

- If DFT-only E RMSE improves strongly while mixed remains poor, the next priority is distillation objective/data contract: teacher energy mixing, teacher-relax weighting, and source-conditioned E/F weights.
- If DFT-only and mixed are both poor, the dominant issue is student projection/representation capacity on the train/deployment manifold. The next architecture step should add front-loaded TECE capacity: learnable radial/channel projection, species-conditioned radial adapters, selected scalar path interactions, or Schur/GN-initialized path expansion.
- If DFT-only improves energy but loses force/throughput, report the tradeoff as a Pareto point rather than collapsing it into a single MAE.
