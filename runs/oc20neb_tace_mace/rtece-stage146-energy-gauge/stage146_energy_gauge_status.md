# Stage146 Energy Gauge Status

Question: why does the Stage144 T3 cavity-vector rTECE point have much worse energy MAE/RMSE than force RMSE, and is this mostly an E0/reference-gauge problem?

Document alignment:
- `TECE_design_space.md` requires atomic-energy distillation to align element baselines / average gauge before using atomic or energy labels.
- `rTECE_review.md` explicitly flags global per-atom energy shift as insufficient for multi-element data and requires E MAE/RMSE/max/bias plus force metrics.

Findings so far:
- Existing same-slice diagnostic was too optimistic: fitting per-element residual E0 on the same held-out slice reduced E RMSE to `6.848 meV/atom`, but that is a diagnostic upper bound, not a deployable correction.
- Disjoint valid split `calib :32 -> eval 32:96`: no calibration E RMSE `80.905`; global `32.327`; per-element `79.010 meV/atom`. Per-element fit overfits the small calibration set.
- Disjoint valid split `calib :128 -> eval 128:384`: no calibration E RMSE `159.182`; global `159.895`; per-element `139.492 meV/atom`. Per-element residual E0 helps somewhat, but does not solve the energy problem.
- Formal full-window split `calib :128 -> eval 128:1024`: no calibration E RMSE `295.887`; global `297.699`; per-element `295.033 meV/atom`. Per-element residual E0 fits the calibration slice to `5.218 meV/atom`, but does not transfer to the broad eval window.
- Forces are unchanged by energy-only calibration: full-window eval F RMSE is `109.196 meV/A`, F MAE is `45.702 meV/A`.

Interpretation:
- The bad energy RMSE is real under the current benchmark protocol.
- It is not fixed by a scalar shift, and simple per-element residual E0 does not transfer across the broad validation window.
- The current train/valid label source and composition/gauge mismatch is therefore a first-order blocker for comparing absolute energies. We should report relative/path energies and benchmark disjoint gauge calibration, while changing training to keep energy references source-consistent.

Next actions:
1. Train an ablation with DFT-only energy anchors or source-consistent energy references, while keeping teacher-relax force distillation separate.
2. Add relative NEB/path energy metrics so the Pareto ranking is not dominated by absolute composition gauge.
3. Continue community baseline comparison after NEP finishes, using the same E/F RMSE/max/bias and physical tests.
