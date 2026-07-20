# Stage146 Energy Gauge Status

Question: why does the Stage144 T3 cavity-vector rTECE point have much worse energy MAE/RMSE than force RMSE, and is this mostly an E0/reference-gauge problem?

Document alignment:
- `TECE_design_space.md` requires atomic-energy distillation to align element baselines / average gauge before using atomic or energy labels.
- `rTECE_review.md` explicitly flags global per-atom energy shift as insufficient for multi-element data and requires E MAE/RMSE/max/bias plus force metrics.
- `rTECE_review.md` also warns that NEB-like data should not be judged only by absolute energy MAE; relative image energies and barrier errors must be tracked.

Findings so far:
- Existing same-slice diagnostic was too optimistic: fitting per-element residual E0 on the same held-out slice reduced E RMSE to `6.848 meV/atom`, but that is a diagnostic upper bound, not a deployable correction.
- Disjoint valid split `calib :32 -> eval 32:96`: no calibration E RMSE `80.905`; global `32.327`; per-element `79.010 meV/atom`. Per-element fit overfits the small calibration set.
- Disjoint valid split `calib :128 -> eval 128:384`: no calibration E RMSE `159.182`; global `159.895`; per-element `139.492 meV/atom`. Per-element residual E0 helps somewhat, but does not solve the energy problem.
- Formal full-window split `calib :128 -> eval 128:1024`: no calibration E RMSE `295.887`; global `297.699`; per-element `295.032 meV/atom`. Per-element residual E0 fits the calibration slice to `5.218 meV/atom`, but does not transfer to the broad eval window.
- On the same full-window split, grouped by `case_id` and ordered by `source_frame`, the window-relative NEB shape is much better: relative image RMSE `8.394 meV/atom`, relative image MAE `5.964 meV/atom`, barrier RMSE `18.386 meV/atom`, barrier MAE `15.232 meV/atom` over 20 groups / 896 images.
- Stage148 decomposition added a stronger diagnostic: raw full-window E RMSE is `295.887 meV/atom`, global per-atom offset only reduces it to `292.482 meV/atom`, but removing one mean offset per NEB `case_id` reduces residual E RMSE to `6.619 meV/atom`; anchoring each case by its first image leaves `17.181 meV/atom`.
- Forces are unchanged by energy-only calibration: full-window eval F RMSE is `109.196 meV/A`, F MAE is `45.702 meV/A`.

Interpretation:
- The bad absolute energy RMSE is real under the current benchmark protocol.
- It is not fixed by a scalar shift, and simple per-element residual E0 does not transfer across the broad validation window.
- The dominant residual is case/slab-level energy reference rather than image-local shape: per-case mean offset nearly removes absolute E error, while first-image anchoring leaves a residual comparable to the barrier error.
- This does not make the absolute energy error acceptable for thermodynamics or composition transfer. It narrows the root cause: next experiments should isolate source-consistent/case-consistent energy references before spending architecture capacity on local PES shape.
- The current Pareto protocol must therefore report both absolute E/F RMSE/max/bias and NEB-relative image/barrier/decomposition metrics. Ranking by absolute energy alone would understate the deployable PES-shape quality; ranking by force alone would miss gauge and thermodynamic-transfer failures.

Next actions:
1. Keep relative/path/decomposition energy metrics in standard rTECE/community benchmark summaries, not only this diagnostic script.
2. Train an ablation with DFT-only energy anchors or source/case-consistent energy references, while keeping teacher-relax force distillation separate.
3. Add a formal stage for case-offset decomposition on rTECE, NEP, and DeepMD outputs so the community baselines are judged under the same absolute/relative/gauge protocol.
4. Continue community baseline comparison after NEP finishes, using the same absolute E/F RMSE/max/bias, relative image/barrier metrics, throughput, and physical tests.
