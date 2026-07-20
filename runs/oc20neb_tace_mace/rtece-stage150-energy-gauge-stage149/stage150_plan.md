# Stage150 Energy Gauge Diagnostic for Stage149

Purpose: determine whether the remaining Stage149 absolute energy RMSE is mostly a transferable baseline/gauge error or a representation/data limitation. This follows rTECE_review.md section 7 and TECE_design_space.md section 8.3: atomic energy/gauge must be separated from total E/F/virial/relative PES shape.

Inputs:
- Stage149 L2 anchor loss-normalized checkpoint.
- Stage149 T3 cavity-vector loss-normalized checkpoint.
- Common OC20NEB valid split.

Protocol:
- Calibration window: valid configs 0:128.
- Evaluation window: valid configs 128:1024.
- Compare no correction, global residual shift, and per-element residual E0 on the evaluation window.
- Report E/F RMSE/MAE/max/bias plus relative image, barrier, group mean-offset and first-anchor metrics.

Interpretation rules:
- If per-element residual calibration collapses E RMSE on disjoint eval without changing relative/barrier metrics, the next architecture step is an explicit environment/case-aware baseline or mixture baseline, not more edge paths.
- If residual calibration does not transfer, the next step is representation/data capacity: teacher-selected scalar paths, active-set projection, and/or broader teacher fake-label coverage.
- These calibrated metrics are diagnostics only, not final benchmark scores.
