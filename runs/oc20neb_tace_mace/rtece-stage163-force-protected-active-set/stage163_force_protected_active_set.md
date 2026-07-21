# Stage163 Force-Protected Active-Set Rerank

Stage162 showed that raw energy RMSE is dominated by case/slab/adsorbate offsets, while Stage161 showed that the `uv` relational scalar can repair projected energy but worsens sampled-force RMSE. Stage163 turns that interpretation into an explicit active-set promotion gate.

Source projection: `stage161_current_baseline_limit64_holdout_edgeframe_sampled_force_relative_projection.json`.

## Force Regression Gate 0%

Promoted candidates: `none`

| candidate | promoted | E RMSE | sampled F RMSE | E gain frac | F gain frac | force regression frac | weighted gain | rejection reasons |
|---|---|---:|---:|---:|---:|---:|---:|---|
| uv | no | 11.239 | 85.290 | 0.703 | -0.079 | 0.079 | 0.624 | force_regression_fraction |
| uv_uqu | no | 14.299 | 103.795 | 0.622 | -0.313 | 0.313 | 0.309 | force_regression_fraction |
| uqu | no | 7954.103 | 89.967 | -209.123 | -0.139 | 0.139 | -209.262 | energy_gain_required, force_regression_fraction |

## Force Regression Gate 10%

Promoted candidates: `uv`

| candidate | promoted | E RMSE | sampled F RMSE | E gain frac | F gain frac | force regression frac | weighted gain | rejection reasons |
|---|---|---:|---:|---:|---:|---:|---:|---|
| uv | yes | 11.239 | 85.290 | 0.703 | -0.079 | 0.079 | 0.624 | - |
| uv_uqu | no | 14.299 | 103.795 | 0.622 | -0.313 | 0.313 | 0.309 | force_regression_fraction |
| uqu | no | 7954.103 | 89.967 | -209.123 | -0.139 | 0.139 | -209.262 | energy_gain_required, force_regression_fraction |

## Force Regression Gate 25%

Promoted candidates: `uv`

| candidate | promoted | E RMSE | sampled F RMSE | E gain frac | F gain frac | force regression frac | weighted gain | rejection reasons |
|---|---|---:|---:|---:|---:|---:|---:|---|
| uv | yes | 11.239 | 85.290 | 0.703 | -0.079 | 0.079 | 0.624 | - |
| uv_uqu | no | 14.299 | 103.795 | 0.622 | -0.313 | 0.313 | 0.309 | force_regression_fraction |
| uqu | no | 7954.103 | 89.967 | -209.123 | -0.139 | 0.139 | -209.262 | energy_gain_required |

## Interpretation

- `uv` is conditional, not blindly promotable. It fails a zero-regression force gate but passes a 10% sampled-force regression gate because its force RMSE regression is about 7.9% while its energy RMSE gain is about 70.3%.
- `uv_uqu` remains rejected under a 25% gate because its sampled-force regression is about 31.3%; this matches the Stage161 dominated-rank interpretation.
- `uqu` remains rejected because its energy projection is catastrophic and its force also regresses.
- The next trainable model experiment should therefore test only the `uv` path, and only under an explicitly force-protected loss/early-selection rule that reports raw E RMSE, group-offset E RMSE, relative-image/barrier RMSE, F RMSE/MAE/max, and physical probes together.
