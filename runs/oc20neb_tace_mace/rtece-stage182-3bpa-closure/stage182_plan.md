# Stage182 3BPA Conventional Closure

Goal: compare the TECE-derived scalar student against NEP4 and DPA1 zero-attention-style baselines on one clean molecular-MD deployment distribution.

## Controlled Variables

- train split: `train_300K`
- train configs: `500`
- benchmark configs per split: `512`
- labels: `energy` and `forces` in eV/eV-A

## Splits

- `test_300K`
- `test_600K`
- `test_1200K`
- `test_dih`

## Interpretation

- If rTECE is much slower than NEP/DPA1 at similar error, the front-end descriptor path is still too expensive.
- If rTECE is much less accurate at similar train coverage, the current scalarized TECE projection lacks needed representation rank.
- If rTECE has competitive force RMSE but poor energy RMSE/max, prioritize energy-gauge/E0 and residual distillation before kernel work.
