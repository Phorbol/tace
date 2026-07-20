# Stage144 Energy Diagnostic

Scope: diagnose why the full Stage144 T3 cavity-vector rTECE point has acceptable force RMSE but poor energy RMSE on the common DFT validation benchmark.

## Observations

- Full DFT benchmark: E RMSE `279.955 meV/atom`, E max `636.649 meV/atom`, E bias `-42.267 meV/atom`; F RMSE `107.972 meV/A`.
- Limit-256 recompute: E RMSE `116.741 meV/atom`, E max `287.049 meV/atom`, F RMSE `93.365 meV/A`.
- A global residual shift does not help on limit-256: E RMSE becomes `130.061 meV/atom`.
- A post-hoc per-element residual E0 fit on the same held-out slice reduces E RMSE to `6.848 meV/atom` and E max to `21.890 meV/atom`.

## E0 Source Consistency

- Stage142 train DFT subset with available `dft_energy`: `2048/2688` configs; per-element E0-only fit RMSE `214.088 meV/atom`.
- Stage142 train teacher subset with available `teacher_energy`: `2048/2688` configs; per-element E0-only fit RMSE `204.611 meV/atom`.
- Mixed training `energy` key over all configs: E0-only fit RMSE `191.393 meV/atom`.
- Common valid DFT split: E0-only fit RMSE `12.817 meV/atom`.
- Common-element valid-minus-train DFT E0 mean absolute difference `2.228 eV`, max `13.471 eV`.

## Interpretation

- The energy failure is not explained by a single scalar bias.
- The limit-256 residual is mostly composition/element-reference correlated: a held-out post-hoc per-element residual E0 nearly removes it. This is diagnostic only and must not be used as a test-set calibration result.
- The augmented Stage142 training source mixes DFT and teacher-relax labels; not every training sample has `dft_energy`. The valid split stores DFT labels in ASE `SinglePointCalculator.results` as `energy/forces`.
- Next priority: make energy gauge explicit in the Pareto protocol: train-source E0, deployment-calibration E0 diagnostics, relative/path energy metrics, and energy-weight/force-weight ablations. Do not rank models by force RMSE alone.
