# Stage129 physical triage summary

## Completed jobs

| job | variant | state | elapsed |
| --- | --- | --- | --- |
| 685077 | l1_active_nrad12_species20_radial_species8_cross3_h64 | COMPLETED 0:0 | 00:00:07 |
| 685078 | l1_active_nrad12_species24_radial_species8_cross3_h64 | COMPLETED 0:0 | 00:00:07 |

These jobs ran the same physical tests used in stage128: dimer scans for C-N, C-O, C-H, N-H, O-H, C-C, N-N over 0.5-5.0 scale, followed by C/N-focused rattle-relax on validation configs 58:66.

## Results

| variant | physical gate | benchmark gate | dimer gate | rattle gate | physical score | rattle max fmax | C/N RMSD | DFT E RMSE | DFT F RMSE | DFT F max |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| species20/cross3 | fail | pass | pass | fail | 17.978 | 14.022 | 0.107 | 300.099 | 107.653 | 2387.490 |
| species24/cross3 | fail | fail | pass | fail | 14.778 | 10.836 | 0.117 | 273.849 | 101.557 | 2697.533 |

## Comparison to stage128

| variant | stage128 rattle fmax | stage129 rattle fmax | delta | stage128 physical score | stage129 physical score | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| species20/cross3 | 10.809 | 14.022 | +3.213 | 14.580 | 17.978 | +3.398 |
| species24/cross3 | 9.971 | 10.836 | +0.865 | 13.897 | 14.778 | +0.881 |

## Interpretation

The stage129 teacher-rattle patch is a negative result for the physical failure mode. Dimer behavior remains clean, so the short-range/ZBL side is not the issue here. The C/N rattle force maxima remain far above the 1.0 eV/A gate and are worse than stage128 for both fixed architectures, while C/N RMSD is roughly unchanged.

This means the next TECE-consistent priority should not be another small local fake-label patch. The evidence points toward either broader teacher trajectory coverage, force/Jacobian-weighted distillation, or a front-loaded trainable representation change that directly controls high-force tails without C/N-specific overfitting.
