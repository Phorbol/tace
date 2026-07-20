# Stage142 Teacher Relax Coverage Results

Stage142 fixed the Stage137/140 L=2 conditioned rTECE architecture and only enlarged teacher-relax coverage. The training set was `2048` base DFT configs plus `640` teacher-relax trajectory frames. Teacher-relax frames were force-only: energy multiplier `0.0`, force multiplier `2.0`; the base set retained the DFT energy anchor.

## Jobs

| job | role | state | exit | elapsed |
|---:|---|---|---|---|
| 685578 | prepare | COMPLETED | 0:0 | 00:04:25 |
| 685579 | train | COMPLETED | 0:0 | 00:11:59 |
| 685580 | physical triage | COMPLETED | 0:0 | 00:00:09 |

## Data And Training Contract

- Weighted train configs: `2688` configs / `139746` atoms.
- Source counts: base `2048`, teacher-relax `640`.
- Energy multipliers: base `1.25`, teacher-relax `0.0`.
- Force multiplier: teacher-relax `2.0`.
- Training: Lightning, batch `8`, valid batch `16`, max steps `20000`, best step `18816`, warmup `500`, early stopping patience `400`.

## Metrics

| target | atoms/s | E MAE | E RMSE | E max | E bias | F MAE | F RMSE | F max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DFT | 1013546 | 239.996 | 304.537 | 730.571 | -17.913 | 46.596 | 107.622 | 1758.278 |
| Teacher | 1014482 | 224.832 | 286.460 | 689.331 | -21.889 | 47.574 | 105.643 | 1721.411 |

Rattle/relax summary: converged fraction `0.0`, mean final RMSD `0.104110 A`, max final RMSD `0.156495 A`, mean max fmax `4.671083 eV/A`, max fmax `11.093983 eV/A`.

## Interpretation

This is a negative or mixed result. Relative to Stage140/141, Stage142 does not create a new DFT RMSE or physical-rattle frontier. It slightly lowers benchmark force max versus some references, but worsens force RMSE, C/N RMSD, and rattle fmax.

The result supports the TECE design-space decomposition: more same-window teacher coverage does not remove architecture/projection error. The next priority is semantic path active-set/projection diagnostics, not another same-family teacher-relax coverage expansion.
