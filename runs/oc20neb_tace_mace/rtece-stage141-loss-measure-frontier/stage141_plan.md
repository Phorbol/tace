# Stage141 loss-measure frontier

This stage holds the Stage140 L=2 conditioned rTECE architecture fixed and varies only the loss/data measure around the best source-balanced distillation row. The purpose is to separate TECE operator-subspace capacity from Sobolev/deployment-measure weighting.

## Document alignment

- `TECE_design_space.md`: evaluate a fixed selected operator basis under energy-force Sobolev/deployment metrics before spending more hardware budget on new paths.
- `rTECE_review.md`: after weighted E0 repair, verify distillation closure with teacher cache/labels, E/F RMSE, max errors, and physical dimer/rattle tests.

## Fixed architecture

- L=2 scalarized moments: radial/species density, vector norm, vector-cross-radial dot, quadrupole norm, quadrupole-cross-radial Frobenius.
- Learnable front-end: radial mixing, species embedding basis, radial-species adapter, learnable cross-radial projection, residual descriptor conditioner.
- No head-width expansion, no new edge path, no kernel optimization in this stage.

## Rows

- `l2_active_nrad12_species24_radial_species8_cross3_cond32_fm1p5_ew2_h64`: teacher force multiplier 1.5, ENERGY_WEIGHT 2.0. lower teacher-relax force measure may reduce force-tail artifacts and rattle relaxed fmax while retaining Stage140 ew2 energy anchor
- `l2_active_nrad12_species24_radial_species8_cross3_cond32_fm2p5_ew2_h64`: teacher force multiplier 2.5, ENERGY_WEIGHT 2.0. higher teacher-relax force measure tests whether physical relaxation and force max improve when Sobolev mass is moved toward teacher trajectories
- `l2_active_nrad12_species24_radial_species8_cross3_cond32_fm2p0_ew3_h64`: teacher force multiplier 2.0, ENERGY_WEIGHT 3.0. keep Stage140 best force-only teacher measure but strengthen global DFT energy anchor to test energy RMSE/max boundary

## Gates and readout

- Primary metrics: DFT E RMSE, F RMSE, E max error, F max error, throughput.
- Physical metrics: dimer smoothness/finite scan and rattle-relax RMSD/fmax, with C/N focus reported but not used as the only long-term gate.
- Selection must explain whether the bottleneck is loss measure, teacher-label volume, or representation capacity/projection error.
