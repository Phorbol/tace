# Stage181 Conventional Molecular MD Validation

This stage adds clean molecular MD deployment splits before continuing renormalized initialization comparisons.

## Datasets

- primary: 3BPA, eV/eV-A extxyz with 300K ID, 600K/1200K OOD, and dihedral PES splits
- secondary: rMD17, npz source requiring kcal/mol to eV conversion before training
- rMD17 smoke wrappers expect converted ethanol train/valid/test extxyz files under `converted_extxyz/`
- 3BPA distribution coverage check compares train_300K against train_mixedT with the same student

## Student

- paths: `atomic.radial_density,atomic.species_basis_density,atomic.local_l0_lowrank_density,atomic.vector_norm,atomic.vector_cross_radial_dot`
- local L0 chemistry rank: `4`

## Required Outputs

- E MAE/RMSE/max per split
- F MAE/RMSE/max per split
- test_600K/test_1200K temperature OOD degradation
- test_dih dihedral PES error
- atoms/s and peak memory per split
