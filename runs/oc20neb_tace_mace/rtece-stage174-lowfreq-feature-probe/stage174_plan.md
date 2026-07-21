# Stage174 Low-Frequency Feature Probe Plan

- stage: `stage174_lowfreq_feature_probe`
- target: `stage165_case_offset_residual_mev_atom`
- split: `group-loocv` by `case_id`
- configs: `/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz`
- Stage165 source: `runs/oc20neb_tace_mace/rtece-stage165-energy-baseline-triage/stage165_energy_baseline_triage.json`
- limit configs: `512`
- ridge grid: `1e-08,1e-06,0.0001,0.01,1,100`

## Question

After Stage173 showed current rTECE path descriptors do not beat an intercept-only unseen-case offset baseline, can deployable low-frequency chemistry/site/front summaries provide a group-heldout coordinate worth promoting into the next trainable rTECE front module?

## Feature Families

| family | requires tags | TECE semantics |
|---|---|---|
| composition_fraction | False | low_frequency_species_composition_scalar |
| tag_composition | True | tag_conditioned_adsorbate_slab_species_scalar |
| geometry_z_profile | False | coarse_cell_and_surface_normal_geometry_scalar |
| composition_tag_geometry | True | tag_conditioned_low_frequency_site_front_scalar |
| pair_histogram | False | coarse_local_pair_radial_histogram_scalar |

## Review Basis

- Stage170: raw absolute E RMSE is dominated by low-frequency case/site/adsorbate offsets.
- Stage173: robust current rTECE semantic path descriptors do not beat the intercept-only group-heldout baseline.
- rTECE_review.md: low-rank species chemistry and per-element references are required, while case/source metadata must not become deployable features.
- TECE_design_space.md: model degradation should retain high-value low-cost scalar coordinates selected by heldout physical/error benefit.
