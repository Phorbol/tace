# Stage183 Training Submissions

submitted_at_utc: `2026-07-21T12:00:02.780261+00:00`

| row | job_id | status_at_record | wrapper |
| --- | ---: | --- | --- |
| stage183_l0_local_species | 689424 | RUNNING | `runs/oc20neb_tace_mace/rtece-stage183-3bpa-representation-ladder/wrappers/stage183_l0_local_species_no_export.sbatch` |
| stage183_l1_cross | 689426 | RUNNING | `runs/oc20neb_tace_mace/rtece-stage183-3bpa-representation-ladder/wrappers/stage183_l1_cross_no_export.sbatch` |
| stage183_l2_atomic_quadrupole | 689423 | RUNNING | `runs/oc20neb_tace_mace/rtece-stage183-3bpa-representation-ladder/wrappers/stage183_l2_atomic_quadrupole_no_export.sbatch` |
| stage183_t3_cavity_vecq | 689425 | RUNNING | `runs/oc20neb_tace_mace/rtece-stage183-3bpa-representation-ladder/wrappers/stage183_t3_cavity_vecq_no_export.sbatch` |

Followup: after these jobs finish, run each matching benchmark wrapper on test_300K, test_600K, test_1200K, and test_dih, then rank by force RMSE with energy RMSE/max, force max, atoms/s, and memory as required secondary axes.
