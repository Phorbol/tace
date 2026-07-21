# Stage184 3BPA Physical Validation Summary

| row | engine | status | test_dih_rmse_f_mev_a | test_300K_rmse_f_mev_a | rattle_mean_final_rmsd_a | rattle_max_fmax_ev_a | dimer_short_repulsive_fraction | atoms/s test_dih |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stage183_l0_local_species | rtece | completed | 227.009 | 178.976 | 0.077 | 16.049 | 1.000 | 2007966.853 |
| stage183_l1_cross | rtece | completed | 212.426 | 135.767 | 0.079 | 15.080 | 1.000 | 1774106.003 |
| nep4_train300k | nep | completed | 116.172 | 120.831 | 0.081 | 13.831 | 0.714 | 51808.491 |
| stage183_l2_atomic_quadrupole | rtece | completed | 160.563 | 119.150 | 0.082 | 14.368 | 1.000 | 1479728.141 |
| stage183_t3_cavity_vecq | rtece | completed | 182.764 | 117.428 | 0.084 | 14.567 | 1.000 | 569490.186 |

Rattle-relax is interpreted as continuous RMSD and force-tail evidence, not as a binary long-term gate.
