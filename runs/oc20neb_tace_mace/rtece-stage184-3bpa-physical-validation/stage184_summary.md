# Stage184 3BPA Physical Validation Summary

| row | engine | status | test_dih_rmse_f_mev_a | test_300K_rmse_f_mev_a | rattle_mean_final_rmsd_a | rattle_max_fmax_ev_a | dimer_short_repulsive_fraction | atoms/s test_dih |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nep4_train300k | nep | missing | 116.172 | 120.831 |  |  |  | 51808.491 |
| stage183_l2_atomic_quadrupole | rtece | missing | 160.563 | 119.150 |  |  |  | 1479728.141 |
| stage183_t3_cavity_vecq | rtece | missing | 182.764 | 117.428 |  |  |  | 569490.186 |
| stage183_l1_cross | rtece | missing | 212.426 | 135.767 |  |  |  | 1774106.003 |
| stage183_l0_local_species | rtece | missing | 227.009 | 178.976 |  |  |  | 2007966.853 |

Rattle-relax is interpreted as continuous RMSD and force-tail evidence, not as a binary long-term gate.
