# Stage182 to Stage183 Analysis

- source: `runs/oc20neb_tace_mace/rtece-stage182-3bpa-closure/stage182_summary.json`
- status: `completed`
- primary failure mode: `dihedral_pes_gap`
- next decision: `representation_ladder_before_more_kernel_work`

## Force RMSE Ratios rTECE / NEP

- `test_300K`: 1.124
- `test_600K`: 1.140
- `test_1200K`: 1.082
- `test_dih`: 1.829

## Energy RMSE Ratios rTECE / NEP

- `test_300K`: 2.326
- `test_600K`: 1.291
- `test_1200K`: 1.072
- `test_dih`: 6.354

## Priority

- `l2_atomic_quadrupole`
- `t3_cavity_edge_relational`

Stage182 shows the L1/local-L0 student is roughly NEP-close on ID/temperature force RMSE but much worse on the dihedral PES split; the clean next test is L2 atomic quadrupoles and then cavity edge-relational sketches before kernel work or head widening.
