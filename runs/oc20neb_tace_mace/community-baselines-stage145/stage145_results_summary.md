# Stage145 Community Baselines Summary

Primary ranking metric: DFT force RMSE; relative image/barrier RMSE is reported for NEB PES shape; case/first-anchor offsets diagnose energy gauge.

| row | engine | conversion | training | DFT F RMSE | DFT E RMSE | rel image RMSE | barrier RMSE | case offset RMSE | first anchor RMSE | atoms/s | physical |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| nep4_mixed_smoke | nep | found | completed | 143.037 | 23.846 | 10.068 | 19.344 | 7.557 | 17.207 | 103226.501 | found |
| deepmd_dpa_like_mixed_smoke | deepmd | found | completed | 109.859 | 325.281 | 17.009 | 39.474 | 12.538 | 31.317 | 10916.840 | found |

Missing values are explicit because conversion, training, benchmark, and physical triage complete at different times.
