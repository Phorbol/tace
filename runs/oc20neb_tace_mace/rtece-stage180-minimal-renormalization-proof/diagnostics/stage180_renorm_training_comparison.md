# Stage180 Renorm Training Comparison

- initializer job: `689028`
- renorm train job: `689033`
- renorm DFT benchmark job: `689120`
- scratch best valid loss: `0.171956` at step `64`
- renorm best valid loss: `0.107413` at step `19776`
- valid loss ratio renorm/scratch: `0.625`

## DFT Benchmark

| arm | E RMSE | E max | F RMSE | F max | rel-image RMSE | barrier RMSE | atoms/s | peak MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `scratch` | 150.241 | 322.693 | 121.982 | 4530.538 | 17.055 | 31.954 | 1223288 | 742.2 |
| `renorm_initialized` | 65.755 | 166.064 | 100.827 | 1638.535 | 8.657 | 12.314 | 1228316 | 742.2 |

## Interpretation

- The same fixed Stage180 student trained from projection-constrained init has lower validation loss than the scratch arm.
- On the matched DFT benchmark, renorm-init improves E RMSE, F RMSE, F max, relative-image RMSE, and barrier RMSE while keeping throughput and memory essentially unchanged.
- This is still not the full physical closure: teacher benchmark, dimer scan, rattle-relax, and scaling diagnostics for the renorm checkpoint remain to be run.
