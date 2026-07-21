# Stage180 Renorm Training Comparison

- initializer job: `689028`
- renorm train job: `689033`
- scratch best valid loss: `0.171956` at step `64`
- renorm best valid loss: `0.107413` at step `19776`
- valid loss ratio renorm/scratch: `0.625`

## Interpretation

- The same fixed Stage180 student trained from projection-constrained init has a lower validation loss than the scratch arm.
- This is not yet a full TECE/rTECE win: matched E/F RMSE/max, physical diagnostics, and throughput/memory must still be run for the renorm checkpoint.
