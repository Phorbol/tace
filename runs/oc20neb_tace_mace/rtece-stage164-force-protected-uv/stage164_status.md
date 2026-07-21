# Stage164 Status

| variant | job | state | artifact status |
|---|---:|---|---|
| `stage164_uv_force_gate_rel0p25_b32` | 687573 | RUNNING after Lightning init | trainable params logged as 55.1K; checkpoints started; waiting for train summary and DFT/teacher benchmarks |

Selection contract: Stage163 `uv` only, sampled force regression gate <= 10%, positive energy gain required.

Submission wrapper avoided `--export`, `--mem`, `--cpus-per-task`, and `set -u`.
