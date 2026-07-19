#!/bin/bash
#SBATCH --job-name=tm-summary
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --output=/home/gengjianrui/bin/logs/tace-mace-summary-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/tace-mace-summary-%j.err

set -euo pipefail
set -x

TACE_ROOT=${TACE_ROOT:-/home/gengjianrui/bin/tace}
TACE_JOB_ID=${TACE_JOB_ID:-675969}
BENCH_JOB_ID=${BENCH_JOB_ID:-675979}
RUN_BASE=${RUN_BASE:-/home/gengjianrui/bin/tace/runs/oc20neb_tace_mace}
MACE_MATRIX=${MACE_MATRIX:-/home/gengjianrui/worktrees/mace-update-boundary/runs/oc20neb_fullcase200_ef_20k/674543/matrix_summary.json}
TACE_RUN=${TACE_RUN:-${RUN_BASE}/${TACE_JOB_ID}}
BENCH_RUN=${BENCH_RUN:-${RUN_BASE}/bench-${BENCH_JOB_ID}}
OUT_JSON=${OUT_JSON:-${RUN_BASE}/comparison_tace-${TACE_JOB_ID}_bench-${BENCH_JOB_ID}.json}
OUT_MD=${OUT_MD:-${RUN_BASE}/comparison_tace-${TACE_JOB_ID}_bench-${BENCH_JOB_ID}.md}

python3 "${TACE_ROOT}/benchmarks/oc20neb_tace_mace/summarize_results.py" \
  --mace-matrix-summary "${MACE_MATRIX}" \
  --mace-benchmark "${BENCH_RUN}/mace_benchmark.json" \
  --tace-benchmark "${BENCH_RUN}/tace_benchmark.json" \
  --tace-log "/home/gengjianrui/bin/logs/tace-oc20neb-${TACE_JOB_ID}.out" \
  --tace-dmon "${TACE_RUN}/nvdmon_job-${TACE_JOB_ID}_tace.log" \
  --benchmark-dmon "${BENCH_RUN}/nvdmon_job-${BENCH_JOB_ID}_bench.log" \
  --tace-job-id "${TACE_JOB_ID}" \
  --output-json "${OUT_JSON}" \
  --output-md "${OUT_MD}"
