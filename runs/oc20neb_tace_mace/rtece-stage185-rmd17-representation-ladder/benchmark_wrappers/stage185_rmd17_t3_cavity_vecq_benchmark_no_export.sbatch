#!/bin/bash
#SBATCH --job-name=rtece-st185-bench-cavity_vector_quadrupole_edge_relational_sketch
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=01:40:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-st185-bench-cavity_vector_quadrupole_edge_relational_sketch-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-st185-bench-cavity_vector_quadrupole_edge_relational_sketch-%j.err

set -eo pipefail
set -x
TACE_ROOT=/home/gengjianrui/bin/tace
TACE_PYTHON=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python
PATH=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin:${PATH}
mkdir -p /home/gengjianrui/bin/logs
cd ${TACE_ROOT}
${TACE_PYTHON} -V

mkdir -p runs/oc20neb_tace_mace/rtece-stage185-rmd17-representation-ladder/diagnostics/stage185_rmd17_t3_cavity_vecq
PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model runs/oc20neb_tace_mace/rtece-stage185-rmd17-representation-ladder/results/stage185_rmd17_t3_cavity_vecq/rtece_scalar_best.pt --configs datasets/rMD17/converted_extxyz/rmd17_ethanol_train.extxyz --output runs/oc20neb_tace_mace/rtece-stage185-rmd17-representation-ladder/diagnostics/stage185_rmd17_t3_cavity_vecq/train_benchmark.json --variant stage185_rmd17_t3_cavity_vecq_train --energy-key energy --forces-key forces --start-config 0 --limit-configs 1000 --measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist
PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model runs/oc20neb_tace_mace/rtece-stage185-rmd17-representation-ladder/results/stage185_rmd17_t3_cavity_vecq/rtece_scalar_best.pt --configs datasets/rMD17/converted_extxyz/rmd17_ethanol_valid.extxyz --output runs/oc20neb_tace_mace/rtece-stage185-rmd17-representation-ladder/diagnostics/stage185_rmd17_t3_cavity_vecq/valid_benchmark.json --variant stage185_rmd17_t3_cavity_vecq_valid --energy-key energy --forces-key forces --start-config 0 --limit-configs 1000 --measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist
PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model runs/oc20neb_tace_mace/rtece-stage185-rmd17-representation-ladder/results/stage185_rmd17_t3_cavity_vecq/rtece_scalar_best.pt --configs datasets/rMD17/converted_extxyz/rmd17_ethanol_test.extxyz --output runs/oc20neb_tace_mace/rtece-stage185-rmd17-representation-ladder/diagnostics/stage185_rmd17_t3_cavity_vecq/test_benchmark.json --variant stage185_rmd17_t3_cavity_vecq_test --energy-key energy --forces-key forces --start-config 0 --limit-configs 1000 --measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist
