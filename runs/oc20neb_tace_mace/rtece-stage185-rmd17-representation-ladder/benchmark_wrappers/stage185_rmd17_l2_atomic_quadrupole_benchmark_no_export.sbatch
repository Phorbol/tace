#!/bin/bash
#SBATCH --job-name=rtece-st185-bench-l2_atomic_quadrupole_without_edge_cost
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=01:40:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-st185-bench-l2_atomic_quadrupole_without_edge_cost-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-st185-bench-l2_atomic_quadrupole_without_edge_cost-%j.err

set -eo pipefail
set -x
TACE_ROOT=/home/gengjianrui/bin/tace
TACE_PYTHON=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python
PATH=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin:${PATH}
mkdir -p /home/gengjianrui/bin/logs
cd ${TACE_ROOT}
${TACE_PYTHON} -V

mkdir -p runs/oc20neb_tace_mace/rtece-stage185-rmd17-representation-ladder/diagnostics/stage185_rmd17_l2_atomic_quadrupole
PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model runs/oc20neb_tace_mace/rtece-stage185-rmd17-representation-ladder/results/stage185_rmd17_l2_atomic_quadrupole/rtece_scalar_best.pt --configs datasets/rMD17/converted_extxyz/rmd17_ethanol_train.extxyz --output runs/oc20neb_tace_mace/rtece-stage185-rmd17-representation-ladder/diagnostics/stage185_rmd17_l2_atomic_quadrupole/train_benchmark.json --variant stage185_rmd17_l2_atomic_quadrupole_train --energy-key energy --forces-key forces --start-config 0 --limit-configs 1000 --measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist
PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model runs/oc20neb_tace_mace/rtece-stage185-rmd17-representation-ladder/results/stage185_rmd17_l2_atomic_quadrupole/rtece_scalar_best.pt --configs datasets/rMD17/converted_extxyz/rmd17_ethanol_valid.extxyz --output runs/oc20neb_tace_mace/rtece-stage185-rmd17-representation-ladder/diagnostics/stage185_rmd17_l2_atomic_quadrupole/valid_benchmark.json --variant stage185_rmd17_l2_atomic_quadrupole_valid --energy-key energy --forces-key forces --start-config 0 --limit-configs 1000 --measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist
PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model runs/oc20neb_tace_mace/rtece-stage185-rmd17-representation-ladder/results/stage185_rmd17_l2_atomic_quadrupole/rtece_scalar_best.pt --configs datasets/rMD17/converted_extxyz/rmd17_ethanol_test.extxyz --output runs/oc20neb_tace_mace/rtece-stage185-rmd17-representation-ladder/diagnostics/stage185_rmd17_l2_atomic_quadrupole/test_benchmark.json --variant stage185_rmd17_l2_atomic_quadrupole_test --energy-key energy --forces-key forces --start-config 0 --limit-configs 1000 --measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist
