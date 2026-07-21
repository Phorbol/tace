#!/bin/bash
#SBATCH --job-name=rtece-st183-bench-cavity_vector_quadrupole_edge_relational_sketch
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=01:40:00
#SBATCH --output=/home/gengjianrui/bin/logs/rtece-st183-bench-cavity_vector_quadrupole_edge_relational_sketch-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/rtece-st183-bench-cavity_vector_quadrupole_edge_relational_sketch-%j.err

set -eo pipefail
set -x
TACE_ROOT=/home/gengjianrui/bin/tace
TACE_PYTHON=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python
PATH=/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin:${PATH}
mkdir -p /home/gengjianrui/bin/logs
cd ${TACE_ROOT}
${TACE_PYTHON} -V

mkdir -p runs/oc20neb_tace_mace/rtece-stage183-3bpa-representation-ladder/diagnostics/stage183_t3_cavity_vecq
PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model runs/oc20neb_tace_mace/rtece-stage183-3bpa-representation-ladder/results/stage183_t3_cavity_vecq/rtece_scalar_best.pt --configs datasets/3BPA/dataset_3BPA/test_300K.xyz --output runs/oc20neb_tace_mace/rtece-stage183-3bpa-representation-ladder/diagnostics/stage183_t3_cavity_vecq/test_300K_benchmark.json --variant stage183_t3_cavity_vecq_test_300K --energy-key energy --forces-key forces --start-config 0 --limit-configs 512 --measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist
PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model runs/oc20neb_tace_mace/rtece-stage183-3bpa-representation-ladder/results/stage183_t3_cavity_vecq/rtece_scalar_best.pt --configs datasets/3BPA/dataset_3BPA/test_600K.xyz --output runs/oc20neb_tace_mace/rtece-stage183-3bpa-representation-ladder/diagnostics/stage183_t3_cavity_vecq/test_600K_benchmark.json --variant stage183_t3_cavity_vecq_test_600K --energy-key energy --forces-key forces --start-config 0 --limit-configs 512 --measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist
PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model runs/oc20neb_tace_mace/rtece-stage183-3bpa-representation-ladder/results/stage183_t3_cavity_vecq/rtece_scalar_best.pt --configs datasets/3BPA/dataset_3BPA/test_1200K.xyz --output runs/oc20neb_tace_mace/rtece-stage183-3bpa-representation-ladder/diagnostics/stage183_t3_cavity_vecq/test_1200K_benchmark.json --variant stage183_t3_cavity_vecq_test_1200K --energy-key energy --forces-key forces --start-config 0 --limit-configs 512 --measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist
PYTHONPATH=${TACE_ROOT}:${PYTHONPATH:-} ${TACE_PYTHON} benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py --model runs/oc20neb_tace_mace/rtece-stage183-3bpa-representation-ladder/results/stage183_t3_cavity_vecq/rtece_scalar_best.pt --configs datasets/3BPA/dataset_3BPA/test_dih.xyz --output runs/oc20neb_tace_mace/rtece-stage183-3bpa-representation-ladder/diagnostics/stage183_t3_cavity_vecq/test_dih_benchmark.json --variant stage183_t3_cavity_vecq_test_dih --energy-key energy --forces-key forces --start-config 0 --limit-configs 512 --measure-passes 5 --device cuda --default-dtype float32 --force-mode autograd --graph-construction-backend matscipy_neighborlist
