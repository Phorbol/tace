#!/usr/bin/env python3
"""Generate stage128 physical-triage sbatch wrappers for rTECE candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


DEFAULT_STAGE127_ROOT = "runs/oc20neb_tace_mace/rtece-stage127-local-cross-species"
DEFAULT_STAGE127_SOURCE = f"{DEFAULT_STAGE127_ROOT}/stage127_interpretation.md"
DEFAULT_VARIANTS = (
    "l1_active_nrad12_species20_radial_species8_cross3_h64",
    "l1_active_nrad12_species24_radial_species8_cross3_h64",
    "l1_active_nrad12_species24_radial_species8_cross4_h64",
)
DEFAULT_DIMER_PAIRS = ("C-N", "C-O", "C-H", "N-H", "O-H", "C-C", "N-N")


def _case_paths(stage127_root: str | Path, variant: str) -> dict[str, str]:
    root = Path(stage127_root)
    row_dir = root / variant / variant
    return {
        "checkpoint": str(row_dir / "rtece_scalar_best.pt"),
        "dft_benchmark": str(row_dir / f"{variant}_dft_benchmark.json"),
        "teacher_benchmark": str(row_dir / f"{variant}_teacher_benchmark.json"),
    }


def stage128_physical_triage_cases(*, stage127_root: str | Path = DEFAULT_STAGE127_ROOT) -> list[dict[str, Any]]:
    """Return the non-dominated stage127 candidates for physical external tests."""
    cases = []
    for variant in DEFAULT_VARIANTS:
        case = {
            "variant": variant,
            "stage127_source": DEFAULT_STAGE127_SOURCE,
            "selection_basis": (
                "stage127 non-dominated local active-set candidate: force-RMSE branch "
                "or energy/robustness branch; cross2 excluded because stage127 marked it dominated"
            ),
        }
        case.update(_case_paths(stage127_root, variant))
        cases.append(case)
    return cases


def _write_wrapper(
    path: Path,
    *,
    variant: str,
    job_name: str,
    run_root: str,
    checkpoint: str,
    dft_benchmark: str,
    teacher_benchmark: str,
    configs: str,
    dimer_pairs: Sequence[str],
    dimer_num_points: int,
    rattle_start_config: int,
    rattle_limit_configs: int,
    rattle_max_steps: int,
    rattle_focus_label: str,
    max_focus_rattle_rmsd_a: float,
) -> None:
    pairs = " ".join(str(pair) for pair in dimer_pairs)
    out_dir = f"{run_root}/{variant}"
    text = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition=16V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=flood-1o2gpu
#SBATCH --time=03:55:00
#SBATCH --output=/home/gengjianrui/bin/logs/{job_name}-%j.out
#SBATCH --error=/home/gengjianrui/bin/logs/{job_name}-%j.err

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1

TACE_ROOT=${{TACE_ROOT:-/home/gengjianrui/bin/tace}}
ENV_DIR=${{ENV_DIR:-/home/gengjianrui/bin/.venvs/tace-mace-cu126}}
TACE_PYTHON=${{TACE_PYTHON:-${{ENV_DIR}}/bin/python}}
VARIANT={variant}
RUN_ROOT={run_root}
OUT_DIR={out_dir}
CHECKPOINT={checkpoint}
DFT_BENCHMARK={dft_benchmark}
TEACHER_BENCHMARK={teacher_benchmark}
CONFIGS={configs}
DIMER_PAIRS="{pairs}"
DIMER_NUM_POINTS={int(dimer_num_points)}
RATTLE_START_CONFIG={int(rattle_start_config)}
RATTLE_LIMIT_CONFIGS={int(rattle_limit_configs)}
RATTLE_MAX_STEPS={int(rattle_max_steps)}
RATTLE_FOCUS_LABEL={rattle_focus_label}
MAX_FOCUS_RATTLE_RMSD_A={float(max_focus_rattle_rmsd_a)}

mkdir -p /home/gengjianrui/bin/logs "${{OUT_DIR}}"
cd "${{TACE_ROOT}}"
export PYTHONPATH="${{TACE_ROOT}}:${{PYTHONPATH:-}}"
export PATH="${{ENV_DIR}}/bin:${{PATH}}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export TORCH_CUDA_ARCH_LIST=${{TORCH_CUDA_ARCH_LIST:-7.0}}

"${{TACE_PYTHON}}" -V
nvidia-smi -L

"${{TACE_PYTHON}}" benchmarks/oc20neb_tace_mace/dimer_scan_rtece.py \
  --checkpoint "${{CHECKPOINT}}" \
  --output-json "${{OUT_DIR}}/${{VARIANT}}_dimer.json" \
  --output-md "${{OUT_DIR}}/${{VARIANT}}_dimer.md" \
  --route "${{VARIANT}}" \
  --pairs ${{DIMER_PAIRS}} \
  --num-points "${{DIMER_NUM_POINTS}}" \
  --min-scale 0.5 \
  --max-scale 5.0 \
  --device cuda \
  --default-dtype float32 \
  --force-mode autograd \
  --neighborlist-backend ase

"${{TACE_PYTHON}}" benchmarks/oc20neb_tace_mace/rattle_relax_rtece.py \
  --checkpoint "${{CHECKPOINT}}" \
  --configs "${{CONFIGS}}" \
  --output-json "${{OUT_DIR}}/${{VARIANT}}_rattle.json" \
  --output-md "${{OUT_DIR}}/${{VARIANT}}_rattle.md" \
  --route "${{VARIANT}}" \
  --start-config "${{RATTLE_START_CONFIG}}" \
  --limit-configs "${{RATTLE_LIMIT_CONFIGS}}" \
  --rattle-std 0.05 \
  --rattle-seed 20260718 \
  --fmax 0.05 \
  --max-steps "${{RATTLE_MAX_STEPS}}" \
  --device cuda \
  --default-dtype float32 \
  --force-mode autograd \
  --neighborlist-backend matscipy

"${{TACE_PYTHON}}" benchmarks/oc20neb_tace_mace/summarize_rtece_physical_pareto.py \
  --case "${{VARIANT}}:${{DFT_BENCHMARK}}:${{TEACHER_BENCHMARK}}:${{OUT_DIR}}/${{VARIANT}}_dimer.json:${{OUT_DIR}}/${{VARIANT}}_rattle.json" \
  --max-dft-f-rmse-mev-a 120.0 \
  --max-dft-e-rmse-mev-atom 320.0 \
  --max-dft-f-max-mev-a 2600.0 \
  --max-dft-e-max-mev-atom 760.0 \
  --rattle-focus-label "${{RATTLE_FOCUS_LABEL}}" \
  --max-focus-rattle-rmsd-a "${{MAX_FOCUS_RATTLE_RMSD_A}}" \
  --max-rattle-fmax-ev-a 1.00 \
  --output-json "${{OUT_DIR}}/${{VARIANT}}_physical_pareto.json" \
  --output-md "${{OUT_DIR}}/${{VARIANT}}_physical_pareto.md"
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_physical_triage_wrappers(
    output_dir: str | Path,
    *,
    run_root: str,
    configs: str,
    cases: Sequence[dict[str, Any]] | None = None,
    stage: str = "stage128_physical_triage",
    source: str = DEFAULT_STAGE127_SOURCE,
    design_basis: str = (
        "Evaluate stage127 non-dominated active-set candidates under physical external tests: "
        "dimer smoothness/short-range repulsion and C_or_N rattle-relax stability."
    ),
    job_name: str = "rtece-phys128",
    dimer_pairs: Sequence[str] = DEFAULT_DIMER_PAIRS,
    dimer_num_points: int = 24,
    rattle_start_config: int = 58,
    rattle_limit_configs: int = 8,
    rattle_max_steps: int = 10,
    rattle_focus_label: str = "C_or_N",
    max_focus_rattle_rmsd_a: float = 0.35,
) -> dict[str, Any]:
    out = Path(output_dir)
    selected = list(cases if cases is not None else stage128_physical_triage_cases())
    rows = []
    for case in selected:
        variant = str(case["variant"])
        wrapper = out / variant / "rtece_physical_triage_no_export.sbatch"
        _write_wrapper(
            wrapper,
            variant=variant,
            job_name=job_name,
            run_root=str(run_root),
            checkpoint=str(case["checkpoint"]),
            dft_benchmark=str(case["dft_benchmark"]),
            teacher_benchmark=str(case["teacher_benchmark"]),
            configs=str(configs),
            dimer_pairs=dimer_pairs,
            dimer_num_points=int(dimer_num_points),
            rattle_start_config=int(rattle_start_config),
            rattle_limit_configs=int(rattle_limit_configs),
            rattle_max_steps=int(rattle_max_steps),
            rattle_focus_label=str(rattle_focus_label),
            max_focus_rattle_rmsd_a=float(max_focus_rattle_rmsd_a),
        )
        row = dict(case)
        row.update(
            {
                "wrapper": str(wrapper),
                "run_dir": f"{run_root}/{variant}",
                "dimer_json": f"{run_root}/{variant}/{variant}_dimer.json",
                "rattle_json": f"{run_root}/{variant}/{variant}_rattle.json",
                "physical_pareto_json": f"{run_root}/{variant}/{variant}_physical_pareto.json",
            }
        )
        rows.append(row)
    payload = {
        "schema_version": "rtece_physical_triage.v1",
        "stage": str(stage),
        "source": str(source),
        "stage127_source": str(source),
        "design_basis": str(design_basis),
        "job_name": str(job_name),
        "run_root": str(run_root),
        "configs": str(configs),
        "dimer_pairs": list(dimer_pairs),
        "dimer_num_points": int(dimer_num_points),
        "rattle_start_config": int(rattle_start_config),
        "rattle_limit_configs": int(rattle_limit_configs),
        "rattle_max_steps": int(rattle_max_steps),
        "rattle_focus_label": str(rattle_focus_label),
        "max_focus_rattle_rmsd_a": float(max_focus_rattle_rmsd_a),
        "cases": rows,
    }
    out.mkdir(parents=True, exist_ok=True)
    index = out / "rtece_physical_triage_index.json"
    index.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_case_spec(spec: str) -> dict[str, Any]:
    """Parse a custom physical-triage case.

    Format: variant:checkpoint:dft_benchmark:teacher_benchmark.
    """
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(
            "--case must use variant:checkpoint:dft_benchmark:teacher_benchmark; "
            f"got {spec!r}"
        )
    variant, checkpoint, dft_benchmark, teacher_benchmark = parts
    return {
        "variant": variant,
        "checkpoint": checkpoint,
        "dft_benchmark": dft_benchmark,
        "teacher_benchmark": teacher_benchmark,
        "selection_basis": "custom physical-triage case",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--stage127-root", default=DEFAULT_STAGE127_ROOT)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Custom case as variant:checkpoint:dft_benchmark:teacher_benchmark. Can be repeated.",
    )
    parser.add_argument("--stage", default="stage128_physical_triage")
    parser.add_argument("--source", default=DEFAULT_STAGE127_SOURCE)
    parser.add_argument(
        "--design-basis",
        default=(
            "Evaluate stage127 non-dominated active-set candidates under physical external tests: "
            "dimer smoothness/short-range repulsion and C_or_N rattle-relax stability."
        ),
    )
    parser.add_argument("--job-name", default="rtece-phys128")
    parser.add_argument("--configs", required=True)
    parser.add_argument("--dimer-pairs", nargs="+", default=list(DEFAULT_DIMER_PAIRS))
    parser.add_argument("--dimer-num-points", type=int, default=24)
    parser.add_argument("--rattle-start-config", type=int, default=58)
    parser.add_argument("--rattle-limit-configs", type=int, default=8)
    parser.add_argument("--rattle-max-steps", type=int, default=10)
    parser.add_argument("--rattle-focus-label", default="C_or_N")
    parser.add_argument("--max-focus-rattle-rmsd-a", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = (
        [parse_case_spec(spec) for spec in args.case]
        if args.case
        else stage128_physical_triage_cases(stage127_root=args.stage127_root)
    )
    payload = write_physical_triage_wrappers(
        args.output_dir,
        run_root=args.run_root,
        configs=args.configs,
        cases=cases,
        stage=args.stage,
        source=args.source,
        design_basis=args.design_basis,
        job_name=args.job_name,
        dimer_pairs=tuple(args.dimer_pairs),
        dimer_num_points=args.dimer_num_points,
        rattle_start_config=args.rattle_start_config,
        rattle_limit_configs=args.rattle_limit_configs,
        rattle_max_steps=args.rattle_max_steps,
        rattle_focus_label=args.rattle_focus_label,
        max_focus_rattle_rmsd_a=args.max_focus_rattle_rmsd_a,
    )
    print(json.dumps({"index": str(args.output_dir / "rtece_physical_triage_index.json"), "stage": payload["stage"], "cases": len(payload["cases"])}))


if __name__ == "__main__":
    main()
