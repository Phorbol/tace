#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_SCRIPT = REPO_ROOT / "benchmarks/oc20neb_tace_mace/rtece_scalar_matrix.sbatch"


def _shell_assign(name: str, value: str | int | float | None) -> str:
    if value is None:
        return ""
    return f"{name}={shlex.quote(str(value))}"


def write_rtece_matrix_wrapper(
    output_dir: str | Path,
    *,
    variants: str,
    run_root: str | None = None,
    train_file: str | None = None,
    train_valid_file: str | None = None,
    dft_valid_file: str | None = None,
    teacher_valid_file: str | None = None,
    limit_configs: int | None = None,
    valid_limit_configs: int | None = None,
    bench_limit_configs: int | None = None,
    max_steps: int | None = None,
    lr: str | float | None = None,
    hidden_channels: str | None = None,
    num_radial: int | None = None,
    seed: int | None = None,
    energy_weight: float | None = None,
    force_weight: float | None = None,
    force_mode: str | None = None,
    measure_passes: int | None = None,
    default_dtype: str | None = None,
    eval_interval: int | None = None,
    checkpoint_name: str | None = None,
    matrix_script: str | Path = MATRIX_SCRIPT,
) -> Path:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    wrapper = target_dir / "rtece_scalar_matrix_no_export.sbatch"
    assignments = [
        _shell_assign("VARIANTS", variants),
        _shell_assign("RUN_ROOT", run_root),
        _shell_assign("TRAIN_FILE", train_file),
        _shell_assign("TRAIN_VALID_FILE", train_valid_file),
        _shell_assign("DFT_VALID_FILE", dft_valid_file),
        _shell_assign("TEACHER_VALID_FILE", teacher_valid_file),
        _shell_assign("LIMIT_CONFIGS", limit_configs),
        _shell_assign("VALID_LIMIT_CONFIGS", valid_limit_configs),
        _shell_assign("BENCH_LIMIT_CONFIGS", bench_limit_configs),
        _shell_assign("MAX_STEPS", max_steps),
        _shell_assign("LR", lr),
        _shell_assign("HIDDEN_CHANNELS", hidden_channels),
        _shell_assign("NUM_RADIAL", num_radial),
        _shell_assign("SEED", seed),
        _shell_assign("ENERGY_WEIGHT", energy_weight),
        _shell_assign("FORCE_WEIGHT", force_weight),
        _shell_assign("FORCE_MODE", force_mode),
        _shell_assign("MEASURE_PASSES", measure_passes),
        _shell_assign("DEFAULT_DTYPE", default_dtype),
        _shell_assign("EVAL_INTERVAL", eval_interval),
        _shell_assign("CHECKPOINT_NAME", checkpoint_name),
    ]
    body = [
        "#!/bin/bash",
        "#SBATCH --job-name=rtece-matrix",
        "#SBATCH --partition=16V100",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --gpus-per-node=1",
        "#SBATCH --qos=flood-1o2gpu",
        "#SBATCH --time=03:55:00",
        "#SBATCH --output=/home/gengjianrui/bin/logs/rtece-matrix-%j.out",
        "#SBATCH --error=/home/gengjianrui/bin/logs/rtece-matrix-%j.err",
        "",
        "set -euo pipefail",
        "# SAI policy: do not submit rTECE jobs with sbatch --export=ALL,... .",
    ]
    body.extend(f"export {line}" for line in assignments if line)
    body.append(f"exec /bin/bash {shlex.quote(str(Path(matrix_script)))}")
    wrapper.write_text("\n".join(body) + "\n", encoding="utf-8")
    return wrapper


def build_sbatch_command(wrapper: str | Path) -> list[str]:
    return ["sbatch", str(wrapper)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit rTECE training/benchmark matrix without Slurm --export.")
    parser.add_argument("--wrapper-dir", type=Path, required=True)
    parser.add_argument("--variants", required=True)
    parser.add_argument("--run-root", default=None)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--train-valid-file", default=None)
    parser.add_argument("--dft-valid-file", default=None)
    parser.add_argument("--teacher-valid-file", default=None)
    parser.add_argument("--limit-configs", type=int, default=None)
    parser.add_argument("--valid-limit-configs", type=int, default=None)
    parser.add_argument("--bench-limit-configs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--lr", default=None)
    parser.add_argument("--hidden-channels", default=None)
    parser.add_argument("--num-radial", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--energy-weight", type=float, default=None)
    parser.add_argument("--force-weight", type=float, default=None)
    parser.add_argument("--force-mode", default=None)
    parser.add_argument("--measure-passes", type=int, default=None)
    parser.add_argument("--default-dtype", default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wrapper = write_rtece_matrix_wrapper(
        args.wrapper_dir,
        variants=args.variants,
        run_root=args.run_root,
        train_file=args.train_file,
        train_valid_file=args.train_valid_file,
        dft_valid_file=args.dft_valid_file,
        teacher_valid_file=args.teacher_valid_file,
        limit_configs=args.limit_configs,
        valid_limit_configs=args.valid_limit_configs,
        bench_limit_configs=args.bench_limit_configs,
        max_steps=args.max_steps,
        lr=args.lr,
        hidden_channels=args.hidden_channels,
        num_radial=args.num_radial,
        seed=args.seed,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        force_mode=args.force_mode,
        measure_passes=args.measure_passes,
        default_dtype=args.default_dtype,
        eval_interval=args.eval_interval,
        checkpoint_name=args.checkpoint_name,
    )
    command = build_sbatch_command(wrapper)
    print(" ".join(shlex.quote(part) for part in command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
