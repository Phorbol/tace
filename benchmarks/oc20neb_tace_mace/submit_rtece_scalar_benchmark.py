#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_SCRIPT = REPO_ROOT / "benchmarks/oc20neb_tace_mace/rtece_scalar_benchmark.sbatch"


def _shell_assign(name: str, value: str | int | float | None) -> str:
    if value is None:
        return ""
    return f"{name}={shlex.quote(str(value))}"


def write_rtece_benchmark_wrapper(
    output_dir: str | Path,
    *,
    model: str,
    configs: str | None = None,
    force_mode: str,
    limit_configs_list: str,
    measure_passes: int,
    out_dir: str | None = None,
    variant: str | None = None,
    start_config: int | None = None,
    default_dtype: str | None = None,
    graph_construction_backend: str | None = None,
    graph_update_backend: str | None = None,
    benchmark_script: str | Path = BENCHMARK_SCRIPT,
) -> Path:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    wrapper = target_dir / "rtece_scalar_benchmark_no_export.sbatch"
    assignments = [
        _shell_assign("MODEL", model),
        _shell_assign("CONFIGS", configs),
        _shell_assign("FORCE_MODE", force_mode),
        _shell_assign("LIMIT_CONFIGS_LIST", limit_configs_list),
        _shell_assign("MEASURE_PASSES", measure_passes),
        _shell_assign("OUT_DIR", out_dir),
        _shell_assign("VARIANT", variant),
        _shell_assign("START_CONFIG", start_config),
        _shell_assign("DEFAULT_DTYPE", default_dtype),
        _shell_assign("GRAPH_CONSTRUCTION_BACKEND", graph_construction_backend),
        _shell_assign("GRAPH_UPDATE_BACKEND", graph_update_backend),
    ]
    body = [
        "#!/bin/bash",
        "#SBATCH --job-name=rtece-bench",
        "#SBATCH --partition=16V100",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        "#SBATCH --gpus-per-node=1",
        "#SBATCH --qos=flood-1o2gpu",
        "#SBATCH --time=01:00:00",
        "#SBATCH --output=/home/gengjianrui/bin/logs/rtece-bench-%j.out",
        "#SBATCH --error=/home/gengjianrui/bin/logs/rtece-bench-%j.err",
        "",
        "set -euo pipefail",
        "# SAI policy: pass parameters inside this wrapper, not through Slurm command-line environment export.",
    ]
    body.extend(f"export {line}" for line in assignments if line)
    body.append(f"exec /bin/bash {shlex.quote(str(Path(benchmark_script)))}")
    wrapper.write_text("\n".join(body) + "\n", encoding="utf-8")
    return wrapper


def build_sbatch_command(wrapper: str | Path) -> list[str]:
    return ["sbatch", str(wrapper)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit rTECE benchmark through a self-contained Slurm wrapper.")
    parser.add_argument("--wrapper-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--configs", default=None)
    parser.add_argument("--force-mode", default="auto")
    parser.add_argument("--limit-configs-list", default="1024")
    parser.add_argument("--measure-passes", type=int, default=5)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--variant", default=None)
    parser.add_argument("--start-config", type=int, default=None)
    parser.add_argument("--default-dtype", default=None)
    parser.add_argument("--graph-construction-backend", default=None)
    parser.add_argument("--graph-update-backend", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wrapper = write_rtece_benchmark_wrapper(
        args.wrapper_dir,
        model=args.model,
        configs=args.configs,
        force_mode=args.force_mode,
        limit_configs_list=args.limit_configs_list,
        measure_passes=args.measure_passes,
        out_dir=args.out_dir,
        variant=args.variant,
        start_config=args.start_config,
        default_dtype=args.default_dtype,
        graph_construction_backend=args.graph_construction_backend,
        graph_update_backend=args.graph_update_backend,
    )
    command = build_sbatch_command(wrapper)
    print(" ".join(shlex.quote(part) for part in command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
