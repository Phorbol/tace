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
    moment_l_max: int | None = None,
    scalar_path_ids: str | None = None,
    species_basis_channels: int | None = None,
    species_basis_mode: str | None = None,
    atomic_cross_radial_sketch_channels: int | None = None,
    atomic_cross_radial_projection: str | None = None,
    atomic_cross_radial_projection_file: str | None = None,
    seed: int | None = None,
    energy_weight: float | None = None,
    force_weight: float | None = None,
    force_focus_elements: str | None = None,
    force_focus_weight: float | None = None,
    use_short_range_repulsion: bool | None = None,
    short_range_repulsion_potential: str | None = None,
    learnable_radial_mixing: bool | None = None,
    radial_species_adapter_channels: int | None = None,
    radial_species_adapter_scope: str | None = None,
    descriptor_conditioner: str | None = None,
    descriptor_conditioner_hidden_channels: int | None = None,
    descriptor_bottleneck_dim: int | None = None,
    short_range_repulsion_strength: float | None = None,
    short_range_repulsion_beta: float | None = None,
    short_range_repulsion_radius_scale: float | None = None,
    force_mode: str | None = None,
    measure_passes: int | None = None,
    default_dtype: str | None = None,
    eval_interval: int | None = None,
    min_eval_step: int | None = None,
    checkpoint_name: str | None = None,
    trainer_backend: str | None = None,
    batch_size: int | None = None,
    valid_batch_size: int | None = None,
    lr_scheduler: str | None = None,
    lr_patience: int | None = None,
    lr_factor: float | None = None,
    lr_warmup_steps: int | None = None,
    early_stopping_patience: int | None = None,
    gradient_clip_val: float | None = None,
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
        _shell_assign("MOMENT_L_MAX", moment_l_max),
        _shell_assign("SCALAR_PATH_IDS", scalar_path_ids),
        _shell_assign("SPECIES_BASIS_CHANNELS", species_basis_channels),
        _shell_assign("SPECIES_BASIS_MODE", species_basis_mode),
        _shell_assign("ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS", atomic_cross_radial_sketch_channels),
        _shell_assign("ATOMIC_CROSS_RADIAL_PROJECTION", atomic_cross_radial_projection),
        _shell_assign("ATOMIC_CROSS_RADIAL_PROJECTION_FILE", atomic_cross_radial_projection_file),
        _shell_assign("SEED", seed),
        _shell_assign("ENERGY_WEIGHT", energy_weight),
        _shell_assign("FORCE_WEIGHT", force_weight),
        _shell_assign("FORCE_FOCUS_ELEMENTS", force_focus_elements),
        _shell_assign("FORCE_FOCUS_WEIGHT", force_focus_weight),
        _shell_assign(
            "USE_SHORT_RANGE_REPULSION",
            1 if use_short_range_repulsion else (0 if use_short_range_repulsion is False else None),
        ),
        _shell_assign("SHORT_RANGE_REPULSION_POTENTIAL", short_range_repulsion_potential),
        _shell_assign(
            "LEARNABLE_RADIAL_MIXING",
            1 if learnable_radial_mixing else (0 if learnable_radial_mixing is False else None),
        ),
        _shell_assign("RADIAL_SPECIES_ADAPTER_CHANNELS", radial_species_adapter_channels),
        _shell_assign("RADIAL_SPECIES_ADAPTER_SCOPE", radial_species_adapter_scope),
        _shell_assign("DESCRIPTOR_CONDITIONER", descriptor_conditioner),
        _shell_assign("DESCRIPTOR_CONDITIONER_HIDDEN_CHANNELS", descriptor_conditioner_hidden_channels),
        _shell_assign("DESCRIPTOR_BOTTLENECK_DIM", descriptor_bottleneck_dim),
        _shell_assign("SHORT_RANGE_REPULSION_STRENGTH", short_range_repulsion_strength),
        _shell_assign("SHORT_RANGE_REPULSION_BETA", short_range_repulsion_beta),
        _shell_assign("SHORT_RANGE_REPULSION_RADIUS_SCALE", short_range_repulsion_radius_scale),
        _shell_assign("FORCE_MODE", force_mode),
        _shell_assign("MEASURE_PASSES", measure_passes),
        _shell_assign("DEFAULT_DTYPE", default_dtype),
        _shell_assign("EVAL_INTERVAL", eval_interval),
        _shell_assign("MIN_EVAL_STEP", min_eval_step),
        _shell_assign("CHECKPOINT_NAME", checkpoint_name),
        _shell_assign("TRAINER_BACKEND", trainer_backend),
        _shell_assign("BATCH_SIZE", batch_size),
        _shell_assign("VALID_BATCH_SIZE", valid_batch_size),
        _shell_assign("LR_SCHEDULER", lr_scheduler),
        _shell_assign("LR_PATIENCE", lr_patience),
        _shell_assign("LR_FACTOR", lr_factor),
        _shell_assign("LR_WARMUP_STEPS", lr_warmup_steps),
        _shell_assign("EARLY_STOPPING_PATIENCE", early_stopping_patience),
        _shell_assign("GRADIENT_CLIP_VAL", gradient_clip_val),
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
        "# SAI policy: pass parameters inside this wrapper, not through Slurm command-line environment export.",
    ]
    body.extend(f"export {line}" for line in assignments if line)
    body.append(f"exec /bin/bash {shlex.quote(str(Path(matrix_script)))}")
    wrapper.write_text("\n".join(body) + "\n", encoding="utf-8")
    return wrapper


def build_sbatch_command(wrapper: str | Path) -> list[str]:
    return ["sbatch", str(wrapper)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit rTECE training/benchmark matrix through a self-contained Slurm wrapper.")
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
    parser.add_argument("--moment-l-max", type=int, choices=(0, 1, 2), default=None)
    parser.add_argument("--scalar-path-ids", default=None)
    parser.add_argument("--species-basis-channels", type=int, default=None)
    parser.add_argument("--species-basis-mode", choices=("fixed_z_power", "learnable_embedding"), default=None)
    parser.add_argument("--atomic-cross-radial-sketch-channels", type=int, default=None)
    parser.add_argument("--atomic-cross-radial-projection", choices=("fixed_shell_mean", "learnable", "pod_fixed"), default=None)
    parser.add_argument("--atomic-cross-radial-projection-file", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--energy-weight", type=float, default=None)
    parser.add_argument("--force-weight", type=float, default=None)
    parser.add_argument("--force-focus-elements", default=None)
    parser.add_argument("--force-focus-weight", type=float, default=None)
    parser.add_argument("--use-short-range-repulsion", action="store_true")
    parser.add_argument("--short-range-repulsion-potential", choices=("softplus_overlap", "zbl"), default=None)
    parser.add_argument("--learnable-radial-mixing", action="store_true")
    parser.add_argument("--radial-species-adapter-channels", type=int, default=None)
    parser.add_argument("--radial-species-adapter-scope", choices=("all", "atomic", "edge"), default=None)
    parser.add_argument("--descriptor-conditioner", choices=("none", "residual_mlp"), default=None)
    parser.add_argument("--descriptor-conditioner-hidden-channels", type=int, default=None)
    parser.add_argument("--descriptor-bottleneck-dim", type=int, default=None)
    parser.add_argument("--short-range-repulsion-strength", type=float, default=None)
    parser.add_argument("--short-range-repulsion-beta", type=float, default=None)
    parser.add_argument("--short-range-repulsion-radius-scale", type=float, default=None)
    parser.add_argument("--force-mode", default=None)
    parser.add_argument("--measure-passes", type=int, default=None)
    parser.add_argument("--default-dtype", default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--min-eval-step", type=int, default=None)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--trainer-backend", choices=("lightning", "step_loop"), default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--valid-batch-size", type=int, default=None)
    parser.add_argument("--lr-scheduler", choices=("plateau", "none"), default=None)
    parser.add_argument("--lr-patience", type=int, default=None)
    parser.add_argument("--lr-factor", type=float, default=None)
    parser.add_argument("--lr-warmup-steps", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--gradient-clip-val", type=float, default=None)
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
        moment_l_max=args.moment_l_max,
        scalar_path_ids=args.scalar_path_ids,
        species_basis_channels=args.species_basis_channels,
        species_basis_mode=args.species_basis_mode,
        atomic_cross_radial_sketch_channels=args.atomic_cross_radial_sketch_channels,
        atomic_cross_radial_projection=args.atomic_cross_radial_projection,
        atomic_cross_radial_projection_file=args.atomic_cross_radial_projection_file,
        seed=args.seed,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        force_focus_elements=args.force_focus_elements,
        force_focus_weight=args.force_focus_weight,
        use_short_range_repulsion=args.use_short_range_repulsion,
        short_range_repulsion_potential=args.short_range_repulsion_potential,
        learnable_radial_mixing=args.learnable_radial_mixing,
        radial_species_adapter_channels=args.radial_species_adapter_channels,
        radial_species_adapter_scope=args.radial_species_adapter_scope,
        descriptor_conditioner=args.descriptor_conditioner,
        descriptor_conditioner_hidden_channels=args.descriptor_conditioner_hidden_channels,
        descriptor_bottleneck_dim=args.descriptor_bottleneck_dim,
        short_range_repulsion_strength=args.short_range_repulsion_strength,
        short_range_repulsion_beta=args.short_range_repulsion_beta,
        short_range_repulsion_radius_scale=args.short_range_repulsion_radius_scale,
        force_mode=args.force_mode,
        measure_passes=args.measure_passes,
        default_dtype=args.default_dtype,
        eval_interval=args.eval_interval,
        min_eval_step=args.min_eval_step,
        checkpoint_name=args.checkpoint_name,
        trainer_backend=args.trainer_backend,
        batch_size=args.batch_size,
        valid_batch_size=args.valid_batch_size,
        lr_scheduler=args.lr_scheduler,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        lr_warmup_steps=args.lr_warmup_steps,
        early_stopping_patience=args.early_stopping_patience,
        gradient_clip_val=args.gradient_clip_val,
    )
    command = build_sbatch_command(wrapper)
    print(" ".join(shlex.quote(part) for part in command))
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
