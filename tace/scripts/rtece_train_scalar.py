from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch

from tace.models.rtece_scalar import (
    RTECEGraph,
    RTECEScalarConfig,
    RTECEScalarModel,
    build_rtece_config,
    build_rtece_config_from_path_ids,
    config_with_moment_l_max,
)
from tace.interface.ase.rtece_calculator import atoms_to_rtece_graph as package_atoms_to_rtece_graph
from tace.models.rtece_workflow import (
    evaluate_loss,
    load_checkpoint as load_checkpoint_with_metadata,
    loss_for_batch,
    save_checkpoint,
    train_steps,
)


def set_training_seed(seed: int) -> None:
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def parse_hidden_channels(value: str) -> tuple[int, ...]:
    channels = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not channels:
        raise ValueError("hidden channel list must contain at least one integer")
    if any(channel < 1 for channel in channels):
        raise ValueError(f"hidden channels must be positive, got {channels}")
    return channels


def parse_scalar_path_ids(value: str | None) -> tuple[str, ...] | None:
    if value is None or not value.strip():
        return None
    paths = tuple(part.strip() for part in value.split(",") if part.strip())
    if not paths:
        raise ValueError("scalar path id list must contain at least one path id")
    return paths


def parse_force_focus_elements(value: str | None) -> tuple[int, ...]:
    if value is None or not str(value).strip():
        return ()
    from ase.data import atomic_numbers

    numbers: list[int] = []
    for part in str(value).split(","):
        token = part.strip()
        if not token:
            continue
        if token.isdigit():
            number = int(token)
        else:
            if token not in atomic_numbers:
                raise ValueError(f"unknown force focus element {token!r}")
            number = int(atomic_numbers[token])
        if number < 1:
            raise ValueError(f"force focus atomic numbers must be positive, got {number}")
        numbers.append(number)
    return tuple(dict.fromkeys(numbers))


def load_atomic_cross_radial_projection_matrix(path: str | Path) -> tuple[tuple[float, ...], ...]:
    payload = json.loads(Path(path).read_text())
    matrix = payload.get("projection_matrix", payload) if isinstance(payload, dict) else payload
    try:
        rows = tuple(tuple(float(value) for value in row) for row in matrix)
    except TypeError as exc:
        raise ValueError("atomic cross-radial projection file must contain a rank-2 numeric matrix") from exc
    if not rows or any(not row for row in rows):
        raise ValueError("atomic cross-radial projection file must contain a non-empty rank-2 numeric matrix")
    return rows


def build_training_config(args: argparse.Namespace) -> RTECEScalarConfig:
    hidden_channels = parse_hidden_channels(args.hidden_channels)
    short_range_kwargs = {
        "use_short_range_repulsion": bool(getattr(args, "use_short_range_repulsion", False)),
        "short_range_repulsion_potential": str(getattr(args, "short_range_repulsion_potential", "softplus_overlap")),
        "short_range_repulsion_strength": float(getattr(args, "short_range_repulsion_strength", 0.0)),
        "short_range_repulsion_beta": float(getattr(args, "short_range_repulsion_beta", 10.0)),
        "short_range_repulsion_radius_scale": float(getattr(args, "short_range_repulsion_radius_scale", 0.75)),
        "learnable_radial_mixing": bool(getattr(args, "learnable_radial_mixing", False)),
        "moment_l_max": getattr(args, "moment_l_max", None),
        "species_basis_mode": str(getattr(args, "species_basis_mode", "fixed_z_power")),
        "atomic_cross_radial_sketch_channels": int(getattr(args, "atomic_cross_radial_sketch_channels", 2)),
        "atomic_cross_radial_projection": str(getattr(args, "atomic_cross_radial_projection", "fixed_shell_mean")),
        "atomic_cross_radial_projection_matrix": getattr(args, "atomic_cross_radial_projection_matrix", None),
        "descriptor_conditioner": str(getattr(args, "descriptor_conditioner", "none")),
        "descriptor_conditioner_hidden_channels": int(getattr(args, "descriptor_conditioner_hidden_channels", 0)),
        "descriptor_bottleneck_dim": int(getattr(args, "descriptor_bottleneck_dim", 0)),
    }
    scalar_path_ids = parse_scalar_path_ids(getattr(args, "scalar_path_ids", None))
    if scalar_path_ids is not None:
        return build_rtece_config_from_path_ids(
            args.variant,
            scalar_path_ids,
            hidden_channels=hidden_channels,
            num_radial=int(args.num_radial),
            species_basis_channels=int(getattr(args, "species_basis_channels", 0)),
            **short_range_kwargs,
        )
    config = replace(
        build_rtece_config(args.variant),
        hidden_channels=hidden_channels,
        num_radial=int(args.num_radial),
        **{key: value for key, value in short_range_kwargs.items() if key != "moment_l_max"},
    )
    return config_with_moment_l_max(config, short_range_kwargs["moment_l_max"])


def load_checkpoint(
    path: str | Path,
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> tuple[RTECEScalarModel, RTECEScalarConfig]:
    model, config, _metadata = load_checkpoint_with_metadata(path, dtype=dtype, device=device)
    return model, config


def _energy_and_forces(atoms):
    if "energy" in atoms.info:
        energy = float(atoms.info["energy"])
    elif atoms.calc is not None and "energy" in getattr(atoms.calc, "results", {}):
        energy = float(atoms.calc.results["energy"])
    else:
        energy = float(atoms.get_potential_energy())

    if "forces" in atoms.arrays:
        forces = np.asarray(atoms.arrays["forces"], dtype=np.float64)
    elif atoms.calc is not None and "forces" in getattr(atoms.calc, "results", {}):
        forces = np.asarray(atoms.calc.results["forces"], dtype=np.float64)
    else:
        forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    return energy, forces


def atoms_to_rtece_graph(
    atoms,
    *,
    cutoff: float,
    device: torch.device,
    dtype: torch.dtype,
    neighborlist_backend: str = "matscipy",
) -> RTECEGraph:
    return package_atoms_to_rtece_graph(
        atoms,
        cutoff=cutoff,
        device=device,
        dtype=dtype,
        neighborlist_backend=neighborlist_backend,
    )


def atoms_to_graph(
    atoms,
    *,
    cutoff: float,
    device: torch.device,
    dtype: torch.dtype,
    neighborlist_backend: str = "matscipy",
) -> tuple[RTECEGraph, torch.Tensor, torch.Tensor]:
    graph = atoms_to_rtece_graph(
        atoms,
        cutoff=cutoff,
        device=device,
        dtype=dtype,
        neighborlist_backend=neighborlist_backend,
    )
    energy_value, forces_value = _energy_and_forces(atoms)
    energy = torch.tensor([energy_value], dtype=dtype, device=device)
    forces = torch.tensor(forces_value, dtype=dtype, device=device)
    return graph, energy, forces


def fit_energy_per_atom_shift(
    samples: list[tuple[RTECEGraph, torch.Tensor, torch.Tensor]],
) -> float:
    if not samples:
        raise ValueError("fit_energy_per_atom_shift requires at least one sample")
    total_energy = 0.0
    total_atoms = 0
    for graph, energy, _ in samples:
        total_energy += float(energy.detach().sum().cpu())
        total_atoms += int(graph.z.numel())
    if total_atoms <= 0:
        raise ValueError("cannot fit energy shift for zero atoms")
    return total_energy / float(total_atoms)


def fit_atomic_energies(
    samples: list[tuple[RTECEGraph, torch.Tensor, torch.Tensor]],
    *,
    ridge: float = 1.0e-12,
) -> dict[int, float]:
    if not samples:
        raise ValueError("fit_atomic_energies requires at least one sample")
    elements = sorted({int(z) for graph, _, _ in samples for z in graph.z.detach().cpu().tolist()})
    if not elements:
        raise ValueError("cannot fit atomic energies for zero atoms")
    rows = []
    targets = []
    for graph, energy, _ in samples:
        z_cpu = graph.z.detach().cpu()
        rows.append([float((z_cpu == z).sum().item()) for z in elements])
        targets.append(float(energy.detach().sum().cpu()))
    design = np.asarray(rows, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    lhs = design.T @ design
    if ridge > 0.0:
        lhs = lhs + float(ridge) * np.eye(lhs.shape[0], dtype=np.float64)
    rhs = design.T @ target
    try:
        values = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        values = np.linalg.lstsq(design, target, rcond=None)[0]
    return {int(z): float(v) for z, v in zip(elements, values, strict=True)}



def load_samples(
    configs: Path,
    *,
    cutoff: float,
    device: torch.device,
    dtype: torch.dtype,
    limit_configs: int | None,
    neighborlist_backend: str = "matscipy",
) -> list[tuple[RTECEGraph, torch.Tensor, torch.Tensor]]:
    import ase.io

    index = ":" if limit_configs is None else f":{int(limit_configs)}"
    atoms_list = ase.io.read(str(configs), index=index)
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    return [
        atoms_to_graph(
            atoms,
            cutoff=cutoff,
            device=device,
            dtype=dtype,
            neighborlist_backend=neighborlist_backend,
        )
        for atoms in atoms_list
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a scalar-sketched rTECE prototype.")
    parser.add_argument(
        "--variant",
        required=True,
        help="Registered rTECE variant name, or a label when --scalar-path-ids is provided.",
    )
    parser.add_argument(
        "--scalar-path-ids",
        default=None,
        help="Comma-separated scalar path ids for manifest-driven rTECE routes.",
    )
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-configs", type=int, default=None)
    parser.add_argument("--valid-limit-configs", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-channels", default="64,64")
    parser.add_argument("--num-radial", type=int, default=8)
    parser.add_argument("--moment-l-max", type=int, choices=(0, 1, 2), default=None)
    parser.add_argument("--species-basis-channels", type=int, default=0)
    parser.add_argument("--species-basis-mode", choices=("fixed_z_power", "learnable_embedding"), default="fixed_z_power")
    parser.add_argument("--atomic-cross-radial-sketch-channels", type=int, default=2)
    parser.add_argument("--atomic-cross-radial-projection", choices=("fixed_shell_mean", "learnable", "pod_fixed"), default="fixed_shell_mean")
    parser.add_argument("--atomic-cross-radial-projection-file", type=Path, default=None)
    parser.add_argument("--descriptor-conditioner", choices=("none", "residual_mlp"), default="none")
    parser.add_argument("--descriptor-conditioner-hidden-channels", type=int, default=0)
    parser.add_argument("--descriptor-bottleneck-dim", type=int, default=0)
    parser.add_argument("--learnable-radial-mixing", action="store_true")
    parser.add_argument("--use-short-range-repulsion", action="store_true")
    parser.add_argument("--short-range-repulsion-potential", choices=("softplus_overlap", "zbl"), default="softplus_overlap")
    parser.add_argument("--short-range-repulsion-strength", type=float, default=0.0)
    parser.add_argument("--short-range-repulsion-beta", type=float, default=10.0)
    parser.add_argument("--short-range-repulsion-radius-scale", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--force-weight", type=float, default=10.0)
    parser.add_argument("--force-focus-elements", default=None)
    parser.add_argument("--force-focus-weight", type=float, default=1.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--default-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--neighborlist-backend", choices=("ase", "vesin", "matscipy"), default="matscipy")
    parser.add_argument("--no-fit-energy-shift", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--min-eval-step", type=int, default=0)
    parser.add_argument("--disable-best-checkpoint", action="store_true")
    parser.add_argument("--trainer-backend", choices=("lightning", "step_loop"), default="lightning")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--valid-batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--accelerator", default=None, help="Lightning accelerator; defaults to gpu for --device cuda, cpu otherwise")
    parser.add_argument("--devices", default="1", help="Lightning devices argument, e.g. 1 or auto")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--gradient-clip-val", type=float, default=0.0)
    parser.add_argument("--lr-scheduler", choices=("plateau", "none"), default="plateau")
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=25)
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--no-progress-bar", action="store_true")
    parser.add_argument("--logger", action="store_true", help="Enable Lightning logger output; disabled by default for Slurm sweeps.")
    parser.add_argument("--no-logger", action="store_true", help="Deprecated compatibility flag; logger is disabled by default.")
    return parser.parse_args()


def _parse_devices(value: str):
    return int(value) if str(value).isdigit() else value


def main() -> None:
    args = parse_args()
    args.atomic_cross_radial_projection_matrix = (
        load_atomic_cross_radial_projection_matrix(args.atomic_cross_radial_projection_file)
        if args.atomic_cross_radial_projection_file is not None
        else None
    )
    if args.trainer_backend == "lightning":
        from tace.lightning.rtece import fit_rtece_lightning

        accelerator = args.accelerator or ("gpu" if args.device == "cuda" else "cpu")
        summary = fit_rtece_lightning(
            variant=args.variant,
            scalar_path_ids=args.scalar_path_ids,
            train_file=args.train_file,
            valid_file=args.valid_file,
            output_dir=args.output_dir,
            limit_configs=args.limit_configs,
            valid_limit_configs=args.valid_limit_configs,
            max_steps=args.max_steps,
            max_epochs=args.max_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            hidden_channels=args.hidden_channels,
            num_radial=args.num_radial,
            moment_l_max=args.moment_l_max,
            species_basis_channels=args.species_basis_channels,
            species_basis_mode=args.species_basis_mode,
            atomic_cross_radial_sketch_channels=args.atomic_cross_radial_sketch_channels,
            atomic_cross_radial_projection=args.atomic_cross_radial_projection,
            atomic_cross_radial_projection_matrix=args.atomic_cross_radial_projection_matrix,
            descriptor_conditioner=args.descriptor_conditioner,
            descriptor_conditioner_hidden_channels=args.descriptor_conditioner_hidden_channels,
            learnable_radial_mixing=args.learnable_radial_mixing,
            use_short_range_repulsion=args.use_short_range_repulsion,
            short_range_repulsion_potential=args.short_range_repulsion_potential,
            short_range_repulsion_strength=args.short_range_repulsion_strength,
            short_range_repulsion_beta=args.short_range_repulsion_beta,
            short_range_repulsion_radius_scale=args.short_range_repulsion_radius_scale,
            seed=args.seed,
            energy_weight=args.energy_weight,
            force_weight=args.force_weight,
            force_focus_elements=args.force_focus_elements,
            force_focus_weight=args.force_focus_weight,
            default_dtype=args.default_dtype,
            neighborlist_backend=args.neighborlist_backend,
            no_fit_energy_shift=args.no_fit_energy_shift,
            batch_size=args.batch_size,
            valid_batch_size=args.valid_batch_size,
            num_workers=args.num_workers,
            accelerator=accelerator,
            devices=_parse_devices(args.devices),
            gradient_clip_val=args.gradient_clip_val,
            lr_scheduler=args.lr_scheduler,
            lr_factor=args.lr_factor,
            lr_patience=args.lr_patience,
            lr_warmup_steps=args.lr_warmup_steps,
            early_stopping_patience=args.early_stopping_patience,
            enable_progress_bar=not args.no_progress_bar,
            logger=bool(args.logger and not args.no_logger),
        )
        summary["eval_interval"] = args.eval_interval
        summary["min_eval_step"] = args.min_eval_step
        (args.output_dir / "train_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    set_training_seed(args.seed)
    dtype = torch.float64 if args.default_dtype == "float64" else torch.float32
    requested = torch.device(args.device)
    device = requested if requested.type == "cpu" or torch.cuda.is_available() else torch.device("cpu")
    config = build_training_config(args)
    samples = load_samples(
        args.train_file,
        cutoff=config.cutoff,
        device=device,
        dtype=dtype,
        limit_configs=args.limit_configs,
        neighborlist_backend=args.neighborlist_backend,
    )
    if not args.no_fit_energy_shift:
        config = replace(config, atomic_energies=fit_atomic_energies(samples))
    model = RTECEScalarModel(config).to(device=device, dtype=dtype)
    valid_samples = None
    best_checkpoint_path = None
    if not args.disable_best_checkpoint and args.eval_interval > 0:
        valid_samples = load_samples(
            args.valid_file,
            cutoff=config.cutoff,
            device=device,
            dtype=dtype,
            limit_configs=args.valid_limit_configs,
            neighborlist_backend=args.neighborlist_backend,
        )
        best_checkpoint_path = args.output_dir / "rtece_scalar_best.pt"
    force_focus_atomic_numbers = parse_force_focus_elements(args.force_focus_elements)
    summary = train_steps(
        model,
        samples,
        max_steps=args.max_steps,
        lr=args.lr,
        valid_samples=valid_samples,
        eval_interval=args.eval_interval,
        min_eval_step=args.min_eval_step,
        best_checkpoint_path=best_checkpoint_path,
        config=config,
        energy_weight=args.energy_weight,
        force_weight=args.force_weight,
        force_focus_atomic_numbers=force_focus_atomic_numbers,
        force_focus_weight=args.force_focus_weight,
    )
    summary.update(
        {
            "trainer_backend": "step_loop",
            "variant": args.variant,
            "train_file": str(args.train_file),
            "valid_file": str(args.valid_file),
            "train_configs": len(samples),
            "valid_configs": len(valid_samples) if valid_samples is not None else 0,
            "energy_per_atom_shift": config.energy_per_atom_shift,
            "atomic_energies": {str(k): float(v) for k, v in (config.atomic_energies or {}).items()},
            "hidden_channels": list(config.hidden_channels),
            "num_radial": config.num_radial,
            "moment_l_max": int(config.moment_l_max) if config.moment_l_max is not None else None,
            "species_basis_channels": int(config.species_basis_channels),
            "species_basis_mode": str(config.species_basis_mode),
            "atomic_cross_radial_sketch_channels": int(config.atomic_cross_radial_sketch_channels),
            "atomic_cross_radial_projection": str(config.atomic_cross_radial_projection),
            "atomic_cross_radial_projection_file": str(args.atomic_cross_radial_projection_file)
            if args.atomic_cross_radial_projection_file is not None
            else None,
            "learnable_radial_mixing": bool(config.learnable_radial_mixing),
            "descriptor_conditioner": str(config.descriptor_conditioner),
            "descriptor_conditioner_hidden_channels": int(config.descriptor_conditioner_hidden_channels),
            "scalar_path_ids": list(config.scalar_path_ids or []),
            "use_short_range_repulsion": bool(config.use_short_range_repulsion),
            "short_range_repulsion_potential": str(config.short_range_repulsion_potential),
            "short_range_repulsion_strength": float(config.short_range_repulsion_strength),
            "short_range_repulsion_beta": float(config.short_range_repulsion_beta),
            "short_range_repulsion_radius_scale": float(config.short_range_repulsion_radius_scale),
            "seed": int(args.seed),
            "energy_weight": args.energy_weight,
            "force_weight": args.force_weight,
            "eval_interval": args.eval_interval,
            "min_eval_step": args.min_eval_step,
            "force_focus_atomic_numbers": list(force_focus_atomic_numbers),
            "force_focus_weight": args.force_focus_weight,
            "device": str(device),
            "default_dtype": args.default_dtype,
            "neighborlist_backend": args.neighborlist_backend,
            "checkpoint": str(args.output_dir / "rtece_scalar.pt"),
            "best_checkpoint": str(best_checkpoint_path) if best_checkpoint_path is not None else None,
        }
    )
    save_checkpoint(args.output_dir / "rtece_scalar.pt", model, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
