from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from tace.interface.ase.rtece_calculator import atoms_to_rtece_graph
from tace.models.rtece_scalar import (
    RTECEGraph,
    RTECEScalarConfig,
    RTECEScalarModel,
    build_rtece_config,
    build_rtece_config_from_path_ids,
    config_with_moment_l_max,
    collate_graphs,
)
from tace.models.rtece_workflow import loss_for_batch, save_checkpoint


def parse_hidden_channels(value: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        channels = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    else:
        channels = tuple(int(part) for part in value)
    if not channels:
        raise ValueError("hidden channel list must contain at least one integer")
    if any(channel < 1 for channel in channels):
        raise ValueError(f"hidden channels must be positive, got {channels}")
    return channels


def parse_scalar_path_ids(value: str | tuple[str, ...] | list[str] | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value.strip():
            return None
        paths = tuple(part.strip() for part in value.split(",") if part.strip())
    else:
        paths = tuple(str(part).strip() for part in value if str(part).strip())
    if not paths:
        raise ValueError("scalar path id list must contain at least one path id")
    return paths


def parse_force_focus_elements(value: str | tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    if value is None or not str(value).strip():
        return ()
    if not isinstance(value, str):
        return tuple(dict.fromkeys(int(number) for number in value))
    from ase.data import atomic_numbers

    numbers: list[int] = []
    for part in value.split(","):
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


def build_training_config(
    *,
    variant: str,
    hidden_channels: str | tuple[int, ...] | list[int] = "64,64",
    num_radial: int = 8,
    moment_l_max: int | None = None,
    scalar_path_ids: str | tuple[str, ...] | list[str] | None = None,
    species_basis_channels: int = 0,
    species_basis_mode: str = "fixed_z_power",
    atomic_cross_radial_sketch_channels: int = 2,
    atomic_cross_radial_projection: str = "fixed_shell_mean",
    atomic_cross_radial_projection_matrix: object | None = None,
    descriptor_conditioner: str = "none",
    descriptor_conditioner_hidden_channels: int = 0,
    descriptor_bottleneck_dim: int = 0,
    radial_species_adapter_channels: int = 0,
    radial_species_adapter_scope: str = "all",
    learnable_radial_mixing: bool = False,
    use_short_range_repulsion: bool = False,
    short_range_repulsion_potential: str = "softplus_overlap",
    short_range_repulsion_strength: float = 0.0,
    short_range_repulsion_beta: float = 10.0,
    short_range_repulsion_radius_scale: float = 0.75,
) -> RTECEScalarConfig:
    hidden = parse_hidden_channels(hidden_channels)
    short_range_kwargs = {
        "use_short_range_repulsion": bool(use_short_range_repulsion),
        "short_range_repulsion_potential": str(short_range_repulsion_potential),
        "short_range_repulsion_strength": float(short_range_repulsion_strength),
        "short_range_repulsion_beta": float(short_range_repulsion_beta),
        "short_range_repulsion_radius_scale": float(short_range_repulsion_radius_scale),
        "learnable_radial_mixing": bool(learnable_radial_mixing),
        "moment_l_max": moment_l_max,
        "species_basis_mode": str(species_basis_mode),
        "atomic_cross_radial_sketch_channels": int(atomic_cross_radial_sketch_channels),
        "atomic_cross_radial_projection": str(atomic_cross_radial_projection),
        "atomic_cross_radial_projection_matrix": atomic_cross_radial_projection_matrix,
        "descriptor_conditioner": str(descriptor_conditioner),
        "descriptor_conditioner_hidden_channels": int(descriptor_conditioner_hidden_channels),
        "descriptor_bottleneck_dim": int(descriptor_bottleneck_dim),
        "radial_species_adapter_channels": int(radial_species_adapter_channels),
        "radial_species_adapter_scope": str(radial_species_adapter_scope),
    }
    paths = parse_scalar_path_ids(scalar_path_ids)
    if paths is not None:
        return build_rtece_config_from_path_ids(
            variant,
            paths,
            hidden_channels=hidden,
            num_radial=int(num_radial),
            species_basis_channels=int(species_basis_channels),
            **short_range_kwargs,
        )
    config = replace(
        build_rtece_config(variant),
        hidden_channels=hidden,
        num_radial=int(num_radial),
        **{key: value for key, value in short_range_kwargs.items() if key != "moment_l_max"},
    )
    return config_with_moment_l_max(config, short_range_kwargs["moment_l_max"])


def _resolve_dtype(default_dtype: str | torch.dtype) -> torch.dtype:
    if default_dtype in {"float64", torch.float64}:
        return torch.float64
    if default_dtype in {"float32", torch.float32}:
        return torch.float32
    raise ValueError("default_dtype must be 'float32' or 'float64'")


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


def atoms_to_graph(
    atoms,
    *,
    cutoff: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
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


def _sample_weights(atoms, *, dtype: torch.dtype, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    energy_weight = float(atoms.info.get("energy_weight", 1.0))
    forces_weight = float(atoms.info.get("forces_weight", 1.0))
    return (
        torch.tensor([energy_weight], dtype=dtype, device=device),
        torch.tensor([forces_weight], dtype=dtype, device=device),
    )


def atoms_to_weighted_graph(
    atoms,
    *,
    cutoff: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    neighborlist_backend: str = "matscipy",
) -> tuple[RTECEGraph, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    graph, energy, forces = atoms_to_graph(
        atoms,
        cutoff=cutoff,
        device=device,
        dtype=dtype,
        neighborlist_backend=neighborlist_backend,
    )
    energy_weight, forces_weight = _sample_weights(atoms, dtype=dtype, device=device)
    return graph, energy, forces, energy_weight, forces_weight


def _energy_fit_weight(sample: tuple) -> float:
    if len(sample) <= 3:
        return 1.0
    weight = float(sample[3].detach().reshape(-1)[0].cpu())
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError(f"energy sample weights must be finite and non-negative, got {weight}")
    return weight


def fit_atomic_energies(
    samples: list[tuple[RTECEGraph, torch.Tensor, torch.Tensor]],
    *,
    ridge: float = 1.0e-12,
) -> dict[int, float]:
    if not samples:
        raise ValueError("fit_atomic_energies requires at least one sample")
    elements = sorted({int(z) for sample in samples for z in sample[0].z.detach().cpu().tolist()})
    if not elements:
        raise ValueError("cannot fit atomic energies for zero atoms")
    rows = []
    targets = []
    weights = []
    for sample in samples:
        graph, energy = sample[:2]
        z_cpu = graph.z.detach().cpu()
        rows.append([float((z_cpu == z).sum().item()) for z in elements])
        targets.append(float(energy.detach().sum().cpu()))
        weights.append(_energy_fit_weight(sample))
    design = np.asarray(rows, dtype=np.float64)
    target = np.asarray(targets, dtype=np.float64)
    sample_weights = np.asarray(weights, dtype=np.float64)
    if float(sample_weights.sum()) <= 0.0:
        raise ValueError("cannot fit atomic energies with zero total energy sample weight")
    weighted_design = design * sample_weights[:, None]
    lhs = design.T @ weighted_design
    if ridge > 0.0:
        lhs = lhs + float(ridge) * np.eye(lhs.shape[0], dtype=np.float64)
    rhs = design.T @ (sample_weights * target)
    try:
        values = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        sqrt_w = np.sqrt(sample_weights)
        values = np.linalg.lstsq(design * sqrt_w[:, None], target * sqrt_w, rcond=None)[0]
    return {int(z): float(value) for z, value in zip(elements, values, strict=True)}


def load_samples(
    configs: str | Path,
    *,
    cutoff: float,
    dtype: torch.dtype,
    limit_configs: int | None = None,
    neighborlist_backend: str = "matscipy",
    include_sample_weights: bool = False,
) -> list[tuple[RTECEGraph, torch.Tensor, torch.Tensor]]:
    import ase.io

    index = ":" if limit_configs is None else f":{int(limit_configs)}"
    atoms_list = ase.io.read(str(configs), index=index)
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    converter = atoms_to_weighted_graph if include_sample_weights else atoms_to_graph
    return [
        converter(
            atoms,
            cutoff=cutoff,
            device="cpu",
            dtype=dtype,
            neighborlist_backend=neighborlist_backend,
        )
        for atoms in atoms_list
    ]


class RTECEDataset(Dataset):
    def __init__(self, samples: list[tuple[RTECEGraph, torch.Tensor, torch.Tensor]]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        return self.samples[index]


def _collate_rtece_samples(samples):
    if len(samples[0]) == 3:
        graphs, energies, forces = zip(*samples, strict=True)
        return collate_graphs(list(graphs)), torch.cat(list(energies), dim=0), torch.cat(list(forces), dim=0)
    graphs, energies, forces, energy_weights, force_weights = zip(*samples, strict=True)
    return (
        collate_graphs(list(graphs)),
        torch.cat(list(energies), dim=0),
        torch.cat(list(forces), dim=0),
        torch.cat(list(energy_weights), dim=0),
        torch.cat(list(force_weights), dim=0),
    )


def _graph_to_device(graph: RTECEGraph, device: torch.device) -> RTECEGraph:
    return RTECEGraph(
        z=graph.z.to(device),
        pos=graph.pos.to(device),
        edge_index=graph.edge_index.to(device),
        batch=graph.batch.to(device),
        cell=graph.cell.to(device) if graph.cell is not None else None,
        edge_shifts=graph.edge_shifts.to(device) if graph.edge_shifts is not None else None,
        edge_batch=graph.edge_batch.to(device) if graph.edge_batch is not None else None,
    )


class RTECEDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_samples: list[tuple[RTECEGraph, torch.Tensor, torch.Tensor]],
        valid_samples: list[tuple[RTECEGraph, torch.Tensor, torch.Tensor]],
        *,
        batch_size: int = 1,
        valid_batch_size: int | None = None,
        num_workers: int = 0,
    ) -> None:
        super().__init__()
        self.train_samples = train_samples
        self.valid_samples = valid_samples
        self.batch_size = int(batch_size)
        self.valid_batch_size = int(valid_batch_size or batch_size)
        self.num_workers = int(num_workers)

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = RTECEDataset(self.train_samples)
        self.val_dataset = RTECEDataset(self.valid_samples)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=_collate_rtece_samples,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.valid_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_rtece_samples,
        )


class RTECELightningModule(L.LightningModule):
    def __init__(
        self,
        model: RTECEScalarModel,
        config: RTECEScalarConfig,
        *,
        lr: float = 1.0e-3,
        weight_decay: float = 0.0,
        energy_weight: float = 1.0,
        force_weight: float = 10.0,
        force_focus_atomic_numbers: tuple[int, ...] = (),
        force_focus_weight: float = 1.0,
        lr_scheduler: str = "plateau",
        lr_factor: float = 0.5,
        lr_patience: int = 25,
        lr_warmup_steps: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model", "config"])
        self.model = model
        self.config = config
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.energy_weight = float(energy_weight)
        self.force_weight = float(force_weight)
        self.force_focus_atomic_numbers = tuple(int(z) for z in force_focus_atomic_numbers)
        self.force_focus_weight = float(force_focus_weight)
        self.lr_scheduler = str(lr_scheduler)
        self.lr_factor = float(lr_factor)
        self.lr_patience = int(lr_patience)
        self.lr_warmup_steps = int(lr_warmup_steps)
        if self.lr_warmup_steps < 0:
            raise ValueError("lr_warmup_steps must be non-negative")

    def on_train_batch_start(self, batch, batch_idx: int) -> None:
        if self.lr_warmup_steps <= 0 or self.trainer is None:
            return
        step = int(self.trainer.global_step)
        if step >= self.lr_warmup_steps:
            return
        scale = float(step + 1) / float(self.lr_warmup_steps)
        for optimizer in self.trainer.optimizers:
            for group in optimizer.param_groups:
                group["lr"] = self.lr * scale

    def _shared_step(self, batch, prefix: str):
        if len(batch) == 3:
            graph, energy, forces = batch
            energy_sample_weights = None
            force_sample_weights = None
        else:
            graph, energy, forces, energy_sample_weights, force_sample_weights = batch
        graph = _graph_to_device(graph, self.device)
        energy = energy.to(self.device)
        forces = forces.to(self.device)
        energy_sample_weights = energy_sample_weights.to(self.device) if energy_sample_weights is not None else None
        force_sample_weights = force_sample_weights.to(self.device) if force_sample_weights is not None else None
        loss = loss_for_batch(
            self.model,
            graph,
            energy,
            forces,
            energy_weight=self.energy_weight,
            force_weight=self.force_weight,
            force_focus_atomic_numbers=self.force_focus_atomic_numbers,
            force_focus_weight=self.force_focus_weight,
            energy_sample_weights=energy_sample_weights,
            force_sample_weights=force_sample_weights,
        )
        with torch.enable_grad():
            output = self.model(graph)
        with torch.no_grad():
            natoms = torch.bincount(graph.batch, minlength=energy.numel()).to(dtype=energy.dtype, device=energy.device).clamp_min(1)
            e_error = (output["energy"].detach() - energy) / natoms
            f_error = output["forces"].detach() - forces
            e_rmse = torch.sqrt(torch.mean(e_error.square())) * 1000.0
            f_rmse = torch.sqrt(torch.mean(f_error.square())) * 1000.0
            e_mae = torch.mean(torch.abs(e_error)) * 1000.0
            f_mae = torch.mean(torch.abs(f_error)) * 1000.0
        batch_size = int(energy.numel())
        self.log(f"{prefix}/loss", loss.detach(), prog_bar=(prefix == "train"), on_step=(prefix == "train"), on_epoch=True, batch_size=batch_size)
        self.log(f"{prefix}/e_rmse_mev_atom", e_rmse, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{prefix}/f_rmse_mev_a", f_rmse, prog_bar=(prefix == "val"), on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{prefix}/e_mae_mev_atom", e_mae, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{prefix}/f_mae_mev_a", f_mae, prog_bar=False, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        with torch.enable_grad():
            return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.lr_scheduler in {"none", "off", "false", "0"}:
            return {"optimizer": optimizer}
        if self.lr_scheduler != "plateau":
            raise ValueError(f"unsupported rTECE lr_scheduler {self.lr_scheduler!r}")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.lr_factor,
            patience=self.lr_patience,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


def _load_lightning_model_state(path: str | Path, config: RTECEScalarConfig, *, dtype: torch.dtype) -> RTECEScalarModel:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = {key[len("model."):]: value for key, value in checkpoint["state_dict"].items() if key.startswith("model.")}
    model = RTECEScalarModel(config).to(dtype=dtype)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def fit_rtece_lightning(
    *,
    variant: str,
    train_file: str | Path,
    valid_file: str | Path,
    output_dir: str | Path,
    scalar_path_ids: str | tuple[str, ...] | list[str] | None = None,
    limit_configs: int | None = None,
    valid_limit_configs: int | None = 64,
    max_steps: int = 1000,
    max_epochs: int | None = None,
    lr: float = 1.0e-3,
    weight_decay: float = 0.0,
    hidden_channels: str | tuple[int, ...] | list[int] = "64,64",
    num_radial: int = 8,
    moment_l_max: int | None = None,
    species_basis_channels: int = 0,
    species_basis_mode: str = "fixed_z_power",
    atomic_cross_radial_sketch_channels: int = 2,
    atomic_cross_radial_projection: str = "fixed_shell_mean",
    atomic_cross_radial_projection_matrix: object | None = None,
    descriptor_conditioner: str = "none",
    descriptor_conditioner_hidden_channels: int = 0,
    descriptor_bottleneck_dim: int = 0,
    radial_species_adapter_channels: int = 0,
    radial_species_adapter_scope: str = "all",
    learnable_radial_mixing: bool = False,
    use_short_range_repulsion: bool = False,
    short_range_repulsion_potential: str = "softplus_overlap",
    short_range_repulsion_strength: float = 0.0,
    short_range_repulsion_beta: float = 10.0,
    short_range_repulsion_radius_scale: float = 0.75,
    seed: int = 0,
    energy_weight: float = 1.0,
    force_weight: float = 10.0,
    force_focus_elements: str | tuple[int, ...] | list[int] | None = None,
    force_focus_weight: float = 1.0,
    default_dtype: str | torch.dtype = "float32",
    neighborlist_backend: str = "matscipy",
    no_fit_energy_shift: bool = False,
    batch_size: int = 1,
    valid_batch_size: int | None = None,
    num_workers: int = 0,
    accelerator: str = "auto",
    devices: int | str = "auto",
    gradient_clip_val: float = 0.0,
    check_val_every_n_epoch: int | None = 1,
    lr_scheduler: str = "plateau",
    lr_factor: float = 0.5,
    lr_patience: int = 25,
    lr_warmup_steps: int = 0,
    early_stopping_patience: int | None = None,
    enable_progress_bar: bool = True,
    logger: bool | Any = False,
) -> dict[str, Any]:
    L.seed_everything(int(seed), workers=True)
    dtype = _resolve_dtype(default_dtype)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    config = build_training_config(
        variant=variant,
        hidden_channels=hidden_channels,
        num_radial=num_radial,
        moment_l_max=moment_l_max,
        scalar_path_ids=scalar_path_ids,
        species_basis_channels=species_basis_channels,
        species_basis_mode=species_basis_mode,
        atomic_cross_radial_sketch_channels=atomic_cross_radial_sketch_channels,
        atomic_cross_radial_projection=atomic_cross_radial_projection,
        atomic_cross_radial_projection_matrix=atomic_cross_radial_projection_matrix,
        descriptor_conditioner=descriptor_conditioner,
        descriptor_conditioner_hidden_channels=descriptor_conditioner_hidden_channels,
        descriptor_bottleneck_dim=descriptor_bottleneck_dim,
        radial_species_adapter_channels=radial_species_adapter_channels,
        radial_species_adapter_scope=radial_species_adapter_scope,
        learnable_radial_mixing=learnable_radial_mixing,
        use_short_range_repulsion=use_short_range_repulsion,
        short_range_repulsion_potential=short_range_repulsion_potential,
        short_range_repulsion_strength=short_range_repulsion_strength,
        short_range_repulsion_beta=short_range_repulsion_beta,
        short_range_repulsion_radius_scale=short_range_repulsion_radius_scale,
    )
    train_samples = load_samples(
        train_file,
        cutoff=config.cutoff,
        dtype=dtype,
        limit_configs=limit_configs,
        neighborlist_backend=neighborlist_backend,
        include_sample_weights=True,
    )
    if not no_fit_energy_shift:
        config = replace(config, atomic_energies=fit_atomic_energies(train_samples))
        train_samples = load_samples(
            train_file,
            cutoff=config.cutoff,
            dtype=dtype,
            limit_configs=limit_configs,
            neighborlist_backend=neighborlist_backend,
            include_sample_weights=True,
        )
    valid_samples = load_samples(
        valid_file,
        cutoff=config.cutoff,
        dtype=dtype,
        limit_configs=valid_limit_configs,
        neighborlist_backend=neighborlist_backend,
        include_sample_weights=True,
    )
    model = RTECEScalarModel(config).to(dtype=dtype)
    datamodule = RTECEDataModule(
        train_samples,
        valid_samples,
        batch_size=batch_size,
        valid_batch_size=valid_batch_size,
        num_workers=num_workers,
    )
    lit_model = RTECELightningModule(
        model,
        config,
        lr=lr,
        weight_decay=weight_decay,
        energy_weight=energy_weight,
        force_weight=force_weight,
        force_focus_atomic_numbers=parse_force_focus_elements(force_focus_elements),
        force_focus_weight=force_focus_weight,
        lr_scheduler=lr_scheduler,
        lr_factor=lr_factor,
        lr_patience=lr_patience,
        lr_warmup_steps=lr_warmup_steps,
    )
    callbacks: list[Any] = [
        ModelCheckpoint(
            dirpath=str(output_path),
            filename="rtece-lightning-best",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        )
    ]
    if early_stopping_patience is not None and int(early_stopping_patience) >= 0:
        callbacks.append(EarlyStopping(monitor="val/loss", mode="min", patience=int(early_stopping_patience)))

    trainer_kwargs: dict[str, Any] = {
        "accelerator": accelerator,
        "devices": devices,
        "max_steps": int(max_steps),
        "logger": logger,
        "callbacks": callbacks,
        "enable_checkpointing": True,
        "enable_progress_bar": bool(enable_progress_bar),
        "gradient_clip_val": float(gradient_clip_val),
        "num_sanity_val_steps": 0,
        "check_val_every_n_epoch": check_val_every_n_epoch,
    }
    if max_epochs is not None:
        trainer_kwargs["max_epochs"] = int(max_epochs)
    trainer = L.Trainer(**trainer_kwargs)
    trainer.fit(lit_model, datamodule=datamodule)

    final_checkpoint = output_path / "rtece_scalar.pt"
    best_checkpoint = output_path / "rtece_scalar_best.pt"
    save_checkpoint(final_checkpoint, lit_model.model.to("cpu"), config, metadata={"trainer_backend": "lightning"})
    checkpoint_callback = callbacks[0]
    best_lightning_path = getattr(checkpoint_callback, "best_model_path", "")
    best_valid_loss = getattr(checkpoint_callback, "best_model_score", None)
    best_step = None
    if best_lightning_path:
        best_model = _load_lightning_model_state(best_lightning_path, config, dtype=dtype)
        best_payload = torch.load(best_lightning_path, map_location="cpu", weights_only=False)
        best_step = int(best_payload.get("global_step", 0))
        save_checkpoint(best_checkpoint, best_model, config, metadata={"trainer_backend": "lightning", "lightning_checkpoint": best_lightning_path})
    else:
        save_checkpoint(best_checkpoint, lit_model.model.to("cpu"), config, metadata={"trainer_backend": "lightning", "lightning_checkpoint": None})

    summary: dict[str, Any] = {
        "trainer_backend": "lightning",
        "steps": int(trainer.global_step),
        "max_steps": int(max_steps),
        "best_valid_loss": float(best_valid_loss.detach().cpu()) if torch.is_tensor(best_valid_loss) else (float(best_valid_loss) if best_valid_loss is not None else None),
        "best_step": best_step,
        "variant": variant,
        "train_file": str(train_file),
        "valid_file": str(valid_file),
        "train_configs": len(train_samples),
        "valid_configs": len(valid_samples),
        "energy_per_atom_shift": config.energy_per_atom_shift,
        "atomic_energies": {str(k): float(v) for k, v in (config.atomic_energies or {}).items()},
        "hidden_channels": list(config.hidden_channels),
        "num_radial": config.num_radial,
        "moment_l_max": int(config.moment_l_max) if config.moment_l_max is not None else None,
        "species_basis_channels": int(config.species_basis_channels),
        "species_basis_mode": str(config.species_basis_mode),
        "atomic_cross_radial_sketch_channels": int(config.atomic_cross_radial_sketch_channels),
        "atomic_cross_radial_projection": str(config.atomic_cross_radial_projection),
        "atomic_cross_radial_projection_matrix": [list(row) for row in config.atomic_cross_radial_projection_matrix]
        if config.atomic_cross_radial_projection_matrix is not None
        else None,
        "learnable_radial_mixing": bool(config.learnable_radial_mixing),
        "radial_species_adapter_channels": int(config.radial_species_adapter_channels),
        "radial_species_adapter_scope": str(config.radial_species_adapter_scope),
        "descriptor_conditioner": str(config.descriptor_conditioner),
        "descriptor_conditioner_hidden_channels": int(config.descriptor_conditioner_hidden_channels),
        "descriptor_bottleneck_dim": int(config.descriptor_bottleneck_dim),
        "scalar_path_ids": list(config.scalar_path_ids or []),
        "use_short_range_repulsion": bool(config.use_short_range_repulsion),
        "short_range_repulsion_potential": str(config.short_range_repulsion_potential),
        "short_range_repulsion_strength": float(config.short_range_repulsion_strength),
        "short_range_repulsion_beta": float(config.short_range_repulsion_beta),
        "short_range_repulsion_radius_scale": float(config.short_range_repulsion_radius_scale),
        "seed": int(seed),
        "energy_weight": float(energy_weight),
        "force_weight": float(force_weight),
        "force_focus_atomic_numbers": list(parse_force_focus_elements(force_focus_elements)),
        "force_focus_weight": float(force_focus_weight),
        "sample_weight_keys": ["energy_weight", "forces_weight"],
        "default_dtype": "float64" if dtype == torch.float64 else "float32",
        "neighborlist_backend": neighborlist_backend,
        "batch_size": int(batch_size),
        "valid_batch_size": int(valid_batch_size or batch_size),
        "lr": float(lr),
        "weight_decay": float(weight_decay),
        "lr_scheduler": lr_scheduler,
        "lr_factor": float(lr_factor),
        "lr_patience": int(lr_patience),
        "lr_warmup_steps": int(lr_warmup_steps),
        "early_stopping_patience": int(early_stopping_patience) if early_stopping_patience is not None else None,
        "gradient_clip_val": float(gradient_clip_val),
        "checkpoint": str(final_checkpoint),
        "best_checkpoint": str(best_checkpoint),
        "lightning_best_checkpoint": str(best_lightning_path) if best_lightning_path else None,
    }
    (output_path / "train_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


__all__ = [
    "RTECEDataModule",
    "RTECEDataset",
    "RTECELightningModule",
    "atoms_to_graph",
    "atoms_to_weighted_graph",
    "build_training_config",
    "fit_atomic_energies",
    "fit_rtece_lightning",
    "load_samples",
    "parse_force_focus_elements",
    "parse_hidden_channels",
    "parse_scalar_path_ids",
]
