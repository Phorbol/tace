from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping
import hashlib
import json
import math

import torch


@dataclass(frozen=True)
class RTECEScalarConfig:
    variant: str
    cutoff: float = 5.0
    num_radial: int = 8
    hidden_channels: tuple[int, ...] = (64, 64)
    moment_l_max: int | None = None
    learnable_radial_mixing: bool = False
    radial_species_adapter_channels: int = 0
    radial_species_adapter_scope: str = "all"
    local_l0_chemistry_rank: int = 0
    max_atomic_number: int = 100
    use_element_density: bool = False
    use_density_quadratic: bool = False
    use_vector_moments: bool = False
    use_atomic_moments: bool = False
    species_basis_channels: int = 0
    species_basis_mode: str = "fixed_z_power"
    num_edge_sketches: int = 0
    use_cavity_edge_sketches: bool = False
    radial_edge_sketch_channels: int = 0
    atomic_cross_radial_sketch_channels: int = 2
    atomic_cross_radial_projection: str = "fixed_shell_mean"
    atomic_cross_radial_projection_matrix: tuple[tuple[float, ...], ...] | None = None
    descriptor_conditioner: str = "none"
    descriptor_conditioner_hidden_channels: int = 0
    descriptor_bottleneck_dim: int = 0
    scalar_path_ids: tuple[str, ...] | None = None
    energy_per_atom_shift: float = 0.0
    atomic_energies: Mapping[int, float] | None = None
    use_short_range_repulsion: bool = False
    short_range_repulsion_potential: str = "softplus_overlap"
    short_range_repulsion_strength: float = 0.0
    short_range_repulsion_beta: float = 10.0
    short_range_repulsion_radius_scale: float = 0.75


_ATOMIC_SCALAR_PATH_IDS = {
    "atomic.radial_density",
    "atomic.element_density",
    "atomic.species_basis_density",
    "atomic.local_l0_lowrank_density",
    "atomic.density_square",
    "atomic.vector_norm",
    "atomic.vector_cross_radial_dot",
    "atomic.quadrupole_norm",
    "atomic.quadrupole_cross_radial_frobenius",
}

_EDGE_SHELL_SCALAR_PATH_DIMS = {
    "edge.full_moment.vector_shell_dot": 2,
    "edge.full_moment.vector_cross_shell_dot": 1,
    "edge.full_moment.quadrupole_shell_frobenius": 2,
    "edge.full_moment.quadrupole_cross_shell_frobenius": 1,
    "edge.full_moment.vector_shell_contrast_projection": 2,
}

_EDGE_SCALAR_PATH_DIMS = {
    "edge.full_moment.vector_dot": 1,
    "edge.cavity.vector_dot": 1,
    "edge.cavity.quadrupole_frobenius": 1,
    "edge.cavity.target_vector_projection": 1,
    "edge.cavity.source_vector_projection": 1,
    "edge.cavity.target_quadrupole_projection": 1,
    "edge.cavity.source_quadrupole_projection": 1,
    "edge.direct.radial": 2,
    **_EDGE_SHELL_SCALAR_PATH_DIMS,
}


def _selected_atomic_path_ids(path_ids: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(path_id for path_id in path_ids if path_id.startswith("atomic."))


def _selected_edge_path_ids(path_ids: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(path_id for path_id in path_ids if path_id.startswith("edge."))


_RTECE_VARIANT_CONFIG_KWARGS: dict[str, dict[str, object]] = {
    "rtece_pair": {},
    "rtece_element_density": {"use_element_density": True},
    "rtece_density_quadratic": {"use_density_quadratic": True},
    "rtece_vector_moments": {"use_vector_moments": True},
    "rtece_atomic_moments": {"use_atomic_moments": True},
    "rtece_species_basis4": {"species_basis_channels": 4},
    "rtece_edge_sketch8": {"use_atomic_moments": True, "num_edge_sketches": 8},
    "rtece_cavity_edge_sketch8": {
        "use_atomic_moments": True,
        "num_edge_sketches": 8,
        "use_cavity_edge_sketches": True,
    },
    "rtece_cavity_radial_edge_sketch14": {
        "use_atomic_moments": True,
        "num_edge_sketches": 14,
        "use_cavity_edge_sketches": True,
        "radial_edge_sketch_channels": 2,
    },
    "rtece_edge_sketch16": {"use_atomic_moments": True, "num_edge_sketches": 16},
}


def available_rtece_variants() -> tuple[str, ...]:
    return tuple(_RTECE_VARIANT_CONFIG_KWARGS)


def build_rtece_config(variant: str) -> RTECEScalarConfig:
    try:
        kwargs = dict(_RTECE_VARIANT_CONFIG_KWARGS[variant])
    except KeyError as exc:
        raise ValueError(f"unknown rTECE scalar variant {variant!r}") from exc
    return RTECEScalarConfig(variant=variant, **kwargs)


def _normalize_moment_l_max(moment_l_max: int | None) -> int | None:
    if moment_l_max is None:
        return None
    value = int(moment_l_max)
    if value not in {0, 1, 2}:
        raise ValueError("moment_l_max must be one of 0, 1, or 2 for the current Cartesian rTECE implementation")
    return value


def _normalize_species_basis_mode(name: str) -> str:
    mode = str(name or "fixed_z_power")
    if mode not in {"fixed_z_power", "learnable_embedding"}:
        raise ValueError("species_basis_mode must be fixed_z_power or learnable_embedding")
    return mode


def _normalize_descriptor_conditioner(name: str, hidden_channels: int) -> tuple[str, int]:
    conditioner = str(name or "none")
    if conditioner not in {"none", "residual_mlp"}:
        raise ValueError("descriptor_conditioner must be none or residual_mlp")
    hidden = int(hidden_channels)
    if conditioner == "none":
        return conditioner, 0
    if hidden <= 0:
        raise ValueError("descriptor_conditioner_hidden_channels must be positive for residual_mlp")
    return conditioner, hidden


def _normalize_descriptor_bottleneck_dim(value: int) -> int:
    dim = int(value)
    if dim < 0:
        raise ValueError("descriptor_bottleneck_dim must be non-negative")
    return dim


def _normalize_radial_species_adapter_channels(value: int) -> int:
    channels = int(value)
    if channels < 0:
        raise ValueError("radial_species_adapter_channels must be non-negative")
    return channels


def _normalize_radial_species_adapter_scope(value: str) -> str:
    scope = str(value or "all")
    if scope not in {"all", "atomic", "edge"}:
        raise ValueError("radial_species_adapter_scope must be all, atomic, or edge")
    return scope


def _normalize_local_l0_chemistry_rank(value: int) -> int:
    rank = int(value)
    if rank < 0:
        raise ValueError("local_l0_chemistry_rank must be non-negative")
    if rank > 4:
        raise ValueError("local_l0_chemistry_rank currently supports ranks 0 through 4")
    return rank


def _radial_species_adapter_label(config: RTECEScalarConfig) -> str:
    scope = _normalize_radial_species_adapter_scope(config.radial_species_adapter_scope)
    base = "learnable_edge_species_radial_adapter"
    return base if scope == "all" else f"{base}_{scope}"


def config_with_moment_l_max(config: RTECEScalarConfig, moment_l_max: int | None) -> RTECEScalarConfig:
    value = _normalize_moment_l_max(moment_l_max)
    if value is None:
        return config
    edge_kwargs: dict[str, object] = {}
    if value < 2:
        edge_kwargs = {
            "num_edge_sketches": 0,
            "use_cavity_edge_sketches": False,
            "radial_edge_sketch_channels": 0,
        }
    return replace(
        config,
        moment_l_max=value,
        use_vector_moments=(value == 1),
        use_atomic_moments=(value >= 2),
        **edge_kwargs,
    )


def _scalar_path_required_ell(path_id: str) -> int:
    if path_id in _EDGE_SHELL_SCALAR_PATH_DIMS:
        return 2
    if "quadrupole" in path_id:
        return 2
    if "vector" in path_id:
        return 1
    return 0


def _edge_paths_required_ell(config: RTECEScalarConfig) -> int:
    if config.num_edge_sketches <= 0:
        return 0
    edge_path_ids = _selected_edge_path_ids(config.scalar_path_ids or ())
    if not edge_path_ids:
        return 2
    return max(_scalar_path_required_ell(path_id) for path_id in edge_path_ids)


def _normalize_atomic_cross_radial_projection_matrix(
    matrix: object,
    *,
    num_sketches: int,
    num_radial: int,
) -> tuple[tuple[float, ...], ...]:
    try:
        rows = tuple(tuple(float(value) for value in row) for row in matrix)  # type: ignore[union-attr]
    except TypeError as exc:
        raise ValueError("atomic_cross_radial_projection_matrix must be a rank-2 numeric sequence") from exc
    expected_shape = (int(num_sketches), int(num_radial))
    if len(rows) != expected_shape[0] or any(len(row) != expected_shape[1] for row in rows):
        actual_shape = (len(rows), len(rows[0]) if rows else 0)
        raise ValueError(
            "atomic_cross_radial_projection_matrix must have shape "
            f"{expected_shape}, got {actual_shape}"
        )
    return rows


def build_rtece_config_from_path_ids(
    variant: str,
    scalar_path_ids: tuple[str, ...] | list[str],
    *,
    cutoff: float = 5.0,
    num_radial: int = 8,
    hidden_channels: tuple[int, ...] = (64, 64),
    moment_l_max: int | None = None,
    learnable_radial_mixing: bool = False,
    radial_species_adapter_channels: int = 0,
    radial_species_adapter_scope: str = "all",
    local_l0_chemistry_rank: int = 0,
    max_atomic_number: int = 100,
    species_basis_channels: int = 0,
    species_basis_mode: str = "fixed_z_power",
    energy_per_atom_shift: float = 0.0,
    atomic_energies: Mapping[int, float] | None = None,
    use_short_range_repulsion: bool = False,
    short_range_repulsion_potential: str = "softplus_overlap",
    short_range_repulsion_strength: float = 0.0,
    short_range_repulsion_beta: float = 10.0,
    short_range_repulsion_radius_scale: float = 0.75,
    atomic_cross_radial_sketch_channels: int = 2,
    atomic_cross_radial_projection: str = "fixed_shell_mean",
    atomic_cross_radial_projection_matrix: object | None = None,
    descriptor_conditioner: str = "none",
    descriptor_conditioner_hidden_channels: int = 0,
    descriptor_bottleneck_dim: int = 0,
) -> RTECEScalarConfig:
    paths = tuple(str(path_id) for path_id in scalar_path_ids)
    if not paths:
        raise ValueError("at least one scalar path id is required")
    supported_paths = _ATOMIC_SCALAR_PATH_IDS | set(_EDGE_SCALAR_PATH_DIMS)
    unsupported = [path_id for path_id in paths if path_id not in supported_paths]
    if unsupported:
        raise ValueError(
            "build_rtece_config_from_path_ids currently supports selected atomic scalar "
            f"and non-radial edge scalar path ids only; unsupported paths: {unsupported}"
        )
    if "atomic.radial_density" not in paths:
        raise ValueError("atomic.radial_density is required as the base scalar path")
    if "atomic.species_basis_density" in paths and species_basis_channels <= 0:
        raise ValueError("species_basis_channels must be positive for atomic.species_basis_density")
    normalized_local_l0_chemistry_rank = _normalize_local_l0_chemistry_rank(local_l0_chemistry_rank)
    if "atomic.local_l0_lowrank_density" in paths and normalized_local_l0_chemistry_rank <= 0:
        raise ValueError("local_l0_chemistry_rank must be positive for atomic.local_l0_lowrank_density")
    if "atomic.local_l0_lowrank_density" not in paths and normalized_local_l0_chemistry_rank:
        raise ValueError("local_l0_chemistry_rank requires atomic.local_l0_lowrank_density")
    normalized_species_basis_mode = _normalize_species_basis_mode(species_basis_mode)
    edge_paths = _selected_edge_path_ids(paths)
    has_cavity_edge_paths = any(path_id.startswith("edge.cavity.") for path_id in edge_paths)
    has_full_edge_paths = any(path_id.startswith("edge.full_moment.") for path_id in edge_paths)
    if has_cavity_edge_paths and has_full_edge_paths:
        raise ValueError("full-moment and cavity edge paths cannot be mixed in one path-id config")
    normalized_l_max = _normalize_moment_l_max(moment_l_max)
    required_l_max = max(_scalar_path_required_ell(path_id) for path_id in paths)
    if normalized_l_max is not None and normalized_l_max < required_l_max:
        raise ValueError(
            f"moment_l_max={normalized_l_max} cannot realize scalar paths requiring ell<={required_l_max}"
        )
    atomic_cross_channels = int(atomic_cross_radial_sketch_channels)
    atomic_cross_projection = str(atomic_cross_radial_projection)
    if atomic_cross_projection not in {"fixed_shell_mean", "learnable", "pod_fixed"}:
        raise ValueError("atomic_cross_radial_projection must be fixed_shell_mean, learnable, or pod_fixed")
    has_atomic_cross_radial_paths = any("cross_radial" in path_id for path_id in _selected_atomic_path_ids(paths))
    if atomic_cross_projection in {"learnable", "pod_fixed"} and not has_atomic_cross_radial_paths:
        raise ValueError(f"atomic_cross_radial_projection={atomic_cross_projection} requires an atomic cross-radial path")
    normalized_atomic_cross_projection_matrix = None
    if atomic_cross_projection == "pod_fixed":
        if atomic_cross_radial_projection_matrix is None:
            raise ValueError("atomic_cross_radial_projection=pod_fixed requires atomic_cross_radial_projection_matrix")
        normalized_atomic_cross_projection_matrix = _normalize_atomic_cross_radial_projection_matrix(
            atomic_cross_radial_projection_matrix,
            num_sketches=atomic_cross_channels,
            num_radial=int(num_radial),
        )
    elif atomic_cross_radial_projection_matrix is not None:
        raise ValueError("atomic_cross_radial_projection_matrix is only valid for atomic_cross_radial_projection=pod_fixed")
    if has_atomic_cross_radial_paths:
        if atomic_cross_channels < 2:
            raise ValueError("atomic_cross_radial_sketch_channels must be at least 2")
        if atomic_cross_channels > int(num_radial):
            raise ValueError("atomic_cross_radial_sketch_channels cannot exceed num_radial")
    normalized_descriptor_conditioner, normalized_descriptor_conditioner_hidden = _normalize_descriptor_conditioner(
        descriptor_conditioner,
        descriptor_conditioner_hidden_channels,
    )
    normalized_descriptor_bottleneck_dim = _normalize_descriptor_bottleneck_dim(descriptor_bottleneck_dim)
    normalized_radial_species_adapter_channels = _normalize_radial_species_adapter_channels(
        radial_species_adapter_channels
    )
    normalized_radial_species_adapter_scope = _normalize_radial_species_adapter_scope(
        radial_species_adapter_scope
    )

    return RTECEScalarConfig(
        variant=variant,
        cutoff=float(cutoff),
        num_radial=int(num_radial),
        hidden_channels=tuple(int(value) for value in hidden_channels),
        moment_l_max=normalized_l_max,
        learnable_radial_mixing=bool(learnable_radial_mixing),
        radial_species_adapter_channels=normalized_radial_species_adapter_channels,
        radial_species_adapter_scope=normalized_radial_species_adapter_scope,
        local_l0_chemistry_rank=normalized_local_l0_chemistry_rank,
        max_atomic_number=int(max_atomic_number),
        use_element_density="atomic.element_density" in paths,
        use_density_quadratic="atomic.density_square" in paths,
        use_vector_moments="atomic.vector_norm" in paths or "atomic.vector_cross_radial_dot" in paths,
        use_atomic_moments=(
            "atomic.quadrupole_norm" in paths
            or "atomic.quadrupole_cross_radial_frobenius" in paths
            or required_l_max >= 2
            or (normalized_l_max is None and bool(edge_paths))
        ),
        species_basis_channels=int(species_basis_channels) if "atomic.species_basis_density" in paths else 0,
        species_basis_mode=normalized_species_basis_mode if "atomic.species_basis_density" in paths else "fixed_z_power",
        num_edge_sketches=sum(int(_EDGE_SCALAR_PATH_DIMS[path_id]) for path_id in edge_paths),
        use_cavity_edge_sketches=has_cavity_edge_paths,
        radial_edge_sketch_channels=0,
        atomic_cross_radial_sketch_channels=atomic_cross_channels,
        atomic_cross_radial_projection=atomic_cross_projection,
        atomic_cross_radial_projection_matrix=normalized_atomic_cross_projection_matrix,
        descriptor_conditioner=normalized_descriptor_conditioner,
        descriptor_conditioner_hidden_channels=normalized_descriptor_conditioner_hidden,
        descriptor_bottleneck_dim=normalized_descriptor_bottleneck_dim,
        scalar_path_ids=paths,
        energy_per_atom_shift=float(energy_per_atom_shift),
        atomic_energies=atomic_energies,
        use_short_range_repulsion=bool(use_short_range_repulsion),
        short_range_repulsion_potential=str(short_range_repulsion_potential),
        short_range_repulsion_strength=float(short_range_repulsion_strength),
        short_range_repulsion_beta=float(short_range_repulsion_beta),
        short_range_repulsion_radius_scale=float(short_range_repulsion_radius_scale),
    )


def build_rtece_config_from_manifest(manifest: Mapping[str, Any]) -> RTECEScalarConfig:
    if manifest.get("schema_version") != "rtece_path_manifest.v1":
        raise ValueError("expected rtece_path_manifest.v1 manifest")
    payload = manifest.get("config")
    if not isinstance(payload, Mapping):
        raise ValueError("rTECE path manifest is missing config payload")
    hidden = payload.get("hidden_channels", RTECEScalarConfig(variant="rtece_pair").hidden_channels)
    if isinstance(hidden, list):
        hidden_channels = tuple(int(value) for value in hidden)
    else:
        hidden_channels = tuple(int(value) for value in hidden)
    return RTECEScalarConfig(
        variant=str(payload.get("variant")),
        cutoff=float(payload.get("cutoff", 5.0)),
        num_radial=int(payload.get("num_radial", 8)),
        hidden_channels=hidden_channels,
        moment_l_max=_normalize_moment_l_max(payload.get("moment_l_max", None)),
        max_atomic_number=int(payload.get("max_atomic_number", 100)),
        learnable_radial_mixing=bool(payload.get("learnable_radial_mixing", False)),
        radial_species_adapter_channels=_normalize_radial_species_adapter_channels(
            int(payload.get("radial_species_adapter_channels", 0))
        ),
        radial_species_adapter_scope=_normalize_radial_species_adapter_scope(
            str(payload.get("radial_species_adapter_scope", "all"))
        ),
        local_l0_chemistry_rank=_normalize_local_l0_chemistry_rank(
            int(payload.get("local_l0_chemistry_rank", 0))
        ),
        use_element_density=bool(payload.get("use_element_density", False)),
        use_density_quadratic=bool(payload.get("use_density_quadratic", False)),
        use_vector_moments=bool(payload.get("use_vector_moments", False)),
        use_atomic_moments=bool(payload.get("use_atomic_moments", False)),
        species_basis_channels=int(payload.get("species_basis_channels", 0)),
        species_basis_mode=_normalize_species_basis_mode(str(payload.get("species_basis_mode", "fixed_z_power"))),
        num_edge_sketches=int(payload.get("num_edge_sketches", 0)),
        use_cavity_edge_sketches=bool(payload.get("use_cavity_edge_sketches", False)),
        radial_edge_sketch_channels=int(payload.get("radial_edge_sketch_channels", 0)),
        atomic_cross_radial_sketch_channels=int(payload.get("atomic_cross_radial_sketch_channels", 2)),
        atomic_cross_radial_projection=str(payload.get("atomic_cross_radial_projection", "fixed_shell_mean")),
        atomic_cross_radial_projection_matrix=_normalize_atomic_cross_radial_projection_matrix(
            payload["atomic_cross_radial_projection_matrix"],
            num_sketches=int(payload.get("atomic_cross_radial_sketch_channels", 2)),
            num_radial=int(payload.get("num_radial", 8)),
        )
        if payload.get("atomic_cross_radial_projection_matrix") is not None
        else None,
        descriptor_conditioner=_normalize_descriptor_conditioner(
            str(payload.get("descriptor_conditioner", "none")),
            int(payload.get("descriptor_conditioner_hidden_channels", 0)),
        )[0],
        descriptor_conditioner_hidden_channels=_normalize_descriptor_conditioner(
            str(payload.get("descriptor_conditioner", "none")),
            int(payload.get("descriptor_conditioner_hidden_channels", 0)),
        )[1],
        descriptor_bottleneck_dim=_normalize_descriptor_bottleneck_dim(
            int(payload.get("descriptor_bottleneck_dim", 0))
        ),
        scalar_path_ids=tuple(str(path_id) for path_id in payload["scalar_path_ids"])
        if "scalar_path_ids" in payload
        else None,
        use_short_range_repulsion=bool((payload.get("short_range_repulsion") or {}).get("enabled", payload.get("use_short_range_repulsion", False))),
        short_range_repulsion_potential=str((payload.get("short_range_repulsion") or {}).get("potential", payload.get("short_range_repulsion_potential", "softplus_overlap"))),
        short_range_repulsion_strength=float((payload.get("short_range_repulsion") or {}).get("strength", payload.get("short_range_repulsion_strength", 0.0))),
        short_range_repulsion_beta=float((payload.get("short_range_repulsion") or {}).get("beta", payload.get("short_range_repulsion_beta", 10.0))),
        short_range_repulsion_radius_scale=float((payload.get("short_range_repulsion") or {}).get("radius_scale", payload.get("short_range_repulsion_radius_scale", 0.75))),
    )


def _num_off_diagonal_shell_pairs(num_shells: int) -> int:
    shells = int(num_shells)
    if shells < 2:
        raise ValueError("cross-radial scalar paths require at least two radial sketch channels")
    return shells * (shells - 1) // 2


def _scalar_path_descriptor_dim(path_id: str, config: RTECEScalarConfig) -> int:
    if path_id in {
        "atomic.radial_density",
        "atomic.element_density",
        "atomic.density_square",
        "atomic.vector_norm",
        "atomic.quadrupole_norm",
    }:
        return int(config.num_radial)
    if path_id == "atomic.species_basis_density":
        if config.species_basis_channels <= 0:
            raise ValueError("atomic.species_basis_density requires species_basis_channels > 0")
        return int(config.num_radial) * int(config.species_basis_channels)
    if path_id == "atomic.local_l0_lowrank_density":
        if config.local_l0_chemistry_rank <= 0:
            raise ValueError("atomic.local_l0_lowrank_density requires local_l0_chemistry_rank > 0")
        return int(config.num_radial) * int(config.local_l0_chemistry_rank)
    if path_id in {"atomic.vector_cross_radial_dot", "atomic.quadrupole_cross_radial_frobenius"}:
        return _num_off_diagonal_shell_pairs(config.atomic_cross_radial_sketch_channels)
    if path_id in _EDGE_SCALAR_PATH_DIMS:
        return int(_EDGE_SCALAR_PATH_DIMS[path_id])
    raise ValueError(f"unsupported scalar path id {path_id!r}")


def descriptor_dim(config: RTECEScalarConfig) -> int:
    if config.scalar_path_ids is not None:
        return sum(_scalar_path_descriptor_dim(path_id, config) for path_id in config.scalar_path_ids)
    dim = config.num_radial
    if config.use_element_density:
        dim += config.num_radial
    if config.use_density_quadratic:
        dim += config.num_radial
    if config.use_vector_moments:
        dim += config.num_radial
    if config.species_basis_channels:
        dim += config.num_radial * int(config.species_basis_channels)
    if config.local_l0_chemistry_rank:
        dim += config.num_radial * int(config.local_l0_chemistry_rank)
    if config.use_atomic_moments:
        dim += 2 * config.num_radial
    dim += config.num_edge_sketches
    return dim


def rtece_route_contract(
    config: RTECEScalarConfig,
    *,
    force_mode: str = "autograd",
    graph_construction_backend: str | None = None,
    graph_update_backend: str | None = None,
) -> dict[str, object]:
    retained = ["radial_density"]
    deleted = [
        "persistent_equivariant_node_state",
        "persistent_equivariant_edge_state",
        "edge_tensor_product_channel_mixing",
        "multi_layer_equivariant_message_passing",
    ]
    if config.use_element_density:
        semantic_tier = "T3_element_conditioned_scalar_density"
        descriptor_family = "element_density"
        retained.append("neighbor_element_density")
    elif config.use_density_quadratic:
        semantic_tier = "T3_scalar_density_quadratic"
        descriptor_family = "density_quadratic"
        retained.append("local_scalar_density_square")
    elif config.use_vector_moments:
        semantic_tier = "T3_vector_moment_scalarization"
        descriptor_family = "vector_moment_norm"
        retained.append("low_order_vector_moment_norm")
    elif config.species_basis_channels:
        semantic_tier = "T3_low_rank_species_density"
        descriptor_family = "species_basis_density"
        retained.append("low_rank_neighbor_species_basis")
        if config.species_basis_mode == "learnable_embedding":
            retained.append("learnable_low_rank_species_basis")
            semantic_tier = f"{semantic_tier}_learnable_species_basis"
            descriptor_family = f"{descriptor_family}_learnable_species_basis"
        if config.use_atomic_moments or config.num_edge_sketches:
            if config.use_cavity_edge_sketches:
                semantic_tier = "T3_species_cavity_edge_scalar_sketch"
                descriptor_family = "species_basis_density_plus_cavity_edge_sketch"
                retained.extend(
                    [
                        "low_order_atomic_moments",
                        "cavity_edge_relational_scalar_sketches",
                    ]
                )
            else:
                semantic_tier = "T3_species_atomic_moment_scalar_sketch"
                descriptor_family = "species_basis_density_plus_atomic_moment_sketch"
                retained.extend(["low_order_atomic_moments", "edge_relational_scalar_sketches"])
    elif config.local_l0_chemistry_rank:
        semantic_tier = "T3_trainable_local_l0_lowrank_density"
        descriptor_family = "local_l0_lowrank_density"
        retained.extend(["trainable_local_l0_chemistry_front", "early_scalarized_local_l0_density"])
    elif config.use_atomic_moments or config.num_edge_sketches:
        if config.scalar_path_ids and any("cross_radial" in path_id for path_id in config.scalar_path_ids):
            retained.append("atomic_cross_radial_invariants")
        if config.use_cavity_edge_sketches:
            if config.radial_edge_sketch_channels:
                semantic_tier = "T3_cavity_radial_edge_scalar_sketch"
                descriptor_family = "cavity_radial_atomic_moment_sketch"
                retained.extend(
                    [
                        "low_order_atomic_moments",
                        "cavity_edge_relational_scalar_sketches",
                        "low_rank_radial_edge_moment_sketches",
                        "cross_radial_edge_invariants",
                        "direct_edge_radial_path",
                    ]
                )
            else:
                semantic_tier = "T3_cavity_edge_scalar_sketch"
                descriptor_family = "cavity_atomic_moment_sketch"
                retained.extend(
                    [
                        "low_order_atomic_moments",
                        "cavity_edge_relational_scalar_sketches",
                        "direct_edge_radial_path",
                    ]
                )
        else:
            semantic_tier = "T3_atomic_moment_scalar_sketch"
            descriptor_family = "atomic_moment_sketch"
            retained.extend(["low_order_atomic_moments", "edge_relational_scalar_sketches"])
    else:
        semantic_tier = "T4_scalar_pair_density"
        descriptor_family = "pair_density"
    if config.scalar_path_ids and any("cross_radial" in path_id for path_id in config.scalar_path_ids):
        if "atomic_cross_radial_invariants" not in retained:
            retained.append("atomic_cross_radial_invariants")
    if config.scalar_path_ids and any("cross_radial" in path_id for path_id in config.scalar_path_ids):
        retained.append(f"atomic_cross_radial_rank_{int(config.atomic_cross_radial_sketch_channels)}")
        if config.atomic_cross_radial_projection == "learnable":
            retained.append("learnable_atomic_cross_radial_projection")
        if config.atomic_cross_radial_projection == "pod_fixed":
            retained.append("pod_fixed_atomic_cross_radial_projection")
    if config.species_basis_channels and config.species_basis_mode == "learnable_embedding":
        if "learnable_low_rank_species_basis" not in retained:
            retained.append("learnable_low_rank_species_basis")
        if "learnable_species_basis" not in semantic_tier:
            semantic_tier = f"{semantic_tier}_learnable_species_basis"
            descriptor_family = f"{descriptor_family}_learnable_species_basis"
    if config.learnable_radial_mixing:
        retained.append("trainable_low_rank_radial_mixing")
        semantic_tier = f"{semantic_tier}_learnable_radial_mixing"
        descriptor_family = f"{descriptor_family}_learnable_radial_mixing"
    if config.local_l0_chemistry_rank:
        if "trainable_local_l0_chemistry_front" not in retained:
            retained.extend(["trainable_local_l0_chemistry_front", "early_scalarized_local_l0_density"])
        if "local_l0_lowrank_density" not in descriptor_family:
            semantic_tier = f"{semantic_tier}_local_l0_lowrank"
            descriptor_family = f"{descriptor_family}_local_l0_lowrank"
    if config.radial_species_adapter_channels:
        adapter_label = _radial_species_adapter_label(config)
        retained.append("trainable_edge_species_radial_basis")
        if config.radial_species_adapter_scope != "all":
            retained.append(f"path_scoped_radial_species_adapter_{config.radial_species_adapter_scope}")
        semantic_tier = f"{semantic_tier}_{adapter_label}"
        descriptor_family = f"{descriptor_family}_{adapter_label}"
    if config.use_short_range_repulsion:
        short_range_group = "short_range_zbl_prior" if config.short_range_repulsion_potential == "zbl" else "short_range_radial_core"
        retained.append(short_range_group)
        semantic_tier = f"{semantic_tier}_with_{short_range_group}"
        descriptor_family = f"{descriptor_family}_plus_{short_range_group}"
    if config.descriptor_conditioner != "none":
        retained.append("trainable_scalar_descriptor_conditioner")
        semantic_tier = f"{semantic_tier}_scalar_conditioned"
        descriptor_family = f"{descriptor_family}_scalar_conditioned"
    if config.descriptor_bottleneck_dim:
        retained.append("trainable_low_rank_descriptor_mixer")
        semantic_tier = f"{semantic_tier}_low_rank_descriptor_mixer"
        descriptor_family = f"{descriptor_family}_low_rank_descriptor_mixer"

    descriptor_realization = "pytorch_edge_scatter"
    force_realization = "autograd_conservative"
    fused_descriptor = False
    fused_force = False
    if force_mode in {"analytic_pair", "analytic_density", "analytic_element_packed"}:
        force_realization = "analytic_scalar_chain_rule"
    if force_mode == "analytic_element_packed":
        descriptor_realization = "packed_pytorch_scatter"
    if force_mode == "analytic_pair_triton_force":
        force_realization = "triton_fused_force"
        fused_force = True
    if force_mode == "analytic_element_triton_force":
        force_realization = "triton_fused_force"
        fused_force = True
    if force_mode == "analytic_element_triton_descriptor_force":
        descriptor_realization = "triton_fused_edge_descriptor"
        force_realization = "triton_fused_descriptor_force"
        fused_descriptor = True
        fused_force = True
    if force_mode == "analytic_element_direct_padded_descriptor_force":
        descriptor_realization = "triton_direct_padded_descriptor"
        force_realization = "triton_direct_padded_descriptor_force"
        fused_descriptor = True
        fused_force = True
    if force_mode == "analytic_element_cell_list_descriptor_force":
        descriptor_realization = "cell_list_fused_descriptor_oracle"
        force_realization = "cell_list_analytic_descriptor_force"
        fused_descriptor = True

    backend = graph_update_backend or graph_construction_backend
    if force_mode == "analytic_element_cell_list_descriptor_force":
        graph_semantics = "direct_active_nopbc"
    elif backend and "torch_radius_nopbc" in backend:
        graph_semantics = "direct_active_nopbc"
    elif backend == "ase_neighborlist":
        graph_semantics = "ase_neighborlist_pbc"
    else:
        graph_semantics = "prebuilt_edge_index"

    if force_mode == "analytic_element_cell_list_descriptor_force":
        edge_state_lifetime = "streaming_cell_candidates_oracle"
    elif force_mode == "analytic_element_direct_padded_descriptor_force":
        edge_state_lifetime = "streaming_padded_candidates"
    elif graph_update_backend == "cached_topology":
        edge_state_lifetime = "persistent_cached_edge_index"
    elif graph_update_backend and "triton_counted" in graph_update_backend:
        edge_state_lifetime = "counted_exact_edge_buffer"
    elif graph_update_backend and "triton_padded" in graph_update_backend:
        edge_state_lifetime = "padded_triton_edge_buffer"
    elif graph_update_backend and "cell" in graph_update_backend:
        edge_state_lifetime = "streaming_cell_candidates"
    elif graph_update_backend:
        edge_state_lifetime = "runtime_materialized_edge_index"
    elif graph_construction_backend:
        edge_state_lifetime = "prebuilt_materialized_edge_index"
    else:
        edge_state_lifetime = "caller_supplied_edge_index"

    pareto_axes = ["semantic_projection", "scalar_head_capacity", "force_realization"]
    if config.moment_l_max is not None:
        pareto_axes.append("angular_bandwidth_l_max")
    if config.scalar_path_ids and any("cross_radial" in path_id for path_id in config.scalar_path_ids):
        pareto_axes.append("radial_rank")
        if config.atomic_cross_radial_projection == "learnable":
            pareto_axes.append("trainable_cross_radial_projection")
        if config.atomic_cross_radial_projection == "pod_fixed":
            pareto_axes.append("pod_fixed_cross_radial_projection")
    if config.species_basis_channels and config.species_basis_mode == "learnable_embedding":
        pareto_axes.append("trainable_species_basis")
    if config.learnable_radial_mixing:
        pareto_axes.append("trainable_feature_extractor")
    if config.local_l0_chemistry_rank:
        pareto_axes.append("trainable_local_l0_front")
    if config.radial_species_adapter_channels:
        pareto_axes.append("trainable_edge_species_radial_basis")
        if config.radial_species_adapter_scope != "all":
            pareto_axes.append("path_scoped_radial_species_adapter")
    if config.descriptor_conditioner != "none":
        pareto_axes.append("scalar_descriptor_conditioning")
    if config.descriptor_bottleneck_dim:
        pareto_axes.append("descriptor_bottleneck")
    if config.use_short_range_repulsion:
        pareto_axes.append("short_range_physical_prior")
    if graph_construction_backend or graph_update_backend:
        pareto_axes.append("topology_provider")
    if fused_descriptor or fused_force:
        pareto_axes.append("kernel_fusion")

    ase_implemented_outputs = ["energy", "free_energy", "forces"]
    missing_output_contracts = ["validated_edge_gradient_virial"]
    if force_mode == "autograd":
        ase_implemented_outputs.append("stress")
        stress_realization = "ase_autograd_finite_strain_inference"
    else:
        missing_output_contracts.append("ase_stress_requires_autograd_force_mode")
        stress_realization = "not_available_for_selected_force_mode"

    return {
        "semantic_tier": semantic_tier,
        "descriptor_family": descriptor_family,
        "retained_tece_groups": retained,
        "deleted_tece_groups": deleted,
        "implemented_outputs": ["energy", "atomic_energy", "forces"],
        "ase_implemented_outputs": ase_implemented_outputs,
        "missing_output_contracts": missing_output_contracts,
        "stress_realization": stress_realization,
        "virial_realization": "not_implemented",
        "descriptor_dim": descriptor_dim(config),
        "num_radial": int(config.num_radial),
        "hidden_channels": list(config.hidden_channels),
        "moment_l_max": int(config.moment_l_max) if config.moment_l_max is not None else None,
        "force_mode": force_mode,
        "force_realization": force_realization,
        "descriptor_realization": descriptor_realization,
        "fused_descriptor": fused_descriptor,
        "fused_force": fused_force,
        "graph_semantics": graph_semantics,
        "graph_construction_backend": graph_construction_backend,
        "graph_update_backend": graph_update_backend,
        "edge_state_lifetime": edge_state_lifetime,
        "energy_reference": "per_element_atomic_energies" if config.atomic_energies else ("global_per_atom_shift" if config.energy_per_atom_shift else "none"),
        "feature_extractor": (
            "learnable_radial_linear_mixing" if config.learnable_radial_mixing else "fixed_radial_basis"
        )
        + ("+trainable_local_l0_chemistry_front" if config.local_l0_chemistry_rank else "")
        + (f"+{_radial_species_adapter_label(config)}" if config.radial_species_adapter_channels else "")
        + ("+learnable_species_basis" if config.species_basis_channels and config.species_basis_mode == "learnable_embedding" else "")
        + ("+residual_scalar_descriptor_conditioner" if config.descriptor_conditioner != "none" else "")
        + ("+low_rank_descriptor_mixer" if config.descriptor_bottleneck_dim else ""),
        "descriptor_readout_dim": int(config.descriptor_bottleneck_dim) if config.descriptor_bottleneck_dim else descriptor_dim(config),
        "pareto_axes": pareto_axes,
    }


def _moment_spec(moment_id: str, *, ell: int, radial_projection: str, chemistry_basis: str = "none") -> dict[str, object]:
    return {
        "id": moment_id,
        "ell": int(ell),
        "radial_projection": radial_projection,
        "chemistry_basis": chemistry_basis,
    }


def _scalar_path_spec(
    path_id: str,
    *,
    placement: str,
    inputs: list[str],
    contraction: str,
    radial_projection: str,
    cavity: bool = False,
    parity: int = 1,
    cutoff_power: int = 0,
    radial_gate: str | None = None,
    cost_group: str,
) -> dict[str, object]:
    return {
        "id": path_id,
        "placement": placement,
        "inputs": inputs,
        "contraction": contraction,
        "radial_projection": radial_projection,
        "cavity": bool(cavity),
        "parity": int(parity),
        "cutoff_power": int(cutoff_power),
        "radial_gate": radial_gate,
        "cost_group": cost_group,
    }


def _atomic_cross_radial_projection_label(config: RTECEScalarConfig) -> str:
    if config.atomic_cross_radial_projection == "learnable":
        return f"learnable_{int(config.atomic_cross_radial_sketch_channels)}x{int(config.num_radial)}_cross_radial_projection"
    if config.atomic_cross_radial_projection == "pod_fixed":
        return f"pod_fixed_{int(config.atomic_cross_radial_sketch_channels)}x{int(config.num_radial)}_cross_radial_projection"
    return f"fixed_{int(config.atomic_cross_radial_sketch_channels)}_shell_mean"


def _config_manifest_payload(config: RTECEScalarConfig) -> dict[str, object]:
    payload: dict[str, object] = {
        "variant": config.variant,
        "cutoff": float(config.cutoff),
        "num_radial": int(config.num_radial),
        "hidden_channels": list(config.hidden_channels),
        "moment_l_max": int(config.moment_l_max) if config.moment_l_max is not None else None,
        "learnable_radial_mixing": bool(config.learnable_radial_mixing),
        "radial_species_adapter_channels": int(config.radial_species_adapter_channels),
        "radial_species_adapter_scope": str(config.radial_species_adapter_scope),
        "local_l0_chemistry_rank": int(config.local_l0_chemistry_rank),
        "max_atomic_number": int(config.max_atomic_number),
        "use_element_density": bool(config.use_element_density),
        "use_density_quadratic": bool(config.use_density_quadratic),
        "use_vector_moments": bool(config.use_vector_moments),
        "use_atomic_moments": bool(config.use_atomic_moments),
        "species_basis_channels": int(config.species_basis_channels),
        "species_basis_mode": str(config.species_basis_mode),
        "num_edge_sketches": int(config.num_edge_sketches),
        "use_cavity_edge_sketches": bool(config.use_cavity_edge_sketches),
        "radial_edge_sketch_channels": int(config.radial_edge_sketch_channels),
        "atomic_cross_radial_sketch_channels": int(config.atomic_cross_radial_sketch_channels),
        "atomic_cross_radial_projection": str(config.atomic_cross_radial_projection),
        "descriptor_conditioner": str(config.descriptor_conditioner),
        "descriptor_conditioner_hidden_channels": int(config.descriptor_conditioner_hidden_channels),
        "descriptor_bottleneck_dim": int(config.descriptor_bottleneck_dim),
        "energy_reference": "per_element_atomic_energies" if config.atomic_energies else ("global_per_atom_shift" if config.energy_per_atom_shift else "none"),
        "short_range_repulsion": {
            "enabled": bool(config.use_short_range_repulsion),
            "potential": str(config.short_range_repulsion_potential),
            "strength": float(config.short_range_repulsion_strength),
            "beta": float(config.short_range_repulsion_beta),
            "radius_scale": float(config.short_range_repulsion_radius_scale),
        },
    }
    if config.atomic_cross_radial_projection_matrix is not None:
        payload["atomic_cross_radial_projection_matrix"] = [
            list(row) for row in config.atomic_cross_radial_projection_matrix
        ]
    if config.scalar_path_ids is not None:
        payload["scalar_path_ids"] = list(config.scalar_path_ids)
    return payload


def rtece_path_manifest(
    config: RTECEScalarConfig,
    *,
    force_mode: str = "autograd",
    graph_construction_backend: str | None = None,
    graph_update_backend: str | None = None,
) -> dict[str, object]:
    route = rtece_route_contract(
        config,
        force_mode=force_mode,
        graph_construction_backend=graph_construction_backend,
        graph_update_backend=graph_update_backend,
    )
    radial_projection = "fixed_two_shell_mean" if config.radial_edge_sketch_channels else "full_radial_mean"
    edge_required_ell = _edge_paths_required_ell(config)
    base_radial_projection = "learnable_identity_initialized_linear_mixing" if config.learnable_radial_mixing else "identity"
    radial_adapter_scope = _normalize_radial_species_adapter_scope(config.radial_species_adapter_scope)
    atomic_radial_adapter_enabled = bool(config.radial_species_adapter_channels) and radial_adapter_scope in {"all", "atomic"}
    radial_density_chemistry_basis = (
        f"learnable_center_neighbor_pair_embedding_{int(config.radial_species_adapter_channels)}"
        if atomic_radial_adapter_enabled
        else "none"
    )
    moments = [
        _moment_spec(
            "moment.l0.radial_density",
            ell=0,
            radial_projection=base_radial_projection,
            chemistry_basis=radial_density_chemistry_basis,
        )
    ]
    if config.use_element_density:
        moments.append(_moment_spec("moment.l0.element_density", ell=0, radial_projection=base_radial_projection, chemistry_basis="atomic_number_first_moment"))
    if config.species_basis_channels:
        chemistry_basis = (
            f"learnable_embedding_{int(config.species_basis_channels)}"
            if config.species_basis_mode == "learnable_embedding"
            else f"fixed_z_power_{int(config.species_basis_channels)}"
        )
        moments.append(_moment_spec("moment.l0.species_basis_density", ell=0, radial_projection=base_radial_projection, chemistry_basis=chemistry_basis))
    if config.local_l0_chemistry_rank:
        moments.append(
            _moment_spec(
                "moment.l0.local_lowrank_density",
                ell=0,
                radial_projection=base_radial_projection,
                chemistry_basis=f"trainable_center_neighbor_z_power_rank_{int(config.local_l0_chemistry_rank)}",
            )
        )
    if config.use_vector_moments or config.use_atomic_moments or edge_required_ell >= 1:
        moments.append(_moment_spec("moment.l1.vector", ell=1, radial_projection=radial_projection))
    if config.use_atomic_moments or edge_required_ell >= 2:
        moments.append(_moment_spec("moment.l2.quadrupole", ell=2, radial_projection=radial_projection))

    scalar_paths = [
        _scalar_path_spec(
            "atomic.radial_density",
            placement="atomic",
            inputs=["moment.l0.radial_density"],
            contraction="identity",
            radial_projection="identity",
            cost_group="atomic_scalar_density",
        )
    ]
    if config.use_element_density:
        scalar_paths.append(
            _scalar_path_spec(
                "atomic.element_density",
                placement="atomic",
                inputs=["moment.l0.element_density"],
                contraction="identity",
                radial_projection="identity",
                cost_group="atomic_chemistry_density",
            )
        )
    if config.species_basis_channels:
        scalar_paths.append(
            _scalar_path_spec(
                "atomic.species_basis_density",
                placement="atomic",
                inputs=["moment.l0.species_basis_density"],
                contraction="identity",
                radial_projection="identity",
                cost_group="atomic_low_rank_species_density",
            )
        )
    if config.local_l0_chemistry_rank:
        scalar_paths.append(
            _scalar_path_spec(
                "atomic.local_l0_lowrank_density",
                placement="atomic",
                inputs=["moment.l0.local_lowrank_density"],
                contraction="center_neighbor_lowrank_l0_density",
                radial_projection="identity",
                cost_group="atomic_trainable_local_l0_density",
            )
        )
    if config.use_density_quadratic:
        scalar_paths.append(
            _scalar_path_spec(
                "atomic.density_square",
                placement="atomic",
                inputs=["moment.l0.radial_density", "moment.l0.radial_density"],
                contraction="square",
                radial_projection="identity",
                cost_group="atomic_scalar_polynomial",
            )
        )
    selected_atomic_paths = set(config.scalar_path_ids or ())
    if config.use_vector_moments or config.use_atomic_moments:
        scalar_paths.append(
            _scalar_path_spec(
                "atomic.vector_norm",
                placement="atomic",
                inputs=["moment.l1.vector"],
                contraction="dot_self",
                radial_projection="diagonal_radial_channels",
                cost_group="atomic_low_order_moments",
            )
        )
    if "atomic.vector_cross_radial_dot" in selected_atomic_paths:
        scalar_paths.append(
            _scalar_path_spec(
                "atomic.vector_cross_radial_dot",
                placement="atomic",
                inputs=["moment.l1.vector"],
                contraction="off_diagonal_shell_dot",
                radial_projection=_atomic_cross_radial_projection_label(config),
                cost_group="atomic_cross_radial_invariants",
            )
        )
    if config.use_atomic_moments:
        scalar_paths.append(
            _scalar_path_spec(
                "atomic.quadrupole_norm",
                placement="atomic",
                inputs=["moment.l2.quadrupole"],
                contraction="frobenius_self",
                radial_projection="diagonal_radial_channels",
                cost_group="atomic_low_order_moments",
            )
        )
    if "atomic.quadrupole_cross_radial_frobenius" in selected_atomic_paths:
        scalar_paths.append(
            _scalar_path_spec(
                "atomic.quadrupole_cross_radial_frobenius",
                placement="atomic",
                inputs=["moment.l2.quadrupole"],
                contraction="off_diagonal_shell_frobenius",
                radial_projection=_atomic_cross_radial_projection_label(config),
                cost_group="atomic_cross_radial_invariants",
            )
        )
    selected_edge_paths = _selected_edge_path_ids(config.scalar_path_ids or ())
    selected_shell_edge_paths = [path_id for path_id in selected_edge_paths if path_id in _EDGE_SHELL_SCALAR_PATH_DIMS]
    if config.num_edge_sketches:
        if config.use_cavity_edge_sketches:
            if config.radial_edge_sketch_channels:
                scalar_paths.extend(
                    [
                        _scalar_path_spec(
                            "edge.cavity.vector_same_radial_dot",
                            placement="edge",
                            inputs=["moment.l1.vector", "moment.l1.vector"],
                            contraction="dot",
                            radial_projection="fixed_two_shell_mean",
                            cavity=True,
                            cutoff_power=1,
                            radial_gate="edge_cutoff_envelope",
                            cost_group="edge_cavity_radial_relations",
                        ),
                        _scalar_path_spec(
                            "edge.cavity.vector_cross_radial_dot",
                            placement="edge",
                            inputs=["moment.l1.vector", "moment.l1.vector"],
                            contraction="cross_radial_dot",
                            radial_projection="fixed_two_shell_mean",
                            cavity=True,
                            cutoff_power=1,
                            radial_gate="edge_cutoff_envelope",
                            cost_group="edge_cavity_radial_relations",
                        ),
                        _scalar_path_spec(
                            "edge.cavity.quadrupole_cross_radial_frobenius",
                            placement="edge",
                            inputs=["moment.l2.quadrupole", "moment.l2.quadrupole"],
                            contraction="cross_radial_frobenius",
                            radial_projection="fixed_two_shell_mean",
                            cavity=True,
                            cutoff_power=1,
                            radial_gate="edge_cutoff_envelope",
                            cost_group="edge_cavity_radial_relations",
                        ),
                    ]
                )
            else:
                scalar_paths.extend(
                    [
                        _scalar_path_spec(
                            "edge.cavity.vector_dot",
                            placement="edge",
                            inputs=["moment.l1.vector", "moment.l1.vector"],
                            contraction="dot",
                            radial_projection="full_radial_mean",
                            cavity=True,
                            cutoff_power=1,
                            radial_gate="edge_cutoff_envelope",
                            cost_group="edge_cavity_relations",
                        ),
                        _scalar_path_spec(
                            "edge.cavity.quadrupole_frobenius",
                            placement="edge",
                            inputs=["moment.l2.quadrupole", "moment.l2.quadrupole"],
                            contraction="frobenius",
                            radial_projection="full_radial_mean",
                            cavity=True,
                            cutoff_power=1,
                            radial_gate="edge_cutoff_envelope",
                            cost_group="edge_cavity_relations",
                        ),
                        _scalar_path_spec(
                            "edge.cavity.target_vector_projection",
                            placement="edge",
                            inputs=["edge.unit_vector", "moment.l1.vector"],
                            contraction="u_dot",
                            radial_projection="full_radial_mean",
                            cavity=True,
                            cutoff_power=1,
                            radial_gate="edge_cutoff_envelope",
                            cost_group="edge_cavity_frame_projections",
                        ),
                        _scalar_path_spec(
                            "edge.cavity.source_vector_projection",
                            placement="edge",
                            inputs=["edge.unit_vector", "moment.l1.vector"],
                            contraction="u_dot",
                            radial_projection="full_radial_mean",
                            cavity=True,
                            cutoff_power=1,
                            radial_gate="edge_cutoff_envelope",
                            cost_group="edge_cavity_frame_projections",
                        ),
                        _scalar_path_spec(
                            "edge.cavity.target_quadrupole_projection",
                            placement="edge",
                            inputs=[
                                "edge.unit_vector",
                                "moment.l2.quadrupole",
                                "edge.unit_vector",
                            ],
                            contraction="uQu",
                            radial_projection="full_radial_mean",
                            cavity=True,
                            cutoff_power=1,
                            radial_gate="edge_cutoff_envelope",
                            cost_group="edge_cavity_frame_projections",
                        ),
                        _scalar_path_spec(
                            "edge.cavity.source_quadrupole_projection",
                            placement="edge",
                            inputs=[
                                "edge.unit_vector",
                                "moment.l2.quadrupole",
                                "edge.unit_vector",
                            ],
                            contraction="uQu",
                            radial_projection="full_radial_mean",
                            cavity=True,
                            cutoff_power=1,
                            radial_gate="edge_cutoff_envelope",
                            cost_group="edge_cavity_frame_projections",
                        ),
                    ]
                )
        else:
            scalar_paths.append(
                _scalar_path_spec(
                    "edge.full_moment.vector_dot",
                    placement="edge",
                    inputs=["moment.l1.vector", "moment.l1.vector"],
                    contraction="dot",
                    radial_projection="full_radial_mean",
                    cutoff_power=1,
                    radial_gate="edge_cutoff_envelope",
                    cost_group="edge_full_moment_relations",
                )
            )
            if config.num_edge_sketches > 8 or selected_shell_edge_paths:
                scalar_paths.extend(
                    [
                        _scalar_path_spec(
                            "edge.full_moment.vector_shell_dot",
                            placement="edge",
                            inputs=["moment.l1.vector", "moment.l1.vector"],
                            contraction="same_shell_dot",
                            radial_projection="two_shell_mean",
                            cutoff_power=1,
                            radial_gate="edge_cutoff_envelope",
                            cost_group="edge_full_moment_radial_shell_relations",
                        ),
                        _scalar_path_spec(
                            "edge.full_moment.vector_cross_shell_dot",
                            placement="edge",
                            inputs=["moment.l1.vector", "moment.l1.vector"],
                            contraction="cross_shell_dot",
                            radial_projection="two_shell_mean",
                            cutoff_power=1,
                            radial_gate="edge_cutoff_envelope",
                            cost_group="edge_full_moment_radial_shell_relations",
                        ),
                        _scalar_path_spec(
                            "edge.full_moment.quadrupole_shell_frobenius",
                            placement="edge",
                            inputs=["moment.l2.quadrupole", "moment.l2.quadrupole"],
                            contraction="same_shell_frobenius",
                            radial_projection="two_shell_mean",
                            cutoff_power=1,
                            radial_gate="edge_cutoff_envelope",
                            cost_group="edge_full_moment_radial_shell_relations",
                        ),
                        _scalar_path_spec(
                            "edge.full_moment.quadrupole_cross_shell_frobenius",
                            placement="edge",
                            inputs=["moment.l2.quadrupole", "moment.l2.quadrupole"],
                            contraction="cross_shell_frobenius",
                            radial_projection="two_shell_mean",
                            cutoff_power=1,
                            radial_gate="edge_cutoff_envelope",
                            cost_group="edge_full_moment_radial_shell_relations",
                        ),
                        _scalar_path_spec(
                            "edge.full_moment.vector_shell_contrast_projection",
                            placement="edge",
                            inputs=["moment.l1.vector"],
                            contraction="edge_frame_shell_contrast",
                            radial_projection="two_shell_mean",
                            cutoff_power=1,
                            radial_gate="edge_cutoff_envelope",
                            cost_group="edge_full_moment_radial_shell_relations",
                        ),
                    ]
                )
        scalar_paths.append(
            _scalar_path_spec(
                "edge.direct.radial",
                placement="edge",
                inputs=["moment.l0.radial_density"],
                contraction="direct_edge_radial",
                radial_projection="selected_direct_channels",
                cutoff_power=1,
                radial_gate="edge_cutoff_envelope",
                cost_group="direct_pair_radial",
            )
        )

    if config.scalar_path_ids is not None:
        scalar_paths_by_id = {str(path["id"]): path for path in scalar_paths}
        missing_paths = [path_id for path_id in config.scalar_path_ids if path_id not in scalar_paths_by_id]
        if missing_paths:
            raise ValueError(f"scalar path ids are not available for this config: {missing_paths}")
        scalar_paths = [scalar_paths_by_id[path_id] for path_id in config.scalar_path_ids]

    output_contract = {
        "implemented": list(route["implemented_outputs"]),
        "ase_implemented": list(route["ase_implemented_outputs"]),
        "missing": {
            "virial": "requires validated edge-gradient virial backend",
        },
        "force_realization": route["force_realization"],
        "stress_realization": route["stress_realization"],
        "virial_realization": route["virial_realization"],
    }
    manifest_core: dict[str, Any] = {
        "schema_version": "rtece_path_manifest.v1",
        "config": _config_manifest_payload(config),
        "route": route,
        "moments": moments,
        "scalar_paths": scalar_paths,
        "output_contract": output_contract,
        "retained_tece_groups": route["retained_tece_groups"],
        "deleted_tece_groups": route["deleted_tece_groups"],
        "compiler_status": "explicit_manifest_not_full_compiler",
    }
    hash_core = dict(manifest_core)
    hash_config = dict(hash_core["config"])
    hash_config.pop("variant", None)
    hash_core["config"] = hash_config
    encoded = json.dumps(hash_core, sort_keys=True, separators=(",", ":"), allow_nan=False)
    manifest = dict(manifest_core)
    manifest["manifest_hash"] = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]
    return manifest


def rtece_variant_registry() -> dict[str, dict[str, object]]:
    registry = {}
    for variant in available_rtece_variants():
        config = build_rtece_config(variant)
        manifest = rtece_path_manifest(config)
        route = manifest["route"]
        scalar_paths = list(manifest["scalar_paths"])
        registry[variant] = {
            "variant": variant,
            "config": manifest["config"],
            "semantic_tier": route["semantic_tier"],
            "descriptor_family": route["descriptor_family"],
            "retained_tece_groups": list(manifest["retained_tece_groups"]),
            "deleted_tece_groups": list(manifest["deleted_tece_groups"]),
            "moment_ids": [str(moment["id"]) for moment in manifest["moments"]],
            "scalar_path_ids": [str(path["id"]) for path in scalar_paths],
            "cost_groups": sorted({str(path["cost_group"]) for path in scalar_paths}),
            "manifest_hash": manifest["manifest_hash"],
        }
    return registry


@dataclass
class RTECEGraph:
    z: torch.Tensor
    pos: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor
    cell: torch.Tensor | None = None
    edge_shifts: torch.Tensor | None = None
    edge_batch: torch.Tensor | None = None


def collate_graphs(graphs: list[RTECEGraph]) -> RTECEGraph:
    if not graphs:
        raise ValueError("collate_graphs requires at least one graph")
    z_parts = []
    pos_parts = []
    edge_parts = []
    batch_parts = []
    cell_parts = []
    edge_shift_parts = []
    edge_batch_parts = []
    has_cell = any(graph.cell is not None for graph in graphs)
    has_edge_shifts = any(graph.edge_shifts is not None for graph in graphs)
    node_offset = 0
    for graph_idx, graph in enumerate(graphs):
        num_nodes = graph.z.shape[0]
        z_parts.append(graph.z)
        pos_parts.append(graph.pos)
        batch_parts.append(torch.full_like(graph.batch, graph_idx))
        if has_cell:
            if graph.cell is None:
                cell_parts.append(graph.pos.new_zeros((1, 3, 3)))
            else:
                cell_parts.append(graph.cell.to(device=graph.pos.device, dtype=graph.pos.dtype).reshape(-1, 3, 3)[:1])
        if graph.edge_index.numel() > 0:
            edge_parts.append(graph.edge_index + node_offset)
            num_edges = graph.edge_index.shape[1]
            if has_edge_shifts:
                if graph.edge_shifts is None:
                    edge_shift_parts.append(graph.edge_index.new_zeros((num_edges, 3)))
                else:
                    edge_shift_parts.append(graph.edge_shifts.to(device=graph.edge_index.device, dtype=torch.long))
            if has_cell or has_edge_shifts:
                edge_batch_parts.append(torch.full((num_edges,), graph_idx, dtype=torch.long, device=graph.edge_index.device))
        node_offset += num_nodes
    if edge_parts:
        edge_index = torch.cat(edge_parts, dim=1)
    else:
        edge_index = graphs[0].edge_index.new_zeros((2, 0))
    cell = torch.cat(cell_parts, dim=0) if cell_parts else None
    edge_shifts = torch.cat(edge_shift_parts, dim=0) if edge_shift_parts else None
    edge_batch = torch.cat(edge_batch_parts, dim=0) if edge_batch_parts else None
    return RTECEGraph(
        z=torch.cat(z_parts, dim=0),
        pos=torch.cat(pos_parts, dim=0),
        edge_index=edge_index,
        batch=torch.cat(batch_parts, dim=0),
        cell=cell,
        edge_shifts=edge_shifts,
        edge_batch=edge_batch,
    )



def atomic_reference_energy(graph: RTECEGraph, config: RTECEScalarConfig) -> torch.Tensor:
    num_graphs = int(graph.batch.max().item()) + 1 if graph.batch.numel() else 1
    if config.atomic_energies:
        values = graph.pos.new_zeros(graph.z.shape[0])
        for z, energy in config.atomic_energies.items():
            values = torch.where(
                graph.z == int(z),
                graph.pos.new_tensor(float(energy)),
                values,
            )
        return scatter_sum(values[:, None], graph.batch, num_graphs).squeeze(-1)
    if config.energy_per_atom_shift:
        atom_counts = scatter_sum(
            torch.ones((graph.z.shape[0], 1), dtype=graph.pos.dtype, device=graph.pos.device),
            graph.batch,
            num_graphs,
        ).squeeze(-1)
        return atom_counts * graph.pos.new_tensor(float(config.energy_per_atom_shift))
    return graph.pos.new_zeros(num_graphs)


def add_atomic_reference_energy(
    energy: torch.Tensor,
    graph: RTECEGraph,
    config: RTECEScalarConfig,
) -> torch.Tensor:
    return energy + atomic_reference_energy(graph, config)


def _covalent_radii_for_z(z: torch.Tensor, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    from ase.data import covalent_radii

    table = torch.as_tensor(covalent_radii, dtype=dtype, device=device)
    z_index = z.to(device=device, dtype=torch.long).clamp(min=0, max=table.numel() - 1)
    radii = table[z_index]
    return torch.where(radii > 0.0, radii, radii.new_full(radii.shape, 0.5))


def _zbl_short_range_energy(graph: RTECEGraph, num_graphs: int) -> torch.Tensor:
    src, dst = graph.edge_index
    _, distances, _unit = compute_pair_geometry(graph)
    x = distances[:, None]
    z = graph.z.to(device=graph.pos.device, dtype=graph.pos.dtype)
    z_src = z[src, None]
    z_dst = z[dst, None]
    c = graph.pos.new_tensor([0.1818, 0.5099, 0.2802, 0.02817])
    a_exp = graph.pos.new_tensor(0.300)
    a_prefactor = graph.pos.new_tensor(0.4543)
    a = a_prefactor * graph.pos.new_tensor(0.529) / (z_src.pow(a_exp) + z_dst.pow(a_exp))
    r_over_a = x / a
    phi = (
        c[0] * torch.exp(-3.2 * r_over_a)
        + c[1] * torch.exp(-0.9423 * r_over_a)
        + c[2] * torch.exp(-0.4028 * r_over_a)
        + c[3] * torch.exp(-0.2016 * r_over_a)
    )
    v_edges = (graph.pos.new_tensor(14.3996) * z_src * z_dst) / x * phi
    radii = _covalent_radii_for_z(graph.z, dtype=graph.pos.dtype, device=graph.pos.device)
    r_max = (radii[src] + radii[dst])[:, None]
    p = graph.pos.new_tensor(5.0)
    y = x / r_max
    envelope = (
        1.0
        - ((p + 1.0) * (p + 2.0) / 2.0) * y.pow(p)
        + p * (p + 2.0) * y.pow(p + 1.0)
        - (p * (p + 1.0) / 2.0) * y.pow(p + 2.0)
    )
    envelope = envelope * (x < r_max)
    node_zbl = scatter_sum(0.5 * v_edges * envelope, dst, graph.z.shape[0]).squeeze(-1)
    return scatter_sum(node_zbl[:, None], graph.batch, num_graphs).squeeze(-1)


def short_range_repulsive_energy(graph: RTECEGraph, config: RTECEScalarConfig) -> torch.Tensor:
    num_graphs = int(graph.batch.max().item()) + 1 if graph.batch.numel() else 1
    if not config.use_short_range_repulsion or graph.edge_index.numel() == 0:
        return graph.pos.new_zeros(num_graphs)
    potential = str(config.short_range_repulsion_potential)
    if potential == "zbl":
        return _zbl_short_range_energy(graph, num_graphs)
    if potential != "softplus_overlap":
        raise ValueError(f"unknown short_range_repulsion_potential {potential!r}")
    if float(config.short_range_repulsion_strength) == 0.0:
        return graph.pos.new_zeros(num_graphs)
    if float(config.short_range_repulsion_beta) <= 0.0:
        raise ValueError("short_range_repulsion_beta must be positive")
    if float(config.short_range_repulsion_radius_scale) <= 0.0:
        raise ValueError("short_range_repulsion_radius_scale must be positive")

    src, dst = graph.edge_index
    _, distances, _unit = compute_pair_geometry(graph)
    radii = _covalent_radii_for_z(graph.z, dtype=graph.pos.dtype, device=graph.pos.device)
    radius_threshold = float(config.short_range_repulsion_radius_scale) * (radii[src] + radii[dst])
    beta = graph.pos.new_tensor(float(config.short_range_repulsion_beta))
    overlap = torch.nn.functional.softplus(beta * (radius_threshold - distances)) / beta
    edge_energy = 0.5 * graph.pos.new_tensor(float(config.short_range_repulsion_strength)) * overlap.square()
    edge_batch = graph.edge_batch if graph.edge_batch is not None else graph.batch[dst]
    return 0.5 * scatter_sum(edge_energy[:, None], edge_batch, num_graphs).squeeze(-1)


def add_short_range_repulsive_energy(
    energy: torch.Tensor,
    graph: RTECEGraph,
    config: RTECEScalarConfig,
) -> torch.Tensor:
    return energy + short_range_repulsive_energy(graph, config)


def compute_pair_geometry(graph: RTECEGraph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src, dst = graph.edge_index
    vectors = graph.pos[dst] - graph.pos[src]
    if graph.cell is not None and graph.edge_shifts is not None and graph.edge_shifts.numel() > 0:
        edge_batch = graph.edge_batch
        if edge_batch is None:
            edge_batch = graph.batch[src]
        cell = graph.cell.to(device=graph.pos.device, dtype=graph.pos.dtype)
        shifts = graph.edge_shifts.to(device=graph.pos.device, dtype=graph.pos.dtype)
        vectors = vectors + torch.einsum("ei,eij->ej", shifts, cell[edge_batch])
    distances = vectors.norm(dim=-1).clamp_min(1e-12)
    unit = vectors / distances[:, None]
    return vectors, distances, unit


def cutoff_envelope(distances: torch.Tensor, cutoff: float) -> torch.Tensor:
    x = (distances / cutoff).clamp(min=0.0, max=1.0)
    envelope = 1.0 - 10.0 * x**3 + 15.0 * x**4 - 6.0 * x**5
    return torch.where(distances < cutoff, envelope, torch.zeros_like(envelope))


def _apply_radial_mixing(
    radial: torch.Tensor,
    radial_mixing: torch.nn.Linear | None = None,
) -> torch.Tensor:
    if radial_mixing is None:
        return radial
    return radial_mixing(radial)


def radial_features_and_derivatives(
    distances: torch.Tensor,
    config: RTECEScalarConfig,
    radial_mixing: torch.nn.Linear | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    centers = torch.linspace(
        0.0,
        config.cutoff,
        config.num_radial,
        device=distances.device,
        dtype=distances.dtype,
    )
    width = config.cutoff / max(config.num_radial - 1, 1)
    delta = distances[:, None] - centers[None, :]
    gaussian = torch.exp(-0.5 * (delta / width) ** 2)
    envelope = cutoff_envelope(distances, config.cutoff)
    gaussian_derivative = gaussian * (-delta / (width**2))
    x = (distances / config.cutoff).clamp(min=0.0, max=1.0)
    envelope_derivative = (-30.0 * x**2 + 60.0 * x**3 - 30.0 * x**4) / config.cutoff
    envelope_derivative = torch.where(
        distances < config.cutoff,
        envelope_derivative,
        torch.zeros_like(envelope_derivative),
    )
    features = gaussian * envelope[:, None]
    derivatives = gaussian_derivative * envelope[:, None] + gaussian * envelope_derivative[:, None]
    if radial_mixing is not None:
        features = _apply_radial_mixing(features, radial_mixing)
        derivatives = derivatives.matmul(radial_mixing.weight.t())
    return features, derivatives


def compute_radial_features_only(
    distances: torch.Tensor,
    config: RTECEScalarConfig,
    radial_mixing: torch.nn.Linear | None = None,
) -> torch.Tensor:
    centers = torch.linspace(
        0.0,
        config.cutoff,
        config.num_radial,
        device=distances.device,
        dtype=distances.dtype,
    )
    width = config.cutoff / max(config.num_radial - 1, 1)
    delta = distances[:, None] - centers[None, :]
    gaussian = torch.exp(-0.5 * (delta / width) ** 2)
    envelope = cutoff_envelope(distances, config.cutoff)
    return _apply_radial_mixing(gaussian * envelope[:, None], radial_mixing)


def compute_radial_features(
    distances: torch.Tensor,
    config: RTECEScalarConfig,
    radial_mixing: torch.nn.Linear | None = None,
) -> torch.Tensor:
    return compute_radial_features_only(distances, config, radial_mixing)


def scatter_sum(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = values.new_zeros((dim_size, *values.shape[1:]))
    out.index_add_(0, index, values)
    return out


def compute_atomic_moments(
    graph: RTECEGraph,
    config: RTECEScalarConfig,
    radial_mixing: torch.nn.Linear | None = None,
    max_ell: int | None = None,
    species_basis_embedding: torch.Tensor | None = None,
    radial_species_adapter: torch.nn.Module | None = None,
    local_l0_chemistry_front: torch.nn.Module | None = None,
) -> dict[str, torch.Tensor | None]:
    _, distances, unit = compute_pair_geometry(graph)
    src, dst = graph.edge_index
    radial = compute_radial_features(distances, config, radial_mixing)
    if radial_species_adapter is not None:
        radial = radial_species_adapter(radial, graph.z[src], graph.z[dst])
    num_nodes = graph.z.shape[0]
    density = scatter_sum(radial, dst, num_nodes)
    neighbor_z = graph.z[src].to(dtype=graph.pos.dtype, device=graph.pos.device) / float(config.max_atomic_number)
    element_density = scatter_sum(radial * neighbor_z[:, None], dst, num_nodes)
    if max_ell is None or int(max_ell) >= 1:
        vector = scatter_sum(radial[:, :, None] * unit[:, None, :], dst, num_nodes)
    else:
        vector = None
    if max_ell is None or int(max_ell) >= 2:
        eye = torch.eye(3, device=graph.pos.device, dtype=graph.pos.dtype)
        quad_unit = unit[:, :, None] * unit[:, None, :] - eye[None, :, :] / 3.0
        quadrupole = scatter_sum(radial[:, :, None, None] * quad_unit[:, None, :, :], dst, num_nodes)
    else:
        quadrupole = None
    local_l0_lowrank_density = None
    if config.local_l0_chemistry_rank:
        if local_l0_chemistry_front is None:
            raise ValueError("local L0 chemistry front is required for local_l0_chemistry_rank > 0")
        local_l0_lowrank_density = local_l0_chemistry_front(
            radial,
            source_z=graph.z[src],
            target_z=graph.z[dst],
            dst=dst,
            num_nodes=num_nodes,
        )
    species_density = None
    if config.species_basis_channels:
        if config.species_basis_mode == "learnable_embedding":
            if species_basis_embedding is None:
                raise ValueError("learnable species basis requires species_basis_embedding")
            embedding = species_basis_embedding.to(device=graph.pos.device, dtype=graph.pos.dtype)
            species_index = graph.z[src].to(device=graph.pos.device, dtype=torch.long)
            if species_index.numel() and int(species_index.max().item()) >= embedding.shape[0]:
                raise ValueError("atomic number exceeds learnable species basis embedding table")
            species_basis = embedding[species_index]
        else:
            powers = torch.arange(
                1,
                int(config.species_basis_channels) + 1,
                device=graph.pos.device,
                dtype=graph.pos.dtype,
            )
            species_basis = neighbor_z[:, None].pow(powers[None, :])
        species_density = scatter_sum(radial[:, :, None] * species_basis[:, None, :], dst, num_nodes)
        species_density = species_density.reshape(num_nodes, config.num_radial * int(config.species_basis_channels))
    return {
        "density": density,
        "element_density": element_density,
        "species_density": species_density,
        "local_l0_lowrank_density": local_l0_lowrank_density,
        "vector": vector,
        "quadrupole": quadrupole,
    }


def _atomic_scalar_path_descriptors(
    path_ids: tuple[str, ...],
    *,
    density: torch.Tensor,
    element_density: torch.Tensor | None = None,
    species_density: torch.Tensor | None = None,
    local_l0_lowrank_density: torch.Tensor | None = None,
    vector_norm: torch.Tensor | None = None,
    quadrupole_norm: torch.Tensor | None = None,
    vector_cross_radial_dot: torch.Tensor | None = None,
    quadrupole_cross_radial_frobenius: torch.Tensor | None = None,
) -> torch.Tensor:
    parts = []
    for path_id in path_ids:
        if path_id == "atomic.radial_density":
            parts.append(density)
        elif path_id == "atomic.element_density":
            if element_density is None:
                raise ValueError("element_density is required for atomic.element_density")
            parts.append(element_density)
        elif path_id == "atomic.species_basis_density":
            if species_density is None:
                raise ValueError("species_density is required for atomic.species_basis_density")
            parts.append(species_density)
        elif path_id == "atomic.local_l0_lowrank_density":
            if local_l0_lowrank_density is None:
                raise ValueError("local_l0_lowrank_density is required for atomic.local_l0_lowrank_density")
            parts.append(local_l0_lowrank_density)
        elif path_id == "atomic.density_square":
            parts.append(density.square())
        elif path_id == "atomic.vector_norm":
            if vector_norm is None:
                raise ValueError("vector_norm is required for atomic.vector_norm")
            parts.append(vector_norm)
        elif path_id == "atomic.vector_cross_radial_dot":
            if vector_cross_radial_dot is None:
                raise ValueError("vector_cross_radial_dot is required for atomic.vector_cross_radial_dot")
            parts.append(vector_cross_radial_dot)
        elif path_id == "atomic.quadrupole_norm":
            if quadrupole_norm is None:
                raise ValueError("quadrupole_norm is required for atomic.quadrupole_norm")
            parts.append(quadrupole_norm)
        elif path_id == "atomic.quadrupole_cross_radial_frobenius":
            if quadrupole_cross_radial_frobenius is None:
                raise ValueError(
                    "quadrupole_cross_radial_frobenius is required for atomic.quadrupole_cross_radial_frobenius"
                )
            parts.append(quadrupole_cross_radial_frobenius)
        else:
            raise ValueError(f"unsupported atomic scalar path id {path_id!r}")
    if not parts:
        return density.new_zeros((density.shape[0], 0))
    return torch.cat(parts, dim=-1) if len(parts) > 1 else parts[0]


def density_scalar_descriptors(
    density: torch.Tensor,
    config: RTECEScalarConfig,
    element_density: torch.Tensor | None = None,
    vector_norm: torch.Tensor | None = None,
) -> torch.Tensor:
    if config.scalar_path_ids is not None:
        atomic_path_ids = _selected_atomic_path_ids(config.scalar_path_ids)
        return _atomic_scalar_path_descriptors(
            atomic_path_ids,
            density=density,
            element_density=element_density if config.use_element_density else None,
            species_density=element_density if config.species_basis_channels else None,
            vector_norm=vector_norm,
        )

    parts = [density]
    if config.use_element_density:
        if element_density is None:
            raise ValueError("element_density is required when use_element_density=True")
        parts.append(element_density)
    if config.use_density_quadratic:
        parts.append(density.square())
    if config.species_basis_channels:
        if element_density is None:
            raise ValueError("species_density is required when species_basis_channels > 0")
        parts.append(element_density)
    if config.local_l0_chemistry_rank:
        raise ValueError("local L0 chemistry front requires scalar_path_ids")
    if config.use_vector_moments:
        if vector_norm is None:
            raise ValueError("vector_norm is required when use_vector_moments=True")
        parts.append(vector_norm)
    return torch.cat(parts, dim=-1) if len(parts) > 1 else density


def _validate_packed_element_density_config(config: RTECEScalarConfig, name: str) -> None:
    if not config.use_element_density:
        raise ValueError(f"{name} requires use_element_density=True")
    if config.use_density_quadratic or config.use_vector_moments or config.species_basis_channels:
        raise ValueError(f"{name} only supports density plus element density")
    if config.use_atomic_moments or config.num_edge_sketches:
        raise ValueError(f"{name} only supports scalar density descriptors")
    if config.scalar_path_ids is not None and config.scalar_path_ids != (
        "atomic.radial_density",
        "atomic.element_density",
    ):
        raise ValueError(f"{name} requires canonical radial/element path order")


def _cell_list_directed_edges_nopbc(graph: RTECEGraph, cutoff: float) -> tuple[torch.Tensor, torch.Tensor]:
    if graph.batch.ndim != 1 or graph.batch.shape[0] != graph.z.shape[0]:
        raise ValueError("cell-list descriptors require one batch id per atom")
    if graph.batch.numel() > 1 and torch.any(graph.batch[1:] < graph.batch[:-1]):
        raise ValueError("cell-list descriptors require atoms sorted by batch")

    num_nodes = graph.z.shape[0]
    if num_nodes == 0:
        empty = torch.zeros(0, dtype=torch.long, device=graph.pos.device)
        return empty, empty

    _, counts = torch.unique_consecutive(graph.batch, return_counts=True)
    starts = torch.cat([counts.new_zeros(1), counts.cumsum(dim=0)[:-1]])
    cutoff_sq = float(cutoff) * float(cutoff)
    src_parts = []
    dst_parts = []

    for start_tensor, count_tensor in zip(starts, counts):
        start = int(start_tensor.item())
        count = int(count_tensor.item())
        if count <= 1:
            continue

        local_pos = graph.pos[start : start + count]
        origin = local_pos.min(dim=0).values
        cell_coords = torch.floor((local_pos - origin) / float(cutoff)).to(dtype=torch.long)
        cell_delta = torch.abs(cell_coords[:, None, :] - cell_coords[None, :, :])
        candidate_mask = torch.all(cell_delta <= 1, dim=-1)
        candidate_mask.fill_diagonal_(False)

        deltas = local_pos[:, None, :] - local_pos[None, :, :]
        distance_sq = deltas.square().sum(dim=-1)
        active_mask = candidate_mask & (distance_sq < cutoff_sq)
        src_local, dst_local = torch.nonzero(active_mask, as_tuple=True)
        if src_local.numel() == 0:
            continue
        src_parts.append(src_local + start)
        dst_parts.append(dst_local + start)

    if not src_parts:
        empty = torch.zeros(0, dtype=torch.long, device=graph.pos.device)
        return empty, empty
    return torch.cat(src_parts), torch.cat(dst_parts)


def packed_element_density_descriptors(
    graph: RTECEGraph,
    config: RTECEScalarConfig,
    radial_mixing: torch.nn.Linear | None = None,
) -> torch.Tensor:
    _validate_packed_element_density_config(config, "packed_element_density_descriptors")
    _, distances, _ = compute_pair_geometry(graph)
    radial = compute_radial_features(distances, config, radial_mixing)
    src, dst = graph.edge_index
    neighbor_z = graph.z[src].to(dtype=graph.pos.dtype, device=graph.pos.device) / float(config.max_atomic_number)
    edge_descriptors = torch.cat([radial, radial * neighbor_z[:, None]], dim=-1)
    return scatter_sum(edge_descriptors, dst, graph.z.shape[0])


def cell_list_packed_element_density_descriptors(
    graph: RTECEGraph,
    config: RTECEScalarConfig,
    radial_mixing: torch.nn.Linear | None = None,
) -> torch.Tensor:
    _validate_packed_element_density_config(config, "cell_list_packed_element_density_descriptors")
    num_nodes = graph.z.shape[0]
    descriptors = graph.pos.new_zeros((num_nodes, 2 * config.num_radial))
    src, dst = _cell_list_directed_edges_nopbc(graph, float(config.cutoff))
    if src.numel() == 0:
        return descriptors

    distances = (graph.pos[dst] - graph.pos[src]).norm(dim=-1).clamp_min(1e-12)
    radial = compute_radial_features(distances, config, radial_mixing)
    neighbor_z = graph.z[src].to(dtype=graph.pos.dtype, device=graph.pos.device) / float(config.max_atomic_number)
    edge_descriptors = torch.cat([radial, radial * neighbor_z[:, None]], dim=-1)
    descriptors.index_add_(0, dst, edge_descriptors)
    return descriptors


def atomic_scalar_descriptors(
    graph: RTECEGraph,
    config: RTECEScalarConfig,
    radial_mixing: torch.nn.Linear | None = None,
    atomic_cross_radial_projection: torch.Tensor | None = None,
    species_basis_embedding: torch.Tensor | None = None,
    radial_species_adapter: torch.nn.Module | None = None,
    local_l0_chemistry_front: torch.nn.Module | None = None,
) -> torch.Tensor:
    moments = compute_atomic_moments(
        graph,
        config,
        radial_mixing,
        species_basis_embedding=species_basis_embedding,
        radial_species_adapter=radial_species_adapter,
        local_l0_chemistry_front=local_l0_chemistry_front,
    )
    density = moments["density"]
    vector = moments["vector"]
    quadrupole = moments["quadrupole"]
    vector_norm = (vector ** 2).sum(dim=-1) if vector is not None else None
    quadrupole_norm = (quadrupole ** 2).sum(dim=(-1, -2)) if quadrupole is not None else None
    atomic_path_ids = _selected_atomic_path_ids(config.scalar_path_ids or ())
    vector_cross_radial_dot = None
    if "atomic.vector_cross_radial_dot" in atomic_path_ids and vector is not None:
        sketch_channels = _cross_radial_sketch_channels(config)
        vector_shells = _project_radial_edge_channels(vector, sketch_channels, atomic_cross_radial_projection)
        vector_cross_radial_dot = _off_diagonal_shell_inner_products(vector_shells)
    quadrupole_cross_radial_frobenius = None
    if "atomic.quadrupole_cross_radial_frobenius" in atomic_path_ids and quadrupole is not None:
        sketch_channels = _cross_radial_sketch_channels(config)
        quadrupole_shells = _project_radial_edge_channels(quadrupole, sketch_channels, atomic_cross_radial_projection)
        quadrupole_cross_radial_frobenius = _off_diagonal_shell_inner_products(quadrupole_shells)
    if config.scalar_path_ids is not None:
        return _atomic_scalar_path_descriptors(
            atomic_path_ids,
            density=density,
            element_density=moments["element_density"],
            species_density=moments["species_density"],
            local_l0_lowrank_density=moments["local_l0_lowrank_density"],
            vector_norm=vector_norm,
            quadrupole_norm=quadrupole_norm,
            vector_cross_radial_dot=vector_cross_radial_dot,
            quadrupole_cross_radial_frobenius=quadrupole_cross_radial_frobenius,
        )

    density_desc = density_scalar_descriptors(
        density,
        config,
        moments["species_density"] if config.species_basis_channels else moments["element_density"],
        vector_norm,
    )
    if not config.use_atomic_moments:
        return density_desc
    return torch.cat([density_desc, vector_norm, quadrupole_norm], dim=-1)



def _fixed_radial_shell_projection_matrix(
    *,
    num_radial: int,
    num_sketches: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if int(num_sketches) <= 0:
        raise ValueError("num_sketches must be positive for an explicit shell projection matrix")
    if int(num_sketches) > int(num_radial):
        raise ValueError("radial edge sketch channels cannot exceed num_radial")
    projection = torch.zeros((int(num_sketches), int(num_radial)), dtype=dtype, device=device)
    radial_indices = torch.arange(int(num_radial), device=device)
    for shell_index, chunk in enumerate(torch.tensor_split(radial_indices, int(num_sketches))):
        projection[shell_index, chunk] = 1.0 / float(chunk.numel())
    return projection


def _project_radial_edge_channels(
    channels: torch.Tensor,
    num_sketches: int,
    projection: torch.Tensor | None = None,
) -> torch.Tensor:
    if num_sketches <= 0:
        return channels.mean(dim=1)
    if channels.ndim < 2:
        raise ValueError("radial edge channels must include a radial dimension")
    num_radial = int(channels.shape[1])
    if num_sketches > num_radial:
        raise ValueError("radial edge sketch channels cannot exceed num_radial")
    if projection is not None:
        if projection.ndim != 2:
            raise ValueError("radial projection must be a rank-2 matrix")
        expected_shape = (int(num_sketches), num_radial)
        if tuple(projection.shape) != expected_shape:
            raise ValueError(f"radial projection must have shape {expected_shape}, got {tuple(projection.shape)}")
        projection = projection.to(device=channels.device, dtype=channels.dtype)
        return torch.einsum("kr,nr...->nk...", projection, channels)
    projection = _fixed_radial_shell_projection_matrix(
        num_radial=num_radial,
        num_sketches=int(num_sketches),
        dtype=channels.dtype,
        device=channels.device,
    )
    return torch.einsum("kr,nr...->nk...", projection, channels)


def _cross_radial_sketch_channels(config: RTECEScalarConfig) -> int:
    channels = int(config.atomic_cross_radial_sketch_channels)
    if channels < 2:
        raise ValueError("atomic_cross_radial_sketch_channels must be at least 2")
    if channels > int(config.num_radial):
        raise ValueError("atomic_cross_radial_sketch_channels cannot exceed num_radial")
    return channels


def _off_diagonal_shell_inner_products(shells: torch.Tensor) -> torch.Tensor:
    if shells.ndim < 3:
        raise ValueError("cross-radial shell contractions require batch, shell, and feature dimensions")
    num_shells = int(shells.shape[1])
    _num_off_diagonal_shell_pairs(num_shells)
    flat = shells.reshape(shells.shape[0], num_shells, -1)
    values = [(flat[:, i] * flat[:, j]).sum(dim=-1) for i in range(num_shells) for j in range(i + 1, num_shells)]
    return torch.stack(values, dim=-1)


def _radial_edge_relational_base(
    vi: torch.Tensor,
    vj: torch.Tensor,
    qi: torch.Tensor,
    qj: torch.Tensor,
    unit: torch.Tensor,
    radial: torch.Tensor,
) -> torch.Tensor:
    num_sketches = int(vi.shape[1])
    pairs = [(i, i) for i in range(num_sketches)]
    pairs.extend((i, j) for i in range(num_sketches) for j in range(num_sketches) if i != j)
    terms = []
    terms.extend((vi[:, i] * vj[:, j]).sum(dim=-1) for i, j in pairs)
    terms.extend((qi[:, i] * qj[:, j]).sum(dim=(-1, -2)) for i, j in pairs)
    terms.extend((unit * vi[:, i]).sum(dim=-1) for i in range(num_sketches))
    terms.extend((unit * vj[:, i]).sum(dim=-1) for i in range(num_sketches))
    terms.append(radial[:, 0])
    terms.append(radial[:, min(1, radial.shape[1] - 1)])
    return torch.stack(terms, dim=-1)


def _full_moment_shell_edge_path_values(
    vector_channels_i: torch.Tensor,
    vector_channels_j: torch.Tensor,
    quadrupole_channels_i: torch.Tensor,
    quadrupole_channels_j: torch.Tensor,
    unit: torch.Tensor,
) -> dict[str, torch.Tensor]:
    vi_shell = _project_radial_edge_channels(vector_channels_i, 2)
    vj_shell = _project_radial_edge_channels(vector_channels_j, 2)
    qi_shell = _project_radial_edge_channels(quadrupole_channels_i, 2)
    qj_shell = _project_radial_edge_channels(quadrupole_channels_j, 2)
    shell_vector_dot_low = (vi_shell[:, 0] * vj_shell[:, 0]).sum(dim=-1)
    shell_vector_dot_high = (vi_shell[:, 1] * vj_shell[:, 1]).sum(dim=-1)
    shell_vector_dot_cross = 0.5 * (
        (vi_shell[:, 0] * vj_shell[:, 1]).sum(dim=-1)
        + (vi_shell[:, 1] * vj_shell[:, 0]).sum(dim=-1)
    )
    shell_quadrupole_frobenius_low = (qi_shell[:, 0] * qj_shell[:, 0]).sum(dim=(-1, -2))
    shell_quadrupole_frobenius_high = (qi_shell[:, 1] * qj_shell[:, 1]).sum(dim=(-1, -2))
    shell_quadrupole_frobenius_cross = 0.5 * (
        (qi_shell[:, 0] * qj_shell[:, 1]).sum(dim=(-1, -2))
        + (qi_shell[:, 1] * qj_shell[:, 0]).sum(dim=(-1, -2))
    )
    target_vector_shell_contrast = (unit * (vi_shell[:, 0] - vi_shell[:, 1])).sum(dim=-1)
    source_vector_shell_contrast = (unit * (vj_shell[:, 0] - vj_shell[:, 1])).sum(dim=-1)
    return {
        "edge.full_moment.vector_shell_dot": torch.stack(
            [shell_vector_dot_low, shell_vector_dot_high],
            dim=-1,
        ),
        "edge.full_moment.vector_cross_shell_dot": shell_vector_dot_cross[:, None],
        "edge.full_moment.quadrupole_shell_frobenius": torch.stack(
            [shell_quadrupole_frobenius_low, shell_quadrupole_frobenius_high],
            dim=-1,
        ),
        "edge.full_moment.quadrupole_cross_shell_frobenius": shell_quadrupole_frobenius_cross[:, None],
        "edge.full_moment.vector_shell_contrast_projection": torch.stack(
            [target_vector_shell_contrast, source_vector_shell_contrast],
            dim=-1,
        ),
    }


def edge_relational_sketches(
    graph: RTECEGraph,
    config: RTECEScalarConfig,
    radial_mixing: torch.nn.Linear | None = None,
    species_basis_embedding: torch.Tensor | None = None,
    radial_species_adapter: torch.nn.Module | None = None,
    local_l0_chemistry_front: torch.nn.Module | None = None,
) -> torch.Tensor:
    if config.num_edge_sketches <= 0:
        return graph.pos.new_zeros((graph.z.shape[0], 0))

    _, distances, unit = compute_pair_geometry(graph)
    radial = compute_radial_features(distances, config, radial_mixing)
    if radial_species_adapter is not None:
        src, dst = graph.edge_index
        radial = radial_species_adapter(radial, graph.z[src], graph.z[dst])
    edge_path_ids = _selected_edge_path_ids(config.scalar_path_ids or ())
    required_ell = _edge_paths_required_ell(config)
    moments = compute_atomic_moments(
        graph,
        config,
        radial_mixing,
        max_ell=required_ell,
        species_basis_embedding=species_basis_embedding,
        radial_species_adapter=radial_species_adapter,
        local_l0_chemistry_front=local_l0_chemistry_front,
    )
    src, dst = graph.edge_index
    vector_channels = moments["vector"]
    quadrupole_channels = moments["quadrupole"]
    vector_channels_i = vector_channels[dst] if vector_channels is not None else None
    vector_channels_j = vector_channels[src] if vector_channels is not None else None
    quadrupole_channels_i = quadrupole_channels[dst] if quadrupole_channels is not None else None
    quadrupole_channels_j = quadrupole_channels[src] if quadrupole_channels is not None else None
    if config.use_cavity_edge_sketches:
        quad_unit = unit[:, :, None] * unit[:, None, :] - torch.eye(
            3,
            device=graph.pos.device,
            dtype=graph.pos.dtype,
        )[None, :, :] / 3.0
        edge_vector = radial[:, :, None] * unit[:, None, :] if required_ell >= 1 else None
        edge_quadrupole = radial[:, :, None, None] * quad_unit[:, None, :, :] if required_ell >= 2 else None
        num_nodes = graph.z.shape[0]
        edge_codes = src * num_nodes + dst
        reverse_codes = dst * num_nodes + src
        has_reverse = torch.isin(reverse_codes, edge_codes).to(dtype=graph.pos.dtype, device=graph.pos.device)
        if edge_vector is not None:
            if vector_channels_i is None or vector_channels_j is None:
                raise RuntimeError("vector cavity edge paths require ell=1 moments")
            vector_channels_i = vector_channels_i - edge_vector
            vector_channels_j = vector_channels_j + has_reverse[:, None, None] * edge_vector
        if edge_quadrupole is not None:
            if quadrupole_channels_i is None or quadrupole_channels_j is None:
                raise RuntimeError("quadrupole cavity edge paths require ell=2 moments")
            quadrupole_channels_i = quadrupole_channels_i - edge_quadrupole
            quadrupole_channels_j = quadrupole_channels_j - has_reverse[:, None, None, None] * edge_quadrupole
    if required_ell >= 1 and (vector_channels_i is None or vector_channels_j is None):
        raise RuntimeError("edge vector paths require ell=1 moments")
    if required_ell >= 2 and (quadrupole_channels_i is None or quadrupole_channels_j is None):
        raise RuntimeError("edge quadrupole paths require ell=2 moments")
    if config.radial_edge_sketch_channels:
        if edge_path_ids:
            raise ValueError("selected radial edge path ids are not implemented yet")
        k_radial = int(config.radial_edge_sketch_channels)
        vi = _project_radial_edge_channels(vector_channels_i, k_radial)
        vj = _project_radial_edge_channels(vector_channels_j, k_radial)
        qi = _project_radial_edge_channels(quadrupole_channels_i, k_radial)
        qj = _project_radial_edge_channels(quadrupole_channels_j, k_radial)
        base = _radial_edge_relational_base(vi, vj, qi, qj, unit, radial)
    else:
        direct_radial = torch.stack([radial[:, 0], radial[:, min(1, radial.shape[1] - 1)]], dim=-1)
        needs_shell_paths = any(path_id in _EDGE_SHELL_SCALAR_PATH_DIMS for path_id in edge_path_ids)
        path_values = None
        if edge_path_ids:
            path_values = {"edge.direct.radial": direct_radial}
            if required_ell >= 1:
                if vector_channels_i is None or vector_channels_j is None:
                    raise RuntimeError("edge vector paths require ell=1 moments")
                vi = vector_channels_i.mean(dim=1)
                vj = vector_channels_j.mean(dim=1)
                vector_dot = (vi * vj).sum(dim=-1)
                vector_path = "edge.cavity.vector_dot" if config.use_cavity_edge_sketches else "edge.full_moment.vector_dot"
                path_values[vector_path] = vector_dot[:, None]
                if config.use_cavity_edge_sketches:
                    path_values["edge.cavity.target_vector_projection"] = (unit * vi).sum(dim=-1, keepdim=True)
                    path_values["edge.cavity.source_vector_projection"] = (unit * vj).sum(dim=-1, keepdim=True)
            if required_ell >= 2:
                if quadrupole_channels_i is None or quadrupole_channels_j is None:
                    raise RuntimeError("edge quadrupole paths require ell=2 moments")
                qi = quadrupole_channels_i.mean(dim=1)
                qj = quadrupole_channels_j.mean(dim=1)
                quadrupole_frobenius = (qi * qj).sum(dim=(-1, -2))
                if config.use_cavity_edge_sketches:
                    path_values["edge.cavity.quadrupole_frobenius"] = quadrupole_frobenius[:, None]
                    path_values["edge.cavity.target_quadrupole_projection"] = torch.einsum(
                        "bi,bij,bj->b", unit, qi, unit
                    )[:, None]
                    path_values["edge.cavity.source_quadrupole_projection"] = torch.einsum(
                        "bi,bij,bj->b", unit, qj, unit
                    )[:, None]
            if (not config.use_cavity_edge_sketches) and needs_shell_paths:
                if radial.shape[1] < 2:
                    raise ValueError("selected shell edge scalar paths require at least two radial channels")
                path_values.update(
                    _full_moment_shell_edge_path_values(
                        vector_channels_i,
                        vector_channels_j,
                        quadrupole_channels_i,
                        quadrupole_channels_j,
                        unit,
                    )
                )
            missing_paths = [path_id for path_id in edge_path_ids if path_id not in path_values]
            if missing_paths:
                raise ValueError(f"edge scalar path ids are not available for this config: {missing_paths}")
            edge_values = torch.cat([path_values[path_id] for path_id in edge_path_ids], dim=-1)
            edge_values = edge_values * cutoff_envelope(distances, config.cutoff)[:, None]
            return scatter_sum(edge_values, dst, graph.z.shape[0])
        if vector_channels_i is None or vector_channels_j is None or quadrupole_channels_i is None or quadrupole_channels_j is None:
            raise RuntimeError("legacy edge sketches require ell=1 and ell=2 moments")
        vi = vector_channels_i.mean(dim=1)
        vj = vector_channels_j.mean(dim=1)
        qi = quadrupole_channels_i.mean(dim=1)
        qj = quadrupole_channels_j.mean(dim=1)
        vector_dot = (vi * vj).sum(dim=-1)
        quadrupole_frobenius = (qi * qj).sum(dim=(-1, -2))
        target_vector_projection = (unit * vi).sum(dim=-1)
        source_vector_projection = (unit * vj).sum(dim=-1)
        target_quadrupole_projection = torch.einsum("bi,bij,bj->b", unit, qi, unit)
        source_quadrupole_projection = torch.einsum("bi,bij,bj->b", unit, qj, unit)
        base = torch.stack(
            [
                vector_dot,
                quadrupole_frobenius,
                target_vector_projection,
                source_vector_projection,
                target_quadrupole_projection,
                source_quadrupole_projection,
                direct_radial[:, 0],
                direct_radial[:, 1],
            ],
            dim=-1,
        )
        if config.num_edge_sketches > base.shape[1]:
            if radial.shape[1] < 2:
                raise ValueError("non-repeated edge sketches above 8 require at least two radial channels")
            shell_values = _full_moment_shell_edge_path_values(
                vector_channels_i,
                vector_channels_j,
                quadrupole_channels_i,
                quadrupole_channels_j,
                unit,
            )
            shell_base = torch.cat(
                [
                    shell_values["edge.full_moment.vector_shell_dot"],
                    shell_values["edge.full_moment.vector_cross_shell_dot"],
                    shell_values["edge.full_moment.quadrupole_shell_frobenius"],
                    shell_values["edge.full_moment.quadrupole_cross_shell_frobenius"],
                    shell_values["edge.full_moment.vector_shell_contrast_projection"],
                ],
                dim=-1,
            )
            base = torch.cat([base, shell_base], dim=-1)
    if config.num_edge_sketches > base.shape[1]:
        raise ValueError(
            f"requested {config.num_edge_sketches} edge sketches, but only {base.shape[1]} non-repeated scalar paths are implemented"
        )
    edge_values = base[:, : config.num_edge_sketches] * cutoff_envelope(distances, config.cutoff)[:, None]
    return scatter_sum(edge_values, dst, graph.z.shape[0])


def rtece_descriptors(
    graph: RTECEGraph,
    config: RTECEScalarConfig,
    radial_mixing: torch.nn.Linear | None = None,
    atomic_cross_radial_projection: torch.Tensor | None = None,
    species_basis_embedding: torch.Tensor | None = None,
    radial_species_adapter: torch.nn.Module | None = None,
    local_l0_chemistry_front: torch.nn.Module | None = None,
) -> torch.Tensor:
    adapter_scope = _normalize_radial_species_adapter_scope(config.radial_species_adapter_scope)
    atomic_radial_species_adapter = (
        radial_species_adapter if radial_species_adapter is not None and adapter_scope in {"all", "atomic"} else None
    )
    edge_radial_species_adapter = (
        radial_species_adapter if radial_species_adapter is not None and adapter_scope in {"all", "edge"} else None
    )
    atomic = atomic_scalar_descriptors(
        graph,
        config,
        radial_mixing,
        atomic_cross_radial_projection,
        species_basis_embedding,
        atomic_radial_species_adapter,
        local_l0_chemistry_front,
    )
    sketches = edge_relational_sketches(
        graph,
        config,
        radial_mixing,
        species_basis_embedding=species_basis_embedding,
        radial_species_adapter=edge_radial_species_adapter,
        local_l0_chemistry_front=local_l0_chemistry_front,
    )
    return torch.cat([atomic, sketches], dim=-1)


def _fixed_z_power_species_embedding_table(
    *,
    max_atomic_number: int,
    species_basis_channels: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    z = torch.arange(int(max_atomic_number) + 1, dtype=dtype, device=device) / float(max_atomic_number)
    powers = torch.arange(1, int(species_basis_channels) + 1, dtype=dtype, device=device)
    return z[:, None].pow(powers[None, :])


def _fixed_local_l0_chemistry_table(
    *,
    max_atomic_number: int,
    rank: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> torch.Tensor:
    z = torch.arange(int(max_atomic_number) + 1, dtype=dtype, device=device) / float(max_atomic_number)
    basis = torch.stack([torch.ones_like(z), z, z.square(), torch.sqrt(z.clamp_min(0.0))], dim=-1)
    return basis[:, : int(rank)]


class LocalL0ChemistryFront(torch.nn.Module):
    def __init__(self, *, max_atomic_number: int, rank: int, num_radial: int) -> None:
        super().__init__()
        local_rank = _normalize_local_l0_chemistry_rank(rank)
        if local_rank <= 0:
            raise ValueError("local L0 chemistry front rank must be positive")
        self.rank = int(local_rank)
        self.num_radial = int(num_radial)
        self.source_embedding = torch.nn.Embedding(int(max_atomic_number) + 1, self.rank, dtype=torch.float64)
        self.target_embedding = torch.nn.Embedding(int(max_atomic_number) + 1, self.rank, dtype=torch.float64)
        with torch.no_grad():
            table = _fixed_local_l0_chemistry_table(
                max_atomic_number=int(max_atomic_number),
                rank=self.rank,
                dtype=self.source_embedding.weight.dtype,
                device=self.source_embedding.weight.device,
            )
            self.source_embedding.weight.copy_(table)
            self.target_embedding.weight.copy_(table)

    def forward(
        self,
        radial: torch.Tensor,
        *,
        source_z: torch.Tensor,
        target_z: torch.Tensor,
        dst: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        max_index = self.source_embedding.num_embeddings - 1
        source_index = source_z.to(device=radial.device, dtype=torch.long).clamp(min=0, max=max_index)
        target_index = target_z.to(device=radial.device, dtype=torch.long).clamp(min=0, max=max_index)
        source = self.source_embedding(source_index).to(dtype=radial.dtype)
        target = self.target_embedding(target_index).to(dtype=radial.dtype)
        edge_basis = source * target
        values = radial[:, :, None] * edge_basis[:, None, :]
        density = scatter_sum(values, dst, int(num_nodes))
        return density.reshape(int(num_nodes), self.num_radial * self.rank)


class RadialSpeciesAdapter(torch.nn.Module):
    def __init__(self, *, max_atomic_number: int, channels: int, num_radial: int) -> None:
        super().__init__()
        adapter_channels = _normalize_radial_species_adapter_channels(channels)
        if adapter_channels <= 0:
            raise ValueError("radial species adapter channels must be positive")
        self.source_embedding = torch.nn.Embedding(int(max_atomic_number) + 1, adapter_channels, dtype=torch.float64)
        self.target_embedding = torch.nn.Embedding(int(max_atomic_number) + 1, adapter_channels, dtype=torch.float64)
        self.projection = torch.nn.Linear(2 * adapter_channels, int(num_radial), bias=False, dtype=torch.float64)
        with torch.no_grad():
            table = _fixed_z_power_species_embedding_table(
                max_atomic_number=int(max_atomic_number),
                species_basis_channels=adapter_channels,
                dtype=self.source_embedding.weight.dtype,
                device=self.source_embedding.weight.device,
            )
            self.source_embedding.weight.copy_(table)
            self.target_embedding.weight.copy_(table)
            self.projection.weight.zero_()

    def forward(self, radial: torch.Tensor, source_z: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
        max_index = self.source_embedding.num_embeddings - 1
        source_index = source_z.to(device=radial.device, dtype=torch.long).clamp(min=0, max=max_index)
        target_index = target_z.to(device=radial.device, dtype=torch.long).clamp(min=0, max=max_index)
        source = self.source_embedding(source_index).to(dtype=radial.dtype)
        target = self.target_embedding(target_index).to(dtype=radial.dtype)
        delta = self.projection(torch.cat([source, target], dim=-1)).to(dtype=radial.dtype)
        return radial * (1.0 + delta)


class RTECEScalarModel(torch.nn.Module):
    def __init__(self, config: RTECEScalarConfig):
        super().__init__()
        self.config = config
        if config.learnable_radial_mixing:
            self.radial_mixing = torch.nn.Linear(config.num_radial, config.num_radial, bias=False)
            torch.nn.init.eye_(self.radial_mixing.weight)
        else:
            self.radial_mixing = None
        if config.local_l0_chemistry_rank:
            self.local_l0_chemistry_front = LocalL0ChemistryFront(
                max_atomic_number=int(config.max_atomic_number),
                rank=int(config.local_l0_chemistry_rank),
                num_radial=int(config.num_radial),
            )
        else:
            self.local_l0_chemistry_front = None
        if config.radial_species_adapter_channels:
            self.radial_species_adapter = RadialSpeciesAdapter(
                max_atomic_number=int(config.max_atomic_number),
                channels=int(config.radial_species_adapter_channels),
                num_radial=int(config.num_radial),
            )
        else:
            self.radial_species_adapter = None
        if config.species_basis_channels and config.species_basis_mode == "learnable_embedding":
            self.species_basis_embedding = torch.nn.Embedding(
                int(config.max_atomic_number) + 1,
                int(config.species_basis_channels),
                dtype=torch.float64,
            )
            with torch.no_grad():
                self.species_basis_embedding.weight.copy_(
                    _fixed_z_power_species_embedding_table(
                        max_atomic_number=int(config.max_atomic_number),
                        species_basis_channels=int(config.species_basis_channels),
                        dtype=self.species_basis_embedding.weight.dtype,
                        device=self.species_basis_embedding.weight.device,
                    )
                )
        else:
            self.species_basis_embedding = None
        has_atomic_cross_radial_paths = bool(
            config.scalar_path_ids and any("cross_radial" in path_id for path_id in config.scalar_path_ids)
        )
        if config.atomic_cross_radial_projection == "learnable" and has_atomic_cross_radial_paths:
            self.atomic_cross_radial_projection = torch.nn.Linear(
                config.num_radial,
                config.atomic_cross_radial_sketch_channels,
                bias=False,
            )
            with torch.no_grad():
                self.atomic_cross_radial_projection.weight.copy_(
                    _fixed_radial_shell_projection_matrix(
                        num_radial=int(config.num_radial),
                        num_sketches=int(config.atomic_cross_radial_sketch_channels),
                        dtype=self.atomic_cross_radial_projection.weight.dtype,
                        device=self.atomic_cross_radial_projection.weight.device,
                    )
                )
            self.register_buffer("atomic_cross_radial_projection_buffer", None)
        elif config.atomic_cross_radial_projection == "pod_fixed" and has_atomic_cross_radial_paths:
            self.atomic_cross_radial_projection = None
            if config.atomic_cross_radial_projection_matrix is None:
                raise ValueError("atomic_cross_radial_projection=pod_fixed requires atomic_cross_radial_projection_matrix")
            self.register_buffer(
                "atomic_cross_radial_projection_buffer",
                torch.tensor(config.atomic_cross_radial_projection_matrix, dtype=torch.get_default_dtype()),
            )
        else:
            self.atomic_cross_radial_projection = None
            self.register_buffer("atomic_cross_radial_projection_buffer", None)
        in_dim = descriptor_dim(config)
        conditioner_name, conditioner_hidden = _normalize_descriptor_conditioner(
            config.descriptor_conditioner,
            config.descriptor_conditioner_hidden_channels,
        )
        if conditioner_name == "residual_mlp":
            self.descriptor_conditioner = torch.nn.Sequential(
                torch.nn.Linear(in_dim, conditioner_hidden),
                torch.nn.SiLU(),
                torch.nn.Linear(conditioner_hidden, in_dim),
            )
            torch.nn.init.zeros_(self.descriptor_conditioner[-1].weight)
            torch.nn.init.zeros_(self.descriptor_conditioner[-1].bias)
        else:
            self.descriptor_conditioner = None
        bottleneck_dim = _normalize_descriptor_bottleneck_dim(config.descriptor_bottleneck_dim)
        if bottleneck_dim:
            self.descriptor_bottleneck = torch.nn.Sequential(
                torch.nn.Linear(in_dim, bottleneck_dim),
                torch.nn.SiLU(),
            )
            readout_dim = bottleneck_dim
        else:
            self.descriptor_bottleneck = None
            readout_dim = in_dim
        layers: list[torch.nn.Module] = []
        prev = readout_dim + 1
        for hidden in config.hidden_channels:
            layers.append(torch.nn.Linear(prev, hidden))
            layers.append(torch.nn.SiLU())
            prev = hidden
        layers.append(torch.nn.Linear(prev, 1))
        self.energy_head = torch.nn.Sequential(*layers)

    def _condition_descriptors(self, descriptors: torch.Tensor) -> torch.Tensor:
        if self.descriptor_conditioner is None:
            return descriptors
        return descriptors + self.descriptor_conditioner(descriptors)

    def _readout_descriptors(self, descriptors: torch.Tensor) -> torch.Tensor:
        descriptors = self._condition_descriptors(descriptors)
        if self.descriptor_bottleneck is None:
            return descriptors
        return self.descriptor_bottleneck(descriptors)

    def _require_inference_mode(self, backend_name: str, graph: RTECEGraph | None = None) -> None:
        if self.training:
            raise RuntimeError(
                f"{backend_name} is an inference-only rTECE force backend; "
                "use model.eval() for benchmark/inference or model(graph) for force training."
            )
        if graph is not None and graph.edge_shifts is not None and graph.edge_shifts.numel() > 0:
            if torch.any(graph.edge_shifts != 0):
                raise ValueError(
                    f"{backend_name} does not support periodic image shifts yet; "
                    "use force_mode='autograd' for PBC inference."
                )
        if self.config.use_short_range_repulsion:
            raise ValueError(
                f"{backend_name} does not include short-range radial-core forces yet; "
                "use force_mode='autograd' for this T4 candidate."
            )
        if self.config.learnable_radial_mixing:
            raise ValueError(
                f"{backend_name} does not include learnable radial-mixing descriptor derivatives yet; "
                "use force_mode='autograd' for trainable-feature rTECE candidates."
            )
        if self.config.local_l0_chemistry_rank:
            raise ValueError(
                f"{backend_name} does not include local L0 chemistry front descriptor derivatives yet; "
                "use force_mode='autograd' for trainable local L0 rTECE candidates."
            )
        if self.config.radial_species_adapter_channels:
            raise ValueError(
                f"{backend_name} does not include radial species adapter descriptor derivatives yet; "
                "use force_mode='autograd' for trainable edge-species radial rTECE candidates."
            )
        if self.config.species_basis_channels and self.config.species_basis_mode == "learnable_embedding":
            raise ValueError(
                f"{backend_name} does not include learnable species-basis descriptor derivatives yet; "
                "use force_mode='autograd' for trainable-species rTECE candidates."
            )
        if self.config.descriptor_conditioner != "none":
            raise ValueError(
                f"{backend_name} does not include descriptor conditioner derivatives yet; "
                "use force_mode='autograd' for scalar-conditioned rTECE candidates."
            )

        if self.config.descriptor_bottleneck_dim:
            raise ValueError(
                f"{backend_name} does not include descriptor bottleneck derivatives yet; "
                "use force_mode='autograd' for low-rank descriptor-mixer rTECE candidates."
            )

    def forward_density_analytic_forces(self, graph: RTECEGraph) -> dict[str, torch.Tensor]:
        self._require_inference_mode("forward_density_analytic_forces", graph)
        if self.config.use_atomic_moments or self.config.num_edge_sketches:
            raise ValueError("forward_density_analytic_forces only supports scalar density/moment descriptors")
        pos = graph.pos
        src, dst = graph.edge_index
        num_nodes = graph.z.shape[0]
        num_graphs = int(graph.batch.max().item()) + 1 if graph.batch.numel() else 1

        vectors = pos[dst] - pos[src]
        distances = vectors.norm(dim=-1).clamp_min(1e-12)
        unit = vectors / distances[:, None]
        radial, radial_derivative = radial_features_and_derivatives(distances, self.config)
        radial_detached = radial.detach()
        density = scatter_sum(radial_detached, dst, num_nodes).requires_grad_(True)
        neighbor_z = graph.z[src].to(dtype=pos.dtype, device=pos.device) / float(self.config.max_atomic_number)
        element_density = None
        if self.config.use_element_density:
            element_density = scatter_sum(radial_detached * neighbor_z[:, None], dst, num_nodes).requires_grad_(True)
        vector = None
        vector_norm = None
        if self.config.use_vector_moments:
            vector = scatter_sum(radial_detached[:, :, None] * unit.detach()[:, None, :], dst, num_nodes)
            vector_norm = (vector.square()).sum(dim=-1).requires_grad_(True)
        descriptors = density_scalar_descriptors(density, self.config, element_density, vector_norm)

        z_scaled = graph.z.to(dtype=pos.dtype, device=pos.device).view(-1, 1)
        z_scaled = z_scaled / float(self.config.max_atomic_number)
        atomic_energy = self.energy_head(torch.cat([z_scaled, descriptors], dim=-1)).squeeze(-1)
        energy = scatter_sum(atomic_energy[:, None], graph.batch, num_graphs).squeeze(-1)
        energy = add_atomic_reference_energy(energy, graph, self.config)

        grad_targets = [density]
        if element_density is not None:
            grad_targets.append(element_density)
        if vector_norm is not None:
            grad_targets.append(vector_norm)
        grads = torch.autograd.grad(
            energy.sum(),
            grad_targets,
            create_graph=False,
            retain_graph=False,
        )
        grad_index = 0
        density_grad = grads[grad_index]
        grad_index += 1
        edge_scale = (density_grad[dst] * radial_derivative).sum(dim=-1)
        if element_density is not None:
            element_density_grad = grads[grad_index]
            grad_index += 1
            edge_scale = edge_scale + (element_density_grad[dst] * neighbor_z[:, None] * radial_derivative).sum(dim=-1)
        grad_vectors = edge_scale[:, None] * unit
        if vector_norm is not None:
            if vector is None:
                raise RuntimeError("vector moments were not computed")
            vector_norm_grad = grads[grad_index]
            edge_vector = vector[dst]
            dot = (edge_vector * unit[:, None, :]).sum(dim=-1)
            parallel = radial_derivative[:, :, None] * dot[:, :, None] * unit[:, None, :]
            transverse = radial_detached[:, :, None] / distances[:, None, None] * (
                edge_vector - dot[:, :, None] * unit[:, None, :]
            )
            vector_grad_vectors = (
                2.0 * vector_norm_grad[dst, :, None] * (parallel + transverse)
            ).sum(dim=1)
            grad_vectors = grad_vectors + vector_grad_vectors
        grad_pos = pos.new_zeros(pos.shape)
        grad_pos.index_add_(0, dst, grad_vectors)
        grad_pos.index_add_(0, src, -grad_vectors)
        forces = -grad_pos
        return {"energy": energy, "atomic_energy": atomic_energy, "forces": forces}

    def forward_pair_analytic_forces(self, graph: RTECEGraph) -> dict[str, torch.Tensor]:
        self._require_inference_mode("forward_pair_analytic_forces", graph)
        if self.config.use_density_quadratic:
            raise ValueError("forward_pair_analytic_forces only supports pure rtece_pair descriptors")
        return self.forward_density_analytic_forces(graph)

    def forward_pair_triton_force_analytic_forces(self, graph: RTECEGraph) -> dict[str, torch.Tensor]:
        self._require_inference_mode("forward_pair_triton_force_analytic_forces", graph)
        if self.config.use_element_density or self.config.use_density_quadratic or self.config.use_vector_moments:
            raise ValueError("forward_pair_triton_force_analytic_forces only supports pure rtece_pair descriptors")
        if self.config.use_atomic_moments or self.config.num_edge_sketches:
            raise ValueError("forward_pair_triton_force_analytic_forces only supports scalar pair descriptors")
        if graph.pos.device.type != "cuda":
            raise RuntimeError("Triton pair force path requires a CUDA graph")
        from tace.models.rtece_triton_kernels import pair_forces_triton

        pos = graph.pos
        src, dst = graph.edge_index
        num_nodes = graph.z.shape[0]
        num_graphs = int(graph.batch.max().item()) + 1 if graph.batch.numel() else 1

        vectors = pos[dst] - pos[src]
        distances = vectors.norm(dim=-1).clamp_min(1e-12)
        radial = compute_radial_features_only(distances, self.config).detach()
        density = scatter_sum(radial, dst, num_nodes).requires_grad_(True)
        descriptors = density_scalar_descriptors(density, self.config)

        z_scaled = graph.z.to(dtype=pos.dtype, device=pos.device).view(-1, 1)
        z_scaled = z_scaled / float(self.config.max_atomic_number)
        atomic_energy = self.energy_head(torch.cat([z_scaled, descriptors], dim=-1)).squeeze(-1)
        energy = scatter_sum(atomic_energy[:, None], graph.batch, num_graphs).squeeze(-1)
        energy = add_atomic_reference_energy(energy, graph, self.config)

        density_grad = torch.autograd.grad(
            energy.sum(),
            density,
            create_graph=False,
            retain_graph=False,
        )[0]
        forces = pair_forces_triton(
            pos=pos,
            edge_index=graph.edge_index,
            density_grad=density_grad,
            cutoff=float(self.config.cutoff),
            num_radial=int(self.config.num_radial),
        )
        return {"energy": energy, "atomic_energy": atomic_energy, "forces": forces}

    def forward_element_density_triton_force_analytic_forces(self, graph: RTECEGraph) -> dict[str, torch.Tensor]:
        self._require_inference_mode("forward_element_density_triton_force_analytic_forces", graph)
        if not self.config.use_element_density:
            raise ValueError("forward_element_density_triton_force_analytic_forces requires use_element_density=True")
        if self.config.use_density_quadratic or self.config.use_vector_moments:
            raise ValueError("Triton element-density path only supports density plus element density")
        if self.config.use_atomic_moments or self.config.num_edge_sketches:
            raise ValueError("Triton element-density path only supports scalar density descriptors")
        if graph.pos.device.type != "cuda":
            raise RuntimeError("Triton element-density force path requires a CUDA graph")
        from tace.models.rtece_triton_kernels import element_density_forces_triton

        pos = graph.pos
        src, dst = graph.edge_index
        num_nodes = graph.z.shape[0]
        num_graphs = int(graph.batch.max().item()) + 1 if graph.batch.numel() else 1

        vectors = pos[dst] - pos[src]
        distances = vectors.norm(dim=-1).clamp_min(1e-12)
        radial = compute_radial_features_only(distances, self.config).detach()
        node_z = graph.z.to(dtype=pos.dtype, device=pos.device) / float(self.config.max_atomic_number)
        neighbor_z = node_z[src]
        density = scatter_sum(radial, dst, num_nodes).requires_grad_(True)
        element_density = scatter_sum(radial * neighbor_z[:, None], dst, num_nodes).requires_grad_(True)
        descriptors = density_scalar_descriptors(density, self.config, element_density)

        z_scaled = node_z.view(-1, 1)
        atomic_energy = self.energy_head(torch.cat([z_scaled, descriptors], dim=-1)).squeeze(-1)
        energy = scatter_sum(atomic_energy[:, None], graph.batch, num_graphs).squeeze(-1)
        energy = add_atomic_reference_energy(energy, graph, self.config)

        density_grad, element_density_grad = torch.autograd.grad(
            energy.sum(),
            [density, element_density],
            create_graph=False,
            retain_graph=False,
        )
        forces = element_density_forces_triton(
            pos=pos,
            edge_index=graph.edge_index,
            node_z=node_z,
            density_grad=density_grad,
            element_density_grad=element_density_grad,
            cutoff=float(self.config.cutoff),
            num_radial=int(self.config.num_radial),
        )
        return {"energy": energy, "atomic_energy": atomic_energy, "forces": forces}

    def forward_element_density_triton_descriptor_force_analytic_forces(self, graph: RTECEGraph) -> dict[str, torch.Tensor]:
        self._require_inference_mode("forward_element_density_triton_descriptor_force_analytic_forces", graph)
        if not self.config.use_element_density:
            raise ValueError("forward_element_density_triton_descriptor_force_analytic_forces requires use_element_density=True")
        if self.config.use_density_quadratic or self.config.use_vector_moments:
            raise ValueError("Triton element-density descriptor+force path only supports density plus element density")
        if self.config.use_atomic_moments or self.config.num_edge_sketches:
            raise ValueError("Triton element-density descriptor+force path only supports scalar density descriptors")
        if graph.pos.device.type != "cuda":
            raise RuntimeError("Triton element-density descriptor+force path requires a CUDA graph")
        from tace.models.rtece_triton_kernels import (
            element_density_descriptors_triton,
            element_density_forces_triton,
        )

        pos = graph.pos
        num_graphs = int(graph.batch.max().item()) + 1 if graph.batch.numel() else 1
        node_z = graph.z.to(dtype=pos.dtype, device=pos.device) / float(self.config.max_atomic_number)
        descriptors = element_density_descriptors_triton(
            pos=pos,
            edge_index=graph.edge_index,
            node_z=node_z,
            cutoff=float(self.config.cutoff),
            num_radial=int(self.config.num_radial),
        ).requires_grad_(True)

        z_scaled = node_z.view(-1, 1)
        atomic_energy = self.energy_head(torch.cat([z_scaled, descriptors], dim=-1)).squeeze(-1)
        energy = scatter_sum(atomic_energy[:, None], graph.batch, num_graphs).squeeze(-1)
        energy = add_atomic_reference_energy(energy, graph, self.config)

        descriptor_grad = torch.autograd.grad(
            energy.sum(),
            descriptors,
            create_graph=False,
            retain_graph=False,
        )[0]
        density_grad, element_density_grad = descriptor_grad.split(self.config.num_radial, dim=-1)
        forces = element_density_forces_triton(
            pos=pos,
            edge_index=graph.edge_index,
            node_z=node_z,
            density_grad=density_grad,
            element_density_grad=element_density_grad,
            cutoff=float(self.config.cutoff),
            num_radial=int(self.config.num_radial),
        )
        return {"energy": energy, "atomic_energy": atomic_energy, "forces": forces}

    def forward_element_density_direct_padded_triton_descriptor_force_analytic_forces(self, graph: RTECEGraph) -> dict[str, torch.Tensor]:
        self._require_inference_mode("forward_element_density_direct_padded_triton_descriptor_force_analytic_forces", graph)
        _validate_packed_element_density_config(
            self.config,
            "forward_element_density_direct_padded_triton_descriptor_force_analytic_forces",
        )
        if graph.pos.device.type != "cuda":
            raise RuntimeError("Triton direct-padded element-density descriptor+force path requires a CUDA graph")
        if graph.pos.dtype != torch.float32:
            raise RuntimeError("Triton direct-padded element-density descriptor+force path currently supports float32 graphs only")
        if graph.batch.ndim != 1 or graph.batch.shape[0] != graph.z.shape[0]:
            raise ValueError("direct-padded Triton path requires one batch id per atom")
        if graph.batch.numel() > 1 and torch.any(graph.batch[1:] < graph.batch[:-1]):
            raise ValueError("direct-padded Triton path requires atoms sorted by batch")

        from tace.models.rtece_triton_kernels import (
            element_density_direct_padded_descriptors_triton,
            element_density_direct_padded_forces_triton,
        )

        pos = graph.pos
        num_graphs = int(graph.batch.max().item()) + 1 if graph.batch.numel() else 1
        if graph.batch.numel():
            _, counts = torch.unique_consecutive(graph.batch, return_counts=True)
            starts = torch.cat([counts.new_zeros(1), counts.cumsum(dim=0)[:-1]])
        else:
            counts = graph.batch.new_zeros(0)
            starts = graph.batch.new_zeros(0)
        node_z = graph.z.to(dtype=pos.dtype, device=pos.device) / float(self.config.max_atomic_number)
        descriptors = element_density_direct_padded_descriptors_triton(
            pos=pos,
            counts=counts.to(device=pos.device),
            starts=starts.to(device=pos.device),
            node_z=node_z,
            cutoff=float(self.config.cutoff),
            num_radial=int(self.config.num_radial),
        ).requires_grad_(True)

        z_scaled = node_z.view(-1, 1)
        atomic_energy = self.energy_head(torch.cat([z_scaled, descriptors], dim=-1)).squeeze(-1)
        energy = scatter_sum(atomic_energy[:, None], graph.batch, num_graphs).squeeze(-1)
        energy = add_atomic_reference_energy(energy, graph, self.config)

        descriptor_grad = torch.autograd.grad(
            energy.sum(),
            descriptors,
            create_graph=False,
            retain_graph=False,
        )[0]
        density_grad, element_density_grad = descriptor_grad.split(self.config.num_radial, dim=-1)
        forces = element_density_direct_padded_forces_triton(
            pos=pos,
            counts=counts.to(device=pos.device),
            starts=starts.to(device=pos.device),
            node_z=node_z,
            density_grad=density_grad,
            element_density_grad=element_density_grad,
            cutoff=float(self.config.cutoff),
            num_radial=int(self.config.num_radial),
        )
        return {"energy": energy, "atomic_energy": atomic_energy, "forces": forces}


    def forward_element_density_cell_list_packed_analytic_forces(self, graph: RTECEGraph) -> dict[str, torch.Tensor]:
        self._require_inference_mode("forward_element_density_cell_list_packed_analytic_forces", graph)
        _validate_packed_element_density_config(self.config, "forward_element_density_cell_list_packed_analytic_forces")
        pos = graph.pos
        src, dst = _cell_list_directed_edges_nopbc(graph, float(self.config.cutoff))
        num_nodes = graph.z.shape[0]
        num_graphs = int(graph.batch.max().item()) + 1 if graph.batch.numel() else 1

        node_z = graph.z.to(dtype=pos.dtype, device=pos.device) / float(self.config.max_atomic_number)
        descriptors = pos.new_zeros((num_nodes, 2 * self.config.num_radial))
        if src.numel() > 0:
            vectors = pos[dst] - pos[src]
            distances = vectors.norm(dim=-1).clamp_min(1e-12)
            radial, radial_derivative = radial_features_and_derivatives(distances, self.config)
            radial_detached = radial.detach()
            neighbor_z = node_z[src]
            edge_descriptors = torch.cat([radial_detached, radial_detached * neighbor_z[:, None]], dim=-1)
            descriptors.index_add_(0, dst, edge_descriptors)
        descriptors = descriptors.requires_grad_(True)

        z_scaled = node_z.view(-1, 1)
        atomic_energy = self.energy_head(torch.cat([z_scaled, descriptors], dim=-1)).squeeze(-1)
        energy = scatter_sum(atomic_energy[:, None], graph.batch, num_graphs).squeeze(-1)
        energy = add_atomic_reference_energy(energy, graph, self.config)

        descriptor_grad = torch.autograd.grad(
            energy.sum(),
            descriptors,
            create_graph=False,
            retain_graph=False,
        )[0]
        forces = pos.new_zeros(pos.shape)
        if src.numel() > 0:
            density_grad, element_density_grad = descriptor_grad.split(self.config.num_radial, dim=-1)
            unit = vectors / distances[:, None]
            edge_scale = (
                (density_grad[dst] + element_density_grad[dst] * neighbor_z[:, None])
                * radial_derivative
            ).sum(dim=-1)
            grad_vectors = edge_scale[:, None] * unit
            grad_pos = pos.new_zeros(pos.shape)
            grad_pos.index_add_(0, dst, grad_vectors)
            grad_pos.index_add_(0, src, -grad_vectors)
            forces = -grad_pos
        return {"energy": energy, "atomic_energy": atomic_energy, "forces": forces}

    def forward_element_density_packed_analytic_forces(self, graph: RTECEGraph) -> dict[str, torch.Tensor]:
        self._require_inference_mode("forward_element_density_packed_analytic_forces", graph)
        if not self.config.use_element_density:
            raise ValueError("forward_element_density_packed_analytic_forces requires use_element_density=True")
        if self.config.use_density_quadratic or self.config.use_vector_moments:
            raise ValueError("packed element-density path only supports density plus element density")
        if self.config.use_atomic_moments or self.config.num_edge_sketches:
            raise ValueError("packed element-density path only supports scalar density descriptors")
        pos = graph.pos
        src, dst = graph.edge_index
        num_nodes = graph.z.shape[0]
        num_graphs = int(graph.batch.max().item()) + 1 if graph.batch.numel() else 1

        vectors = pos[dst] - pos[src]
        distances = vectors.norm(dim=-1).clamp_min(1e-12)
        unit = vectors / distances[:, None]
        radial, radial_derivative = radial_features_and_derivatives(distances, self.config)
        radial_detached = radial.detach()
        neighbor_z = graph.z[src].to(dtype=pos.dtype, device=pos.device) / float(self.config.max_atomic_number)
        edge_descriptors = torch.cat([radial_detached, radial_detached * neighbor_z[:, None]], dim=-1)
        descriptors = scatter_sum(edge_descriptors, dst, num_nodes).requires_grad_(True)

        z_scaled = graph.z.to(dtype=pos.dtype, device=pos.device).view(-1, 1)
        z_scaled = z_scaled / float(self.config.max_atomic_number)
        atomic_energy = self.energy_head(torch.cat([z_scaled, descriptors], dim=-1)).squeeze(-1)
        energy = scatter_sum(atomic_energy[:, None], graph.batch, num_graphs).squeeze(-1)
        energy = add_atomic_reference_energy(energy, graph, self.config)

        descriptor_grad = torch.autograd.grad(
            energy.sum(),
            descriptors,
            create_graph=False,
            retain_graph=False,
        )[0]
        density_grad, element_density_grad = descriptor_grad.split(self.config.num_radial, dim=-1)
        edge_scale = (
            (density_grad[dst] + element_density_grad[dst] * neighbor_z[:, None])
            * radial_derivative
        ).sum(dim=-1)
        grad_vectors = edge_scale[:, None] * unit
        grad_pos = pos.new_zeros(pos.shape)
        grad_pos.index_add_(0, dst, grad_vectors)
        grad_pos.index_add_(0, src, -grad_vectors)
        forces = -grad_pos
        return {"energy": energy, "atomic_energy": atomic_energy, "forces": forces}

    def _atomic_cross_radial_projection_weight(self) -> torch.Tensor | None:
        if self.atomic_cross_radial_projection is not None:
            return self.atomic_cross_radial_projection.weight
        return self.atomic_cross_radial_projection_buffer

    def _species_basis_embedding_weight(self) -> torch.Tensor | None:
        if self.species_basis_embedding is None:
            return None
        return self.species_basis_embedding.weight

    def forward(self, graph: RTECEGraph, *, compute_forces: bool = True) -> dict[str, torch.Tensor]:
        pos = graph.pos
        if compute_forces and not pos.requires_grad:
            pos = pos.detach().clone().requires_grad_(True)
            graph = RTECEGraph(
                z=graph.z,
                pos=pos,
                edge_index=graph.edge_index,
                batch=graph.batch,
                cell=graph.cell,
                edge_shifts=graph.edge_shifts,
                edge_batch=graph.edge_batch,
            )

        z_scaled = graph.z.to(dtype=pos.dtype, device=pos.device).view(-1, 1)
        z_scaled = z_scaled / float(self.config.max_atomic_number)
        descriptors = rtece_descriptors(
            graph,
            self.config,
            self.radial_mixing,
            self._atomic_cross_radial_projection_weight(),
            self._species_basis_embedding_weight(),
            self.radial_species_adapter,
            self.local_l0_chemistry_front,
        )
        descriptors = self._readout_descriptors(descriptors)
        atomic_input = torch.cat([z_scaled, descriptors], dim=-1)
        atomic_energy = self.energy_head(atomic_input).squeeze(-1)
        num_graphs = int(graph.batch.max().item()) + 1 if graph.batch.numel() else 1
        energy = scatter_sum(atomic_energy[:, None], graph.batch, num_graphs).squeeze(-1)
        energy = add_atomic_reference_energy(energy, graph, self.config)
        energy = add_short_range_repulsive_energy(energy, graph, self.config)
        output = {"energy": energy, "atomic_energy": atomic_energy}
        if not compute_forces:
            return output
        forces = -torch.autograd.grad(
            energy.sum(),
            pos,
            create_graph=self.training,
            retain_graph=True,
        )[0]
        output["forces"] = forces
        return output


__all__ = [
    "RTECEScalarConfig",
    "RTECEGraph",
    "RTECEScalarModel",
    "RadialSpeciesAdapter",
    "LocalL0ChemistryFront",
    "available_rtece_variants",
    "build_rtece_config",
    "config_with_moment_l_max",
    "build_rtece_config_from_manifest",
    "build_rtece_config_from_path_ids",
    "descriptor_dim",
    "collate_graphs",
    "compute_pair_geometry",
    "atomic_reference_energy",
    "add_atomic_reference_energy",
    "short_range_repulsive_energy",
    "add_short_range_repulsive_energy",
    "cutoff_envelope",
    "radial_features_and_derivatives",
    "compute_radial_features_only",
    "compute_radial_features",
    "scatter_sum",
    "compute_atomic_moments",
    "density_scalar_descriptors",
    "packed_element_density_descriptors",
    "cell_list_packed_element_density_descriptors",
    "atomic_scalar_descriptors",
    "_project_radial_edge_channels",
    "edge_relational_sketches",
    "rtece_descriptors",
    "rtece_path_manifest",
    "rtece_route_contract",
    "rtece_variant_registry",
]
