from __future__ import annotations

from dataclasses import dataclass
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
    max_atomic_number: int = 100
    use_element_density: bool = False
    use_density_quadratic: bool = False
    use_vector_moments: bool = False
    use_atomic_moments: bool = False
    species_basis_channels: int = 0
    num_edge_sketches: int = 0
    use_cavity_edge_sketches: bool = False
    radial_edge_sketch_channels: int = 0
    scalar_path_ids: tuple[str, ...] | None = None
    energy_per_atom_shift: float = 0.0
    atomic_energies: Mapping[int, float] | None = None


_ATOMIC_SCALAR_PATH_IDS = {
    "atomic.radial_density",
    "atomic.element_density",
    "atomic.species_basis_density",
    "atomic.density_square",
    "atomic.vector_norm",
    "atomic.quadrupole_norm",
}


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


def build_rtece_config_from_path_ids(
    variant: str,
    scalar_path_ids: tuple[str, ...] | list[str],
    *,
    cutoff: float = 5.0,
    num_radial: int = 8,
    hidden_channels: tuple[int, ...] = (64, 64),
    max_atomic_number: int = 100,
    species_basis_channels: int = 0,
    energy_per_atom_shift: float = 0.0,
    atomic_energies: Mapping[int, float] | None = None,
) -> RTECEScalarConfig:
    paths = tuple(str(path_id) for path_id in scalar_path_ids)
    if not paths:
        raise ValueError("at least one scalar path id is required")
    unsupported = [path_id for path_id in paths if path_id not in _ATOMIC_SCALAR_PATH_IDS]
    if unsupported:
        raise ValueError(
            "build_rtece_config_from_path_ids currently supports atomic scalar path ids only; "
            f"unsupported paths: {unsupported}"
        )
    if "atomic.radial_density" not in paths:
        raise ValueError("atomic.radial_density is required as the base scalar path")
    if "atomic.species_basis_density" in paths and species_basis_channels <= 0:
        raise ValueError("species_basis_channels must be positive for atomic.species_basis_density")

    return RTECEScalarConfig(
        variant=variant,
        cutoff=float(cutoff),
        num_radial=int(num_radial),
        hidden_channels=tuple(int(value) for value in hidden_channels),
        max_atomic_number=int(max_atomic_number),
        use_element_density="atomic.element_density" in paths,
        use_density_quadratic="atomic.density_square" in paths,
        use_vector_moments="atomic.vector_norm" in paths,
        use_atomic_moments="atomic.quadrupole_norm" in paths,
        species_basis_channels=int(species_basis_channels) if "atomic.species_basis_density" in paths else 0,
        num_edge_sketches=0,
        use_cavity_edge_sketches=False,
        radial_edge_sketch_channels=0,
        scalar_path_ids=paths,
        energy_per_atom_shift=float(energy_per_atom_shift),
        atomic_energies=atomic_energies,
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
        max_atomic_number=int(payload.get("max_atomic_number", 100)),
        use_element_density=bool(payload.get("use_element_density", False)),
        use_density_quadratic=bool(payload.get("use_density_quadratic", False)),
        use_vector_moments=bool(payload.get("use_vector_moments", False)),
        use_atomic_moments=bool(payload.get("use_atomic_moments", False)),
        species_basis_channels=int(payload.get("species_basis_channels", 0)),
        num_edge_sketches=int(payload.get("num_edge_sketches", 0)),
        use_cavity_edge_sketches=bool(payload.get("use_cavity_edge_sketches", False)),
        radial_edge_sketch_channels=int(payload.get("radial_edge_sketch_channels", 0)),
        scalar_path_ids=tuple(str(path_id) for path_id in payload["scalar_path_ids"])
        if "scalar_path_ids" in payload
        else None,
    )


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
    elif config.use_atomic_moments or config.num_edge_sketches:
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
    if graph_construction_backend or graph_update_backend:
        pareto_axes.append("topology_provider")
    if fused_descriptor or fused_force:
        pareto_axes.append("kernel_fusion")

    return {
        "semantic_tier": semantic_tier,
        "descriptor_family": descriptor_family,
        "retained_tece_groups": retained,
        "deleted_tece_groups": deleted,
        "descriptor_dim": descriptor_dim(config),
        "num_radial": int(config.num_radial),
        "hidden_channels": list(config.hidden_channels),
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
    cost_group: str,
) -> dict[str, object]:
    return {
        "id": path_id,
        "placement": placement,
        "inputs": inputs,
        "contraction": contraction,
        "radial_projection": radial_projection,
        "cavity": bool(cavity),
        "cost_group": cost_group,
    }


def _config_manifest_payload(config: RTECEScalarConfig) -> dict[str, object]:
    payload: dict[str, object] = {
        "variant": config.variant,
        "cutoff": float(config.cutoff),
        "num_radial": int(config.num_radial),
        "hidden_channels": list(config.hidden_channels),
        "max_atomic_number": int(config.max_atomic_number),
        "use_element_density": bool(config.use_element_density),
        "use_density_quadratic": bool(config.use_density_quadratic),
        "use_vector_moments": bool(config.use_vector_moments),
        "use_atomic_moments": bool(config.use_atomic_moments),
        "species_basis_channels": int(config.species_basis_channels),
        "num_edge_sketches": int(config.num_edge_sketches),
        "use_cavity_edge_sketches": bool(config.use_cavity_edge_sketches),
        "radial_edge_sketch_channels": int(config.radial_edge_sketch_channels),
        "energy_reference": "per_element_atomic_energies" if config.atomic_energies else ("global_per_atom_shift" if config.energy_per_atom_shift else "none"),
    }
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
    moments = [_moment_spec("moment.l0.radial_density", ell=0, radial_projection="identity")]
    if config.use_element_density:
        moments.append(_moment_spec("moment.l0.element_density", ell=0, radial_projection="identity", chemistry_basis="atomic_number_first_moment"))
    if config.species_basis_channels:
        moments.append(_moment_spec("moment.l0.species_basis_density", ell=0, radial_projection="identity", chemistry_basis=f"fixed_z_power_{int(config.species_basis_channels)}"))
    if config.use_vector_moments or config.use_atomic_moments or config.num_edge_sketches:
        moments.append(_moment_spec("moment.l1.vector", ell=1, radial_projection=radial_projection))
    if config.use_atomic_moments or config.num_edge_sketches:
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
                            cost_group="edge_cavity_radial_relations",
                        ),
                        _scalar_path_spec(
                            "edge.cavity.vector_cross_radial_dot",
                            placement="edge",
                            inputs=["moment.l1.vector", "moment.l1.vector"],
                            contraction="cross_radial_dot",
                            radial_projection="fixed_two_shell_mean",
                            cavity=True,
                            cost_group="edge_cavity_radial_relations",
                        ),
                        _scalar_path_spec(
                            "edge.cavity.quadrupole_cross_radial_frobenius",
                            placement="edge",
                            inputs=["moment.l2.quadrupole", "moment.l2.quadrupole"],
                            contraction="cross_radial_frobenius",
                            radial_projection="fixed_two_shell_mean",
                            cavity=True,
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
                            cost_group="edge_cavity_relations",
                        ),
                        _scalar_path_spec(
                            "edge.cavity.quadrupole_frobenius",
                            placement="edge",
                            inputs=["moment.l2.quadrupole", "moment.l2.quadrupole"],
                            contraction="frobenius",
                            radial_projection="full_radial_mean",
                            cavity=True,
                            cost_group="edge_cavity_relations",
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
                    cost_group="edge_full_moment_relations",
                )
            )
        scalar_paths.append(
            _scalar_path_spec(
                "edge.direct.radial",
                placement="edge",
                inputs=["moment.l0.radial_density"],
                contraction="direct_edge_radial",
                radial_projection="selected_direct_channels",
                cost_group="direct_pair_radial",
            )
        )

    if config.scalar_path_ids is not None:
        scalar_paths_by_id = {str(path["id"]): path for path in scalar_paths}
        missing_paths = [path_id for path_id in config.scalar_path_ids if path_id not in scalar_paths_by_id]
        if missing_paths:
            raise ValueError(f"scalar path ids are not available for this config: {missing_paths}")
        scalar_paths = [scalar_paths_by_id[path_id] for path_id in config.scalar_path_ids]

    manifest_core: dict[str, Any] = {
        "schema_version": "rtece_path_manifest.v1",
        "config": _config_manifest_payload(config),
        "route": route,
        "moments": moments,
        "scalar_paths": scalar_paths,
        "retained_tece_groups": route["retained_tece_groups"],
        "deleted_tece_groups": route["deleted_tece_groups"],
        "compiler_status": "explicit_manifest_not_full_compiler",
    }
    encoded = json.dumps(manifest_core, sort_keys=True, separators=(",", ":"), allow_nan=False)
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


def radial_features_and_derivatives(
    distances: torch.Tensor,
    config: RTECEScalarConfig,
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
    return features, derivatives


def compute_radial_features_only(distances: torch.Tensor, config: RTECEScalarConfig) -> torch.Tensor:
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
    return gaussian * envelope[:, None]


def compute_radial_features(distances: torch.Tensor, config: RTECEScalarConfig) -> torch.Tensor:
    return compute_radial_features_only(distances, config)


def scatter_sum(values: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = values.new_zeros((dim_size, *values.shape[1:]))
    out.index_add_(0, index, values)
    return out


def compute_atomic_moments(graph: RTECEGraph, config: RTECEScalarConfig) -> dict[str, torch.Tensor]:
    _, distances, unit = compute_pair_geometry(graph)
    radial = compute_radial_features(distances, config)
    src, dst = graph.edge_index
    num_nodes = graph.z.shape[0]
    density = scatter_sum(radial, dst, num_nodes)
    neighbor_z = graph.z[src].to(dtype=graph.pos.dtype, device=graph.pos.device) / float(config.max_atomic_number)
    element_density = scatter_sum(radial * neighbor_z[:, None], dst, num_nodes)
    vector = scatter_sum(radial[:, :, None] * unit[:, None, :], dst, num_nodes)
    eye = torch.eye(3, device=graph.pos.device, dtype=graph.pos.dtype)
    quad_unit = unit[:, :, None] * unit[:, None, :] - eye[None, :, :] / 3.0
    quadrupole = scatter_sum(radial[:, :, None, None] * quad_unit[:, None, :, :], dst, num_nodes)
    species_density = None
    if config.species_basis_channels:
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
        "vector": vector,
        "quadrupole": quadrupole,
    }


def _atomic_scalar_path_descriptors(
    path_ids: tuple[str, ...],
    *,
    density: torch.Tensor,
    element_density: torch.Tensor | None = None,
    species_density: torch.Tensor | None = None,
    vector_norm: torch.Tensor | None = None,
    quadrupole_norm: torch.Tensor | None = None,
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
        elif path_id == "atomic.density_square":
            parts.append(density.square())
        elif path_id == "atomic.vector_norm":
            if vector_norm is None:
                raise ValueError("vector_norm is required for atomic.vector_norm")
            parts.append(vector_norm)
        elif path_id == "atomic.quadrupole_norm":
            if quadrupole_norm is None:
                raise ValueError("quadrupole_norm is required for atomic.quadrupole_norm")
            parts.append(quadrupole_norm)
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
        return _atomic_scalar_path_descriptors(
            config.scalar_path_ids,
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


def packed_element_density_descriptors(graph: RTECEGraph, config: RTECEScalarConfig) -> torch.Tensor:
    _validate_packed_element_density_config(config, "packed_element_density_descriptors")
    _, distances, _ = compute_pair_geometry(graph)
    radial = compute_radial_features(distances, config)
    src, dst = graph.edge_index
    neighbor_z = graph.z[src].to(dtype=graph.pos.dtype, device=graph.pos.device) / float(config.max_atomic_number)
    edge_descriptors = torch.cat([radial, radial * neighbor_z[:, None]], dim=-1)
    return scatter_sum(edge_descriptors, dst, graph.z.shape[0])


def cell_list_packed_element_density_descriptors(
    graph: RTECEGraph,
    config: RTECEScalarConfig,
) -> torch.Tensor:
    _validate_packed_element_density_config(config, "cell_list_packed_element_density_descriptors")
    num_nodes = graph.z.shape[0]
    descriptors = graph.pos.new_zeros((num_nodes, 2 * config.num_radial))
    src, dst = _cell_list_directed_edges_nopbc(graph, float(config.cutoff))
    if src.numel() == 0:
        return descriptors

    distances = (graph.pos[dst] - graph.pos[src]).norm(dim=-1).clamp_min(1e-12)
    radial = compute_radial_features(distances, config)
    neighbor_z = graph.z[src].to(dtype=graph.pos.dtype, device=graph.pos.device) / float(config.max_atomic_number)
    edge_descriptors = torch.cat([radial, radial * neighbor_z[:, None]], dim=-1)
    descriptors.index_add_(0, dst, edge_descriptors)
    return descriptors


def atomic_scalar_descriptors(graph: RTECEGraph, config: RTECEScalarConfig) -> torch.Tensor:
    moments = compute_atomic_moments(graph, config)
    density = moments["density"]
    vector_norm = (moments["vector"] ** 2).sum(dim=-1)
    quadrupole_norm = (moments["quadrupole"] ** 2).sum(dim=(-1, -2))
    if config.scalar_path_ids is not None:
        return _atomic_scalar_path_descriptors(
            config.scalar_path_ids,
            density=density,
            element_density=moments["element_density"],
            species_density=moments["species_density"],
            vector_norm=vector_norm,
            quadrupole_norm=quadrupole_norm,
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



def _project_radial_edge_channels(channels: torch.Tensor, num_sketches: int) -> torch.Tensor:
    if num_sketches <= 0:
        return channels.mean(dim=1)
    if channels.ndim < 2:
        raise ValueError("radial edge channels must include a radial dimension")
    num_radial = int(channels.shape[1])
    if num_sketches > num_radial:
        raise ValueError("radial edge sketch channels cannot exceed num_radial")
    chunks = torch.tensor_split(channels, int(num_sketches), dim=1)
    return torch.stack([chunk.mean(dim=1) for chunk in chunks], dim=1)


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


def edge_relational_sketches(graph: RTECEGraph, config: RTECEScalarConfig) -> torch.Tensor:
    if config.num_edge_sketches <= 0:
        return graph.pos.new_zeros((graph.z.shape[0], 0))

    _, distances, unit = compute_pair_geometry(graph)
    radial = compute_radial_features(distances, config)
    moments = compute_atomic_moments(graph, config)
    src, dst = graph.edge_index
    vector_channels_i = moments["vector"][dst]
    vector_channels_j = moments["vector"][src]
    quadrupole_channels_i = moments["quadrupole"][dst]
    quadrupole_channels_j = moments["quadrupole"][src]
    if config.use_cavity_edge_sketches:
        quad_unit = unit[:, :, None] * unit[:, None, :] - torch.eye(
            3,
            device=graph.pos.device,
            dtype=graph.pos.dtype,
        )[None, :, :] / 3.0
        edge_vector = radial[:, :, None] * unit[:, None, :]
        edge_quadrupole = radial[:, :, None, None] * quad_unit[:, None, :, :]
        vector_channels_i = vector_channels_i - edge_vector
        quadrupole_channels_i = quadrupole_channels_i - edge_quadrupole
        num_nodes = graph.z.shape[0]
        edge_codes = src * num_nodes + dst
        reverse_codes = dst * num_nodes + src
        has_reverse = torch.isin(reverse_codes, edge_codes).to(dtype=graph.pos.dtype, device=graph.pos.device)
        vector_channels_j = vector_channels_j + has_reverse[:, None, None] * edge_vector
        quadrupole_channels_j = quadrupole_channels_j - has_reverse[:, None, None, None] * edge_quadrupole
    if config.radial_edge_sketch_channels:
        k_radial = int(config.radial_edge_sketch_channels)
        vi = _project_radial_edge_channels(vector_channels_i, k_radial)
        vj = _project_radial_edge_channels(vector_channels_j, k_radial)
        qi = _project_radial_edge_channels(quadrupole_channels_i, k_radial)
        qj = _project_radial_edge_channels(quadrupole_channels_j, k_radial)
        base = _radial_edge_relational_base(vi, vj, qi, qj, unit, radial)
    else:
        vi = vector_channels_i.mean(dim=1)
        vj = vector_channels_j.mean(dim=1)
        qi = quadrupole_channels_i.mean(dim=1)
        qj = quadrupole_channels_j.mean(dim=1)
        base = torch.stack(
            [
                (vi * vj).sum(dim=-1),
                (qi * qj).sum(dim=(-1, -2)),
                (unit * vi).sum(dim=-1),
                (unit * vj).sum(dim=-1),
                torch.einsum("bi,bij,bj->b", unit, qi, unit),
                torch.einsum("bi,bij,bj->b", unit, qj, unit),
                radial[:, 0],
                radial[:, min(1, radial.shape[1] - 1)],
            ],
            dim=-1,
        )
    if config.num_edge_sketches > base.shape[1]:
        repeats = math.ceil(config.num_edge_sketches / base.shape[1])
        base = base.repeat(1, repeats)
    edge_values = base[:, : config.num_edge_sketches] * cutoff_envelope(distances, config.cutoff)[:, None]
    return scatter_sum(edge_values, dst, graph.z.shape[0])


def rtece_descriptors(graph: RTECEGraph, config: RTECEScalarConfig) -> torch.Tensor:
    atomic = atomic_scalar_descriptors(graph, config)
    sketches = edge_relational_sketches(graph, config)
    return torch.cat([atomic, sketches], dim=-1)



class RTECEScalarModel(torch.nn.Module):
    def __init__(self, config: RTECEScalarConfig):
        super().__init__()
        self.config = config
        in_dim = descriptor_dim(config)
        layers: list[torch.nn.Module] = []
        prev = in_dim + 1
        for hidden in config.hidden_channels:
            layers.append(torch.nn.Linear(prev, hidden))
            layers.append(torch.nn.SiLU())
            prev = hidden
        layers.append(torch.nn.Linear(prev, 1))
        self.energy_head = torch.nn.Sequential(*layers)

    def _require_inference_mode(self, backend_name: str) -> None:
        if self.training:
            raise RuntimeError(
                f"{backend_name} is an inference-only rTECE force backend; "
                "use model.eval() for benchmark/inference or model(graph) for force training."
            )

    def forward_density_analytic_forces(self, graph: RTECEGraph) -> dict[str, torch.Tensor]:
        self._require_inference_mode("forward_density_analytic_forces")
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
        self._require_inference_mode("forward_pair_analytic_forces")
        if self.config.use_density_quadratic:
            raise ValueError("forward_pair_analytic_forces only supports pure rtece_pair descriptors")
        return self.forward_density_analytic_forces(graph)

    def forward_pair_triton_force_analytic_forces(self, graph: RTECEGraph) -> dict[str, torch.Tensor]:
        self._require_inference_mode("forward_pair_triton_force_analytic_forces")
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
        self._require_inference_mode("forward_element_density_triton_force_analytic_forces")
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
        self._require_inference_mode("forward_element_density_triton_descriptor_force_analytic_forces")
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
        self._require_inference_mode("forward_element_density_direct_padded_triton_descriptor_force_analytic_forces")
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
        self._require_inference_mode("forward_element_density_cell_list_packed_analytic_forces")
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
        self._require_inference_mode("forward_element_density_packed_analytic_forces")
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

    def forward(self, graph: RTECEGraph) -> dict[str, torch.Tensor]:
        pos = graph.pos
        if not pos.requires_grad:
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
        descriptors = rtece_descriptors(graph, self.config)
        atomic_input = torch.cat([z_scaled, descriptors], dim=-1)
        atomic_energy = self.energy_head(atomic_input).squeeze(-1)
        num_graphs = int(graph.batch.max().item()) + 1 if graph.batch.numel() else 1
        energy = scatter_sum(atomic_energy[:, None], graph.batch, num_graphs).squeeze(-1)
        energy = add_atomic_reference_energy(energy, graph, self.config)
        forces = -torch.autograd.grad(
            energy.sum(),
            pos,
            create_graph=self.training,
            retain_graph=True,
        )[0]
        return {"energy": energy, "atomic_energy": atomic_energy, "forces": forces}


__all__ = [
    "RTECEScalarConfig",
    "RTECEGraph",
    "RTECEScalarModel",
    "available_rtece_variants",
    "build_rtece_config",
    "build_rtece_config_from_manifest",
    "build_rtece_config_from_path_ids",
    "descriptor_dim",
    "collate_graphs",
    "compute_pair_geometry",
    "atomic_reference_energy",
    "add_atomic_reference_energy",
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
