from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
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
    num_edge_sketches: int = 0
    energy_per_atom_shift: float = 0.0
    atomic_energies: Mapping[int, float] | None = None


def build_rtece_config(variant: str) -> RTECEScalarConfig:
    if variant == "rtece_pair":
        return RTECEScalarConfig(variant=variant)
    if variant == "rtece_element_density":
        return RTECEScalarConfig(variant=variant, use_element_density=True)
    if variant == "rtece_density_quadratic":
        return RTECEScalarConfig(variant=variant, use_density_quadratic=True)
    if variant == "rtece_vector_moments":
        return RTECEScalarConfig(variant=variant, use_vector_moments=True)
    if variant == "rtece_atomic_moments":
        return RTECEScalarConfig(variant=variant, use_atomic_moments=True)
    if variant == "rtece_edge_sketch8":
        return RTECEScalarConfig(variant=variant, use_atomic_moments=True, num_edge_sketches=8)
    if variant == "rtece_edge_sketch16":
        return RTECEScalarConfig(variant=variant, use_atomic_moments=True, num_edge_sketches=16)
    raise ValueError(f"unknown rTECE scalar variant {variant!r}")


def descriptor_dim(config: RTECEScalarConfig) -> int:
    dim = config.num_radial
    if config.use_element_density:
        dim += config.num_radial
    if config.use_density_quadratic:
        dim += config.num_radial
    if config.use_vector_moments:
        dim += config.num_radial
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
    elif config.use_atomic_moments or config.num_edge_sketches:
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


@dataclass
class RTECEGraph:
    z: torch.Tensor
    pos: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor


def collate_graphs(graphs: list[RTECEGraph]) -> RTECEGraph:
    if not graphs:
        raise ValueError("collate_graphs requires at least one graph")
    z_parts = []
    pos_parts = []
    edge_parts = []
    batch_parts = []
    node_offset = 0
    for graph_idx, graph in enumerate(graphs):
        num_nodes = graph.z.shape[0]
        z_parts.append(graph.z)
        pos_parts.append(graph.pos)
        batch_parts.append(torch.full_like(graph.batch, graph_idx))
        if graph.edge_index.numel() > 0:
            edge_parts.append(graph.edge_index + node_offset)
        node_offset += num_nodes
    if edge_parts:
        edge_index = torch.cat(edge_parts, dim=1)
    else:
        edge_index = graphs[0].edge_index.new_zeros((2, 0))
    return RTECEGraph(
        z=torch.cat(z_parts, dim=0),
        pos=torch.cat(pos_parts, dim=0),
        edge_index=edge_index,
        batch=torch.cat(batch_parts, dim=0),
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
    return {
        "density": density,
        "element_density": element_density,
        "vector": vector,
        "quadrupole": quadrupole,
    }


def density_scalar_descriptors(
    density: torch.Tensor,
    config: RTECEScalarConfig,
    element_density: torch.Tensor | None = None,
    vector_norm: torch.Tensor | None = None,
) -> torch.Tensor:
    parts = [density]
    if config.use_element_density:
        if element_density is None:
            raise ValueError("element_density is required when use_element_density=True")
        parts.append(element_density)
    if config.use_density_quadratic:
        parts.append(density.square())
    if config.use_vector_moments:
        if vector_norm is None:
            raise ValueError("vector_norm is required when use_vector_moments=True")
        parts.append(vector_norm)
    return torch.cat(parts, dim=-1) if len(parts) > 1 else density


def _validate_packed_element_density_config(config: RTECEScalarConfig, name: str) -> None:
    if not config.use_element_density:
        raise ValueError(f"{name} requires use_element_density=True")
    if config.use_density_quadratic or config.use_vector_moments:
        raise ValueError(f"{name} only supports density plus element density")
    if config.use_atomic_moments or config.num_edge_sketches:
        raise ValueError(f"{name} only supports scalar density descriptors")


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
    density_desc = density_scalar_descriptors(
        density,
        config,
        moments["element_density"],
        vector_norm,
    )
    if not config.use_atomic_moments:
        return density_desc
    quadrupole_norm = (moments["quadrupole"] ** 2).sum(dim=(-1, -2))
    return torch.cat([density_desc, vector_norm, quadrupole_norm], dim=-1)



def edge_relational_sketches(graph: RTECEGraph, config: RTECEScalarConfig) -> torch.Tensor:
    if config.num_edge_sketches <= 0:
        return graph.pos.new_zeros((graph.z.shape[0], 0))

    _, distances, unit = compute_pair_geometry(graph)
    radial = compute_radial_features(distances, config)
    moments = compute_atomic_moments(graph, config)
    src, dst = graph.edge_index
    vi = moments["vector"][dst].mean(dim=1)
    vj = moments["vector"][src].mean(dim=1)
    qi = moments["quadrupole"][dst].mean(dim=1)
    qj = moments["quadrupole"][src].mean(dim=1)
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
            graph = RTECEGraph(z=graph.z, pos=pos, edge_index=graph.edge_index, batch=graph.batch)

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
    "build_rtece_config",
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
    "edge_relational_sketches",
    "rtece_descriptors",
    "rtece_route_contract",
]
