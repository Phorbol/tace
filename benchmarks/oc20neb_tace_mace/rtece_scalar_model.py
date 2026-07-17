from __future__ import annotations

from dataclasses import dataclass
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


def compute_radial_features(distances: torch.Tensor, config: RTECEScalarConfig) -> torch.Tensor:
    features, _ = radial_features_and_derivatives(distances, config)
    return features


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

    def forward_density_analytic_forces(self, graph: RTECEGraph) -> dict[str, torch.Tensor]:
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
        if self.config.energy_per_atom_shift:
            atom_counts = scatter_sum(
                torch.ones_like(atomic_energy[:, None]),
                graph.batch,
                num_graphs,
            ).squeeze(-1)
            energy = energy + atom_counts * pos.new_tensor(float(self.config.energy_per_atom_shift))

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
        if self.config.use_density_quadratic:
            raise ValueError("forward_pair_analytic_forces only supports pure rtece_pair descriptors")
        return self.forward_density_analytic_forces(graph)

    def forward_element_density_packed_analytic_forces(self, graph: RTECEGraph) -> dict[str, torch.Tensor]:
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
        if self.config.energy_per_atom_shift:
            atom_counts = scatter_sum(
                torch.ones_like(atomic_energy[:, None]),
                graph.batch,
                num_graphs,
            ).squeeze(-1)
            energy = energy + atom_counts * pos.new_tensor(float(self.config.energy_per_atom_shift))

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
        if self.config.energy_per_atom_shift:
            atom_counts = scatter_sum(
                torch.ones_like(atomic_energy[:, None]),
                graph.batch,
                num_graphs,
            ).squeeze(-1)
            shift = pos.new_tensor(float(self.config.energy_per_atom_shift))
            energy = energy + atom_counts * shift
        forces = -torch.autograd.grad(
            energy.sum(),
            pos,
            create_graph=self.training,
            retain_graph=True,
        )[0]
        return {"energy": energy, "atomic_energy": atomic_energy, "forces": forces}
