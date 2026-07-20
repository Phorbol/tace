#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from benchmarks.oc20neb_tace_mace.benchmark_models import (
    cuda_memory,
    log,
    reference_arrays,
    summarize_errors,
)
from benchmarks.oc20neb_tace_mace.relative_energy_metrics import (
    atoms_group_values,
    relative_energy_group_metrics,
)
from benchmarks.oc20neb_tace_mace.rtece_scalar_model import (
    RTECEGraph,
    RTECEScalarConfig,
    collate_graphs,
    rtece_path_manifest,
    rtece_route_contract,
)
from benchmarks.oc20neb_tace_mace.train_rtece_scalar import atoms_to_graph, atoms_to_rtece_graph, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a scalar-sketched rTECE prototype.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--default-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--start-config", type=int, default=0)
    parser.add_argument("--limit-configs", type=int, default=128)
    parser.add_argument("--measure-passes", type=int, default=3)
    parser.add_argument(
        "--force-mode",
        choices=("auto", "autograd", "analytic_pair", "analytic_pair_triton_force", "analytic_element_triton_force", "analytic_element_triton_descriptor_force", "analytic_element_direct_padded_descriptor_force", "analytic_element_cell_list_descriptor_force", "analytic_density", "analytic_element_packed"),
        default="auto",
    )
    parser.add_argument(
        "--include-graph-construction",
        action="store_true",
        help="Include ASE neighbor-list graph construction inside the timed loop.",
    )
    parser.add_argument(
        "--batch-graph-construction",
        action="store_true",
        help="When graph construction is timed, rebuild all graphs and collate before one batched model pass.",
    )
    parser.add_argument(
        "--replay-cached-graph",
        action="store_true",
        help="Prebuild the graph once, then time position refresh plus one batched model pass.",
    )
    parser.add_argument(
        "--trajectory-replay-steps",
        type=int,
        default=0,
        help="Run a synthetic MD-style cached-graph trajectory with this many force steps per timed pass.",
    )
    parser.add_argument(
        "--trajectory-rebuild-interval",
        type=int,
        default=0,
        help="In trajectory replay mode, rebuild the batched graph every K force steps; 0 disables rebuilds.",
    )
    parser.add_argument(
        "--trajectory-displacement-std",
        type=float,
        default=0.0,
        help="Deterministic per-step synthetic displacement scale in Angstrom for trajectory replay.",
    )
    parser.add_argument(
        "--trajectory-skin-margin",
        type=float,
        default=0.0,
        help="In trajectory replay mode, rebuild when max displacement since graph build exceeds half this skin margin.",
    )
    parser.add_argument(
        "--trajectory-validity-only",
        action="store_true",
        help="In trajectory replay mode, time only position generation plus graph-cache validity checks, without model evaluation or graph rebuilds.",
    )
    parser.add_argument(
        "--trajectory-update-only",
        action="store_true",
        help="In trajectory replay mode, time validity checks plus provider graph updates/rebuilds, without model evaluation.",
    )
    parser.add_argument(
        "--graph-update-backend",
        choices=(
            "ase_neighborlist",
            "cached_topology",
            "torch_radius_nopbc",
            "torch_radius_nopbc_grouped",
            "torch_radius_nopbc_grouped_chunked",
            "torch_radius_nopbc_grouped_by_size",
            "torch_radius_nopbc_ragged",
            "torch_radius_nopbc_triton_padded",
            "torch_radius_nopbc_triton_counted",
        ),
        default="ase_neighborlist",
        help="Graph update backend used when trajectory replay invalidates the cached graph.",
    )
    parser.add_argument(
        "--graph-update-chunk-configs",
        type=int,
        default=128,
        help="Number of configurations per chunk for chunked graph update backends.",
    )
    parser.add_argument(
        "--graph-construction-backend",
        choices=("ase_neighborlist", "matscipy_neighborlist", "torch_radius_nopbc"),
        default="ase_neighborlist",
        help="Graph construction backend for initial and timed atom-to-graph builds.",
    )
    return parser.parse_args()


def extxyz_index(*, start_config: int, limit_configs: int | None) -> str:
    if start_config < 0:
        raise ValueError(f"start_config must be non-negative, got {start_config}")
    if limit_configs is not None and limit_configs < 1:
        raise ValueError(f"limit_configs must be positive when set, got {limit_configs}")
    if limit_configs is None:
        return ":" if start_config == 0 else f"{start_config}:"
    stop_config = start_config + limit_configs
    return f":{stop_config}" if start_config == 0 else f"{start_config}:{stop_config}"


def choose_rtece_force_mode(
    requested: str,
    config: RTECEScalarConfig,
    *,
    device_type: str,
    dtype: torch.dtype,
) -> str:
    if requested != "auto":
        return requested
    eligible_element_density = (
        bool(config.use_element_density)
        and not bool(config.use_density_quadratic)
        and not bool(config.use_vector_moments)
        and not bool(config.use_atomic_moments)
        and int(config.num_edge_sketches) == 0
        and device_type == "cuda"
        and dtype == torch.float32
    )
    if eligible_element_density:
        return "analytic_element_triton_descriptor_force"
    return "autograd"


def validate_graph_construction_args(
    *,
    include_graph_construction: bool,
    batch_graph_construction: bool,
    replay_cached_graph: bool,
    trajectory_replay_steps: int = 0,
    trajectory_rebuild_interval: int = 0,
    trajectory_skin_margin: float = 0.0,
    trajectory_validity_only: bool = False,
    trajectory_update_only: bool = False,
) -> None:
    if trajectory_replay_steps < 0:
        raise ValueError(f"--trajectory-replay-steps must be non-negative, got {trajectory_replay_steps}")
    if trajectory_rebuild_interval < 0:
        raise ValueError(f"--trajectory-rebuild-interval must be non-negative, got {trajectory_rebuild_interval}")
    if trajectory_skin_margin < 0.0:
        raise ValueError(f"--trajectory-skin-margin must be non-negative, got {trajectory_skin_margin}")
    if batch_graph_construction and not include_graph_construction:
        raise ValueError("--batch-graph-construction requires --include-graph-construction")
    if replay_cached_graph and include_graph_construction:
        raise ValueError("--replay-cached-graph is separate from --include-graph-construction timing")
    if replay_cached_graph and batch_graph_construction:
        raise ValueError("--replay-cached-graph cannot be combined with --batch-graph-construction")
    if trajectory_replay_steps > 0:
        if include_graph_construction or batch_graph_construction or replay_cached_graph:
            raise ValueError("--trajectory-replay-steps is separate from graph-construction and cached-graph replay modes")
        if trajectory_validity_only and trajectory_rebuild_interval:
            raise ValueError("--trajectory-validity-only cannot be combined with --trajectory-rebuild-interval")
        if trajectory_validity_only and trajectory_update_only:
            raise ValueError("--trajectory-validity-only cannot be combined with --trajectory-update-only")
        if trajectory_rebuild_interval == 0:
            return
        if trajectory_rebuild_interval < 1:
            raise ValueError("--trajectory-rebuild-interval must be positive when enabled")
    elif trajectory_rebuild_interval:
        raise ValueError("--trajectory-rebuild-interval requires --trajectory-replay-steps")
    elif trajectory_skin_margin > 0.0:
        raise ValueError("--trajectory-skin-margin requires --trajectory-replay-steps")
    elif trajectory_validity_only:
        raise ValueError("--trajectory-validity-only requires --trajectory-replay-steps")
    elif trajectory_update_only:
        raise ValueError("--trajectory-update-only requires --trajectory-replay-steps")


class GraphUpdateBackend:
    def __init__(self, name: str, rebuild_fn, *, static_metadata: dict[str, object] | None = None):
        self.name = str(name)
        self._rebuild_fn = rebuild_fn
        self.static_metadata = dict(static_metadata or {})
        self.last_metadata = dict(self.static_metadata)
        self.rebuild_count = 0
        self.rebuild_times_s: list[float] = []
        self.total_rebuild_time_s = 0.0

    def rebuild(self, positions: torch.Tensor):
        start = time.perf_counter()
        result = self._rebuild_fn(positions)
        elapsed = time.perf_counter() - start
        self.rebuild_count += 1
        self.rebuild_times_s.append(float(elapsed))
        self.total_rebuild_time_s += float(elapsed)
        metadata = dict(self.static_metadata)
        if isinstance(result, RTECEGraph):
            metadata["num_directed_edges"] = int(result.edge_index.shape[1])
        self.last_metadata = metadata
        return result


def direct_radius_backend_static_metadata(
    template: RTECEGraph,
    *,
    backend_name: str,
    chunk_configs: int = 128,
) -> dict[str, object]:
    batch = template.batch
    if batch.ndim != 1 or batch.shape[0] != template.z.shape[0]:
        raise ValueError("template batch must be a one-dimensional tensor with one entry per atom")
    if batch.numel() == 0:
        return {
            "provider_family": "direct_radius_nopbc",
            "num_configs": 0,
            "num_atoms": 0,
            "max_atoms_per_config": 0,
            "num_chunks": 0,
            "max_chunk_configs": 0,
            "exact_pair_slots": 0,
            "padded_pair_slots": 0,
            "padding_overhead_ratio": 0.0,
        }
    if torch.any(batch[1:] < batch[:-1]):
        raise ValueError("template batch must be sorted by configuration")
    _, counts = torch.unique_consecutive(batch, return_counts=True)
    counts_list = [int(v) for v in counts.detach().cpu().tolist()]
    num_configs = len(counts_list)
    exact_pair_slots = int(sum(count * count for count in counts_list))
    max_atoms = int(max(counts_list))

    if backend_name == "torch_radius_nopbc":
        num_chunks = num_configs
        max_chunk_configs = 1 if num_configs else 0
        padded_pair_slots = exact_pair_slots
    elif backend_name == "torch_radius_nopbc_ragged":
        num_chunks = 1 if num_configs else 0
        max_chunk_configs = num_configs
        padded_pair_slots = exact_pair_slots
    elif backend_name == "torch_radius_nopbc_grouped":
        num_chunks = 1 if num_configs else 0
        max_chunk_configs = num_configs
        padded_pair_slots = int(num_configs * max_atoms * max_atoms)
    elif backend_name in {"torch_radius_nopbc_triton_padded", "torch_radius_nopbc_triton_counted"}:
        num_chunks = 1 if num_configs else 0
        max_chunk_configs = num_configs
        padded_pair_slots = int(num_configs * max_atoms * max_atoms)
    elif backend_name == "torch_radius_nopbc_grouped_chunked":
        if chunk_configs < 1:
            raise ValueError(f"chunk_configs must be positive, got {chunk_configs}")
        num_chunks = 0
        max_chunk_configs = 0
        padded_pair_slots = 0
        chunk_size = int(chunk_configs)
        for chunk_start in range(0, num_configs, chunk_size):
            chunk_counts = counts_list[chunk_start : chunk_start + chunk_size]
            chunk_len = len(chunk_counts)
            chunk_max = max(chunk_counts)
            num_chunks += 1
            max_chunk_configs = max(max_chunk_configs, chunk_len)
            padded_pair_slots += int(chunk_len * chunk_max * chunk_max)
    elif backend_name == "torch_radius_nopbc_grouped_by_size":
        unique_counts = sorted(set(counts_list))
        num_chunks = len(unique_counts)
        max_chunk_configs = max(counts_list.count(count) for count in unique_counts) if unique_counts else 0
        padded_pair_slots = int(sum(counts_list.count(count) * count * count for count in unique_counts))
    else:
        raise ValueError(f"unknown direct-radius backend for metadata: {backend_name}")

    return {
        "provider_family": "direct_radius_nopbc",
        "num_configs": int(num_configs),
        "num_atoms": int(batch.numel()),
        "max_atoms_per_config": int(max_atoms),
        "num_chunks": int(num_chunks),
        "max_chunk_configs": int(max_chunk_configs),
        "exact_pair_slots": int(exact_pair_slots),
        "padded_pair_slots": int(padded_pair_slots),
        "padding_overhead_ratio": float(padded_pair_slots / exact_pair_slots) if exact_pair_slots else 0.0,
    }


def cell_list_oracle_work_metadata(template: RTECEGraph, positions: torch.Tensor, *, cutoff: float) -> dict[str, object]:
    if cutoff <= 0.0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    if tuple(positions.shape) != tuple(template.pos.shape):
        raise ValueError(f"positions shape {tuple(positions.shape)} does not match graph shape {tuple(template.pos.shape)}")
    batch = template.batch
    if batch.ndim != 1 or batch.shape[0] != template.z.shape[0]:
        raise ValueError("template batch must be a one-dimensional tensor with one entry per atom")
    if batch.numel() == 0:
        return {
            "provider_family": "direct_radius_cell_list_oracle",
            "num_configs": 0,
            "num_atoms": 0,
            "max_atoms_per_config": 0,
            "exact_pair_slots": 0,
            "padded_pair_slots": 0,
            "all_pair_nonself_slots": 0,
            "cell_candidate_directed_pairs": 0,
            "active_directed_edges": 0,
            "max_cell_occupancy": 0,
            "candidate_to_padded_ratio": 0.0,
            "candidate_to_all_pair_nonself_ratio": 0.0,
            "candidate_to_active_ratio": 0.0,
            "active_to_padded_ratio": 0.0,
        }
    if torch.any(batch[1:] < batch[:-1]):
        raise ValueError("template batch must be sorted by configuration")

    pos_cpu = positions.detach().to(device="cpu")
    batch_cpu = batch.detach().to(device="cpu")
    _, counts = torch.unique_consecutive(batch_cpu, return_counts=True)
    starts = torch.cat([batch_cpu.new_zeros(1), counts.cumsum(dim=0)[:-1]])
    counts_list = [int(v) for v in counts.tolist()]
    max_atoms = max(counts_list) if counts_list else 0
    exact_pair_slots = int(sum(count * count for count in counts_list))
    padded_pair_slots = int(len(counts_list) * max_atoms * max_atoms) if counts_list else 0
    all_pair_nonself_slots = int(sum(count * (count - 1) for count in counts_list))
    cutoff_sq = float(cutoff) * float(cutoff)

    cell_candidate_directed_pairs = 0
    active_directed_edges = 0
    max_cell_occupancy = 0
    for start_tensor, count_tensor in zip(starts, counts, strict=True):
        start = int(start_tensor.item())
        count = int(count_tensor.item())
        if count <= 1:
            max_cell_occupancy = max(max_cell_occupancy, count)
            continue
        block = pos_cpu[start : start + count]
        origin = block.min(dim=0).values
        cell_coords = torch.floor((block - origin) / float(cutoff)).to(dtype=torch.long)
        _, cell_counts = torch.unique(cell_coords, dim=0, return_counts=True)
        if cell_counts.numel() > 0:
            max_cell_occupancy = max(max_cell_occupancy, int(cell_counts.max().item()))

        cell_delta = torch.abs(cell_coords[:, None, :] - cell_coords[None, :, :])
        candidate_mask = torch.all(cell_delta <= 1, dim=-1)
        candidate_mask.fill_diagonal_(False)
        cell_candidate_directed_pairs += int(candidate_mask.sum().item())

        delta = block[:, None, :] - block[None, :, :]
        active_mask = delta.square().sum(dim=-1) < cutoff_sq
        active_mask.fill_diagonal_(False)
        active_directed_edges += int(active_mask.sum().item())

    return {
        "provider_family": "direct_radius_cell_list_oracle",
        "num_configs": int(len(counts_list)),
        "num_atoms": int(batch.numel()),
        "max_atoms_per_config": int(max_atoms),
        "exact_pair_slots": int(exact_pair_slots),
        "padded_pair_slots": int(padded_pair_slots),
        "all_pair_nonself_slots": int(all_pair_nonself_slots),
        "cell_candidate_directed_pairs": int(cell_candidate_directed_pairs),
        "active_directed_edges": int(active_directed_edges),
        "max_cell_occupancy": int(max_cell_occupancy),
        "candidate_to_padded_ratio": float(cell_candidate_directed_pairs / padded_pair_slots) if padded_pair_slots else 0.0,
        "candidate_to_all_pair_nonself_ratio": (
            float(cell_candidate_directed_pairs / all_pair_nonself_slots) if all_pair_nonself_slots else 0.0
        ),
        "candidate_to_active_ratio": float(cell_candidate_directed_pairs / active_directed_edges) if active_directed_edges else 0.0,
        "active_to_padded_ratio": float(active_directed_edges / padded_pair_slots) if padded_pair_slots else 0.0,
    }


def make_graph_update_backend(
    *,
    backend_name: str,
    rebuild_fn,
    template_graph: RTECEGraph | None = None,
    cutoff: float | None = None,
    chunk_configs: int = 128,
) -> GraphUpdateBackend:
    if backend_name == "ase_neighborlist":
        return GraphUpdateBackend("ase_neighborlist", rebuild_fn)
    if backend_name == "cached_topology":
        if template_graph is None:
            raise ValueError("cached_topology graph update backend requires a template graph")
        return GraphUpdateBackend(
            "cached_topology",
            lambda positions: replay_graph_positions(template_graph, positions),
            static_metadata={
                "provider_family": "cached_topology",
                "num_atoms": int(template_graph.pos.shape[0]),
                "cached_num_directed_edges": int(template_graph.edge_index.shape[1]),
            },
        )
    if backend_name in {
        "torch_radius_nopbc",
        "torch_radius_nopbc_grouped",
        "torch_radius_nopbc_grouped_chunked",
        "torch_radius_nopbc_grouped_by_size",
        "torch_radius_nopbc_ragged",
        "torch_radius_nopbc_triton_padded",
        "torch_radius_nopbc_triton_counted",
    }:
        if template_graph is None:
            raise ValueError(f"{backend_name} graph update backend requires a template graph")
        if cutoff is None:
            raise ValueError(f"{backend_name} graph update backend requires a cutoff")
        radius_fns = {
            "torch_radius_nopbc": torch_radius_nopbc_graph,
            "torch_radius_nopbc_grouped": torch_radius_nopbc_grouped_graph,
            "torch_radius_nopbc_grouped_by_size": torch_radius_nopbc_grouped_by_size_graph,
            "torch_radius_nopbc_ragged": torch_radius_nopbc_ragged_graph,
            "torch_radius_nopbc_triton_padded": torch_radius_nopbc_triton_padded_graph,
            "torch_radius_nopbc_triton_counted": torch_radius_nopbc_triton_counted_graph,
        }
        if backend_name == "torch_radius_nopbc_grouped_chunked":
            return GraphUpdateBackend(
                backend_name,
                lambda positions: torch_radius_nopbc_grouped_chunked_graph(
                    template_graph,
                    positions,
                    cutoff=float(cutoff),
                    chunk_configs=int(chunk_configs),
                ),
                static_metadata=direct_radius_backend_static_metadata(
                    template_graph,
                    backend_name=backend_name,
                    chunk_configs=int(chunk_configs),
                ),
            )
        radius_fn = radius_fns[backend_name]
        return GraphUpdateBackend(
            backend_name,
            lambda positions: radius_fn(template_graph, positions, cutoff=float(cutoff)),
            static_metadata=direct_radius_backend_static_metadata(
                template_graph,
                backend_name=backend_name,
                chunk_configs=int(chunk_configs),
            ),
        )
    raise ValueError(f"unknown graph update backend: {backend_name}")


class TrajectoryGraphCacheProvider:
    def __init__(self, reference_positions: torch.Tensor, *, skin_margin: float):
        self.reference_positions = reference_positions.detach().clone()
        self.skin_margin = float(skin_margin)
        self.rebuild_count = 0
        self.last_rebuild_step: int | None = None
        self.rebuild_steps: list[int] = []
        self.rebuild_causes: list[str] = []

    def check(self, positions: torch.Tensor) -> dict[str, object]:
        probe = graph_cache_displacement_probe(
            self.reference_positions,
            positions,
            skin_margin=self.skin_margin,
        )
        return {
            "max_displacement": probe["max_displacement"],
            "threshold": probe["threshold"],
            "needs_rebuild": probe["rebuild_required"],
        }

    def mark_rebuilt(self, positions: torch.Tensor, *, step: int | None = None, cause: str | None = None) -> None:
        self.reference_positions = positions.detach().clone()
        self.rebuild_count += 1
        self.last_rebuild_step = step
        if step is not None:
            self.rebuild_steps.append(int(step))
            self.rebuild_causes.append(str(cause or "unspecified"))


def graph_cache_displacement_probe(
    reference_positions: torch.Tensor,
    positions: torch.Tensor,
    *,
    skin_margin: float,
) -> dict[str, object]:
    if tuple(reference_positions.shape) != tuple(positions.shape):
        raise ValueError(
            f"reference positions shape {tuple(reference_positions.shape)} does not match positions shape {tuple(positions.shape)}"
        )
    displacement = (positions - reference_positions).norm(dim=-1)
    max_displacement = displacement.max() if displacement.numel() else reference_positions.new_tensor(0.0)
    if skin_margin <= 0.0:
        return {
            "max_displacement": max_displacement,
            "threshold": None,
            "rebuild_required": False,
        }
    threshold = 0.5 * float(skin_margin)
    return {
        "max_displacement": max_displacement,
        "threshold": threshold,
        "rebuild_required": bool(float(max_displacement.detach().cpu()) > threshold),
    }


def synthetic_trajectory_positions(
    base_positions: torch.Tensor,
    *,
    step: int,
    displacement_std: float,
) -> torch.Tensor:
    if step == 0 or displacement_std == 0.0:
        return base_positions
    atom_index = torch.arange(base_positions.shape[0], device=base_positions.device, dtype=base_positions.dtype)
    phases = atom_index[:, None] * base_positions.new_tensor([12.9898, 78.233, 37.719])
    phases = phases + float(step) * base_positions.new_tensor([0.37, 0.53, 0.71])
    return base_positions + float(displacement_std) * torch.sin(phases)


def atoms_to_geometry_graph(
    atoms,
    *,
    cutoff: float,
    device: torch.device,
    dtype: torch.dtype,
) -> RTECEGraph:
    return atoms_to_graph(atoms, cutoff=cutoff, device=device, dtype=dtype)[0]


def replay_graph_positions(template: RTECEGraph, positions: torch.Tensor) -> RTECEGraph:
    if tuple(positions.shape) != tuple(template.pos.shape):
        raise ValueError(f"cached positions shape {tuple(positions.shape)} does not match graph shape {tuple(template.pos.shape)}")
    return RTECEGraph(
        z=template.z,
        pos=positions.to(device=template.pos.device, dtype=template.pos.dtype),
        edge_index=template.edge_index,
        batch=template.batch,
        cell=template.cell,
        edge_shifts=template.edge_shifts,
        edge_batch=template.edge_batch,
    )


def torch_radius_nopbc_graph(template: RTECEGraph, positions: torch.Tensor, *, cutoff: float) -> RTECEGraph:
    if cutoff <= 0.0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    if tuple(positions.shape) != tuple(template.pos.shape):
        raise ValueError(f"positions shape {tuple(positions.shape)} does not match graph shape {tuple(template.pos.shape)}")
    pos = positions.to(device=template.pos.device, dtype=template.pos.dtype)
    batch = template.batch
    if batch.ndim != 1 or batch.shape[0] != template.z.shape[0]:
        raise ValueError("template batch must be a one-dimensional tensor with one entry per atom")
    if batch.numel() == 0:
        edge_index = template.edge_index.new_zeros((2, 0))
    else:
        changes = torch.nonzero(batch[1:] != batch[:-1], as_tuple=False).flatten() + 1
        starts = torch.cat([batch.new_tensor([0]), changes])
        stops = torch.cat([changes, batch.new_tensor([batch.numel()])])
        edge_parts = []
        cutoff_sq = float(cutoff) * float(cutoff)
        for start_tensor, stop_tensor in zip(starts, stops, strict=True):
            start = int(start_tensor.detach().cpu())
            stop = int(stop_tensor.detach().cpu())
            count = stop - start
            if count <= 1:
                continue
            block = pos[start:stop]
            delta = block[:, None, :] - block[None, :, :]
            dist_sq = delta.square().sum(dim=-1)
            mask = dist_sq < cutoff_sq
            mask.fill_diagonal_(False)
            src, dst = torch.nonzero(mask, as_tuple=True)
            if src.numel() > 0:
                edge_parts.append(torch.stack([src + start, dst + start], dim=0))
        if edge_parts:
            edge_index = torch.cat(edge_parts, dim=1).to(device=template.edge_index.device, dtype=template.edge_index.dtype)
        else:
            edge_index = template.edge_index.new_zeros((2, 0))
    return RTECEGraph(
        z=template.z,
        pos=pos,
        edge_index=edge_index,
        batch=template.batch,
    )


def torch_radius_nopbc_grouped_graph(template: RTECEGraph, positions: torch.Tensor, *, cutoff: float) -> RTECEGraph:
    if cutoff <= 0.0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    if tuple(positions.shape) != tuple(template.pos.shape):
        raise ValueError(f"positions shape {tuple(positions.shape)} does not match graph shape {tuple(template.pos.shape)}")
    pos = positions.to(device=template.pos.device, dtype=template.pos.dtype)
    batch = template.batch
    if batch.ndim != 1 or batch.shape[0] != template.z.shape[0]:
        raise ValueError("template batch must be a one-dimensional tensor with one entry per atom")
    if batch.numel() == 0:
        edge_index = template.edge_index.new_zeros((2, 0))
    else:
        if torch.any(batch[1:] < batch[:-1]):
            raise ValueError("template batch must be sorted by configuration")
        _, counts = torch.unique_consecutive(batch, return_counts=True)
        starts = torch.cat([batch.new_zeros(1), counts.cumsum(dim=0)[:-1]])
        max_count = int(counts.max().detach().cpu())
        if max_count <= 1:
            edge_index = template.edge_index.new_zeros((2, 0))
        else:
            local = torch.arange(max_count, device=pos.device)
            counts_device = counts.to(device=pos.device)
            starts_device = starts.to(device=pos.device)
            valid = local.unsqueeze(0) < counts_device.unsqueeze(1)
            flat = starts_device.unsqueeze(1) + local.unsqueeze(0)
            safe_flat = flat.clamp(max=max(pos.shape[0] - 1, 0))
            padded = pos.new_zeros((counts.shape[0], max_count, 3))
            padded[valid] = pos[safe_flat[valid]]
            delta = padded[:, :, None, :] - padded[:, None, :, :]
            dist_sq = delta.square().sum(dim=-1)
            mask = dist_sq < float(cutoff) * float(cutoff)
            mask &= valid[:, :, None] & valid[:, None, :]
            diag = torch.eye(max_count, dtype=torch.bool, device=pos.device).unsqueeze(0)
            mask &= ~diag
            graph_idx, src_local, dst_local = torch.nonzero(mask, as_tuple=True)
            if src_local.numel() > 0:
                src = starts_device[graph_idx] + src_local
                dst = starts_device[graph_idx] + dst_local
                edge_index = torch.stack([src, dst], dim=0).to(device=template.edge_index.device, dtype=template.edge_index.dtype)
            else:
                edge_index = template.edge_index.new_zeros((2, 0))
    return RTECEGraph(
        z=template.z,
        pos=pos,
        edge_index=edge_index,
        batch=template.batch,
    )


def torch_radius_nopbc_grouped_chunked_graph(
    template: RTECEGraph,
    positions: torch.Tensor,
    *,
    cutoff: float,
    chunk_configs: int = 128,
) -> RTECEGraph:
    if cutoff <= 0.0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    if chunk_configs < 1:
        raise ValueError(f"chunk_configs must be positive, got {chunk_configs}")
    if tuple(positions.shape) != tuple(template.pos.shape):
        raise ValueError(f"positions shape {tuple(positions.shape)} does not match graph shape {tuple(template.pos.shape)}")
    pos = positions.to(device=template.pos.device, dtype=template.pos.dtype)
    batch = template.batch
    if batch.ndim != 1 or batch.shape[0] != template.z.shape[0]:
        raise ValueError("template batch must be a one-dimensional tensor with one entry per atom")
    if batch.numel() == 0:
        edge_index = template.edge_index.new_zeros((2, 0))
    else:
        if torch.any(batch[1:] < batch[:-1]):
            raise ValueError("template batch must be sorted by configuration")
        _, counts = torch.unique_consecutive(batch, return_counts=True)
        starts = torch.cat([batch.new_zeros(1), counts.cumsum(dim=0)[:-1]])
        edge_parts = []
        cutoff_sq = float(cutoff) * float(cutoff)
        num_graphs = int(counts.numel())
        for chunk_start in range(0, num_graphs, int(chunk_configs)):
            chunk_stop = min(chunk_start + int(chunk_configs), num_graphs)
            counts_chunk = counts[chunk_start:chunk_stop]
            starts_chunk = starts[chunk_start:chunk_stop].to(device=pos.device)
            max_count = int(counts_chunk.max().detach().cpu())
            if max_count <= 1:
                continue
            local = torch.arange(max_count, device=pos.device)
            counts_device = counts_chunk.to(device=pos.device)
            valid = local.unsqueeze(0) < counts_device.unsqueeze(1)
            flat = starts_chunk.unsqueeze(1) + local.unsqueeze(0)
            safe_flat = flat.clamp(max=max(pos.shape[0] - 1, 0))
            padded = pos.new_zeros((chunk_stop - chunk_start, max_count, 3))
            padded[valid] = pos[safe_flat[valid]]
            delta = padded[:, :, None, :] - padded[:, None, :, :]
            dist_sq = delta.square().sum(dim=-1)
            mask = dist_sq < cutoff_sq
            mask &= valid[:, :, None] & valid[:, None, :]
            mask &= ~torch.eye(max_count, dtype=torch.bool, device=pos.device).unsqueeze(0)
            graph_idx, src_local, dst_local = torch.nonzero(mask, as_tuple=True)
            if src_local.numel() == 0:
                continue
            src = starts_chunk[graph_idx] + src_local
            dst = starts_chunk[graph_idx] + dst_local
            edge_parts.append(torch.stack([src, dst], dim=0).to(device=template.edge_index.device, dtype=template.edge_index.dtype))
        if edge_parts:
            edge_index = torch.cat(edge_parts, dim=1).to(device=template.edge_index.device, dtype=template.edge_index.dtype)
        else:
            edge_index = template.edge_index.new_zeros((2, 0))
    return RTECEGraph(
        z=template.z,
        pos=pos,
        edge_index=edge_index,
        batch=template.batch,
    )


def torch_radius_nopbc_ragged_graph(template: RTECEGraph, positions: torch.Tensor, *, cutoff: float) -> RTECEGraph:
    if cutoff <= 0.0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    if tuple(positions.shape) != tuple(template.pos.shape):
        raise ValueError(f"positions shape {tuple(positions.shape)} does not match graph shape {tuple(template.pos.shape)}")
    pos = positions.to(device=template.pos.device, dtype=template.pos.dtype)
    batch = template.batch
    if batch.ndim != 1 or batch.shape[0] != template.z.shape[0]:
        raise ValueError("template batch must be a one-dimensional tensor with one entry per atom")
    if batch.numel() == 0:
        edge_index = template.edge_index.new_zeros((2, 0))
    else:
        if torch.any(batch[1:] < batch[:-1]):
            raise ValueError("template batch must be sorted by configuration")
        _, counts = torch.unique_consecutive(batch, return_counts=True)
        starts = torch.cat([batch.new_zeros(1), counts.cumsum(dim=0)[:-1]])
        counts_device = counts.to(device=pos.device)
        starts_device = starts.to(device=pos.device)
        pair_counts = counts_device * counts_device
        total_pair_slots = int(pair_counts.sum().detach().cpu())
        if total_pair_slots == 0:
            edge_index = template.edge_index.new_zeros((2, 0))
        else:
            pair_starts = torch.cat([pair_counts.new_zeros(1), pair_counts.cumsum(dim=0)[:-1]])
            graph_ids = torch.repeat_interleave(
                torch.arange(counts_device.numel(), device=pos.device, dtype=starts_device.dtype),
                pair_counts,
            )
            pair_offsets = torch.arange(total_pair_slots, device=pos.device, dtype=starts_device.dtype)
            local_pair = pair_offsets - torch.repeat_interleave(pair_starts, pair_counts)
            graph_counts = counts_device[graph_ids]
            src_local = torch.div(local_pair, graph_counts, rounding_mode="floor")
            dst_local = local_pair - src_local * graph_counts
            src = starts_device[graph_ids] + src_local
            dst = starts_device[graph_ids] + dst_local
            delta = pos[src] - pos[dst]
            dist_sq = delta.square().sum(dim=-1)
            keep = (src_local != dst_local) & (dist_sq < float(cutoff) * float(cutoff))
            edge_index = torch.stack([src[keep], dst[keep]], dim=0).to(
                device=template.edge_index.device,
                dtype=template.edge_index.dtype,
            )
    return RTECEGraph(
        z=template.z,
        pos=pos,
        edge_index=edge_index,
        batch=template.batch,
    )


def torch_radius_nopbc_triton_padded_graph(template: RTECEGraph, positions: torch.Tensor, *, cutoff: float) -> RTECEGraph:
    if cutoff <= 0.0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    if tuple(positions.shape) != tuple(template.pos.shape):
        raise ValueError(f"positions shape {tuple(positions.shape)} does not match graph shape {tuple(template.pos.shape)}")
    pos = positions.to(device=template.pos.device, dtype=template.pos.dtype)
    if pos.device.type != "cuda" or pos.dtype != torch.float32:
        return torch_radius_nopbc_grouped_graph(template, positions, cutoff=cutoff)
    batch = template.batch
    if batch.ndim != 1 or batch.shape[0] != template.z.shape[0]:
        raise ValueError("template batch must be a one-dimensional tensor with one entry per atom")
    if batch.numel() == 0:
        edge_index = template.edge_index.new_zeros((2, 0))
    else:
        if torch.any(batch[1:] < batch[:-1]):
            raise ValueError("template batch must be sorted by configuration")
        _, counts = torch.unique_consecutive(batch, return_counts=True)
        starts = torch.cat([batch.new_zeros(1), counts.cumsum(dim=0)[:-1]])
        from benchmarks.oc20neb_tace_mace.rtece_triton_kernels import direct_radius_padded_edges_triton

        edge_index = direct_radius_padded_edges_triton(
            pos=pos,
            counts=counts.to(device=pos.device),
            starts=starts.to(device=pos.device),
            cutoff=float(cutoff),
        ).to(device=template.edge_index.device, dtype=template.edge_index.dtype)
    return RTECEGraph(
        z=template.z,
        pos=pos,
        edge_index=edge_index,
        batch=template.batch,
    )


def torch_radius_nopbc_triton_counted_graph(template: RTECEGraph, positions: torch.Tensor, *, cutoff: float) -> RTECEGraph:
    if cutoff <= 0.0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    if tuple(positions.shape) != tuple(template.pos.shape):
        raise ValueError(f"positions shape {tuple(positions.shape)} does not match graph shape {tuple(template.pos.shape)}")
    pos = positions.to(device=template.pos.device, dtype=template.pos.dtype)
    if pos.device.type != "cuda" or pos.dtype != torch.float32:
        return torch_radius_nopbc_grouped_graph(template, positions, cutoff=cutoff)
    batch = template.batch
    if batch.ndim != 1 or batch.shape[0] != template.z.shape[0]:
        raise ValueError("template batch must be a one-dimensional tensor with one entry per atom")
    if batch.numel() == 0:
        edge_index = template.edge_index.new_zeros((2, 0))
    else:
        if torch.any(batch[1:] < batch[:-1]):
            raise ValueError("template batch must be sorted by configuration")
        _, counts = torch.unique_consecutive(batch, return_counts=True)
        starts = torch.cat([batch.new_zeros(1), counts.cumsum(dim=0)[:-1]])
        from benchmarks.oc20neb_tace_mace.rtece_triton_kernels import direct_radius_counted_edges_triton

        edge_index = direct_radius_counted_edges_triton(
            pos=pos,
            counts=counts.to(device=pos.device),
            starts=starts.to(device=pos.device),
            cutoff=float(cutoff),
        ).to(device=template.edge_index.device, dtype=template.edge_index.dtype)
    return RTECEGraph(
        z=template.z,
        pos=pos,
        edge_index=edge_index,
        batch=template.batch,
    )


def torch_radius_nopbc_grouped_by_size_graph(template: RTECEGraph, positions: torch.Tensor, *, cutoff: float) -> RTECEGraph:
    if cutoff <= 0.0:
        raise ValueError(f"cutoff must be positive, got {cutoff}")
    if tuple(positions.shape) != tuple(template.pos.shape):
        raise ValueError(f"positions shape {tuple(positions.shape)} does not match graph shape {tuple(template.pos.shape)}")
    pos = positions.to(device=template.pos.device, dtype=template.pos.dtype)
    batch = template.batch
    if batch.ndim != 1 or batch.shape[0] != template.z.shape[0]:
        raise ValueError("template batch must be a one-dimensional tensor with one entry per atom")
    if batch.numel() == 0:
        edge_index = template.edge_index.new_zeros((2, 0))
    else:
        if torch.any(batch[1:] < batch[:-1]):
            raise ValueError("template batch must be sorted by configuration")
        _, counts = torch.unique_consecutive(batch, return_counts=True)
        starts = torch.cat([batch.new_zeros(1), counts.cumsum(dim=0)[:-1]])
        edge_parts_by_graph: list[torch.Tensor | None] = [None] * int(counts.numel())
        cutoff_sq = float(cutoff) * float(cutoff)
        for count_tensor in torch.unique(counts, sorted=True):
            count = int(count_tensor.detach().cpu())
            if count <= 1:
                continue
            graph_idx = torch.nonzero(counts == count_tensor, as_tuple=False).flatten()
            starts_group = starts[graph_idx].to(device=pos.device)
            local = torch.arange(count, device=pos.device)
            block_index = starts_group[:, None] + local[None, :]
            blocks = pos[block_index]
            delta = blocks[:, :, None, :] - blocks[:, None, :, :]
            dist_sq = delta.square().sum(dim=-1)
            mask = dist_sq < cutoff_sq
            mask &= ~torch.eye(count, dtype=torch.bool, device=pos.device).unsqueeze(0)
            rel_graph_idx, src_local, dst_local = torch.nonzero(mask, as_tuple=True)
            if src_local.numel() == 0:
                continue
            src = starts_group[rel_graph_idx] + src_local
            dst = starts_group[rel_graph_idx] + dst_local
            edges = torch.stack([src, dst], dim=0).to(device=template.edge_index.device, dtype=template.edge_index.dtype)
            graph_ids_cpu = graph_idx[rel_graph_idx].detach().cpu()
            for graph_id in torch.unique(graph_ids_cpu, sorted=True).tolist():
                keep = graph_ids_cpu == int(graph_id)
                edge_parts_by_graph[int(graph_id)] = edges[:, keep]
        edge_parts = [part for part in edge_parts_by_graph if part is not None and part.numel() > 0]
        if edge_parts:
            edge_index = torch.cat(edge_parts, dim=1).to(device=template.edge_index.device, dtype=template.edge_index.dtype)
        else:
            edge_index = template.edge_index.new_zeros((2, 0))
    return RTECEGraph(
        z=template.z,
        pos=pos,
        edge_index=edge_index,
        batch=template.batch,
    )


def atoms_to_torch_radius_nopbc_graph(
    atoms,
    *,
    cutoff: float,
    device: torch.device,
    dtype: torch.dtype,
) -> RTECEGraph:
    z = torch.tensor(atoms.numbers, dtype=torch.long, device=device)
    pos = torch.tensor(atoms.positions, dtype=dtype, device=device)
    template = RTECEGraph(
        z=z,
        pos=pos,
        edge_index=torch.zeros((2, 0), dtype=torch.long, device=device),
        batch=torch.zeros(len(atoms), dtype=torch.long, device=device),
    )
    return torch_radius_nopbc_graph(template, pos, cutoff=float(cutoff))


def build_atom_graph(
    atoms,
    *,
    graph_construction_backend: str,
    cutoff: float,
    device: torch.device,
    dtype: torch.dtype,
) -> RTECEGraph:
    if graph_construction_backend == "ase_neighborlist":
        return atoms_to_rtece_graph(atoms, cutoff=cutoff, device=device, dtype=dtype, neighborlist_backend="ase")
    if graph_construction_backend == "matscipy_neighborlist":
        return atoms_to_rtece_graph(atoms, cutoff=cutoff, device=device, dtype=dtype, neighborlist_backend="matscipy")
    if graph_construction_backend == "torch_radius_nopbc":
        return atoms_to_torch_radius_nopbc_graph(atoms, cutoff=cutoff, device=device, dtype=dtype)
    raise ValueError(f"unknown graph construction backend: {graph_construction_backend}")


def prediction_error_payload(
    first_outputs: list[dict[str, np.ndarray]],
    ref_e,
    ref_f,
    natoms,
    *,
    group_ids: list[str] | None = None,
    image_indices: list[float] | None = None,
) -> dict[str, object]:
    if not first_outputs:
        return {
            "prediction_errors_available": False,
            "relative_energy_errors_available": False,
            "mae_e_mev_atom": None,
            "rmse_e_mev_atom": None,
            "max_abs_e_mev_atom": None,
            "mae_f_mev_a": None,
            "rmse_f_mev_a": None,
            "max_abs_f_mev_a": None,
        }
    pred_e = np.concatenate([item["energy"].reshape(-1) for item in first_outputs], axis=0)
    pred_f = np.concatenate([item["forces"].reshape(-1, 3) for item in first_outputs], axis=0)
    payload: dict[str, object] = {
        "prediction_errors_available": True,
        "relative_energy_errors_available": False,
        **summarize_errors(pred_e, pred_f, ref_e, ref_f, natoms),
    }
    if group_ids is not None:
        relative = relative_energy_group_metrics(
            pred_e,
            ref_e,
            natoms,
            group_ids,
            image_indices=image_indices,
        )
        payload.update(relative)
        payload["relative_energy_errors_available"] = True
    return payload


def load_atoms_window(configs: Path, *, start_config: int, limit_configs: int | None):
    import ase.io

    index = extxyz_index(start_config=start_config, limit_configs=limit_configs)
    log(f"reading configs {configs} index={index}")
    atoms_list = ase.io.read(str(configs), index=index)
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    if not atoms_list:
        raise ValueError(f"no configurations read from {configs} index={index}")
    log(f"read {len(atoms_list)} configs")
    return atoms_list


def main() -> None:
    args = parse_args()
    validate_graph_construction_args(
        include_graph_construction=bool(args.include_graph_construction),
        batch_graph_construction=bool(args.batch_graph_construction),
        replay_cached_graph=bool(args.replay_cached_graph),
        trajectory_replay_steps=int(args.trajectory_replay_steps),
        trajectory_rebuild_interval=int(args.trajectory_rebuild_interval),
        trajectory_skin_margin=float(args.trajectory_skin_margin),
        trajectory_validity_only=bool(args.trajectory_validity_only),
        trajectory_update_only=bool(args.trajectory_update_only),
    )
    dtype = torch.float64 if args.default_dtype == "float64" else torch.float32
    requested = torch.device(args.device)
    device = requested if requested.type == "cpu" or torch.cuda.is_available() else torch.device("cpu")
    model, config = load_checkpoint(args.model, dtype=dtype, device=device)
    force_mode = choose_rtece_force_mode(args.force_mode, config, device_type=device.type, dtype=dtype)
    for param in model.parameters():
        param.requires_grad_(False)

    atoms_list = load_atoms_window(
        args.configs,
        start_config=args.start_config,
        limit_configs=args.limit_configs,
    )
    ref_e, ref_f, natoms = reference_arrays(atoms_list, "energy", "forces")
    prebuilt_graph = None
    replay_template = None
    replay_positions = None
    trajectory_template = None
    trajectory_base_positions = None
    trajectory_provider = None
    graph_update_backend = None
    trajectory_report_rebuild_steps: list[int] = []
    trajectory_report_rebuild_causes: list[str] = []
    trajectory_report_max_displacement = 0.0
    if not args.include_graph_construction:
        prebuilt_graph = collate_graphs(
            [
                build_atom_graph(atoms, graph_construction_backend=args.graph_construction_backend, cutoff=config.cutoff, device=device, dtype=dtype)
                for atoms in atoms_list
            ]
        )
        if args.trajectory_replay_steps > 0:
            trajectory_template = prebuilt_graph
            trajectory_base_positions = prebuilt_graph.pos.detach().clone()
            trajectory_provider = TrajectoryGraphCacheProvider(
                trajectory_base_positions,
                skin_margin=float(args.trajectory_skin_margin),
            )
            prebuilt_graph = None
        elif args.replay_cached_graph:
            replay_template = prebuilt_graph
            replay_positions = prebuilt_graph.pos.detach().clone()
            prebuilt_graph = None

    def rebuild_batched_graph_from_positions(positions: torch.Tensor) -> RTECEGraph:
        positions_cpu = positions.detach().cpu().numpy()
        rebuilt = []
        offset = 0
        for atoms, count in zip(atoms_list, natoms, strict=True):
            count_int = int(count)
            atoms_copy = atoms.copy()
            atoms_copy.positions = positions_cpu[offset : offset + count_int]
            rebuilt.append(atoms_to_geometry_graph(atoms_copy, cutoff=config.cutoff, device=device, dtype=dtype))
            offset += count_int
        return collate_graphs(rebuilt)

    graph_update_backend = make_graph_update_backend(
        backend_name=args.graph_update_backend,
        rebuild_fn=rebuild_batched_graph_from_positions,
        template_graph=trajectory_template,
        cutoff=float(config.cutoff),
        chunk_configs=int(args.graph_update_chunk_configs),
    )

    def reset_trajectory_runtime_state() -> None:
        nonlocal trajectory_provider, graph_update_backend
        if trajectory_template is None or trajectory_base_positions is None:
            return
        trajectory_provider = TrajectoryGraphCacheProvider(
            trajectory_base_positions,
            skin_margin=float(args.trajectory_skin_margin),
        )
        graph_update_backend = make_graph_update_backend(
            backend_name=args.graph_update_backend,
            rebuild_fn=rebuild_batched_graph_from_positions,
            template_graph=trajectory_template,
            cutoff=float(config.cutoff),
            chunk_configs=int(args.graph_update_chunk_configs),
        )

    def run_model(graph):
        if force_mode == "analytic_pair":
            return model.forward_pair_analytic_forces(graph)
        if force_mode == "analytic_pair_triton_force":
            return model.forward_pair_triton_force_analytic_forces(graph)
        if force_mode == "analytic_element_triton_force":
            return model.forward_element_density_triton_force_analytic_forces(graph)
        if force_mode == "analytic_element_triton_descriptor_force":
            return model.forward_element_density_triton_descriptor_force_analytic_forces(graph)
        if force_mode == "analytic_element_direct_padded_descriptor_force":
            return model.forward_element_density_direct_padded_triton_descriptor_force_analytic_forces(graph)
        if force_mode == "analytic_element_cell_list_descriptor_force":
            return model.forward_element_density_cell_list_packed_analytic_forces(graph)
        if force_mode == "analytic_density":
            return model.forward_density_analytic_forces(graph)
        if force_mode == "analytic_element_packed":
            return model.forward_element_density_packed_analytic_forces(graph)
        return model(graph)

    def forward_once(collect: bool):
        if prebuilt_graph is not None:
            out = run_model(prebuilt_graph)
            if not collect:
                return []
            return [
                {
                    "energy": out["energy"].detach().cpu().numpy(),
                    "forces": out["forces"].detach().cpu().numpy(),
                }
            ]

        if replay_template is not None:
            if replay_positions is None:
                raise RuntimeError("cached replay positions were not initialized")
            graph = replay_graph_positions(replay_template, replay_positions.clone())
            out = run_model(graph)
            if not collect:
                return []
            return [
                {
                    "energy": out["energy"].detach().cpu().numpy(),
                    "forces": out["forces"].detach().cpu().numpy(),
                }
            ]

        if trajectory_template is not None:
            nonlocal trajectory_provider, trajectory_report_rebuild_steps, trajectory_report_rebuild_causes, trajectory_report_max_displacement
            if trajectory_base_positions is None or trajectory_provider is None:
                raise RuntimeError("trajectory replay positions were not initialized")
            graph_template = trajectory_template
            pass_rebuild_steps: list[int] = []
            pass_rebuild_causes: list[str] = []
            pass_max_displacement = 0.0
            collected = []
            for step in range(int(args.trajectory_replay_steps)):
                positions = synthetic_trajectory_positions(
                    trajectory_base_positions,
                    step=step,
                    displacement_std=float(args.trajectory_displacement_std),
                )
                probe = trajectory_provider.check(positions)
                pass_max_displacement = max(
                    pass_max_displacement,
                    float(probe["max_displacement"].detach().cpu()),
                )
                rebuild_cause = None
                if step > 0 and args.trajectory_rebuild_interval and step % int(args.trajectory_rebuild_interval) == 0:
                    rebuild_cause = "interval"
                if step > 0 and bool(probe["needs_rebuild"]):
                    rebuild_cause = "skin" if rebuild_cause is None else f"{rebuild_cause}+skin"
                if rebuild_cause is not None:
                    if args.trajectory_validity_only:
                        trajectory_provider.mark_rebuilt(positions, step=step, cause=rebuild_cause)
                    else:
                        graph_template = graph_update_backend.rebuild(positions)
                        trajectory_provider.mark_rebuilt(positions, step=step, cause=rebuild_cause)
                        graph = graph_template
                    pass_rebuild_steps.append(step)
                    pass_rebuild_causes.append(rebuild_cause)
                elif not args.trajectory_validity_only:
                    graph = replay_graph_positions(graph_template, positions)
                if args.trajectory_validity_only or args.trajectory_update_only:
                    continue
                out = run_model(graph)
                if collect and step == 0:
                    collected.append(
                        {
                            "energy": out["energy"].detach().cpu().numpy(),
                            "forces": out["forces"].detach().cpu().numpy(),
                        }
                    )
            if collect:
                trajectory_report_rebuild_steps = pass_rebuild_steps
                trajectory_report_rebuild_causes = pass_rebuild_causes
                trajectory_report_max_displacement = pass_max_displacement
            return collected

        if args.batch_graph_construction:
            graph = collate_graphs(
                [
                    build_atom_graph(atoms, graph_construction_backend=args.graph_construction_backend, cutoff=config.cutoff, device=device, dtype=dtype)
                    for atoms in atoms_list
                ]
            )
            out = run_model(graph)
            if not collect:
                return []
            return [
                {
                    "energy": out["energy"].detach().cpu().numpy(),
                    "forces": out["forces"].detach().cpu().numpy(),
                }
            ]

        outputs = []
        for atoms in atoms_list:
            graph = build_atom_graph(
                atoms,
                graph_construction_backend=args.graph_construction_backend,
                cutoff=config.cutoff,
                device=device,
                dtype=dtype,
            )
            out = run_model(graph)
            if collect:
                outputs.append(
                    {
                        "energy": out["energy"].detach().cpu().numpy(),
                        "forces": out["forces"].detach().cpu().numpy(),
                    }
                )
        return outputs

    reset_trajectory_runtime_state()
    forward_once(False)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    pass_times = []
    first_outputs = []
    for pass_idx in range(max(1, args.measure_passes)):
        reset_trajectory_runtime_state()
        start = time.perf_counter()
        outputs = forward_once(pass_idx == 0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        pass_times.append(time.perf_counter() - start)
        if pass_idx == 0:
            first_outputs = outputs

    group_ids, image_indices = atoms_group_values(atoms_list, "case_id", "source_frame")
    error_payload = prediction_error_payload(
        first_outputs,
        ref_e,
        ref_f,
        natoms,
        group_ids=group_ids,
        image_indices=image_indices,
    )
    seconds_per_pass = float(np.mean(pass_times))
    atoms = int(natoms.sum())
    force_steps_per_pass = int(args.trajectory_replay_steps) if int(args.trajectory_replay_steps) > 0 else 1
    atom_steps_per_pass = atoms * force_steps_per_pass
    config_steps_per_pass = len(atoms_list) * force_steps_per_pass
    architecture_route = rtece_route_contract(model.config)
    architecture_path_manifest = rtece_path_manifest(model.config)
    runtime_route = rtece_route_contract(
        model.config,
        force_mode=force_mode,
        graph_construction_backend=args.graph_construction_backend,
        graph_update_backend=graph_update_backend.name if graph_update_backend is not None else None,
    )
    runtime_path_manifest = rtece_path_manifest(
        model.config,
        force_mode=force_mode,
        graph_construction_backend=args.graph_construction_backend,
        graph_update_backend=graph_update_backend.name if graph_update_backend is not None else None,
    )
    payload = {
        "backend": "rtece_scalar",
        "variant": args.variant,
        "model": str(args.model),
        "configs_path": str(args.configs),
        "start_config": args.start_config,
        "extxyz_index": extxyz_index(
            start_config=args.start_config,
            limit_configs=args.limit_configs,
        ),
        "configs": len(atoms_list),
        "atoms": atoms,
        "device": str(device),
        "default_dtype": args.default_dtype,
        "model_class": model.__class__.__name__,
        "requested_force_mode": args.force_mode,
        "force_mode": force_mode,
        "tece_architecture_route": architecture_route,
        "tece_architecture_path_manifest": architecture_path_manifest,
        "tece_architecture_path_manifest_hash": architecture_path_manifest["manifest_hash"],
        "tece_route": runtime_route,
        "tece_path_manifest": runtime_path_manifest,
        "tece_path_manifest_hash": runtime_path_manifest["manifest_hash"],
        "hidden_channels": list(model.config.hidden_channels),
        "num_radial": int(model.config.num_radial),
        "moment_l_max": int(model.config.moment_l_max) if model.config.moment_l_max is not None else None,
        "learnable_radial_mixing": bool(model.config.learnable_radial_mixing),
        "cutoff": float(model.config.cutoff),
        "max_atomic_number": int(model.config.max_atomic_number),
        "energy_per_atom_shift": float(model.config.energy_per_atom_shift),
        "atomic_energies": {str(k): float(v) for k, v in (model.config.atomic_energies or {}).items()},
        "short_range_repulsion_potential": str(model.config.short_range_repulsion_potential),
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "includes_graph_construction": bool(args.include_graph_construction),
        "batched_graph_construction": bool(args.batch_graph_construction),
        "replay_cached_graph": bool(args.replay_cached_graph),
        "trajectory_replay": trajectory_template is not None,
        "trajectory_replay_steps": int(args.trajectory_replay_steps),
        "trajectory_rebuild_interval": int(args.trajectory_rebuild_interval),
        "trajectory_displacement_std": float(args.trajectory_displacement_std),
        "trajectory_validity_only": bool(args.trajectory_validity_only),
        "trajectory_update_only": bool(args.trajectory_update_only),
        "trajectory_skin_margin": float(args.trajectory_skin_margin),
        "trajectory_skin_threshold": 0.5 * float(args.trajectory_skin_margin) if float(args.trajectory_skin_margin) > 0.0 else None,
        "trajectory_rebuild_count": len(trajectory_report_rebuild_steps),
        "trajectory_rebuild_steps": trajectory_report_rebuild_steps,
        "trajectory_rebuild_causes": trajectory_report_rebuild_causes,
        "trajectory_max_displacement_since_rebuild_a": trajectory_report_max_displacement,
        "graph_construction_backend": args.graph_construction_backend,
        "graph_update_backend": graph_update_backend.name if graph_update_backend is not None else None,
        "graph_update_chunk_configs": int(args.graph_update_chunk_configs),
        "graph_update_backend_metadata": graph_update_backend.last_metadata if graph_update_backend is not None else None,
        "graph_update_rebuild_count": graph_update_backend.rebuild_count if graph_update_backend is not None else 0,
        "graph_update_total_s": graph_update_backend.total_rebuild_time_s if graph_update_backend is not None else 0.0,
        "graph_update_times_s": graph_update_backend.rebuild_times_s if graph_update_backend is not None else [],
        "prebuilt_batched_graph": prebuilt_graph is not None,
        "measure_passes": args.measure_passes,
        "force_steps_per_pass": force_steps_per_pass,
        "atom_steps_per_pass": atom_steps_per_pass,
        "pass_times_s": pass_times,
        "seconds_per_pass": seconds_per_pass,
        "atoms_per_second": atom_steps_per_pass / seconds_per_pass,
        "configs_per_second": config_steps_per_pass / seconds_per_pass,
        **error_payload,
        **cuda_memory(torch),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
