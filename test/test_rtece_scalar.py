from __future__ import annotations

import math

import torch

from benchmarks.oc20neb_tace_mace.rtece_scalar_model import (
    RTECEGraph,
    atomic_scalar_descriptors,
    build_rtece_config,
    descriptor_dim,
    edge_relational_sketches,
    rtece_descriptors,
)


def rotation_z(theta: float) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )


def complete_directed_edges(num_nodes: int) -> torch.Tensor:
    edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def test_build_rtece_config_defines_ordered_variants():
    pair = build_rtece_config("rtece_pair")
    atomic = build_rtece_config("rtece_atomic_moments")
    sketch8 = build_rtece_config("rtece_edge_sketch8")
    sketch16 = build_rtece_config("rtece_edge_sketch16")

    assert pair.variant == "rtece_pair"
    assert pair.use_atomic_moments is False
    assert pair.num_edge_sketches == 0
    assert atomic.use_atomic_moments is True
    assert atomic.num_edge_sketches == 0
    assert sketch8.use_atomic_moments is True
    assert sketch8.num_edge_sketches == 8
    assert sketch16.use_atomic_moments is True
    assert sketch16.num_edge_sketches == 16
    assert descriptor_dim(pair) < descriptor_dim(atomic) < descriptor_dim(sketch8) < descriptor_dim(sketch16)


def test_atomic_scalar_descriptors_are_rotation_invariant():
    config = build_rtece_config("rtece_atomic_moments")
    z = torch.tensor([6, 8, 1], dtype=torch.long)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.7, 0.2, 0.1], [-0.3, 0.6, -0.2]],
        dtype=torch.float64,
    )
    edge_index = complete_directed_edges(3)
    batch = torch.zeros(3, dtype=torch.long)

    graph = RTECEGraph(z=z, pos=pos, edge_index=edge_index, batch=batch)
    rotated = RTECEGraph(
        z=z,
        pos=pos @ rotation_z(0.37).T,
        edge_index=edge_index,
        batch=batch,
    )

    desc = atomic_scalar_descriptors(graph, config)
    desc_rot = atomic_scalar_descriptors(rotated, config)

    assert torch.allclose(desc, desc_rot, atol=1e-10, rtol=1e-10)



def test_edge_relational_sketches_are_rotation_invariant():
    config = build_rtece_config("rtece_edge_sketch8")
    z = torch.tensor([6, 8, 1, 1], dtype=torch.long)
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.7, 0.2, 0.1],
            [-0.3, 0.6, -0.2],
            [0.4, -0.5, 0.3],
        ],
        dtype=torch.float64,
    )
    edge_index = complete_directed_edges(4)
    batch = torch.zeros(4, dtype=torch.long)
    graph = RTECEGraph(z=z, pos=pos, edge_index=edge_index, batch=batch)
    rotated = RTECEGraph(
        z=z,
        pos=pos @ rotation_z(1.11).T,
        edge_index=edge_index,
        batch=batch,
    )

    sketches = edge_relational_sketches(graph, config)
    sketches_rot = edge_relational_sketches(rotated, config)
    full = rtece_descriptors(graph, config)
    full_rot = rtece_descriptors(rotated, config)

    assert sketches.shape == (4, config.num_edge_sketches)
    assert torch.allclose(sketches, sketches_rot, atol=1e-10, rtol=1e-10)
    assert torch.allclose(full, full_rot, atol=1e-10, rtol=1e-10)
