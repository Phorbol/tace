from __future__ import annotations

import json
import math
import subprocess
import sys

import pytest
import torch

from benchmarks.oc20neb_tace_mace.rtece_scalar_model import (
    RTECEGraph,
    RTECEScalarConfig,
    RTECEScalarModel,
    collate_graphs,
    atomic_scalar_descriptors,
    build_rtece_config,
    cell_list_packed_element_density_descriptors,
    descriptor_dim,
    edge_relational_sketches,
    packed_element_density_descriptors,
    rtece_descriptors,
    rtece_route_contract,
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


def test_rtece_scalar_model_has_formal_tace_models_entrypoint():
    from benchmarks.oc20neb_tace_mace import rtece_scalar_model as benchmark_rtece
    from tace.models import (
        RTECEGraph as CoreRTECEGraph,
        RTECEScalarConfig as CoreRTECEScalarConfig,
        RTECEScalarModel as CoreRTECEScalarModel,
        build_rtece_config as core_build_rtece_config,
    )
    from tace.models.rtece_scalar import packed_element_density_descriptors as core_packed_descriptors

    assert CoreRTECEGraph is benchmark_rtece.RTECEGraph
    assert CoreRTECEScalarConfig is benchmark_rtece.RTECEScalarConfig
    assert CoreRTECEScalarModel is benchmark_rtece.RTECEScalarModel
    assert core_build_rtece_config is benchmark_rtece.build_rtece_config
    assert core_packed_descriptors is benchmark_rtece.packed_element_density_descriptors


def test_rtece_triton_kernels_have_formal_tace_models_entrypoint():
    from benchmarks.oc20neb_tace_mace import rtece_triton_kernels as benchmark_kernels
    from tace.models import rtece_triton_kernels as core_kernels

    assert core_kernels.pair_forces_triton is benchmark_kernels.pair_forces_triton
    assert core_kernels.element_density_descriptors_triton is benchmark_kernels.element_density_descriptors_triton
    assert core_kernels.element_density_forces_triton is benchmark_kernels.element_density_forces_triton
    assert core_kernels.element_density_direct_padded_descriptors_triton is benchmark_kernels.element_density_direct_padded_descriptors_triton
    assert core_kernels.element_density_direct_padded_forces_triton is benchmark_kernels.element_density_direct_padded_forces_triton
    assert core_kernels.direct_radius_padded_edges_triton is benchmark_kernels.direct_radius_padded_edges_triton
    assert core_kernels.direct_radius_counted_edges_triton is benchmark_kernels.direct_radius_counted_edges_triton


def test_rtece_route_contract_classifies_semantic_and_runtime_degradation():
    pair = rtece_route_contract(
        RTECEScalarConfig(variant="rtece_pair"),
        force_mode="analytic_pair",
        graph_construction_backend="torch_radius_nopbc",
    )
    element = rtece_route_contract(
        RTECEScalarConfig(
            variant="rtece_element_density",
            use_element_density=True,
            num_radial=8,
            hidden_channels=(24, 24),
        ),
        force_mode="analytic_element_triton_descriptor_force",
        graph_construction_backend="torch_radius_nopbc",
        graph_update_backend="torch_radius_nopbc_triton_counted",
    )

    assert pair["semantic_tier"] == "T4_scalar_pair_density"
    assert pair["descriptor_family"] == "pair_density"
    assert pair["force_realization"] == "analytic_scalar_chain_rule"
    assert pair["fused_force"] is False
    assert pair["graph_semantics"] == "direct_active_nopbc"
    assert "persistent_equivariant_edge_state" in pair["deleted_tece_groups"]

    assert element["semantic_tier"] == "T3_element_conditioned_scalar_density"
    assert element["descriptor_family"] == "element_density"
    assert element["descriptor_realization"] == "triton_fused_edge_descriptor"
    assert element["force_realization"] == "triton_fused_descriptor_force"
    assert element["fused_descriptor"] is True
    assert element["fused_force"] is True
    assert element["edge_state_lifetime"] == "counted_exact_edge_buffer"
    assert "neighbor_element_density" in element["retained_tece_groups"]


def test_build_rtece_config_defines_ordered_variants():
    pair = build_rtece_config("rtece_pair")
    element = build_rtece_config("rtece_element_density")
    quadratic = build_rtece_config("rtece_density_quadratic")
    vector = build_rtece_config("rtece_vector_moments")
    atomic = build_rtece_config("rtece_atomic_moments")
    sketch8 = build_rtece_config("rtece_edge_sketch8")
    sketch16 = build_rtece_config("rtece_edge_sketch16")

    assert pair.variant == "rtece_pair"
    assert pair.use_atomic_moments is False
    assert pair.num_edge_sketches == 0
    assert element.variant == "rtece_element_density"
    assert element.use_element_density is True
    assert element.use_density_quadratic is False
    assert element.use_atomic_moments is False
    assert element.num_edge_sketches == 0
    assert quadratic.variant == "rtece_density_quadratic"
    assert quadratic.use_density_quadratic is True
    assert quadratic.use_atomic_moments is False
    assert quadratic.num_edge_sketches == 0
    assert vector.variant == "rtece_vector_moments"
    assert vector.use_vector_moments is True
    assert vector.use_atomic_moments is False
    assert vector.num_edge_sketches == 0
    assert atomic.use_atomic_moments is True
    assert atomic.num_edge_sketches == 0
    assert sketch8.use_atomic_moments is True
    assert sketch8.num_edge_sketches == 8
    assert sketch16.use_atomic_moments is True
    assert sketch16.num_edge_sketches == 16
    assert (
        descriptor_dim(pair)
        < descriptor_dim(element)
        == descriptor_dim(quadratic)
        == descriptor_dim(vector)
        < descriptor_dim(atomic)
        < descriptor_dim(sketch8)
        < descriptor_dim(sketch16)
    )


def test_packed_element_density_descriptors_match_split_descriptors():
    config = build_rtece_config("rtece_element_density")
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
    graph = RTECEGraph(
        z=z,
        pos=pos,
        edge_index=complete_directed_edges(4),
        batch=torch.zeros(4, dtype=torch.long),
    )

    split = atomic_scalar_descriptors(graph, config)
    packed = packed_element_density_descriptors(graph, config)

    assert packed.shape == (4, 2 * config.num_radial)
    assert torch.allclose(packed, split, atol=1e-10, rtol=1e-10)


def test_cell_list_packed_element_density_descriptors_match_direct_radius_edges():
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import torch_radius_nopbc_graph

    config = RTECEScalarConfig(
        variant="rtece_element_density",
        cutoff=1.0,
        num_radial=4,
        use_element_density=True,
    )
    z = torch.tensor([6, 8, 1, 7, 1], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [1.8, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    template = RTECEGraph(
        z=z,
        pos=pos,
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=batch,
    )
    direct_graph = torch_radius_nopbc_graph(template, pos, cutoff=config.cutoff)

    direct = packed_element_density_descriptors(direct_graph, config)
    fused = cell_list_packed_element_density_descriptors(template, config)

    assert fused.shape == (5, 2 * config.num_radial)
    assert torch.allclose(fused, direct, atol=1e-10, rtol=1e-10)


def test_cell_list_packed_element_density_forces_match_packed_and_ignore_edge_index():
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import torch_radius_nopbc_graph

    config = RTECEScalarConfig(
        variant="rtece_element_density",
        cutoff=1.1,
        num_radial=4,
        hidden_channels=(8,),
        use_element_density=True,
    )
    model = RTECEScalarModel(config).double().eval()
    z = torch.tensor([6, 8, 1, 7, 1], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.7, 0.1, 0.0],
            [1.4, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    template = RTECEGraph(
        z=z,
        pos=pos,
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=batch,
    )
    direct_graph = torch_radius_nopbc_graph(template, pos, cutoff=config.cutoff)
    bogus_graph = RTECEGraph(
        z=z,
        pos=pos,
        edge_index=complete_directed_edges(5),
        batch=batch,
    )

    packed = model.forward_element_density_packed_analytic_forces(direct_graph)
    cell_list = model.forward_element_density_cell_list_packed_analytic_forces(bogus_graph)

    assert torch.allclose(cell_list["energy"], packed["energy"], atol=1e-10, rtol=1e-10)
    assert torch.allclose(cell_list["atomic_energy"], packed["atomic_energy"], atol=1e-10, rtol=1e-10)
    assert torch.allclose(cell_list["forces"], packed["forces"], atol=1e-10, rtol=1e-10)


def test_cell_list_packed_element_density_descriptors_ignore_input_edge_index():
    config = RTECEScalarConfig(
        variant="rtece_element_density",
        cutoff=1.0,
        num_radial=4,
        use_element_density=True,
    )
    z = torch.tensor([6, 8, 1, 7], dtype=torch.long)
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [2.2, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    batch = torch.tensor([0, 0, 0, 1], dtype=torch.long)
    empty_graph = RTECEGraph(
        z=z,
        pos=pos,
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=batch,
    )
    bogus_cross_batch_edges = RTECEGraph(
        z=z,
        pos=pos,
        edge_index=complete_directed_edges(4),
        batch=batch,
    )

    empty = cell_list_packed_element_density_descriptors(empty_graph, config)
    bogus = cell_list_packed_element_density_descriptors(bogus_cross_batch_edges, config)

    assert torch.allclose(bogus, empty, atol=1e-10, rtol=1e-10)
    assert torch.allclose(empty[3], torch.zeros_like(empty[3]))


def test_element_density_descriptors_are_rotation_invariant_and_element_sensitive():
    config = build_rtece_config("rtece_element_density")
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
    changed_neighbor_element = RTECEGraph(
        z=torch.tensor([6, 1, 1], dtype=torch.long),
        pos=pos,
        edge_index=edge_index,
        batch=batch,
    )

    desc = atomic_scalar_descriptors(graph, config)
    desc_rot = atomic_scalar_descriptors(rotated, config)
    desc_changed = atomic_scalar_descriptors(changed_neighbor_element, config)

    assert desc.shape[-1] == 2 * config.num_radial
    assert torch.allclose(desc, desc_rot, atol=1e-10, rtol=1e-10)
    assert not torch.allclose(desc, desc_changed, atol=1e-10, rtol=1e-10)


def test_density_quadratic_descriptors_are_rotation_invariant():
    config = build_rtece_config("rtece_density_quadratic")
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

    assert desc.shape[-1] == 2 * config.num_radial
    assert torch.allclose(desc, desc_rot, atol=1e-10, rtol=1e-10)


def test_vector_moment_descriptors_are_rotation_invariant_and_geometry_sensitive():
    config = build_rtece_config("rtece_vector_moments")
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
    bent = pos.clone()
    bent[2] = torch.tensor([-0.1, 0.8, -0.4], dtype=torch.float64)
    edge_index = complete_directed_edges(4)
    batch = torch.zeros(4, dtype=torch.long)

    graph = RTECEGraph(z=z, pos=pos, edge_index=edge_index, batch=batch)
    rotated = RTECEGraph(
        z=z,
        pos=pos @ rotation_z(0.73).T,
        edge_index=edge_index,
        batch=batch,
    )
    geometry_changed = RTECEGraph(z=z, pos=bent, edge_index=edge_index, batch=batch)

    desc = atomic_scalar_descriptors(graph, config)
    desc_rot = atomic_scalar_descriptors(rotated, config)
    desc_changed = atomic_scalar_descriptors(geometry_changed, config)

    assert desc.shape[-1] == 2 * config.num_radial
    assert torch.allclose(desc, desc_rot, atol=1e-10, rtol=1e-10)
    assert not torch.allclose(desc, desc_changed, atol=1e-10, rtol=1e-10)


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


def test_rtece_scalar_model_returns_conservative_forces():
    config = build_rtece_config("rtece_edge_sketch8")
    model = RTECEScalarModel(config).double()
    z = torch.tensor([6, 8, 1], dtype=torch.long)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.7, 0.2, 0.1], [-0.3, 0.6, -0.2]],
        dtype=torch.float64,
    )
    graph = RTECEGraph(
        z=z,
        pos=pos,
        edge_index=complete_directed_edges(3),
        batch=torch.zeros(3, dtype=torch.long),
    )

    out = model(graph)

    assert out["energy"].shape == (1,)
    assert out["atomic_energy"].shape == (3,)
    assert out["forces"].shape == (3, 3)
    assert torch.isfinite(out["energy"]).all()
    assert torch.isfinite(out["forces"]).all()


def test_rtece_scalar_model_energy_is_permutation_invariant_for_complete_graph():
    config = build_rtece_config("rtece_edge_sketch8")
    model = RTECEScalarModel(config).double()
    z = torch.tensor([6, 8, 1], dtype=torch.long)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.7, 0.2, 0.1], [-0.3, 0.6, -0.2]],
        dtype=torch.float64,
    )
    graph = RTECEGraph(
        z=z,
        pos=pos,
        edge_index=complete_directed_edges(3),
        batch=torch.zeros(3, dtype=torch.long),
    )
    perm = torch.tensor([2, 0, 1], dtype=torch.long)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(3)
    edge_index_perm = inv[complete_directed_edges(3)]
    graph_perm = RTECEGraph(
        z=z[perm],
        pos=pos[perm],
        edge_index=edge_index_perm,
        batch=torch.zeros(3, dtype=torch.long),
    )

    e = model(graph)["energy"]
    e_perm = model(graph_perm)["energy"]

    assert torch.allclose(e, e_perm, atol=1e-10, rtol=1e-10)


def test_rtece_route_contract_classifies_direct_padded_streaming_backend():
    route = rtece_route_contract(
        RTECEScalarConfig(
            variant="rtece_element_density",
            use_element_density=True,
            hidden_channels=(16, 16),
        ),
        force_mode="analytic_element_direct_padded_descriptor_force",
        graph_construction_backend="torch_radius_nopbc",
    )

    assert route["semantic_tier"] == "T3_element_conditioned_scalar_density"
    assert route["descriptor_realization"] == "triton_direct_padded_descriptor"
    assert route["force_realization"] == "triton_direct_padded_descriptor_force"
    assert route["fused_descriptor"] is True
    assert route["fused_force"] is True
    assert route["edge_state_lifetime"] == "streaming_padded_candidates"


def test_core_rtece_workflow_predicts_cell_list_descriptor_force_mode():
    from tace.models.rtece_workflow import predict

    config = RTECEScalarConfig(
        variant="rtece_element_density",
        cutoff=1.0,
        num_radial=4,
        hidden_channels=(8,),
        use_element_density=True,
    )
    model = RTECEScalarModel(config).double().eval()
    graph = RTECEGraph(
        z=torch.tensor([6, 8], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=torch.zeros(2, dtype=torch.long),
    )

    out = predict(
        model,
        graph,
        force_mode="analytic_element_cell_list_descriptor_force",
        include_route=True,
    )

    assert out["energy"].shape == (1,)
    assert out["forces"].shape == (2, 3)
    assert out["tece_route"]["descriptor_realization"] == "cell_list_fused_descriptor_oracle"
    assert out["tece_route"]["edge_state_lifetime"] == "streaming_cell_candidates_oracle"


def test_core_rtece_workflow_rejects_inference_only_force_modes_while_training():
    from tace.models.rtece_workflow import predict

    config = RTECEScalarConfig(
        variant="rtece_element_density",
        num_radial=4,
        hidden_channels=(8,),
        use_element_density=True,
    )
    model = RTECEScalarModel(config).double()
    model.train()
    graph = RTECEGraph(
        z=torch.tensor([6, 8], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64),
        edge_index=complete_directed_edges(2),
        batch=torch.zeros(2, dtype=torch.long),
    )

    with pytest.raises(RuntimeError, match="inference-only"):
        predict(model, graph, force_mode="analytic_element_packed")


def test_core_rtece_workflow_saves_loads_and_predicts_with_route_metadata(tmp_path):
    from tace.models.rtece_workflow import load_checkpoint, predict, save_checkpoint

    config = RTECEScalarConfig(
        variant="rtece_element_density",
        use_element_density=True,
        hidden_channels=(8,),
        num_radial=4,
    )
    model = RTECEScalarModel(config).double().eval()
    path = tmp_path / "core_rtece.pt"
    graph = RTECEGraph(
        z=torch.tensor([6, 8], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        batch=torch.zeros(2, dtype=torch.long),
    )

    save_checkpoint(
        path,
        model,
        config,
        force_mode="analytic_density",
        graph_construction_backend="torch_radius_nopbc",
    )
    loaded_model, loaded_config, metadata = load_checkpoint(path, dtype=torch.float64)
    prediction = predict(
        loaded_model,
        graph,
        force_mode="autograd",
        graph_construction_backend="torch_radius_nopbc",
        include_route=True,
    )

    assert loaded_config == config
    assert metadata["tece_route"]["semantic_tier"] == "T3_element_conditioned_scalar_density"
    assert metadata["tece_route"]["force_realization"] == "analytic_scalar_chain_rule"
    assert prediction["energy"].shape == (1,)
    assert prediction["forces"].shape == (2, 3)
    assert prediction["tece_route"]["force_realization"] == "autograd_conservative"


def test_rtece_benchmark_training_reuses_core_workflow_api():
    from benchmarks.oc20neb_tace_mace import train_rtece_scalar
    from tace.models import rtece_workflow

    assert train_rtece_scalar.save_checkpoint is rtece_workflow.save_checkpoint
    assert train_rtece_scalar.load_checkpoint_with_metadata is rtece_workflow.load_checkpoint
    assert train_rtece_scalar.loss_for_batch is rtece_workflow.loss_for_batch
    assert train_rtece_scalar.evaluate_loss is rtece_workflow.evaluate_loss
    assert train_rtece_scalar.train_steps is rtece_workflow.train_steps


def test_rtece_checkpoint_roundtrip(tmp_path):
    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import save_checkpoint, load_checkpoint

    config = build_rtece_config("rtece_pair")
    model = RTECEScalarModel(config).double()
    path = tmp_path / "rtece_scalar.pt"

    save_checkpoint(path, model, config)
    loaded_model, loaded_config = load_checkpoint(path, dtype=torch.float64)

    assert loaded_config.variant == "rtece_pair"
    assert isinstance(loaded_model, RTECEScalarModel)


def test_rtece_tiny_training_step_reduces_finite_loss(tmp_path):
    from ase import Atoms
    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import atoms_to_graph, train_steps

    atoms = Atoms(
        "H2O",
        positions=[[0.0, 0.0, 0.0], [0.75, 0.0, 0.0], [-0.25, 0.65, 0.0]],
    )
    atoms.info["energy"] = -1.0
    atoms.arrays["forces"] = torch.zeros((3, 3), dtype=torch.float64).numpy()
    config = build_rtece_config("rtece_pair")
    model = RTECEScalarModel(config).double()
    graph, energy, forces = atoms_to_graph(
        atoms,
        cutoff=config.cutoff,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    summary = train_steps(model, [(graph, energy, forces)], max_steps=2, lr=1e-3)

    assert summary["steps"] == 2
    assert torch.isfinite(torch.tensor(summary["final_loss"]))


def test_rtece_benchmark_row_is_summary_compatible():
    from benchmarks.oc20neb_tace_mace.summarize_tece_distill import make_student_row

    dft = {
        "model": "rtece_scalar.pt",
        "atoms_per_second": 100000.0,
        "configs_per_second": 1000.0,
        "seconds_per_pass": 0.1,
        "peak_allocated_mb": 64.0,
        "peak_reserved_mb": 80.0,
        "num_parameters": 1234,
        "mae_e_mev_atom": 10.0,
        "rmse_e_mev_atom": 20.0,
        "mae_f_mev_a": 40.0,
        "rmse_f_mev_a": 80.0,
    }
    teacher = dict(dft)
    teacher["mae_f_mev_a"] = 39.0

    row = make_student_row("rtece_edge_sketch8", dft_benchmark=dft, teacher_benchmark=teacher)

    assert row["variant"] == "rtece_edge_sketch8"
    assert row["atoms_per_second"] == 100000.0
    assert row["dft_f_mae_mev_a"] == 40.0
    assert row["teacher_f_mae_mev_a"] == 39.0


def test_parse_hidden_channels_accepts_ordered_capacity_axis():
    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import parse_hidden_channels

    assert parse_hidden_channels("32") == (32,)
    assert parse_hidden_channels("32,64") == (32, 64)
    assert parse_hidden_channels("32, 64,128") == (32, 64, 128)


def test_loss_for_batch_honors_force_weight_axis():
    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import loss_for_batch

    config = build_rtece_config("rtece_pair")
    model = RTECEScalarModel(config).double()
    for param in model.parameters():
        param.data.zero_()
    graph = RTECEGraph(
        z=torch.tensor([1, 1], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64),
        edge_index=complete_directed_edges(2),
        batch=torch.zeros(2, dtype=torch.long),
    )
    ref_energy = torch.zeros(1, dtype=torch.float64)
    ref_forces = torch.ones((2, 3), dtype=torch.float64)

    assert torch.allclose(
        loss_for_batch(model, graph, ref_energy, ref_forces, force_weight=0.0),
        torch.tensor(0.0, dtype=torch.float64),
    )
    assert torch.allclose(
        loss_for_batch(model, graph, ref_energy, ref_forces, force_weight=2.0),
        torch.tensor(2.0, dtype=torch.float64),
    )

def test_rtece_scripts_are_directly_executable():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    for script in (
        "benchmarks/oc20neb_tace_mace/train_rtece_scalar.py",
        "benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py",
        "benchmarks/oc20neb_tace_mace/profile_rtece_scalar.py",
    ):
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr


def test_collate_graphs_matches_individual_energies():
    config = build_rtece_config("rtece_edge_sketch8")
    model = RTECEScalarModel(config).double().eval()
    g1 = RTECEGraph(
        z=torch.tensor([1, 1], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64),
        edge_index=complete_directed_edges(2),
        batch=torch.zeros(2, dtype=torch.long),
    )
    g2 = RTECEGraph(
        z=torch.tensor([6, 1, 1], dtype=torch.long),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [0.6, 0.1, 0.0], [-0.2, 0.5, 0.0]],
            dtype=torch.float64,
        ),
        edge_index=complete_directed_edges(3),
        batch=torch.zeros(3, dtype=torch.long),
    )

    e1 = model(g1)["energy"]
    e2 = model(g2)["energy"]
    batched = model(collate_graphs([g1, g2]))["energy"]

    assert batched.shape == (2,)
    assert torch.allclose(batched, torch.cat([e1, e2]), atol=1e-10, rtol=1e-10)


def test_atomic_energies_add_tace_style_element_reference_energy():
    config = RTECEScalarConfig(
        variant="rtece_pair",
        use_atomic_moments=False,
        num_edge_sketches=0,
        atomic_energies={1: -0.5, 6: -3.0},
    )
    model = RTECEScalarModel(config).double()
    for param in model.parameters():
        param.data.zero_()
    graph = RTECEGraph(
        z=torch.tensor([1, 6, 1], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [1.4, 0.0, 0.0]], dtype=torch.float64),
        edge_index=complete_directed_edges(3),
        batch=torch.zeros(3, dtype=torch.long),
    )

    out = model(graph)

    assert torch.allclose(out["energy"], torch.tensor([-4.0], dtype=torch.float64))


def test_energy_per_atom_shift_remains_legacy_zeroth_order_energy():
    config = RTECEScalarConfig(
        variant="rtece_pair",
        use_atomic_moments=False,
        num_edge_sketches=0,
        energy_per_atom_shift=1.25,
    )
    model = RTECEScalarModel(config).double()
    for param in model.parameters():
        param.data.zero_()
    graph = RTECEGraph(
        z=torch.tensor([1, 1], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64),
        edge_index=complete_directed_edges(2),
        batch=torch.zeros(2, dtype=torch.long),
    )

    out = model(graph)

    assert torch.allclose(out["energy"], torch.tensor([2.5], dtype=torch.float64))


def test_fit_atomic_energies_solves_per_element_reference_least_squares():
    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import fit_atomic_energies

    g1 = RTECEGraph(
        z=torch.tensor([1, 1], dtype=torch.long),
        pos=torch.zeros((2, 3), dtype=torch.float64),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=torch.zeros(2, dtype=torch.long),
    )
    g2 = RTECEGraph(
        z=torch.tensor([6, 1], dtype=torch.long),
        pos=torch.zeros((2, 3), dtype=torch.float64),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=torch.zeros(2, dtype=torch.long),
    )
    samples = [
        (g1, torch.tensor([-1.0], dtype=torch.float64), torch.zeros((2, 3), dtype=torch.float64)),
        (g2, torch.tensor([-3.5], dtype=torch.float64), torch.zeros((2, 3), dtype=torch.float64)),
    ]

    assert fit_atomic_energies(samples) == pytest.approx({1: -0.5, 6: -3.0})


def test_fit_energy_per_atom_shift_uses_total_energy_per_total_atom():
    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import fit_energy_per_atom_shift

    g1 = RTECEGraph(
        z=torch.tensor([1, 1], dtype=torch.long),
        pos=torch.zeros((2, 3), dtype=torch.float64),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=torch.zeros(2, dtype=torch.long),
    )
    g2 = RTECEGraph(
        z=torch.tensor([1, 1, 1], dtype=torch.long),
        pos=torch.zeros((3, 3), dtype=torch.float64),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=torch.zeros(3, dtype=torch.long),
    )
    samples = [
        (g1, torch.tensor([2.0], dtype=torch.float64), torch.zeros((2, 3), dtype=torch.float64)),
        (g2, torch.tensor([6.0], dtype=torch.float64), torch.zeros((3, 3), dtype=torch.float64)),
    ]

    assert fit_energy_per_atom_shift(samples) == 1.6



def test_train_rtece_scalar_cli_fits_atomic_energies_by_default(tmp_path):
    import ase.io
    from ase import Atoms

    train_file = tmp_path / "train.extxyz"
    out_dir = tmp_path / "out"
    h2 = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    h2.info["energy"] = -1.0
    h2.arrays["forces"] = torch.zeros((2, 3), dtype=torch.float64).numpy()
    ch = Atoms("CH", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    ch.info["energy"] = -3.5
    ch.arrays["forces"] = torch.zeros((2, 3), dtype=torch.float64).numpy()
    ase.io.write(str(train_file), [h2, ch], format="extxyz")

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/oc20neb_tace_mace/train_rtece_scalar.py",
            "--variant",
            "rtece_pair",
            "--train-file",
            str(train_file),
            "--valid-file",
            str(train_file),
            "--output-dir",
            str(out_dir),
            "--limit-configs",
            "2",
            "--max-steps",
            "1",
            "--eval-interval",
            "0",
            "--disable-best-checkpoint",
            "--hidden-channels",
            "4",
            "--num-radial",
            "2",
            "--device",
            "cpu",
            "--default-dtype",
            "float64",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    summary = json.loads((out_dir / "train_summary.json").read_text())
    assert summary["atomic_energies"] == pytest.approx({"1": -0.5, "6": -3.0})
    assert summary["energy_per_atom_shift"] == 0.0


def test_train_steps_saves_best_validation_checkpoint(tmp_path):
    from ase import Atoms
    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import atoms_to_graph, load_checkpoint, train_steps

    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    atoms.info["energy"] = -0.5
    atoms.arrays["forces"] = torch.zeros((2, 3), dtype=torch.float64).numpy()
    config = build_rtece_config("rtece_pair")
    model = RTECEScalarModel(config).double()
    sample = atoms_to_graph(atoms, cutoff=config.cutoff, device=torch.device("cpu"), dtype=torch.float64)
    best_path = tmp_path / "rtece_scalar_best.pt"

    summary = train_steps(
        model,
        [sample],
        max_steps=2,
        lr=1e-3,
        valid_samples=[sample],
        eval_interval=1,
        best_checkpoint_path=best_path,
        config=config,
    )
    loaded_model, loaded_config = load_checkpoint(best_path, dtype=torch.float64)

    assert best_path.exists()
    assert loaded_config.variant == "rtece_pair"
    assert isinstance(loaded_model, RTECEScalarModel)
    assert summary["best_step"] in (1, 2)
    assert torch.isfinite(torch.tensor(summary["best_valid_loss"]))


def test_pair_analytic_forces_match_autograd_forces():
    config = RTECEScalarConfig(
        variant="rtece_pair",
        use_atomic_moments=False,
        num_edge_sketches=0,
        energy_per_atom_shift=-0.25,
    )
    model = RTECEScalarModel(config).double().eval()
    graph = RTECEGraph(
        z=torch.tensor([6, 8, 1, 1], dtype=torch.long),
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.7, 0.2, 0.1],
                [-0.3, 0.6, -0.2],
                [0.4, -0.5, 0.3],
            ],
            dtype=torch.float64,
        ),
        edge_index=complete_directed_edges(4),
        batch=torch.zeros(4, dtype=torch.long),
    )

    autograd_out = model(graph)
    analytic_out = model.forward_pair_analytic_forces(graph)

    assert torch.allclose(analytic_out["energy"], autograd_out["energy"], atol=1e-10, rtol=1e-10)
    assert torch.allclose(
        analytic_out["forces"],
        autograd_out["forces"],
        atol=1e-8,
        rtol=1e-8,
    )


def test_triton_pair_force_path_requires_cuda_graph():
    config = RTECEScalarConfig(
        variant="rtece_pair",
        use_atomic_moments=False,
        num_edge_sketches=0,
        energy_per_atom_shift=-0.25,
    )
    model = RTECEScalarModel(config).double().eval()
    graph = RTECEGraph(
        z=torch.tensor([6, 8], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.2, 0.1]], dtype=torch.float64),
        edge_index=complete_directed_edges(2),
        batch=torch.zeros(2, dtype=torch.long),
    )

    try:
        model.forward_pair_triton_force_analytic_forces(graph)
    except RuntimeError as exc:
        assert "CUDA" in str(exc) or "Triton" in str(exc)
    else:
        raise AssertionError("Triton pair force path should reject CPU graphs")


def test_triton_element_density_force_path_requires_cuda_graph():
    config = RTECEScalarConfig(
        variant="rtece_element_density",
        use_element_density=True,
        use_atomic_moments=False,
        num_edge_sketches=0,
        energy_per_atom_shift=-0.25,
    )
    model = RTECEScalarModel(config).double().eval()
    graph = RTECEGraph(
        z=torch.tensor([6, 8], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.2, 0.1]], dtype=torch.float64),
        edge_index=complete_directed_edges(2),
        batch=torch.zeros(2, dtype=torch.long),
    )

    try:
        model.forward_element_density_triton_force_analytic_forces(graph)
    except RuntimeError as exc:
        assert "CUDA" in str(exc) or "Triton" in str(exc)
    else:
        raise AssertionError("Triton element-density force path should reject CPU graphs")


def test_triton_direct_padded_descriptor_force_path_requires_cuda_graph():
    config = RTECEScalarConfig(
        variant="rtece_element_density",
        use_element_density=True,
        use_atomic_moments=False,
        num_edge_sketches=0,
        energy_per_atom_shift=-0.25,
    )
    model = RTECEScalarModel(config).double().eval()
    graph = RTECEGraph(
        z=torch.tensor([6, 8], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.2, 0.1]], dtype=torch.float64),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=torch.zeros(2, dtype=torch.long),
    )

    try:
        model.forward_element_density_direct_padded_triton_descriptor_force_analytic_forces(graph)
    except RuntimeError as exc:
        assert "CUDA" in str(exc) or "Triton" in str(exc)
    else:
        raise AssertionError("Triton direct-padded descriptor+force path should reject CPU graphs")


def test_triton_element_density_descriptor_force_path_requires_cuda_graph():
    config = RTECEScalarConfig(
        variant="rtece_element_density",
        use_element_density=True,
        use_atomic_moments=False,
        num_edge_sketches=0,
        energy_per_atom_shift=-0.25,
    )
    model = RTECEScalarModel(config).double().eval()
    graph = RTECEGraph(
        z=torch.tensor([6, 8], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.2, 0.1]], dtype=torch.float64),
        edge_index=complete_directed_edges(2),
        batch=torch.zeros(2, dtype=torch.long),
    )

    try:
        model.forward_element_density_triton_descriptor_force_analytic_forces(graph)
    except RuntimeError as exc:
        assert "CUDA" in str(exc) or "Triton" in str(exc)
    else:
        raise AssertionError("Triton element-density descriptor+force path should reject CPU graphs")


def test_density_analytic_forces_match_autograd_forces_for_element_descriptors():
    config = RTECEScalarConfig(
        variant="rtece_element_density",
        use_element_density=True,
        use_atomic_moments=False,
        num_edge_sketches=0,
        energy_per_atom_shift=-0.25,
    )
    model = RTECEScalarModel(config).double().eval()
    graph = RTECEGraph(
        z=torch.tensor([6, 8, 1, 1], dtype=torch.long),
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.7, 0.2, 0.1],
                [-0.3, 0.6, -0.2],
                [0.4, -0.5, 0.3],
            ],
            dtype=torch.float64,
        ),
        edge_index=complete_directed_edges(4),
        batch=torch.zeros(4, dtype=torch.long),
    )

    autograd_out = model(graph)
    analytic_out = model.forward_density_analytic_forces(graph)

    assert torch.allclose(analytic_out["energy"], autograd_out["energy"], atol=1e-10, rtol=1e-10)
    assert torch.allclose(
        analytic_out["forces"],
        autograd_out["forces"],
        atol=1e-8,
        rtol=1e-8,
    )


def test_triton_direct_padded_descriptor_force_matches_edge_index_triton_when_cuda_available():
    if not torch.cuda.is_available():
        return
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import torch_radius_nopbc_graph

    config = RTECEScalarConfig(
        variant="rtece_element_density",
        cutoff=1.0,
        num_radial=4,
        hidden_channels=(8,),
        use_element_density=True,
    )
    model = RTECEScalarModel(config).to(device="cuda", dtype=torch.float32).eval()
    z = torch.tensor([6, 8, 1, 7], dtype=torch.long, device="cuda")
    batch = torch.tensor([0, 0, 0, 1], dtype=torch.long, device="cuda")
    pos = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.7, 0.1, 0.0],
            [1.4, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    template = RTECEGraph(
        z=z,
        pos=pos,
        edge_index=torch.zeros((2, 0), dtype=torch.long, device="cuda"),
        batch=batch,
    )
    direct_graph = torch_radius_nopbc_graph(template, pos, cutoff=config.cutoff)

    edge_index_out = model.forward_element_density_triton_descriptor_force_analytic_forces(direct_graph)
    direct_out = model.forward_element_density_direct_padded_triton_descriptor_force_analytic_forces(template)

    torch.cuda.synchronize()
    assert torch.allclose(direct_out["energy"], edge_index_out["energy"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(direct_out["atomic_energy"], edge_index_out["atomic_energy"], atol=1e-5, rtol=1e-5)
    assert torch.allclose(direct_out["forces"], edge_index_out["forces"], atol=1e-4, rtol=1e-4)


def test_packed_element_density_analytic_forces_match_autograd_forces():
    config = RTECEScalarConfig(
        variant="rtece_element_density",
        use_element_density=True,
        use_atomic_moments=False,
        num_edge_sketches=0,
        energy_per_atom_shift=-0.25,
    )
    model = RTECEScalarModel(config).double().eval()
    graph = RTECEGraph(
        z=torch.tensor([6, 8, 1, 1], dtype=torch.long),
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.7, 0.2, 0.1],
                [-0.3, 0.6, -0.2],
                [0.4, -0.5, 0.3],
            ],
            dtype=torch.float64,
        ),
        edge_index=complete_directed_edges(4),
        batch=torch.zeros(4, dtype=torch.long),
    )

    autograd_out = model(graph)
    packed_out = model.forward_element_density_packed_analytic_forces(graph)

    assert torch.allclose(packed_out["energy"], autograd_out["energy"], atol=1e-10, rtol=1e-10)
    assert torch.allclose(
        packed_out["forces"],
        autograd_out["forces"],
        atol=1e-8,
        rtol=1e-8,
    )


def test_density_analytic_forces_match_autograd_forces_for_vector_moments():
    config = RTECEScalarConfig(
        variant="rtece_vector_moments",
        use_vector_moments=True,
        use_atomic_moments=False,
        num_edge_sketches=0,
        energy_per_atom_shift=-0.25,
    )
    model = RTECEScalarModel(config).double().eval()
    graph = RTECEGraph(
        z=torch.tensor([6, 8, 1, 1], dtype=torch.long),
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.7, 0.2, 0.1],
                [-0.3, 0.6, -0.2],
                [0.4, -0.5, 0.3],
            ],
            dtype=torch.float64,
        ),
        edge_index=complete_directed_edges(4),
        batch=torch.zeros(4, dtype=torch.long),
    )

    autograd_out = model(graph)
    analytic_out = model.forward_density_analytic_forces(graph)

    assert torch.allclose(analytic_out["energy"], autograd_out["energy"], atol=1e-10, rtol=1e-10)
    assert torch.allclose(
        analytic_out["forces"],
        autograd_out["forces"],
        atol=1e-8,
        rtol=1e-8,
    )


def test_density_analytic_forces_match_autograd_forces_for_quadratic_descriptors():
    config = RTECEScalarConfig(
        variant="rtece_density_quadratic",
        use_density_quadratic=True,
        use_atomic_moments=False,
        num_edge_sketches=0,
        energy_per_atom_shift=-0.25,
    )
    model = RTECEScalarModel(config).double().eval()
    graph = RTECEGraph(
        z=torch.tensor([6, 8, 1, 1], dtype=torch.long),
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.7, 0.2, 0.1],
                [-0.3, 0.6, -0.2],
                [0.4, -0.5, 0.3],
            ],
            dtype=torch.float64,
        ),
        edge_index=complete_directed_edges(4),
        batch=torch.zeros(4, dtype=torch.long),
    )

    autograd_out = model(graph)
    analytic_out = model.forward_density_analytic_forces(graph)

    assert torch.allclose(analytic_out["energy"], autograd_out["energy"], atol=1e-10, rtol=1e-10)
    assert torch.allclose(
        analytic_out["forces"],
        autograd_out["forces"],
        atol=1e-8,
        rtol=1e-8,
    )

def test_rtece_auto_force_mode_prefers_fused_element_density_only_when_eligible():
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import choose_rtece_force_mode

    element = RTECEScalarConfig(
        variant="rtece_element_density",
        use_element_density=True,
        use_atomic_moments=False,
        num_edge_sketches=0,
    )
    pair = RTECEScalarConfig(variant="rtece_pair")
    vector = RTECEScalarConfig(variant="rtece_vector_moments", use_vector_moments=True)

    assert (
        choose_rtece_force_mode(
            "auto",
            element,
            device_type="cuda",
            dtype=torch.float32,
        )
        == "analytic_element_triton_descriptor_force"
    )
    assert choose_rtece_force_mode("auto", element, device_type="cpu", dtype=torch.float32) == "autograd"
    assert choose_rtece_force_mode("auto", element, device_type="cuda", dtype=torch.float64) == "autograd"
    assert choose_rtece_force_mode("auto", pair, device_type="cuda", dtype=torch.float32) == "autograd"
    assert choose_rtece_force_mode("auto", vector, device_type="cuda", dtype=torch.float32) == "autograd"
    assert (
        choose_rtece_force_mode(
            "analytic_element_triton_force",
            element,
            device_type="cuda",
            dtype=torch.float32,
        )
        == "analytic_element_triton_force"
    )


def test_batched_graph_construction_requires_graph_construction_timing():
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import validate_graph_construction_args

    validate_graph_construction_args(
        include_graph_construction=True,
        batch_graph_construction=True,
        replay_cached_graph=False,
        trajectory_replay_steps=0,
        trajectory_rebuild_interval=0,
    )
    validate_graph_construction_args(
        include_graph_construction=False,
        batch_graph_construction=False,
        replay_cached_graph=True,
        trajectory_replay_steps=0,
        trajectory_rebuild_interval=0,
    )
    validate_graph_construction_args(
        include_graph_construction=False,
        batch_graph_construction=False,
        replay_cached_graph=False,
        trajectory_replay_steps=4,
        trajectory_rebuild_interval=2,
    )
    try:
        validate_graph_construction_args(
            include_graph_construction=False,
            batch_graph_construction=True,
            replay_cached_graph=False,
            trajectory_replay_steps=0,
            trajectory_rebuild_interval=0,
        )
    except ValueError as exc:
        assert "--include-graph-construction" in str(exc)
    else:
        raise AssertionError("batched graph construction should require graph-construction timing")
    try:
        validate_graph_construction_args(
            include_graph_construction=True,
            batch_graph_construction=False,
            replay_cached_graph=True,
            trajectory_replay_steps=0,
            trajectory_rebuild_interval=0,
        )
    except ValueError as exc:
        assert "--replay-cached-graph" in str(exc)
    else:
        raise AssertionError("cached graph replay should be separate from graph-construction timing")
    try:
        validate_graph_construction_args(
            include_graph_construction=False,
            batch_graph_construction=False,
            replay_cached_graph=True,
            trajectory_replay_steps=4,
            trajectory_rebuild_interval=0,
        )
    except ValueError as exc:
        assert "--trajectory-replay-steps" in str(exc)
    else:
        raise AssertionError("trajectory replay should be separate from cached graph replay")
    try:
        validate_graph_construction_args(
            include_graph_construction=False,
            batch_graph_construction=False,
            replay_cached_graph=False,
            trajectory_replay_steps=0,
            trajectory_rebuild_interval=2,
        )
    except ValueError as exc:
        assert "--trajectory-rebuild-interval" in str(exc)
    else:
        raise AssertionError("trajectory rebuild interval should require trajectory replay")


def test_synthetic_trajectory_positions_keep_initial_frame_exact():
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import synthetic_trajectory_positions

    base = torch.tensor(
        [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]],
        dtype=torch.float64,
    )

    first = synthetic_trajectory_positions(base, step=0, displacement_std=0.05)
    moved = synthetic_trajectory_positions(base, step=1, displacement_std=0.05)
    static = synthetic_trajectory_positions(base, step=3, displacement_std=0.0)

    assert torch.equal(first, base)
    assert torch.equal(static, base)
    assert moved.shape == base.shape
    assert moved.dtype == base.dtype
    assert moved.device == base.device
    assert not torch.equal(moved, base)


def test_trajectory_cache_displacement_probe_uses_half_skin_margin():
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import graph_cache_displacement_probe

    reference = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    moved = reference.clone()
    moved[1, 0] += 0.24

    valid = graph_cache_displacement_probe(reference, moved, skin_margin=0.50)
    invalid = graph_cache_displacement_probe(reference, moved, skin_margin=0.40)
    disabled = graph_cache_displacement_probe(reference, moved, skin_margin=0.0)

    assert valid["max_displacement"] == torch.tensor(0.24, dtype=torch.float64)
    assert valid["threshold"] == 0.25
    assert valid["rebuild_required"] is False
    assert invalid["threshold"] == 0.20
    assert invalid["rebuild_required"] is True
    assert disabled["rebuild_required"] is False
    assert disabled["threshold"] is None


def test_trajectory_graph_cache_provider_tracks_rebuild_state():
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import TrajectoryGraphCacheProvider

    reference = torch.zeros((2, 3), dtype=torch.float64)
    provider = TrajectoryGraphCacheProvider(reference, skin_margin=0.4)

    inside = reference.clone()
    inside[0, 0] = 0.19
    outside = reference.clone()
    outside[0, 0] = 0.21

    valid = provider.check(inside)
    invalid = provider.check(outside)

    assert valid["needs_rebuild"] is False
    assert valid["threshold"] == 0.2
    assert invalid["needs_rebuild"] is True
    assert invalid["max_displacement"] == torch.tensor(0.21, dtype=torch.float64)

    provider.mark_rebuilt(outside)
    assert provider.rebuild_count == 1
    assert provider.last_rebuild_step is None
    assert provider.check(outside)["needs_rebuild"] is False

    outside[1, 1] = 0.22
    provider.mark_rebuilt(outside, step=7, cause="skin")
    assert provider.rebuild_count == 2
    assert provider.last_rebuild_step == 7
    assert provider.rebuild_steps == [7]
    assert provider.rebuild_causes == ["skin"]


def test_cached_topology_update_backend_reuses_edges_and_updates_positions():
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import make_graph_update_backend
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import RTECEGraph

    template = RTECEGraph(
        z=torch.tensor([6, 8], dtype=torch.long),
        pos=torch.zeros((2, 3), dtype=torch.float64),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        batch=torch.zeros(2, dtype=torch.long),
    )
    new_positions = torch.tensor([[0.1, 0.2, 0.3], [1.0, 1.1, 1.2]], dtype=torch.float64)

    backend = make_graph_update_backend(
        backend_name="cached_topology",
        rebuild_fn=lambda positions: (_ for _ in ()).throw(AssertionError("ASE rebuild should not run")),
        template_graph=template,
    )
    graph = backend.rebuild(new_positions)

    assert graph.z is template.z
    assert graph.edge_index is template.edge_index
    assert graph.batch is template.batch
    assert torch.equal(graph.pos, new_positions)
    assert graph.pos is not template.pos
    assert backend.name == "cached_topology"
    assert backend.rebuild_count == 1


def test_atoms_to_torch_radius_nopbc_graph_builds_direct_edges():
    from ase import Atoms
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import atoms_to_torch_radius_nopbc_graph

    atoms = Atoms(
        numbers=[6, 8, 1],
        positions=[[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [2.2, 0.0, 0.0]],
        pbc=True,
        cell=[3.0, 3.0, 3.0],
    )

    graph = atoms_to_torch_radius_nopbc_graph(
        atoms,
        cutoff=1.0,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )

    assert graph.z.tolist() == [6, 8, 1]
    assert graph.pos.dtype == torch.float64
    assert graph.batch.tolist() == [0, 0, 0]
    assert graph.edge_index.tolist() == [[0, 1], [1, 0]]


def test_cell_list_oracle_work_metadata_counts_candidate_and_active_pairs():
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import cell_list_oracle_work_metadata
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import RTECEGraph

    template = RTECEGraph(
        z=torch.tensor([6, 8, 1, 7, 1], dtype=torch.long),
        pos=torch.zeros((5, 3), dtype=torch.float64),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
    )
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [1.8, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    metadata = cell_list_oracle_work_metadata(template, positions, cutoff=1.0)

    assert metadata["num_configs"] == 2
    assert metadata["num_atoms"] == 5
    assert metadata["exact_pair_slots"] == 13
    assert metadata["padded_pair_slots"] == 18
    assert metadata["all_pair_nonself_slots"] == 8
    assert metadata["cell_candidate_directed_pairs"] == 8
    assert metadata["active_directed_edges"] == 6
    assert metadata["max_cell_occupancy"] == 2
    assert metadata["candidate_to_padded_ratio"] == 8 / 18
    assert metadata["candidate_to_active_ratio"] == 8 / 6


def test_torch_radius_nopbc_grouped_matches_loop_edges():
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import (
        torch_radius_nopbc_graph,
        torch_radius_nopbc_grouped_graph,
        torch_radius_nopbc_grouped_by_size_graph,
        torch_radius_nopbc_grouped_chunked_graph,
        torch_radius_nopbc_ragged_graph,
        torch_radius_nopbc_triton_padded_graph,
        torch_radius_nopbc_triton_counted_graph,
    )
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import RTECEGraph

    template = RTECEGraph(
        z=torch.tensor([6, 8, 1, 7, 1], dtype=torch.long),
        pos=torch.zeros((5, 3), dtype=torch.float64),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
    )
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [2.2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    loop_graph = torch_radius_nopbc_graph(template, positions, cutoff=1.0)
    grouped_graph = torch_radius_nopbc_grouped_graph(template, positions, cutoff=1.0)
    by_size_graph = torch_radius_nopbc_grouped_by_size_graph(template, positions, cutoff=1.0)
    chunked_graph = torch_radius_nopbc_grouped_chunked_graph(template, positions, cutoff=1.0, chunk_configs=1)
    ragged_graph = torch_radius_nopbc_ragged_graph(template, positions, cutoff=1.0)
    triton_padded_graph = torch_radius_nopbc_triton_padded_graph(template, positions, cutoff=1.0)
    triton_counted_graph = torch_radius_nopbc_triton_counted_graph(template, positions, cutoff=1.0)

    assert grouped_graph.z is template.z
    assert grouped_graph.batch is template.batch
    assert torch.equal(grouped_graph.pos, positions)
    assert torch.equal(grouped_graph.edge_index, loop_graph.edge_index)
    assert by_size_graph.z is template.z
    assert by_size_graph.batch is template.batch
    assert torch.equal(by_size_graph.pos, positions)
    assert torch.equal(by_size_graph.edge_index, loop_graph.edge_index)
    assert chunked_graph.z is template.z
    assert chunked_graph.batch is template.batch
    assert torch.equal(chunked_graph.pos, positions)
    assert torch.equal(chunked_graph.edge_index, loop_graph.edge_index)
    assert ragged_graph.z is template.z
    assert ragged_graph.batch is template.batch
    assert torch.equal(ragged_graph.pos, positions)
    assert torch.equal(ragged_graph.edge_index, loop_graph.edge_index)
    assert triton_padded_graph.z is template.z
    assert triton_padded_graph.batch is template.batch
    assert torch.equal(triton_padded_graph.pos, positions)
    assert torch.equal(triton_padded_graph.edge_index, loop_graph.edge_index)
    assert triton_counted_graph.z is template.z
    assert triton_counted_graph.batch is template.batch
    assert torch.equal(triton_counted_graph.pos, positions)
    assert torch.equal(triton_counted_graph.edge_index, loop_graph.edge_index)


def test_torch_radius_nopbc_update_backend_rebuilds_edges_within_each_batch():
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import make_graph_update_backend
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import RTECEGraph

    template = RTECEGraph(
        z=torch.tensor([6, 8, 1, 7], dtype=torch.long),
        pos=torch.zeros((4, 3), dtype=torch.float64),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=torch.tensor([0, 0, 0, 1], dtype=torch.long),
    )
    new_positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [2.2, 0.0, 0.0],
            [0.1, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    backend = make_graph_update_backend(
        backend_name="torch_radius_nopbc",
        rebuild_fn=lambda positions: (_ for _ in ()).throw(AssertionError("ASE rebuild should not run")),
        template_graph=template,
        cutoff=1.0,
    )
    graph = backend.rebuild(new_positions)

    assert graph.z is template.z
    assert graph.batch is template.batch
    assert torch.equal(graph.pos, new_positions)
    assert graph.pos is not template.pos
    assert graph.edge_index.tolist() == [[0, 1], [1, 0]]
    assert backend.name == "torch_radius_nopbc"
    assert backend.rebuild_count == 1


def test_chunked_torch_radius_update_backend_honors_chunk_size():
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import make_graph_update_backend
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import RTECEGraph

    template = RTECEGraph(
        z=torch.tensor([6, 8, 1, 7, 1], dtype=torch.long),
        pos=torch.zeros((5, 3), dtype=torch.float64),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
    )
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [2.2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    backend = make_graph_update_backend(
        backend_name="torch_radius_nopbc_grouped_chunked",
        rebuild_fn=lambda positions: (_ for _ in ()).throw(AssertionError("ASE rebuild should not run")),
        template_graph=template,
        cutoff=1.0,
        chunk_configs=1,
    )
    graph = backend.rebuild(positions)

    assert graph.edge_index.tolist() == [[0, 1, 3, 4], [1, 0, 4, 3]]


def test_chunked_torch_radius_update_backend_records_provider_work_metadata():
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import make_graph_update_backend
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import RTECEGraph

    template = RTECEGraph(
        z=torch.tensor([6, 8, 1, 7, 1], dtype=torch.long),
        pos=torch.zeros((5, 3), dtype=torch.float64),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
    )
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [2.2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    backend = make_graph_update_backend(
        backend_name="torch_radius_nopbc_grouped_chunked",
        rebuild_fn=lambda positions: (_ for _ in ()).throw(AssertionError("ASE rebuild should not run")),
        template_graph=template,
        cutoff=1.0,
        chunk_configs=1,
    )

    graph = backend.rebuild(positions)

    assert backend.static_metadata["num_configs"] == 2
    assert backend.static_metadata["max_atoms_per_config"] == 3
    assert backend.static_metadata["num_chunks"] == 2
    assert backend.static_metadata["padded_pair_slots"] == 13
    assert backend.last_metadata["num_directed_edges"] == graph.edge_index.shape[1] == 4

def test_grouped_chunked_torch_radius_update_backend_rebuilds_edges():
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import make_graph_update_backend
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import RTECEGraph

    template = RTECEGraph(
        z=torch.tensor([6, 8, 1, 7, 1], dtype=torch.long),
        pos=torch.zeros((5, 3), dtype=torch.float64),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
    )
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [2.2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    backend = make_graph_update_backend(
        backend_name="torch_radius_nopbc_grouped_chunked",
        rebuild_fn=lambda positions: (_ for _ in ()).throw(AssertionError("ASE rebuild should not run")),
        template_graph=template,
        cutoff=1.0,
    )
    graph = backend.rebuild(positions)

    assert backend.name == "torch_radius_nopbc_grouped_chunked"
    assert graph.edge_index.tolist() == [[0, 1, 3, 4], [1, 0, 4, 3]]
    assert backend.rebuild_count == 1


def test_grouped_by_size_torch_radius_update_backend_rebuilds_edges():
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import make_graph_update_backend
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import RTECEGraph

    template = RTECEGraph(
        z=torch.tensor([6, 8, 1, 7, 1], dtype=torch.long),
        pos=torch.zeros((5, 3), dtype=torch.float64),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
    )
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [2.2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    backend = make_graph_update_backend(
        backend_name="torch_radius_nopbc_grouped_by_size",
        rebuild_fn=lambda positions: (_ for _ in ()).throw(AssertionError("ASE rebuild should not run")),
        template_graph=template,
        cutoff=1.0,
    )
    graph = backend.rebuild(positions)

    assert backend.name == "torch_radius_nopbc_grouped_by_size"
    assert graph.edge_index.tolist() == [[0, 1, 3, 4], [1, 0, 4, 3]]
    assert backend.rebuild_count == 1


def test_triton_counted_torch_radius_update_backend_rebuilds_edges():
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import make_graph_update_backend
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import RTECEGraph

    template = RTECEGraph(
        z=torch.tensor([6, 8, 1, 7, 1], dtype=torch.long),
        pos=torch.zeros((5, 3), dtype=torch.float64),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
    )
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [2.2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    backend = make_graph_update_backend(
        backend_name="torch_radius_nopbc_triton_counted",
        rebuild_fn=lambda positions: (_ for _ in ()).throw(AssertionError("ASE rebuild should not run")),
        template_graph=template,
        cutoff=1.0,
    )
    graph = backend.rebuild(positions)

    assert backend.name == "torch_radius_nopbc_triton_counted"
    assert graph.edge_index.tolist() == [[0, 1, 3, 4], [1, 0, 4, 3]]
    assert backend.static_metadata["padded_pair_slots"] == 18
    assert backend.last_metadata["num_directed_edges"] == 4


def test_triton_padded_torch_radius_update_backend_rebuilds_edges():
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import make_graph_update_backend
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import RTECEGraph

    template = RTECEGraph(
        z=torch.tensor([6, 8, 1, 7, 1], dtype=torch.long),
        pos=torch.zeros((5, 3), dtype=torch.float64),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
    )
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [2.2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    backend = make_graph_update_backend(
        backend_name="torch_radius_nopbc_triton_padded",
        rebuild_fn=lambda positions: (_ for _ in ()).throw(AssertionError("ASE rebuild should not run")),
        template_graph=template,
        cutoff=1.0,
    )
    graph = backend.rebuild(positions)

    assert backend.name == "torch_radius_nopbc_triton_padded"
    assert graph.edge_index.tolist() == [[0, 1, 3, 4], [1, 0, 4, 3]]
    assert backend.static_metadata["padded_pair_slots"] == 18
    assert backend.last_metadata["num_directed_edges"] == 4


def test_ragged_torch_radius_update_backend_rebuilds_edges_and_records_exact_work():
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import make_graph_update_backend
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import RTECEGraph

    template = RTECEGraph(
        z=torch.tensor([6, 8, 1, 7, 1], dtype=torch.long),
        pos=torch.zeros((5, 3), dtype=torch.float64),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
    )
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [2.2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    backend = make_graph_update_backend(
        backend_name="torch_radius_nopbc_ragged",
        rebuild_fn=lambda positions: (_ for _ in ()).throw(AssertionError("ASE rebuild should not run")),
        template_graph=template,
        cutoff=1.0,
    )
    graph = backend.rebuild(positions)

    assert backend.name == "torch_radius_nopbc_ragged"
    assert graph.edge_index.tolist() == [[0, 1, 3, 4], [1, 0, 4, 3]]
    assert backend.static_metadata["exact_pair_slots"] == 13
    assert backend.static_metadata["padded_pair_slots"] == 13
    assert backend.static_metadata["padding_overhead_ratio"] == 1.0
    assert backend.last_metadata["num_directed_edges"] == 4


def test_grouped_torch_radius_update_backend_rebuilds_edges():
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import make_graph_update_backend
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import RTECEGraph

    template = RTECEGraph(
        z=torch.tensor([6, 8, 1, 7, 1], dtype=torch.long),
        pos=torch.zeros((5, 3), dtype=torch.float64),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
    )
    positions = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.9, 0.0, 0.0],
            [2.2, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.8, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )

    backend = make_graph_update_backend(
        backend_name="torch_radius_nopbc_grouped",
        rebuild_fn=lambda positions: (_ for _ in ()).throw(AssertionError("ASE rebuild should not run")),
        template_graph=template,
        cutoff=1.0,
    )
    graph = backend.rebuild(positions)

    assert backend.name == "torch_radius_nopbc_grouped"
    assert graph.edge_index.tolist() == [[0, 1, 3, 4], [1, 0, 4, 3]]
    assert backend.rebuild_count == 1


def test_graph_update_backend_records_rebuild_timing():
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import GraphUpdateBackend

    calls = []

    def rebuild(positions):
        calls.append(positions.clone())
        return {"nodes": positions.shape[0]}

    backend = GraphUpdateBackend("fake", rebuild)
    positions = torch.zeros((3, 3), dtype=torch.float64)

    result = backend.rebuild(positions)

    assert result == {"nodes": 3}
    assert len(calls) == 1
    assert backend.name == "fake"
    assert backend.rebuild_count == 1
    assert len(backend.rebuild_times_s) == 1
    assert backend.total_rebuild_time_s >= 0.0


def test_prediction_error_payload_marks_missing_predictions_as_null():
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import prediction_error_payload

    ref_e = __import__("numpy").array([1.0])
    ref_f = __import__("numpy").zeros((2, 3))
    natoms = __import__("numpy").array([2])

    payload = prediction_error_payload([], ref_e, ref_f, natoms)

    assert payload["prediction_errors_available"] is False
    assert payload["mae_e_mev_atom"] is None
    assert payload["mae_f_mev_a"] is None
    assert payload["rmse_e_mev_atom"] is None
    assert payload["rmse_f_mev_a"] is None


def test_prediction_error_payload_summarizes_available_predictions():
    import numpy as np
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import prediction_error_payload

    ref_e = np.array([1.0])
    ref_f = np.zeros((2, 3))
    natoms = np.array([2])
    outputs = [{"energy": ref_e.copy(), "forces": ref_f.copy()}]

    payload = prediction_error_payload(outputs, ref_e, ref_f, natoms)

    assert payload["prediction_errors_available"] is True
    assert payload["mae_e_mev_atom"] == 0.0
    assert payload["mae_f_mev_a"] == 0.0


def test_rtece_benchmark_help_exposes_force_mode():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py", "--help"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--force-mode" in result.stdout
    assert "--batch-graph-construction" in result.stdout
    assert "--replay-cached-graph" in result.stdout
    assert "--trajectory-replay-steps" in result.stdout
    assert "--trajectory-rebuild-interval" in result.stdout
    assert "--trajectory-displacement-std" in result.stdout
    assert "--trajectory-skin-margin" in result.stdout
    assert "--trajectory-validity-only" in result.stdout
    assert "--trajectory-update-only" in result.stdout
    assert "--graph-update-backend" in result.stdout
    assert "--graph-update-chunk-configs" in result.stdout
    assert "--graph-construction-backend" in result.stdout
    assert "torch_radius_nopbc" in result.stdout
    assert "torch_radius_nopbc_grouped" in result.stdout
    assert "torch_radius_nopbc_grouped_chunked" in result.stdout
    assert "torch_radius_nopbc_grouped_by_size" in result.stdout
    assert "torch_radius_nopbc_ragged" in result.stdout
    assert "torch_radius_nopbc_triton_padded" in result.stdout
    assert "torch_radius_nopbc_triton_counted" in result.stdout
    assert "auto" in result.stdout
    assert "analytic_pair" in result.stdout
    assert "analytic_pair_triton_force" in result.stdout
    assert "analytic_element_triton_force" in result.stdout
    assert "analytic_element_triton_descriptor_force" in result.stdout
    assert "analytic_density" in result.stdout
    assert "analytic_element_packed" in result.stdout


def test_rtece_train_help_exposes_num_radial():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "benchmarks/oc20neb_tace_mace/train_rtece_scalar.py", "--help"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--num-radial" in result.stdout
    assert "--seed" in result.stdout


def test_summary_extracts_force_throughput_pareto_front():
    from benchmarks.oc20neb_tace_mace.summarize_tece_distill import pareto_front_rows

    rows = [
        {"variant": "fast", "atoms_per_second": 18.0, "dft_f_mae_mev_a": 42.0},
        {"variant": "accurate", "atoms_per_second": 16.0, "dft_f_mae_mev_a": 36.0},
        {"variant": "dominated", "atoms_per_second": 12.0, "dft_f_mae_mev_a": 46.0},
        {"variant": "missing", "atoms_per_second": None, "dft_f_mae_mev_a": 10.0},
    ]

    front = pareto_front_rows(rows, error_key="dft_f_mae_mev_a")

    assert [row["variant"] for row in front] == ["fast", "accurate"]


def test_profile_rows_from_events_sorts_by_device_then_cpu_time():
    from types import SimpleNamespace
    from benchmarks.oc20neb_tace_mace.profile_rtece_scalar import profile_rows_from_events

    events = [
        SimpleNamespace(key="cpu_heavy", cpu_time_total=100.0, self_cpu_time_total=60.0, count=2),
        SimpleNamespace(key="cuda_heavy", cpu_time_total=10.0, self_cpu_time_total=5.0, count=1, device_time_total=300.0),
        SimpleNamespace(key="cuda_light", cpu_time_total=20.0, self_cpu_time_total=10.0, count=4, device_time_total=30.0),
    ]

    rows = profile_rows_from_events(events, limit=2)

    assert [row["name"] for row in rows] == ["cuda_heavy", "cuda_light"]
    assert rows[0]["device_time_total_us"] == 300.0
    assert rows[0]["cpu_time_total_us"] == 10.0
    assert rows[0]["count"] == 1

def test_rtece_benchmark_row_preserves_force_mode():
    from benchmarks.oc20neb_tace_mace.summarize_tece_distill import make_student_row

    dft = {
        "model": "rtece_scalar.pt",
        "force_mode": "analytic_pair",
        "hidden_channels": [16, 16],
        "num_radial": 4,
        "atomic_energies": {"1": -0.5, "6": -3.0},
        "atoms_per_second": 100000.0,
        "configs_per_second": 1000.0,
        "seconds_per_pass": 0.1,
        "peak_allocated_mb": 64.0,
        "peak_reserved_mb": 80.0,
        "num_parameters": 1234,
        "mae_e_mev_atom": 10.0,
        "rmse_e_mev_atom": 20.0,
        "mae_f_mev_a": 40.0,
        "rmse_f_mev_a": 80.0,
    }
    teacher = dict(dft)

    row = make_student_row("rtece_pair", dft_benchmark=dft, teacher_benchmark=teacher)

    assert row["force_mode"] == "analytic_pair"
    assert row["hidden_channels"] == [16, 16]
    assert row["num_radial"] == 4
    assert row["atomic_energies"] == {"1": -0.5, "6": -3.0}
    assert row["tece_route"]["energy_reference"] == "per_element_atomic_energies"


def test_rtece_summary_attaches_tece_route_contract():
    from benchmarks.oc20neb_tace_mace.summarize_tece_distill import (
        format_markdown,
        make_student_row,
    )

    dft = {
        "model": "rtece_scalar.pt",
        "variant": "rtece_element_density",
        "force_mode": "analytic_element_triton_descriptor_force",
        "graph_construction_backend": "torch_radius_nopbc",
        "graph_update_backend": "torch_radius_nopbc_triton_counted",
        "hidden_channels": [24, 24],
        "num_radial": 8,
        "atoms_per_second": 50929958.0,
        "configs_per_second": 83000.0,
        "seconds_per_pass": 0.1,
        "peak_allocated_mb": 212.1,
        "peak_reserved_mb": 240.0,
        "num_parameters": 1057,
        "mae_e_mev_atom": 1400.0,
        "rmse_e_mev_atom": 2000.0,
        "mae_f_mev_a": 30.19,
        "rmse_f_mev_a": 111.5,
    }
    teacher = dict(dft)
    teacher["mae_f_mev_a"] = 35.84

    row = make_student_row("radial8h24", dft_benchmark=dft, teacher_benchmark=teacher)
    markdown = format_markdown([row], baselines=[])

    assert row["tece_route"]["semantic_tier"] == "T3_element_conditioned_scalar_density"
    assert row["tece_route"]["force_realization"] == "triton_fused_descriptor_force"
    assert row["tece_route"]["edge_state_lifetime"] == "counted_exact_edge_buffer"
    assert "| variant | TECE route | graph backend | force mode |" in markdown
    assert "| radial8h24 | T3_element_conditioned_scalar_density" in markdown


def test_rtece_summary_preserves_graph_construction_backend():
    from benchmarks.oc20neb_tace_mace.summarize_tece_distill import (
        format_markdown,
        make_student_row,
    )

    dft = {
        "model": "rtece_scalar.pt",
        "force_mode": "analytic_element_triton_descriptor_force",
        "graph_construction_backend": "torch_radius_nopbc",
        "hidden_channels": [16, 16],
        "num_radial": 4,
        "atoms_per_second": 87762046.0,
        "configs_per_second": 350.0,
        "seconds_per_pass": 0.1,
        "peak_allocated_mb": 150.8,
        "peak_reserved_mb": 180.0,
        "num_parameters": 449,
        "mae_e_mev_atom": 1400.0,
        "rmse_e_mev_atom": 2000.0,
        "mae_f_mev_a": 43.85,
        "rmse_f_mev_a": 114.9,
    }
    teacher = dict(dft)
    teacher["mae_f_mev_a"] = 47.71

    row = make_student_row("radial4h16", dft_benchmark=dft, teacher_benchmark=teacher)
    markdown = format_markdown([row], baselines=[])

    assert row["graph_construction_backend"] == "torch_radius_nopbc"
    assert "| variant | TECE route | graph backend | force mode |" in markdown
    assert "| radial4h16 | T3_element_conditioned_scalar_density | torch_radius_nopbc | analytic_element_triton_descriptor_force |" in markdown


def test_rtece_matrix_sbatch_separates_training_and_benchmark_validation_files():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    script = (root / "benchmarks/oc20neb_tace_mace/rtece_scalar_matrix.sbatch").read_text()

    assert "TRAIN_VALID_FILE=${TRAIN_VALID_FILE:-${DFT_VALID_FILE}}" in script
    assert '--valid-file "${TRAIN_VALID_FILE}"' in script
    assert '--configs "${DFT_VALID_FILE}"' in script


def test_rtece_benchmark_sbatch_forwards_graph_backend_controls():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    script = (root / "benchmarks/oc20neb_tace_mace/rtece_scalar_benchmark.sbatch").read_text()

    assert "GRAPH_CONSTRUCTION_BACKEND=${GRAPH_CONSTRUCTION_BACKEND:-torch_radius_nopbc}" in script
    assert "GRAPH_UPDATE_BACKEND=${GRAPH_UPDATE_BACKEND:-ase_neighborlist}" in script
    assert '--graph-construction-backend "${GRAPH_CONSTRUCTION_BACKEND}"' in script
    assert '--graph-update-backend "${GRAPH_UPDATE_BACKEND}"' in script


def test_rtece_benchmark_submit_helper_generates_wrapper_without_sbatch_export(tmp_path):
    from benchmarks.oc20neb_tace_mace.submit_rtece_scalar_benchmark import (
        build_sbatch_command,
        write_rtece_benchmark_wrapper,
    )

    wrapper = write_rtece_benchmark_wrapper(
        tmp_path,
        model="/tmp/rtece.pt",
        force_mode="analytic_element_direct_padded_descriptor_force",
        limit_configs_list="64",
        measure_passes=1,
        out_dir="/tmp/rtece-bench",
    )
    command = build_sbatch_command(wrapper)
    text = wrapper.read_text()

    assert "--export" not in command
    assert "#SBATCH --gpus-per-node=1" in text
    assert "#SBATCH --qos=flood-1o2gpu" in text
    assert "--mem" not in text
    assert "--cpus-per-task" not in text
    assert "MODEL=/tmp/rtece.pt" in text
    assert "FORCE_MODE=analytic_element_direct_padded_descriptor_force" in text
    assert "exec /bin/bash" in text
    assert "rtece_scalar_benchmark.sbatch" in text


def test_rtece_matrix_sbatch_forwards_benchmark_force_mode():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    script = (root / "benchmarks/oc20neb_tace_mace/rtece_scalar_matrix.sbatch").read_text()

    assert "FORCE_MODE=${FORCE_MODE:-autograd}" in script
    assert '--force-mode "${FORCE_MODE}"' in script


def test_rtece_matrix_sbatch_forwards_num_radial():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    script = (root / "benchmarks/oc20neb_tace_mace/rtece_scalar_matrix.sbatch").read_text()

    assert "NUM_RADIAL=${NUM_RADIAL:-8}" in script
    assert '--num-radial "${NUM_RADIAL}"' in script


def test_rtece_matrix_sbatch_forwards_seed():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    script = (root / "benchmarks/oc20neb_tace_mace/rtece_scalar_matrix.sbatch").read_text()

    assert "SEED=${SEED:-0}" in script
    assert '--seed "${SEED}"' in script


def test_extxyz_index_supports_offset_windows():
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import extxyz_index

    assert extxyz_index(start_config=0, limit_configs=8) == ":8"
    assert extxyz_index(start_config=16, limit_configs=8) == "16:24"
    assert extxyz_index(start_config=16, limit_configs=None) == "16:"


def test_rtece_benchmark_help_exposes_start_config():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py", "--help"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--start-config" in result.stdout
