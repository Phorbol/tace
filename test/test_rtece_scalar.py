from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import replace

import pytest
import torch

from benchmarks.oc20neb_tace_mace.rtece_scalar_model import (
    RTECEGraph,
    RTECEScalarConfig,
    RTECEScalarModel,
    collate_graphs,
    compute_pair_geometry,
    compute_atomic_moments,
    atomic_scalar_descriptors,
    build_rtece_config_from_path_ids,
    build_rtece_config,
    cell_list_packed_element_density_descriptors,
    descriptor_dim,
    edge_relational_sketches,
    _project_radial_edge_channels,
    packed_element_density_descriptors,
    rtece_descriptors,
    short_range_repulsive_energy,
    rtece_path_manifest,
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
        rtece_path_manifest as core_rtece_path_manifest,
    )
    from tace.models.rtece_scalar import packed_element_density_descriptors as core_packed_descriptors

    assert CoreRTECEGraph is benchmark_rtece.RTECEGraph
    assert CoreRTECEScalarConfig is benchmark_rtece.RTECEScalarConfig
    assert CoreRTECEScalarModel is benchmark_rtece.RTECEScalarModel
    assert core_build_rtece_config is benchmark_rtece.build_rtece_config
    assert core_rtece_path_manifest is benchmark_rtece.rtece_path_manifest
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



def test_fit_atomic_energies_respects_energy_sample_weights():
    from tace.lightning.rtece import fit_atomic_energies

    graph_a = RTECEGraph(
        z=torch.tensor([1], dtype=torch.long),
        pos=torch.zeros((1, 3), dtype=torch.float64),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=torch.zeros(1, dtype=torch.long),
    )
    graph_b = RTECEGraph(
        z=torch.tensor([1], dtype=torch.long),
        pos=torch.zeros((1, 3), dtype=torch.float64),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=torch.zeros(1, dtype=torch.long),
    )
    samples = [
        (
            graph_a,
            torch.tensor([2.0], dtype=torch.float64),
            torch.zeros((1, 3), dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
        ),
        (
            graph_b,
            torch.tensor([100.0], dtype=torch.float64),
            torch.zeros((1, 3), dtype=torch.float64),
            torch.tensor([0.0], dtype=torch.float64),
            torch.tensor([1.0], dtype=torch.float64),
        ),
    ]

    atomic_energies = fit_atomic_energies(samples)

    assert atomic_energies[1] == pytest.approx(2.0)


def test_rtece_forward_can_skip_force_autograd_for_energy_only_diagnostics():
    config = build_rtece_config_from_path_ids(
        "energy_only",
        ("atomic.radial_density", "atomic.vector_norm", "edge.direct.radial"),
        cutoff=2.0,
        num_radial=4,
        hidden_channels=(8,),
        moment_l_max=1,
    )
    model = RTECEScalarModel(config).double().eval()
    graph = RTECEGraph(
        z=torch.tensor([6, 8, 1], dtype=torch.long),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [0.7, 0.2, 0.1], [-0.3, 0.6, -0.2]],
            dtype=torch.float64,
        ),
        edge_index=complete_directed_edges(3),
        batch=torch.zeros(3, dtype=torch.long),
    )

    with_forces = model(graph)
    energy_only = model(graph, compute_forces=False)

    assert set(energy_only) == {"energy", "atomic_energy"}
    assert torch.allclose(energy_only["energy"], with_forces["energy"], atol=1e-12, rtol=1e-12)
    assert torch.allclose(energy_only["atomic_energy"], with_forces["atomic_energy"], atol=1e-12, rtol=1e-12)


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
    assert "stress" not in pair["ase_implemented_outputs"]
    assert pair["stress_realization"] == "not_available_for_selected_force_mode"
    assert "ase_stress_requires_autograd_force_mode" in pair["missing_output_contracts"]
    assert "persistent_equivariant_edge_state" in pair["deleted_tece_groups"]

    assert element["semantic_tier"] == "T3_element_conditioned_scalar_density"
    assert element["descriptor_family"] == "element_density"
    assert element["descriptor_realization"] == "triton_fused_edge_descriptor"
    assert element["force_realization"] == "triton_fused_descriptor_force"
    assert element["fused_descriptor"] is True
    assert element["fused_force"] is True
    assert element["edge_state_lifetime"] == "counted_exact_edge_buffer"
    assert "neighbor_element_density" in element["retained_tece_groups"]

    cavity = rtece_route_contract(build_rtece_config("rtece_cavity_edge_sketch8"))
    assert cavity["semantic_tier"] == "T3_cavity_edge_scalar_sketch"
    assert cavity["descriptor_family"] == "cavity_atomic_moment_sketch"
    assert "cavity_edge_relational_scalar_sketches" in cavity["retained_tece_groups"]
    assert "direct_edge_radial_path" in cavity["retained_tece_groups"]

    radial_cavity = rtece_route_contract(build_rtece_config("rtece_cavity_radial_edge_sketch14"))
    assert radial_cavity["semantic_tier"] == "T3_cavity_radial_edge_scalar_sketch"
    assert radial_cavity["descriptor_family"] == "cavity_radial_atomic_moment_sketch"
    assert "low_rank_radial_edge_moment_sketches" in radial_cavity["retained_tece_groups"]
    assert "cross_radial_edge_invariants" in radial_cavity["retained_tece_groups"]


def test_build_rtece_config_defines_ordered_variants():
    pair = build_rtece_config("rtece_pair")
    element = build_rtece_config("rtece_element_density")
    quadratic = build_rtece_config("rtece_density_quadratic")
    vector = build_rtece_config("rtece_vector_moments")
    atomic = build_rtece_config("rtece_atomic_moments")
    sketch8 = build_rtece_config("rtece_edge_sketch8")
    cavity8 = build_rtece_config("rtece_cavity_edge_sketch8")
    radial_cavity14 = build_rtece_config("rtece_cavity_radial_edge_sketch14")
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
    assert cavity8.use_atomic_moments is True
    assert cavity8.use_cavity_edge_sketches is True
    assert cavity8.num_edge_sketches == 8
    assert descriptor_dim(cavity8) == descriptor_dim(sketch8)
    assert radial_cavity14.use_atomic_moments is True
    assert radial_cavity14.use_cavity_edge_sketches is True
    assert radial_cavity14.radial_edge_sketch_channels == 2
    assert radial_cavity14.num_edge_sketches == 14
    assert descriptor_dim(radial_cavity14) > descriptor_dim(cavity8)
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



def test_compute_pair_geometry_uses_periodic_edge_shifts():
    graph = RTECEGraph(
        z=torch.tensor([1, 1], dtype=torch.long),
        pos=torch.tensor([[0.1, 0.0, 0.0], [4.9, 0.0, 0.0]], dtype=torch.float64),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        batch=torch.zeros(2, dtype=torch.long),
        cell=torch.eye(3, dtype=torch.float64).unsqueeze(0) * 5.0,
        edge_shifts=torch.tensor([[-1, 0, 0], [1, 0, 0]], dtype=torch.long),
        edge_batch=torch.zeros(2, dtype=torch.long),
    )

    vectors, distances, unit = compute_pair_geometry(graph)

    assert torch.allclose(vectors, torch.tensor([[-0.2, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=torch.float64))
    assert torch.allclose(distances, torch.tensor([0.2, 0.2], dtype=torch.float64))
    assert torch.allclose(unit, torch.tensor([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64))


def test_train_atoms_to_graph_preserves_tace_matscipy_periodic_shifts():
    from ase import Atoms
    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import atoms_to_graph

    atoms = Atoms(
        "H2",
        positions=[[0.1, 0.0, 0.0], [4.9, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    atoms.info["energy"] = 0.0
    atoms.arrays["forces"] = torch.zeros((2, 3), dtype=torch.float64).numpy()

    graph, _energy, _forces = atoms_to_graph(
        atoms,
        cutoff=0.5,
        device=torch.device("cpu"),
        dtype=torch.float64,
        neighborlist_backend="matscipy",
    )
    _vectors, distances, _unit = compute_pair_geometry(graph)

    assert graph.cell is not None
    assert graph.edge_shifts is not None
    assert graph.edge_batch is not None
    assert graph.edge_index.shape[1] == 2
    assert torch.allclose(distances, torch.tensor([0.2, 0.2], dtype=torch.float64), atol=1e-12)


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



def test_species_basis_descriptors_distinguish_equal_z_sum_neighbors():
    element_config = RTECEScalarConfig(
        variant="rtece_element_density",
        cutoff=2.0,
        num_radial=3,
        use_element_density=True,
        max_atomic_number=10,
    )
    species_config = RTECEScalarConfig(
        variant="rtece_species_basis4",
        cutoff=2.0,
        num_radial=3,
        species_basis_channels=2,
        max_atomic_number=10,
    )
    edge_index = torch.tensor([[1, 2], [0, 0]], dtype=torch.long)
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0], [-0.8, 0.0, 0.0]], dtype=torch.float64)
    cc = RTECEGraph(
        z=torch.tensor([1, 6, 6], dtype=torch.long),
        pos=pos,
        edge_index=edge_index,
        batch=torch.zeros(3, dtype=torch.long),
    )
    bn = RTECEGraph(
        z=torch.tensor([1, 5, 7], dtype=torch.long),
        pos=pos,
        edge_index=edge_index,
        batch=torch.zeros(3, dtype=torch.long),
    )

    element_cc = atomic_scalar_descriptors(cc, element_config)[0]
    element_bn = atomic_scalar_descriptors(bn, element_config)[0]
    species_cc = atomic_scalar_descriptors(cc, species_config)[0]
    species_bn = atomic_scalar_descriptors(bn, species_config)[0]

    assert torch.allclose(element_cc, element_bn, atol=1e-12, rtol=1e-12)
    assert not torch.allclose(species_cc, species_bn, atol=1e-12, rtol=1e-12)


def test_rtece_path_manifest_has_stable_path_ids_and_hash():
    radial = build_rtece_config("rtece_cavity_radial_edge_sketch14")
    species = build_rtece_config("rtece_species_basis4")

    manifest = rtece_path_manifest(radial, force_mode="autograd")
    manifest_again = rtece_path_manifest(radial, force_mode="autograd")
    species_manifest = rtece_path_manifest(species, force_mode="autograd")

    assert manifest["schema_version"] == "rtece_path_manifest.v1"
    assert manifest["manifest_hash"] == manifest_again["manifest_hash"]
    assert manifest["manifest_hash"] != species_manifest["manifest_hash"]
    assert len(manifest["manifest_hash"]) == 16
    assert manifest["route"]["semantic_tier"] == "T3_cavity_radial_edge_scalar_sketch"
    assert any(item["id"] == "moment.l1.vector" for item in manifest["moments"])
    assert any(
        item["id"] == "edge.cavity.vector_cross_radial_dot"
        and item["placement"] == "edge"
        and item["cavity"] is True
        and item["radial_projection"] == "fixed_two_shell_mean"
        for item in manifest["scalar_paths"]
    )
    assert any(item["id"] == "edge.direct.radial" for item in manifest["scalar_paths"])
    assert "persistent_equivariant_edge_state" in manifest["deleted_tece_groups"]


def test_rtece_route_and_manifest_expose_ase_stress_and_unimplemented_virial_contract():
    config = build_rtece_config("rtece_cavity_radial_edge_sketch14")

    route = rtece_route_contract(config)
    manifest = rtece_path_manifest(config)

    assert route["implemented_outputs"] == ["energy", "atomic_energy", "forces"]
    assert route["ase_implemented_outputs"] == ["energy", "free_energy", "forces", "stress"]
    assert route["stress_realization"] == "ase_autograd_finite_strain_inference"
    assert route["virial_realization"] == "not_implemented"
    assert "validated_edge_gradient_virial" in route["missing_output_contracts"]
    assert "stress" not in route["implemented_outputs"]
    assert "ase_stress" not in route["missing_output_contracts"]
    assert manifest["output_contract"]["implemented"] == ["energy", "atomic_energy", "forces"]
    assert manifest["output_contract"]["ase_implemented"] == ["energy", "free_energy", "forces", "stress"]
    assert manifest["output_contract"]["missing"] == {"virial": "requires validated edge-gradient virial backend"}
    assert manifest["compiler_status"] == "explicit_manifest_not_full_compiler"


def test_rtece_path_manifest_scalar_paths_have_compiler_semantics():
    manifest = rtece_path_manifest(build_rtece_config("rtece_cavity_radial_edge_sketch14"))

    required = {"id", "placement", "inputs", "contraction", "cavity", "parity", "cutoff_power", "radial_gate", "cost_group"}
    for path in manifest["scalar_paths"]:
        assert required <= set(path), path
        assert path["parity"] in (-1, 1)
        assert isinstance(path["cutoff_power"], int)
        assert path["cutoff_power"] >= 0
    edge_direct = next(path for path in manifest["scalar_paths"] if path["id"] == "edge.direct.radial")
    assert edge_direct["radial_gate"] == "edge_cutoff_envelope"
    assert edge_direct["cutoff_power"] == 1


def test_rtece_moment_l_max_controls_atomic_descriptor_bandwidth():
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import config_with_moment_l_max

    base = RTECEScalarConfig(variant="lmax", num_radial=3, hidden_channels=(4,))

    l0 = config_with_moment_l_max(base, 0)
    l1 = config_with_moment_l_max(base, 1)
    l2 = config_with_moment_l_max(base, 2)

    assert descriptor_dim(l0) == 3
    assert [item["ell"] for item in rtece_path_manifest(l0)["moments"]] == [0]
    assert [item["id"] for item in rtece_path_manifest(l0)["scalar_paths"]] == ["atomic.radial_density"]
    assert descriptor_dim(l1) == 6
    assert [item["ell"] for item in rtece_path_manifest(l1)["moments"]] == [0, 1]
    assert "atomic.vector_norm" in [item["id"] for item in rtece_path_manifest(l1)["scalar_paths"]]
    assert descriptor_dim(l2) == 9
    assert [item["ell"] for item in rtece_path_manifest(l2)["moments"]] == [0, 1, 2]
    assert "atomic.quadrupole_norm" in [item["id"] for item in rtece_path_manifest(l2)["scalar_paths"]]
    assert rtece_route_contract(l2)["moment_l_max"] == 2
    assert "angular_bandwidth_l_max" in rtece_route_contract(l2)["pareto_axes"]


def test_rtece_path_id_moment_l_max_keeps_vector_cavity_at_l1():
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import build_rtece_config_from_path_ids

    config = build_rtece_config_from_path_ids(
        "l1_cavity",
        ("atomic.radial_density", "edge.cavity.vector_dot", "edge.direct.radial"),
        num_radial=3,
        moment_l_max=1,
    )
    graph = RTECEGraph(
        z=torch.tensor([1, 6, 8], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [0.2, 0.9, 0.0]], dtype=torch.float64),
        edge_index=complete_directed_edges(3),
        batch=torch.zeros(3, dtype=torch.long),
    )

    manifest = rtece_path_manifest(config)
    descriptors = rtece_descriptors(graph, config)

    assert config.moment_l_max == 1
    assert config.use_atomic_moments is False
    assert [item["ell"] for item in manifest["moments"]] == [0, 1]
    assert descriptors.shape == (3, descriptor_dim(config))


def test_rtece_moment_l_max_rejects_unsupported_orders():
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import config_with_moment_l_max

    with pytest.raises(ValueError, match="moment_l_max"):
        config_with_moment_l_max(RTECEScalarConfig(variant="bad"), 3)


def test_rtece_learnable_radial_mixing_initializes_as_fixed_feature_extractor():
    fixed = RTECEScalarModel(RTECEScalarConfig(variant="fixed", num_radial=3, hidden_channels=(4,))).double()
    learnable_config = RTECEScalarConfig(
        variant="learnable",
        num_radial=3,
        hidden_channels=(4,),
        learnable_radial_mixing=True,
    )
    learnable = RTECEScalarModel(learnable_config).double()
    learnable.energy_head.load_state_dict(fixed.energy_head.state_dict())
    graph = RTECEGraph(
        z=torch.tensor([1, 6, 8], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [0.2, 0.9, 0.0]], dtype=torch.float64),
        edge_index=complete_directed_edges(3),
        batch=torch.zeros(3, dtype=torch.long),
    )

    fixed_out = fixed(graph)
    learnable_out = learnable(graph)

    assert "radial_mixing.weight" in learnable.state_dict()
    assert torch.allclose(learnable.radial_mixing.weight, torch.eye(3, dtype=torch.float64))
    assert torch.allclose(learnable_out["energy"], fixed_out["energy"], atol=1e-12)
    assert torch.allclose(learnable_out["forces"], fixed_out["forces"], atol=1e-12)


def test_rtece_learnable_radial_mixing_is_manifested_as_trainable_feature_path():
    config = RTECEScalarConfig(
        variant="learnable_radial",
        num_radial=4,
        hidden_channels=(8,),
        learnable_radial_mixing=True,
    )

    route = rtece_route_contract(config)
    manifest = rtece_path_manifest(config)

    assert route["feature_extractor"] == "learnable_radial_linear_mixing"
    assert "trainable_low_rank_radial_mixing" in route["retained_tece_groups"]
    assert "trainable_feature_extractor" in route["pareto_axes"]
    assert manifest["config"]["learnable_radial_mixing"] is True
    assert manifest["moments"][0]["radial_projection"] == "learnable_identity_initialized_linear_mixing"


def test_rtece_radial_species_adapter_initializes_as_fixed_feature_extractor():
    fixed_config = RTECEScalarConfig(variant="fixed", num_radial=4, hidden_channels=(4,))
    adapted_config = RTECEScalarConfig(
        variant="radial_species_adapter",
        num_radial=4,
        hidden_channels=(4,),
        radial_species_adapter_channels=3,
    )
    fixed = RTECEScalarModel(fixed_config).double()
    adapted = RTECEScalarModel(adapted_config).double()
    adapted.energy_head.load_state_dict(fixed.energy_head.state_dict())
    graph = RTECEGraph(
        z=torch.tensor([1, 6, 8], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [0.2, 0.9, 0.0]], dtype=torch.float64),
        edge_index=complete_directed_edges(3),
        batch=torch.zeros(3, dtype=torch.long),
    )

    fixed_out = fixed(graph)
    adapted_out = adapted(graph)

    assert "radial_species_adapter.projection.weight" in adapted.state_dict()
    assert torch.count_nonzero(adapted.radial_species_adapter.projection.weight) == 0
    assert torch.allclose(adapted_out["energy"], fixed_out["energy"], atol=1e-12)
    assert torch.allclose(adapted_out["forces"], fixed_out["forces"], atol=1e-12)


def test_rtece_radial_species_adapter_is_manifested_and_checkpointed(tmp_path):
    from tace.models.rtece_workflow import load_checkpoint, save_checkpoint

    config = RTECEScalarConfig(
        variant="radial_species_adapter",
        num_radial=4,
        hidden_channels=(8,),
        radial_species_adapter_channels=5,
    )
    model = RTECEScalarModel(config).double()
    path = tmp_path / "rtece_radial_species_adapter.pt"

    manifest = rtece_path_manifest(config)
    route = rtece_route_contract(config)
    save_checkpoint(path, model, config)
    _loaded_model, loaded_config, metadata = load_checkpoint(path, dtype=torch.float64)

    assert loaded_config == config
    assert route["feature_extractor"] == "fixed_radial_basis+learnable_edge_species_radial_adapter"
    assert "trainable_edge_species_radial_basis" in route["retained_tece_groups"]
    assert "trainable_edge_species_radial_basis" in route["pareto_axes"]
    assert manifest["config"]["radial_species_adapter_channels"] == 5
    assert manifest["moments"][0]["chemistry_basis"] == "learnable_center_neighbor_pair_embedding_5"
    assert metadata["tece_path_manifest"]["config"]["radial_species_adapter_channels"] == 5


def test_rtece_radial_species_adapter_scope_selects_atomic_or_edge_paths():
    config = build_rtece_config_from_path_ids(
        "scope_atomic",
        ("atomic.radial_density", "edge.direct.radial"),
        num_radial=4,
        hidden_channels=(8,),
        moment_l_max=0,
        radial_species_adapter_channels=3,
        radial_species_adapter_scope="atomic",
    )
    model = RTECEScalarModel(config).double()
    with torch.no_grad():
        model.radial_species_adapter.projection.weight.fill_(0.05)
    graph = RTECEGraph(
        z=torch.tensor([1, 6, 8], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [0.2, 0.9, 0.0]], dtype=torch.float64),
        edge_index=complete_directed_edges(3),
        batch=torch.zeros(3, dtype=torch.long),
    )

    scoped = rtece_descriptors(graph, config, radial_species_adapter=model.radial_species_adapter)
    expected = torch.cat(
        [
            atomic_scalar_descriptors(graph, config, radial_species_adapter=model.radial_species_adapter),
            edge_relational_sketches(graph, config, radial_species_adapter=None),
        ],
        dim=-1,
    )
    all_scope = rtece_descriptors(
        graph,
        replace(config, radial_species_adapter_scope="all"),
        radial_species_adapter=model.radial_species_adapter,
    )

    assert torch.allclose(scoped, expected)
    assert not torch.allclose(scoped, all_scope)

    edge_config = replace(config, variant="scope_edge", radial_species_adapter_scope="edge")
    edge_scoped = rtece_descriptors(graph, edge_config, radial_species_adapter=model.radial_species_adapter)
    edge_expected = torch.cat(
        [
            atomic_scalar_descriptors(graph, edge_config, radial_species_adapter=None),
            edge_relational_sketches(graph, edge_config, radial_species_adapter=model.radial_species_adapter),
        ],
        dim=-1,
    )
    assert torch.allclose(edge_scoped, edge_expected)


def test_rtece_zbl_short_range_prior_matches_tace_zbl_basis():
    from tace.models.radial import ZBLBasis

    graph = RTECEGraph(
        z=torch.tensor([1, 6], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.35, 0.0, 0.0]], dtype=torch.float64),
        edge_index=complete_directed_edges(2),
        batch=torch.zeros(2, dtype=torch.long),
    )
    config = RTECEScalarConfig(
        variant="zbl",
        use_short_range_repulsion=True,
        short_range_repulsion_potential="zbl",
    )
    _, distances, _ = compute_pair_geometry(graph)
    node_attrs = torch.eye(2, dtype=torch.float64)
    atomic_numbers = torch.tensor([1, 6], dtype=torch.long)
    zbl = ZBLBasis("c2poly", trainable=False).to(dtype=torch.float64)

    expected = zbl(distances[:, None], node_attrs, graph.edge_index, atomic_numbers).sum().view(1)

    assert torch.allclose(short_range_repulsive_energy(graph, config), expected)


def test_rtece_short_range_repulsive_core_adds_conservative_repulsion():
    config = RTECEScalarConfig(
        variant="rtece_pair",
        hidden_channels=(),
        num_radial=4,
        use_short_range_repulsion=True,
        short_range_repulsion_strength=10.0,
        short_range_repulsion_beta=20.0,
        short_range_repulsion_radius_scale=1.0,
    )
    model = RTECEScalarModel(config).double()
    for parameter in model.energy_head.parameters():
        torch.nn.init.zeros_(parameter)

    short = RTECEGraph(
        z=torch.tensor([1, 1], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=torch.float64),
        edge_index=complete_directed_edges(2),
        batch=torch.zeros(2, dtype=torch.long),
    )
    long = RTECEGraph(
        z=short.z,
        pos=torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=torch.float64),
        edge_index=short.edge_index,
        batch=short.batch,
    )

    short_out = model(short)
    long_out = model(long)

    assert short_out["energy"].item() > long_out["energy"].item() + 0.5
    assert short_out["forces"][0, 0].item() < 0.0
    assert torch.allclose(short_out["forces"].sum(dim=0), torch.zeros(3, dtype=torch.float64), atol=1e-10)


def test_rtece_analytic_force_backend_rejects_training_mode():
    config = RTECEScalarConfig(variant="rtece_pair")
    model = RTECEScalarModel(config).double().train()
    graph = RTECEGraph(
        z=torch.tensor([1, 1], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64),
        edge_index=complete_directed_edges(2),
        batch=torch.zeros(2, dtype=torch.long),
    )

    with pytest.raises(RuntimeError, match="inference-only rTECE force backend"):
        model.forward_pair_analytic_forces(graph)


def test_rtece_analytic_force_backend_rejects_periodic_image_shifts():
    config = RTECEScalarConfig(variant="rtece_pair")
    model = RTECEScalarModel(config).double().eval()
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    graph = RTECEGraph(
        z=torch.tensor([1, 1], dtype=torch.long),
        pos=torch.tensor([[0.1, 0.0, 0.0], [4.9, 0.0, 0.0]], dtype=torch.float64),
        edge_index=edge_index,
        batch=torch.zeros(2, dtype=torch.long),
        cell=torch.eye(3, dtype=torch.float64).unsqueeze(0) * 5.0,
        edge_shifts=torch.tensor([[1, 0, 0], [-1, 0, 0]], dtype=torch.long),
        edge_batch=torch.zeros(edge_index.shape[1], dtype=torch.long),
    )

    with pytest.raises(ValueError, match="periodic image shifts"):
        model.forward_pair_analytic_forces(graph)


def test_rtece_short_range_repulsive_core_rejects_analytic_force_backend():
    config = RTECEScalarConfig(
        variant="rtece_pair",
        use_short_range_repulsion=True,
        short_range_repulsion_strength=1.0,
    )
    model = RTECEScalarModel(config).double().eval()
    graph = RTECEGraph(
        z=torch.tensor([1, 1], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0]], dtype=torch.float64),
        edge_index=complete_directed_edges(2),
        batch=torch.zeros(2, dtype=torch.long),
    )

    with pytest.raises(ValueError, match="short-range radial-core forces"):
        model.forward_pair_analytic_forces(graph)


def test_rtece_short_range_repulsive_core_is_manifested_and_checkpointed(tmp_path):
    from tace.models.rtece_workflow import load_checkpoint, save_checkpoint

    config = RTECEScalarConfig(
        variant="rtece_pair",
        hidden_channels=(4,),
        use_short_range_repulsion=True,
        short_range_repulsion_strength=3.0,
        short_range_repulsion_beta=8.0,
        short_range_repulsion_radius_scale=0.75,
    )
    model = RTECEScalarModel(config).double()
    path = tmp_path / "rtece_core.pt"

    manifest = rtece_path_manifest(config)
    save_checkpoint(path, model, config)
    _loaded_model, loaded_config, metadata = load_checkpoint(path, dtype=torch.float64)

    assert loaded_config == config
    assert manifest["config"]["short_range_repulsion"]["enabled"] is True
    assert manifest["config"]["short_range_repulsion"]["strength"] == pytest.approx(3.0)
    assert "short_range_radial_core" in manifest["retained_tece_groups"]
    assert "short_range_physical_prior" in manifest["route"]["pareto_axes"]
    assert metadata["tece_path_manifest"]["config"]["short_range_repulsion"]["radius_scale"] == pytest.approx(0.75)


def test_rtece_variant_registry_exposes_path_spec_architectures():
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import (
        available_rtece_variants,
        rtece_variant_registry,
    )

    variants = available_rtece_variants()
    registry = rtece_variant_registry()

    assert variants == tuple(registry)
    assert "rtece_cavity_radial_edge_sketch14" in registry
    radial = registry["rtece_cavity_radial_edge_sketch14"]
    assert radial["semantic_tier"] == "T3_cavity_radial_edge_scalar_sketch"
    assert "low_rank_radial_edge_moment_sketches" in radial["retained_tece_groups"]
    assert "edge.cavity.vector_cross_radial_dot" in radial["scalar_path_ids"]
    assert radial["config"]["radial_edge_sketch_channels"] == 2

    for variant in variants:
        config = build_rtece_config(variant)
        manifest = rtece_path_manifest(config)
        spec = registry[variant]
        assert spec["semantic_tier"] == manifest["route"]["semantic_tier"]
        assert set(spec["scalar_path_ids"]) == {path["id"] for path in manifest["scalar_paths"]}


def test_rtece_config_from_manifest_reconstructs_architecture_hash():
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import build_rtece_config_from_manifest

    config = RTECEScalarConfig(
        variant="rtece_cavity_radial_edge_sketch14",
        cutoff=4.5,
        num_radial=6,
        hidden_channels=(16, 32),
        use_atomic_moments=True,
        num_edge_sketches=14,
        use_cavity_edge_sketches=True,
        radial_edge_sketch_channels=2,
    )
    manifest = rtece_path_manifest(config, force_mode="autograd")
    rebuilt = build_rtece_config_from_manifest(manifest)

    assert rebuilt == config
    assert rtece_path_manifest(rebuilt, force_mode="autograd")["manifest_hash"] == manifest["manifest_hash"]


def test_rtece_formal_models_import_does_not_eager_load_optional_tace_backends():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    script = "import importlib.abc\nimport json\nimport sys\n\noptional = {\n    'tace.models._cart',\n    'tace.models._e3nn',\n    'tace.models.adapter',\n    'tace.models.compile',\n}\nattempts = []\n\nclass OptionalBackendProbe(importlib.abc.MetaPathFinder):\n    def find_spec(self, fullname, path=None, target=None):\n        if fullname in optional or any(fullname.startswith(name + '.') for name in optional):\n            attempts.append(fullname)\n            raise ImportError(f'blocked optional backend {fullname}')\n        return None\n\nsys.meta_path.insert(0, OptionalBackendProbe())\nfrom tace.models import RTECEScalarModel  # noqa: F401\nprint(json.dumps(attempts))"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


def test_rtece_route_registry_has_formal_tace_models_entrypoint():
    from benchmarks.oc20neb_tace_mace import rtece_scalar_model as benchmark_rtece
    from tace.models import (
        available_rtece_variants as core_available_rtece_variants,
        build_rtece_config_from_manifest as core_build_rtece_config_from_manifest,
        rtece_variant_registry as core_rtece_variant_registry,
    )

    assert core_available_rtece_variants is benchmark_rtece.available_rtece_variants
    assert core_build_rtece_config_from_manifest is benchmark_rtece.build_rtece_config_from_manifest
    assert core_rtece_variant_registry is benchmark_rtece.rtece_variant_registry


def test_rtece_path_id_config_drives_atomic_descriptor_order_and_dim():
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import build_rtece_config_from_path_ids

    config = build_rtece_config_from_path_ids(
        "rtece_path_density_square_vector",
        ("atomic.radial_density", "atomic.density_square", "atomic.vector_norm"),
        num_radial=4,
        hidden_channels=(8,),
    )
    graph = RTECEGraph(
        z=torch.tensor([6, 1, 8], dtype=torch.long),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [0.6, 0.0, 0.0], [0.0, 0.8, 0.0]],
            dtype=torch.float64,
        ),
        edge_index=complete_directed_edges(3),
        batch=torch.zeros(3, dtype=torch.long),
    )

    moments = compute_atomic_moments(graph, config)
    descriptors = rtece_descriptors(graph, config)

    assert config.scalar_path_ids == ("atomic.radial_density", "atomic.density_square", "atomic.vector_norm")
    assert config.use_density_quadratic is True
    assert config.use_vector_moments is True
    assert config.use_atomic_moments is False
    assert descriptor_dim(config) == 12
    assert descriptors.shape == (3, 12)
    assert torch.allclose(descriptors[:, :4], moments["density"])
    assert torch.allclose(descriptors[:, 4:8], moments["density"].square())
    assert torch.allclose(descriptors[:, 8:12], (moments["vector"] ** 2).sum(dim=-1))


def test_rtece_path_id_config_selects_atomic_cross_radial_moment_invariants():
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import build_rtece_config_from_path_ids

    config = build_rtece_config_from_path_ids(
        "rtece_atomic_cross_radial",
        (
            "atomic.radial_density",
            "atomic.vector_cross_radial_dot",
            "atomic.quadrupole_cross_radial_frobenius",
        ),
        num_radial=4,
        hidden_channels=(8,),
        moment_l_max=2,
    )
    graph = RTECEGraph(
        z=torch.tensor([6, 1, 8], dtype=torch.long),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [0.7, 0.1, 0.0], [0.2, 0.9, 0.1]],
            dtype=torch.float64,
        ),
        edge_index=complete_directed_edges(3),
        batch=torch.zeros(3, dtype=torch.long),
    )

    moments = compute_atomic_moments(graph, config)
    vector_shells = _project_radial_edge_channels(moments["vector"], 2)
    quadrupole_shells = _project_radial_edge_channels(moments["quadrupole"], 2)
    expected_vector_cross = (vector_shells[:, 0] * vector_shells[:, 1]).sum(dim=-1, keepdim=True)
    expected_quadrupole_cross = (quadrupole_shells[:, 0] * quadrupole_shells[:, 1]).sum(
        dim=(-1, -2),
    ).unsqueeze(-1)

    descriptors = rtece_descriptors(graph, config)
    manifest = rtece_path_manifest(config)

    assert descriptor_dim(config) == 6
    assert descriptors.shape == (3, 6)
    assert torch.allclose(descriptors[:, :4], moments["density"])
    assert torch.allclose(descriptors[:, 4:5], expected_vector_cross)
    assert torch.allclose(descriptors[:, 5:6], expected_quadrupole_cross)
    assert [path["id"] for path in manifest["scalar_paths"]] == [
        "atomic.radial_density",
        "atomic.vector_cross_radial_dot",
        "atomic.quadrupole_cross_radial_frobenius",
    ]
    assert "atomic_cross_radial_invariants" in manifest["retained_tece_groups"]


def test_rtece_path_id_manifest_reconstructs_selected_atomic_paths():
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import (
        build_rtece_config_from_manifest,
        build_rtece_config_from_path_ids,
    )

    config = build_rtece_config_from_path_ids(
        "rtece_path_species_density",
        ("atomic.radial_density", "atomic.species_basis_density"),
        num_radial=3,
        species_basis_channels=2,
    )
    manifest = rtece_path_manifest(config)
    rebuilt = build_rtece_config_from_manifest(manifest)

    assert [path["id"] for path in manifest["scalar_paths"]] == [
        "atomic.radial_density",
        "atomic.species_basis_density",
    ]
    assert manifest["config"]["scalar_path_ids"] == ["atomic.radial_density", "atomic.species_basis_density"]
    assert rebuilt == config
    assert rtece_path_manifest(rebuilt)["manifest_hash"] == manifest["manifest_hash"]


def test_rtece_path_id_constructor_rejects_unfactored_radial_edge_paths():
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import build_rtece_config_from_path_ids

    with pytest.raises(ValueError, match="non-radial edge scalar"):
        build_rtece_config_from_path_ids(
            "rtece_path_edge_radial_cross",
            ("atomic.radial_density", "edge.cavity.vector_cross_radial_dot"),
        )


def test_rtece_path_id_constructor_has_formal_tace_models_entrypoint():
    from benchmarks.oc20neb_tace_mace import rtece_scalar_model as benchmark_rtece
    from tace.models import build_rtece_config_from_path_ids as core_build_rtece_config_from_path_ids

    assert core_build_rtece_config_from_path_ids is benchmark_rtece.build_rtece_config_from_path_ids


def test_rtece_path_id_custom_order_rejects_packed_element_descriptor_backend():
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import build_rtece_config_from_path_ids

    config = build_rtece_config_from_path_ids(
        "rtece_path_element_first",
        ("atomic.element_density", "atomic.radial_density"),
        num_radial=2,
    )
    graph = RTECEGraph(
        z=torch.tensor([6, 1], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64),
        edge_index=complete_directed_edges(2),
        batch=torch.zeros(2, dtype=torch.long),
    )

    with pytest.raises(ValueError, match="canonical radial/element path order"):
        packed_element_density_descriptors(graph, config)


def test_rtece_path_id_config_selects_named_cavity_edge_column():
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import build_rtece_config_from_path_ids

    selected = build_rtece_config_from_path_ids(
        "rtece_path_cavity_vector_dot",
        ("atomic.radial_density", "edge.cavity.vector_dot"),
        num_radial=4,
        hidden_channels=(8,),
    )
    full = build_rtece_config("rtece_cavity_edge_sketch8")
    full = RTECEScalarConfig(
        variant=full.variant,
        cutoff=full.cutoff,
        num_radial=4,
        hidden_channels=full.hidden_channels,
        max_atomic_number=full.max_atomic_number,
        use_atomic_moments=full.use_atomic_moments,
        num_edge_sketches=full.num_edge_sketches,
        use_cavity_edge_sketches=full.use_cavity_edge_sketches,
    )
    graph = RTECEGraph(
        z=torch.tensor([6, 1, 8], dtype=torch.long),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [0.7, 0.1, 0.0], [0.2, 0.9, 0.1]],
            dtype=torch.float64,
        ),
        edge_index=complete_directed_edges(3),
        batch=torch.zeros(3, dtype=torch.long),
    )

    selected_descriptors = rtece_descriptors(graph, selected)
    full_edge_sketches = edge_relational_sketches(graph, full)
    manifest = rtece_path_manifest(selected)

    assert selected.scalar_path_ids == ("atomic.radial_density", "edge.cavity.vector_dot")
    assert selected.use_atomic_moments is True
    assert selected.use_cavity_edge_sketches is True
    assert selected.num_edge_sketches == 1
    assert descriptor_dim(selected) == 5
    assert selected_descriptors.shape == (3, 5)
    assert torch.allclose(selected_descriptors[:, :4], compute_atomic_moments(graph, selected)["density"])
    assert torch.allclose(selected_descriptors[:, 4:5], full_edge_sketches[:, :1])
    assert [path["id"] for path in manifest["scalar_paths"]] == [
        "atomic.radial_density",
        "edge.cavity.vector_dot",
    ]


def test_rtece_path_id_config_selects_full_moment_shell_edge_columns():
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import build_rtece_config_from_path_ids

    selected = build_rtece_config_from_path_ids(
        "rtece_path_full_shell_vector_dot",
        ("atomic.radial_density", "edge.full_moment.vector_shell_dot"),
        num_radial=4,
        hidden_channels=(8,),
    )
    full = RTECEScalarConfig(
        variant="rtece_edge_sketch16_reference",
        num_radial=4,
        hidden_channels=(8,),
        use_atomic_moments=True,
        num_edge_sketches=16,
    )
    graph = RTECEGraph(
        z=torch.tensor([6, 1, 8], dtype=torch.long),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [0.7, 0.1, 0.0], [0.2, 0.9, 0.1]],
            dtype=torch.float64,
        ),
        edge_index=complete_directed_edges(3),
        batch=torch.zeros(3, dtype=torch.long),
    )

    selected_descriptors = rtece_descriptors(graph, selected)
    full_edge_sketches = edge_relational_sketches(graph, full)
    manifest = rtece_path_manifest(selected)

    assert selected.scalar_path_ids == ("atomic.radial_density", "edge.full_moment.vector_shell_dot")
    assert selected.num_edge_sketches == 2
    assert descriptor_dim(selected) == 6
    assert selected_descriptors.shape == (3, 6)
    assert torch.allclose(selected_descriptors[:, :4], compute_atomic_moments(graph, selected)["density"])
    assert torch.allclose(selected_descriptors[:, 4:6], full_edge_sketches[:, 8:10])
    assert [path["id"] for path in manifest["scalar_paths"]] == [
        "atomic.radial_density",
        "edge.full_moment.vector_shell_dot",
    ]


def test_rtece_path_id_config_selects_mixed_full_shell_and_direct_edge_paths():
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import build_rtece_config_from_path_ids

    selected = build_rtece_config_from_path_ids(
        "rtece_path_full_shell_cross_direct",
        (
            "atomic.radial_density",
            "edge.full_moment.vector_cross_shell_dot",
            "edge.direct.radial",
        ),
        num_radial=4,
        hidden_channels=(8,),
    )
    full = RTECEScalarConfig(
        variant="rtece_edge_sketch16_reference",
        num_radial=4,
        hidden_channels=(8,),
        use_atomic_moments=True,
        num_edge_sketches=16,
    )
    graph = RTECEGraph(
        z=torch.tensor([6, 1, 8], dtype=torch.long),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [0.7, 0.1, 0.0], [0.2, 0.9, 0.1]],
            dtype=torch.float64,
        ),
        edge_index=complete_directed_edges(3),
        batch=torch.zeros(3, dtype=torch.long),
    )

    selected_edge = edge_relational_sketches(graph, selected)
    full_edge = edge_relational_sketches(graph, full)

    assert selected.num_edge_sketches == 3
    assert descriptor_dim(selected) == 7
    assert torch.allclose(selected_edge[:, :1], full_edge[:, 10:11])
    assert torch.allclose(selected_edge[:, 1:3], full_edge[:, 6:8])


def test_rtece_path_id_config_selects_cavity_edge_frame_projection_paths():
    selected = build_rtece_config_from_path_ids(
        "rtece_path_cavity_edge_frame_projections",
        (
            "atomic.radial_density",
            "edge.cavity.target_vector_projection",
            "edge.cavity.source_vector_projection",
            "edge.cavity.target_quadrupole_projection",
            "edge.cavity.source_quadrupole_projection",
        ),
        num_radial=4,
        hidden_channels=(8,),
    )
    full = RTECEScalarConfig(
        variant="rtece_cavity_edge_sketch8_reference",
        num_radial=4,
        hidden_channels=(8,),
        use_atomic_moments=True,
        use_cavity_edge_sketches=True,
        num_edge_sketches=8,
    )
    graph = RTECEGraph(
        z=torch.tensor([6, 1, 8], dtype=torch.long),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [0.7, 0.1, 0.0], [0.2, 0.9, 0.1]],
            dtype=torch.float64,
        ),
        edge_index=complete_directed_edges(3),
        batch=torch.zeros(3, dtype=torch.long),
    )

    selected_edge = edge_relational_sketches(graph, selected)
    full_edge = edge_relational_sketches(graph, full)

    assert selected.scalar_path_ids == (
        "atomic.radial_density",
        "edge.cavity.target_vector_projection",
        "edge.cavity.source_vector_projection",
        "edge.cavity.target_quadrupole_projection",
        "edge.cavity.source_quadrupole_projection",
    )
    assert selected.use_cavity_edge_sketches is True
    assert selected.num_edge_sketches == 4
    assert descriptor_dim(selected) == 8
    assert torch.allclose(selected_edge, full_edge[:, 2:6])


def test_rtece_cavity_edge_frame_projection_manifest_names_total_m0_contractions():
    config = build_rtece_config_from_path_ids(
        "rtece_path_cavity_edge_frame_manifest",
        (
            "atomic.radial_density",
            "edge.cavity.target_vector_projection",
            "edge.cavity.target_quadrupole_projection",
        ),
        num_radial=4,
        hidden_channels=(8,),
    )
    manifest = rtece_path_manifest(config)
    paths = {path["id"]: path for path in manifest["scalar_paths"]}

    assert paths["edge.cavity.target_vector_projection"]["contraction"] == "u_dot"
    assert paths["edge.cavity.target_vector_projection"]["inputs"] == [
        "edge.unit_vector",
        "moment.l1.vector",
    ]
    assert paths["edge.cavity.target_vector_projection"]["cavity"] is True
    assert paths["edge.cavity.target_quadrupole_projection"]["contraction"] == "uQu"
    assert paths["edge.cavity.target_quadrupole_projection"]["inputs"] == [
        "edge.unit_vector",
        "moment.l2.quadrupole",
        "edge.unit_vector",
    ]
    assert paths["edge.cavity.target_quadrupole_projection"]["cavity"] is True


def test_rtece_path_id_edge_manifest_reconstructs_selected_edge_paths():
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import (
        build_rtece_config_from_manifest,
        build_rtece_config_from_path_ids,
    )

    config = build_rtece_config_from_path_ids(
        "rtece_path_cavity_vector_direct_radial",
        ("atomic.radial_density", "edge.cavity.vector_dot", "edge.direct.radial"),
        num_radial=3,
    )
    manifest = rtece_path_manifest(config)
    rebuilt = build_rtece_config_from_manifest(manifest)

    assert descriptor_dim(config) == 3 + 1 + 2
    assert manifest["config"]["scalar_path_ids"] == [
        "atomic.radial_density",
        "edge.cavity.vector_dot",
        "edge.direct.radial",
    ]
    assert rebuilt == config
    assert rtece_path_manifest(rebuilt)["manifest_hash"] == manifest["manifest_hash"]


def test_rtece_atomic_cross_radial_sketch_channels_control_descriptor_rank():
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import build_rtece_config_from_path_ids

    config = build_rtece_config_from_path_ids(
        "l1_cross_k3",
        ("atomic.radial_density", "atomic.vector_norm", "atomic.vector_cross_radial_dot"),
        num_radial=6,
        moment_l_max=1,
        atomic_cross_radial_sketch_channels=3,
    )
    graph = RTECEGraph(
        z=torch.tensor([6, 1, 8], dtype=torch.long),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [0.7, 0.1, 0.0], [0.2, 0.9, 0.1]],
            dtype=torch.float64,
        ),
        edge_index=complete_directed_edges(3),
        batch=torch.zeros(3, dtype=torch.long),
    )

    desc = atomic_scalar_descriptors(graph, config)
    manifest = rtece_path_manifest(config)
    cross_path = next(path for path in manifest["scalar_paths"] if path["id"] == "atomic.vector_cross_radial_dot")

    assert config.atomic_cross_radial_sketch_channels == 3
    assert descriptor_dim(config) == 6 + 6 + 3
    assert desc.shape == (3, 15)
    assert cross_path["radial_projection"] == "fixed_3_shell_mean"
    assert cross_path["contraction"] == "off_diagonal_shell_dot"
    assert manifest["config"]["atomic_cross_radial_sketch_channels"] == 3
    assert "radial_rank" in manifest["route"]["pareto_axes"]


def test_rtece_path_manifest_hash_is_independent_of_variant_label():
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import build_rtece_config_from_path_ids

    paths = ("atomic.radial_density", "edge.cavity.vector_dot", "edge.direct.radial")
    left = build_rtece_config_from_path_ids("left_label", paths, num_radial=3)
    right = build_rtece_config_from_path_ids("right_label", paths, num_radial=3)

    left_manifest = rtece_path_manifest(left)
    right_manifest = rtece_path_manifest(right)

    assert left_manifest["config"]["variant"] == "left_label"
    assert right_manifest["config"]["variant"] == "right_label"
    assert left_manifest["manifest_hash"] == right_manifest["manifest_hash"]


def test_edge_sketch16_manifest_names_shell_resolved_scalar_paths():
    config = build_rtece_config("rtece_edge_sketch16")
    manifest = rtece_path_manifest(config)
    path_ids = [path["id"] for path in manifest["scalar_paths"]]

    assert "edge.full_moment.vector_dot" in path_ids
    assert "edge.full_moment.vector_shell_dot" in path_ids
    assert "edge.full_moment.vector_cross_shell_dot" in path_ids
    assert "edge.full_moment.quadrupole_shell_frobenius" in path_ids
    assert "edge.full_moment.quadrupole_cross_shell_frobenius" in path_ids
    assert "edge.full_moment.vector_shell_contrast_projection" in path_ids
    assert len(path_ids) == len(set(path_ids))


def test_rtece_projection_residual_metrics_detects_spanned_and_deleted_components():
    from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import projection_residual_metrics

    source = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, -1.0]],
        dtype=torch.float64,
    )
    spanned_target = torch.cat([source, (2.0 * source[:, :1] - source[:, 1:2])], dim=-1)
    unspanned_target = torch.cat([spanned_target, torch.tensor([[0.0], [1.0], [0.0], [-1.0]], dtype=torch.float64)], dim=-1)

    spanned = projection_residual_metrics(source, spanned_target)
    unspanned = projection_residual_metrics(source, unspanned_target)

    assert spanned["source_dim"] == 2
    assert spanned["target_dim"] == 3
    assert spanned["relative_residual"] < 1e-10
    assert unspanned["target_dim"] == 4
    assert unspanned["relative_residual"] > spanned["relative_residual"]


def test_rtece_projection_residual_metrics_accepts_sample_weights():
    from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import projection_residual_metrics

    source = torch.ones((3, 1), dtype=torch.float64)
    target = torch.tensor([[0.0], [10.0], [10.0]], dtype=torch.float64)

    unweighted = projection_residual_metrics(source, target)
    weighted = projection_residual_metrics(source, target, sample_weights=torch.tensor([100.0, 1.0, 1.0]))

    assert weighted["weighted"] is True
    assert weighted["weight_sum"] == pytest.approx(102.0)
    assert weighted["relative_residual"] > unweighted["relative_residual"]


def test_rtece_head_jacobian_weights_follow_energy_head_descriptor_gradient():
    from benchmarks.oc20neb_tace_mace.make_rtece_projection_weights import rtece_head_jacobian_sample_weights

    config = RTECEScalarConfig(num_radial=2, hidden_channels=(), variant="linear_sensitivity")
    model = RTECEScalarModel(config).to(dtype=torch.float64)
    linear = model.energy_head[0]
    with torch.no_grad():
        linear.weight.zero_()
        linear.bias.zero_()
        linear.weight[0, 1:] = torch.tensor([3.0, 4.0], dtype=torch.float64)

    graph = RTECEGraph(
        z=torch.tensor([6, 1], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=torch.float64),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        batch=torch.zeros(2, dtype=torch.long),
    )

    payload = rtece_head_jacobian_sample_weights(model, [graph], normalize="none")

    assert payload["weight_source"] == "rtece_head_jacobian_l2"
    assert payload["num_samples"] == 2
    assert payload["sample_weights"] == pytest.approx([5.0, 5.0])
    assert payload["normalization"] == "none"


def test_rtece_head_jacobian_weights_mean_normalize(tmp_path):
    from benchmarks.oc20neb_tace_mace.make_rtece_projection_weights import write_sample_weight_json

    payload = {
        "weight_source": "rtece_head_jacobian_l2",
        "sample_weights": [2.0, 4.0],
        "num_samples": 2,
        "normalization": "mean1",
    }
    path = tmp_path / "weights.json"

    write_sample_weight_json(path, payload)

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == "rtece_projection_sample_weights.v1"
    assert saved["weight_source"] == "rtece_head_jacobian_l2"
    assert saved["sample_weights"] == [2.0, 4.0]


def test_rtece_force_residual_weights_measure_model_force_error():
    from benchmarks.oc20neb_tace_mace.make_rtece_projection_weights import force_residual_sample_weight_payload

    predicted = [torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64)]
    reference = [torch.tensor([[0.0, 0.0, 0.0], [0.0, -1.0, 4.0]], dtype=torch.float64)]

    payload = force_residual_sample_weight_payload(
        predicted,
        reference,
        target_force_source="teacher_forces",
        normalize="none",
    )

    assert payload["weight_source"] == "rtece_force_residual_l2:teacher_forces"
    assert payload["sample_weights"] == pytest.approx([1.0, 5.0])
    assert payload["num_samples"] == 2


def test_rtece_force_residual_weights_mean_normalize():
    from benchmarks.oc20neb_tace_mace.make_rtece_projection_weights import force_residual_sample_weight_payload

    predicted = [torch.tensor([[1.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=torch.float64)]
    reference = [torch.zeros((2, 3), dtype=torch.float64)]

    payload = force_residual_sample_weight_payload(predicted, reference, target_force_source="dft_forces")

    assert payload["normalization"] == "mean1"
    assert payload["sample_weights"] == pytest.approx([0.5, 1.5])
    assert payload["weight_mean"] == pytest.approx(1.0)


def test_rtece_projection_weight_stratification_reports_element_and_focus_groups():
    from benchmarks.oc20neb_tace_mace.stratify_rtece_projection_weights import stratify_symbol_weights

    summary = stratify_symbol_weights(
        symbols=["C", "N", "H", "Cu"],
        weights=[4.0, 2.0, 1.0, 3.0],
        top_fraction=0.5,
    )
    elements = {row["label"]: row for row in summary["elements"]}
    groups = {row["label"]: row for row in summary["focus_groups"]}

    assert summary["num_samples"] == 4
    assert summary["total_weight"] == pytest.approx(10.0)
    assert elements["C"]["count"] == 1
    assert elements["C"]["weight_fraction"] == pytest.approx(0.4)
    assert groups["C_or_N"]["count"] == 2
    assert groups["C_or_N"]["weight_sum"] == pytest.approx(6.0)
    assert groups["CHNO"]["count"] == 3
    assert groups["not_CHNO"]["weight_fraction"] == pytest.approx(0.3)
    assert groups["C_or_N"]["top_weight_fraction"] == pytest.approx(4.0 / 7.0)


def test_rtece_projection_weight_stratification_rejects_misaligned_weights():
    from benchmarks.oc20neb_tace_mace.stratify_rtece_projection_weights import stratify_symbol_weights

    with pytest.raises(ValueError, match="symbols/weights length mismatch"):
        stratify_symbol_weights(symbols=["C", "N"], weights=[1.0])


def test_rtece_dimer_scan_distances_follow_covalent_radius_scale():
    from benchmarks.oc20neb_tace_mace.dimer_scan_rtece import dimer_distances_from_covalent_radii

    distances = dimer_distances_from_covalent_radii("C", "N", num_points=4, min_scale=0.5, max_scale=5.0)

    assert len(distances) == 4
    assert distances[0] == pytest.approx(0.5 * (0.76 + 0.71))
    assert distances[-1] == pytest.approx(5.0 * (0.76 + 0.71))
    assert distances == sorted(distances)


def test_rtece_dimer_scan_summary_reports_smoothness_and_nonfinite_counts():
    from benchmarks.oc20neb_tace_mace.dimer_scan_rtece import summarize_dimer_scan_rows

    rows = [
        {"distance_a": 1.0, "energy_eV": -1.0, "force_parallel_ev_a": 0.5, "max_force_norm_ev_a": 0.5},
        {"distance_a": 2.0, "energy_eV": -1.4, "force_parallel_ev_a": 0.2, "max_force_norm_ev_a": 0.2},
        {"distance_a": 3.0, "energy_eV": -1.3, "force_parallel_ev_a": -0.1, "max_force_norm_ev_a": 0.1},
    ]

    summary = summarize_dimer_scan_rows(rows)

    assert summary["num_points"] == 3
    assert summary["num_nonfinite_energy"] == 0
    assert summary["num_nonfinite_force"] == 0
    assert summary["energy_range_eV"] == pytest.approx(0.4)
    assert summary["max_abs_force_ev_a"] == pytest.approx(0.5)
    assert summary["max_abs_energy_step_eV"] == pytest.approx(0.4)
    assert summary["max_abs_force_step_ev_a"] == pytest.approx(0.3)
    assert summary["short_minus_long_energy_eV"] == pytest.approx(0.3)
    assert summary["short_force_parallel_ev_a"] == pytest.approx(0.5)
    assert summary["short_force_repulsive"] is False


def test_rtece_rattle_relax_focus_group_classification():
    from benchmarks.oc20neb_tace_mace.rattle_relax_rtece import classify_focus_groups

    assert classify_focus_groups(["Cu", "C", "H"]) == ["C_or_N", "CHNO"]
    assert classify_focus_groups(["Cu", "O", "H"]) == ["CHNO", "CHNO_no_CN"]
    assert classify_focus_groups(["Cu", "Ag"]) == ["not_CHNO"]


def test_rtece_rattle_relax_summary_tracks_rmsd_and_force_spikes():
    from benchmarks.oc20neb_tace_mace.rattle_relax_rtece import (
        positions_rmsd,
        summarize_relax_records,
    )

    ref = torch.zeros((2, 3), dtype=torch.float64)
    shifted = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64)
    assert positions_rmsd(shifted, ref) == pytest.approx((5.0 / 2.0) ** 0.5)

    records = [
        {"focus_groups": ["C_or_N", "CHNO"], "converged": True, "initial_rmsd_a": 0.1, "final_rmsd_a": 0.2, "max_fmax_ev_a": 1.5},
        {"focus_groups": ["not_CHNO"], "converged": False, "initial_rmsd_a": 0.2, "final_rmsd_a": 0.5, "max_fmax_ev_a": 3.0},
    ]
    summary = summarize_relax_records(records)
    groups = {row["label"]: row for row in summary["focus_groups"]}

    assert summary["num_configs"] == 2
    assert summary["converged_fraction"] == pytest.approx(0.5)
    assert summary["mean_final_rmsd_a"] == pytest.approx(0.35)
    assert summary["max_fmax_ev_a"] == pytest.approx(3.0)
    assert groups["C_or_N"]["count"] == 1
    assert groups["C_or_N"]["mean_final_rmsd_a"] == pytest.approx(0.2)
    assert groups["not_CHNO"]["converged_fraction"] == pytest.approx(0.0)


def test_make_rattle_distill_configs_preserves_training_labels_and_records_source_metadata():
    import numpy as np
    from ase import Atoms

    from benchmarks.oc20neb_tace_mace.make_rattle_distill_configs import make_rattle_distill_configs
    from benchmarks.oc20neb_tace_mace.rattle_relax_rtece import positions_rmsd

    atoms = Atoms(
        "CH",
        positions=[[0.0, 0.0, 0.0], [0.8, 0.1, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    atoms.info["energy"] = -1.25
    atoms.info["case_id"] = "case-7"
    atoms.arrays["forces"] = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=np.float64)

    rattled, summary = make_rattle_distill_configs(
        [atoms],
        copies_per_config=2,
        rattle_std_a=0.05,
        seed=123,
    )

    assert len(rattled) == 2
    assert summary["configs"] == 2
    assert summary["source_configs"] == 1
    assert summary["mean_initial_rmsd_a"] > 0.0
    assert rattled[0].info["energy"] == pytest.approx(-1.25)
    assert np.allclose(rattled[0].arrays["forces"], atoms.arrays["forces"])
    assert rattled[0].info["rattle_source_config_index"] == 0
    assert rattled[0].info["rattle_copy_index"] == 0
    assert rattled[0].info["rattle_source_case_id"] == "case-7"
    assert positions_rmsd(rattled[0].positions, atoms.positions) == pytest.approx(rattled[0].info["rattle_initial_rmsd_a"])
    assert np.allclose((rattled[0].positions - atoms.positions).mean(axis=0), np.zeros(3), atol=1e-14)
    assert not np.allclose(rattled[0].positions, rattled[1].positions)


def test_make_rattle_distill_configs_cli_runs_from_repo_script_path(tmp_path):
    import json
    import numpy as np
    import ase.io
    from ase import Atoms

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    source = tmp_path / "source.extxyz"
    output = tmp_path / "rattled.extxyz"
    summary = tmp_path / "summary.json"
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]])
    atoms.info["energy"] = -0.5
    atoms.arrays["forces"] = np.zeros((2, 3), dtype=np.float64)
    ase.io.write(source, [atoms], format="extxyz")

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/oc20neb_tace_mace/make_rattle_distill_configs.py",
            "--input",
            str(source),
            "--output",
            str(output),
            "--summary",
            str(summary),
            "--copies-per-config",
            "1",
            "--rattle-std-a",
            "0.01",
            "--seed",
            "9",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["configs"] == 1
    assert output.exists()
    assert summary.exists()


def test_make_rattle_distill_configs_cli_supports_start_config_window(tmp_path):
    import json
    import numpy as np
    import ase.io
    from ase import Atoms

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    source = tmp_path / "source.extxyz"
    output = tmp_path / "rattled.extxyz"
    summary = tmp_path / "summary.json"
    frames = []
    for idx in range(4):
        atoms = Atoms("H", positions=[[float(idx), 0.0, 0.0]])
        atoms.info["energy"] = float(-idx)
        atoms.info["case_id"] = f"case-{idx}"
        atoms.arrays["forces"] = np.zeros((1, 3), dtype=np.float64)
        frames.append(atoms)
    ase.io.write(source, frames, format="extxyz")

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/oc20neb_tace_mace/make_rattle_distill_configs.py",
            "--input",
            str(source),
            "--output",
            str(output),
            "--summary",
            str(summary),
            "--start-config",
            "2",
            "--limit-configs",
            "1",
            "--copies-per-config",
            "1",
            "--rattle-std-a",
            "0.01",
            "--seed",
            "9",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["start_config"] == 2
    assert payload["source_configs"] == 1
    generated = ase.io.read(output, index=":")
    assert generated[0].info["rattle_source_absolute_config_index"] == 2
    assert generated[0].info["rattle_source_case_id"] == "case-2"


def test_make_rattle_distill_configs_requires_standard_energy_force_targets():
    from ase import Atoms

    from benchmarks.oc20neb_tace_mace.make_rattle_distill_configs import make_rattle_distill_configs

    atoms = Atoms("H", positions=[[0.0, 0.0, 0.0]])

    with pytest.raises(KeyError, match="energy"):
        make_rattle_distill_configs([atoms], copies_per_config=1, rattle_std_a=0.01, seed=5)


def test_rtece_physical_pareto_row_combines_benchmark_dimer_and_rattle_gates():
    from benchmarks.oc20neb_tace_mace.summarize_rtece_physical_pareto import (
        make_physical_pareto_row,
        physical_pareto_front_rows,
    )

    benchmark = {
        "variant": "radial_core_balanced",
        "atoms_per_second": 3.1e6,
        "mae_f_mev_a": 29.0,
        "rmse_f_mev_a": 101.0,
        "mae_e_mev_atom": 168.0,
        "tece_architecture_path_manifest_hash": "abc123",
    }
    teacher = {"mae_f_mev_a": 34.0, "rmse_f_mev_a": 102.0}
    dimer = {
        "pair_summaries": [
            {"pair": "C-N", "summary": {"short_force_repulsive": True, "has_nonfinite": False, "short_force_parallel_ev_a": -0.25}},
            {"pair": "N-H", "summary": {"short_force_repulsive": True, "has_nonfinite": False, "short_force_parallel_ev_a": -0.20}},
        ]
    }
    rattle = {
        "summary": {
            "converged_fraction": 0.0,
            "mean_final_rmsd_a": 0.18,
            "max_fmax_ev_a": 0.34,
            "focus_groups": [
                {"label": "C_or_N", "mean_final_rmsd_a": 0.18, "max_fmax_ev_a": 0.34},
                {"label": "CHNO_no_CN", "mean_final_rmsd_a": 0.17, "max_fmax_ev_a": 0.06},
            ],
        }
    }

    row = make_physical_pareto_row(
        "radial_core_balanced",
        dft_benchmark=benchmark,
        teacher_benchmark=teacher,
        dimer_scan=dimer,
        rattle_relax=rattle,
        max_dft_f_mae_mev_a=35.0,
        max_cn_rattle_rmsd_a=0.20,
        max_rattle_fmax_ev_a=0.40,
    )

    assert row["schema_version"] == "rtece_physical_pareto_row.v1"
    assert row["dimer_short_repulsive_fraction"] == pytest.approx(1.0)
    assert row["dimer_gate_pass"] is True
    assert row["cn_rattle_final_rmsd_a"] == pytest.approx(0.18)
    assert row["rattle_gate_pass"] is True
    assert row["benchmark_gate_pass"] is True
    assert row["physical_gate_pass"] is True
    assert row["tece_path_manifest_hash"] == "abc123"
    assert row["physical_score"] < 3.0

    dominated = dict(row, variant="dominated", atoms_per_second=2.0e6, physical_score=row["physical_score"] + 0.5)
    fast_tradeoff = dict(row, variant="fast_tradeoff", atoms_per_second=4.0e6, physical_score=row["physical_score"] + 0.2)
    front = physical_pareto_front_rows([row, dominated, fast_tradeoff])
    assert [item["variant"] for item in front] == ["fast_tradeoff", "radial_core_balanced"]


def test_rtece_physical_pareto_accepts_non_cn_rattle_focus_label():
    from benchmarks.oc20neb_tace_mace.summarize_rtece_physical_pareto import format_markdown, make_physical_pareto_row

    benchmark = {"atoms_per_second": 3.0e6, "mae_f_mev_a": 29.0, "rmse_f_mev_a": 101.0}
    dimer = {
        "pair_summaries": [
            {"pair": "O-Cu", "summary": {"short_force_repulsive": True, "has_nonfinite": False, "short_force_parallel_ev_a": -0.20}}
        ]
    }
    rattle = {
        "summary": {
            "mean_final_rmsd_a": 0.16,
            "max_fmax_ev_a": 0.20,
            "focus_groups": [
                {"label": "O_or_Cu", "mean_final_rmsd_a": 0.12, "max_fmax_ev_a": 0.20},
                {"label": "C_or_N", "mean_final_rmsd_a": 0.30, "max_fmax_ev_a": 0.20},
            ],
        }
    }

    row = make_physical_pareto_row(
        "generic_focus",
        dft_benchmark=benchmark,
        dimer_scan=dimer,
        rattle_relax=rattle,
        rattle_focus_label="O_or_Cu",
        max_focus_rattle_rmsd_a=0.15,
    )

    assert row["rattle_focus_label"] == "O_or_Cu"
    assert row["focus_rattle_final_rmsd_a"] == pytest.approx(0.12)
    assert row["cn_rattle_final_rmsd_a"] == pytest.approx(0.30)
    assert row["max_cn_rattle_rmsd_a"] is None
    assert row["rattle_gate_pass"] is True
    markdown = format_markdown([row], [row])
    assert "O_or_Cu RMSD" in markdown
    assert "configurable stress-test dimension" in markdown


def test_rtece_physical_pareto_penalizes_negative_dimer_energy_lift():
    from benchmarks.oc20neb_tace_mace.summarize_rtece_physical_pareto import make_physical_pareto_row

    benchmark = {"atoms_per_second": 3.0e6, "mae_f_mev_a": 29.0, "rmse_f_mev_a": 101.0}
    rattle = {
        "summary": {
            "mean_final_rmsd_a": 0.16,
            "max_fmax_ev_a": 0.20,
            "focus_groups": [{"label": "C_or_N", "mean_final_rmsd_a": 0.16, "max_fmax_ev_a": 0.20}],
        }
    }
    repulsive_but_negative_energy = {
        "pair_summaries": [
            {
                "pair": "C-N",
                "summary": {
                    "short_force_repulsive": True,
                    "has_nonfinite": False,
                    "short_force_parallel_ev_a": -0.25,
                    "short_minus_long_energy_eV": -0.02,
                },
            }
        ]
    }

    row = make_physical_pareto_row(
        "negative_energy_lift",
        dft_benchmark=benchmark,
        dimer_scan=repulsive_but_negative_energy,
        rattle_relax=rattle,
    )

    assert row["dimer_gate_pass"] is True
    assert row["dimer_min_short_energy_lift_eV"] == pytest.approx(-0.02)
    assert row["dimer_energy_shape_penalty"] == pytest.approx(0.02)
    assert row["physical_score"] == pytest.approx(101.0 / 120.0 + 0.16 / 0.20 + 0.20 / 0.40 + 0.02 / 0.05)


def test_rtece_physical_pareto_uses_rmse_and_max_error_in_score():
    from benchmarks.oc20neb_tace_mace.summarize_rtece_physical_pareto import make_physical_pareto_row

    benchmark = {
        "atoms_per_second": 3.0e6,
        "mae_e_mev_atom": 5.0,
        "rmse_e_mev_atom": 50.0,
        "max_abs_e_mev_atom": 250.0,
        "mae_f_mev_a": 10.0,
        "rmse_f_mev_a": 80.0,
        "max_abs_f_mev_a": 300.0,
    }
    rattle = {
        "summary": {
            "mean_final_rmsd_a": 0.10,
            "max_fmax_ev_a": 0.20,
            "focus_groups": [{"label": "C_or_N", "mean_final_rmsd_a": 0.10, "max_fmax_ev_a": 0.20}],
        }
    }
    dimer = {
        "pair_summaries": [
            {
                "pair": "C-N",
                "summary": {
                    "short_force_repulsive": True,
                    "has_nonfinite": False,
                    "short_force_parallel_ev_a": -0.25,
                    "short_minus_long_energy_eV": 0.01,
                },
            }
        ]
    }

    row = make_physical_pareto_row(
        "rmse_weighted",
        dft_benchmark=benchmark,
        dimer_scan=dimer,
        rattle_relax=rattle,
        max_dft_f_rmse_mev_a=100.0,
        max_dft_e_rmse_mev_atom=100.0,
        max_dft_f_max_mev_a=600.0,
        max_dft_e_max_mev_atom=500.0,
    )

    assert row["benchmark_gate_pass"] is True
    assert row["dft_e_rmse_mev_atom"] == pytest.approx(50.0)
    assert row["dft_f_rmse_mev_a"] == pytest.approx(80.0)
    assert row["dft_e_max_abs_mev_atom"] == pytest.approx(250.0)
    assert row["dft_f_max_abs_mev_a"] == pytest.approx(300.0)
    assert row["benchmark_score"] == pytest.approx(0.8 + 0.5 + 0.5 + 0.5)
    assert row["physical_score"] == pytest.approx(0.8 + 0.5 + 0.5 + 0.5 + 0.10 / 0.20 + 0.20 / 0.40)


def test_rtece_force_error_stratification_reports_element_and_focus_groups():
    from benchmarks.oc20neb_tace_mace.stratify_rtece_force_errors import stratify_symbol_force_errors

    predicted = torch.tensor(
        [
            [0.001, -0.001, 0.0],
            [0.0, 0.002, 0.0],
            [0.0, 0.0, 0.003],
            [0.004, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    reference = torch.zeros((4, 3), dtype=torch.float64)

    summary = stratify_symbol_force_errors(
        symbols=["C", "N", "H", "Cu"],
        predicted_forces=predicted,
        reference_forces=reference,
        target_force_source="teacher_forces",
    )
    elements = {row["label"]: row for row in summary["elements"]}
    groups = {row["label"]: row for row in summary["focus_groups"]}

    assert summary["num_atoms"] == 4
    assert summary["num_force_components"] == 12
    assert summary["target_force_source"] == "teacher_forces"
    assert elements["C"]["count"] == 1
    assert elements["C"]["component_count"] == 3
    assert elements["C"]["mae_f_mev_a"] == pytest.approx(2.0 / 3.0)
    assert elements["C"]["rmse_f_mev_a"] == pytest.approx((2.0 / 3.0) ** 0.5)
    assert groups["C_or_N"]["count"] == 2
    assert groups["C_or_N"]["mae_f_mev_a"] == pytest.approx(4.0 / 6.0)
    assert groups["C_or_N"]["rmse_f_mev_a"] == pytest.approx(1.0)
    assert groups["not_CHNO"]["mae_f_mev_a"] == pytest.approx(4.0 / 3.0)


def test_rtece_force_error_stratification_adds_cn_selection_proxy():
    from benchmarks.oc20neb_tace_mace.stratify_rtece_force_errors import stratify_symbol_force_errors

    predicted = torch.tensor(
        [
            [0.006, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    reference = torch.zeros((2, 3), dtype=torch.float64)

    summary = stratify_symbol_force_errors(
        symbols=["C", "Cu"],
        predicted_forces=predicted,
        reference_forces=reference,
        target_force_source="teacher_forces",
        focus_selection_label="C_or_N",
        focus_excess_weight=2.0,
    )

    assert summary["selection_focus_label"] == "C_or_N"
    assert summary["selection_focus_mae_f_mev_a"] == pytest.approx(2.0)
    assert summary["selection_focus_excess_mae_f_mev_a"] == pytest.approx(1.0)
    assert summary["selection_score_mev_a"] == pytest.approx(3.0)

    from benchmarks.oc20neb_tace_mace.stratify_rtece_force_errors import format_markdown

    markdown = format_markdown(summary)
    assert "selection score: 3.000 meV/A" in markdown
    assert "selection focus: `C_or_N` excess weight 2.0" in markdown


def test_rtece_projection_loads_sample_weight_json(tmp_path):
    from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import _load_sample_weights_json

    path = tmp_path / "sample_weights.json"
    path.write_text(
        json.dumps({"weight_source": "teacher_force_l2", "sample_weights": [0.5, 2.0, 3.5]}),
        encoding="utf-8",
    )

    weights, source = _load_sample_weights_json(path)

    assert source == "teacher_force_l2"
    assert weights.dtype == torch.float64
    assert weights.tolist() == pytest.approx([0.5, 2.0, 3.5])


def test_rtece_projection_diagnostic_builds_species_path_config():
    from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import build_projection_config

    config = build_projection_config(
        "weighted_species_cavity",
        ("atomic.radial_density", "atomic.species_basis_density", "edge.cavity.vector_dot"),
        num_radial=4,
        species_basis_channels=3,
    )

    assert config.variant == "weighted_species_cavity"
    assert config.species_basis_channels == 3
    assert descriptor_dim(config) == 4 + 12 + 1
    assert config.scalar_path_ids == (
        "atomic.radial_density",
        "atomic.species_basis_density",
        "edge.cavity.vector_dot",
    )


def test_rtece_projection_config_accepts_atomic_cross_radial_sketch_channels():
    from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import build_projection_config

    config = build_projection_config(
        "l1_cross_k3_projection",
        ("atomic.radial_density", "atomic.vector_norm", "atomic.vector_cross_radial_dot"),
        num_radial=6,
        atomic_cross_radial_sketch_channels=3,
    )

    assert config.atomic_cross_radial_sketch_channels == 3
    assert descriptor_dim(config) == 6 + 6 + 3


def test_rtece_projection_cli_help_does_not_import_torch_or_tace():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    script = "import importlib.abc\nimport sys\nblocked_roots = ('torch', 'tace')\nclass HeavyImportProbe(importlib.abc.MetaPathFinder):\n    def find_spec(self, fullname, path=None, target=None):\n        if fullname in blocked_roots or any(fullname.startswith(root + '.') for root in blocked_roots):\n            raise ImportError(f'blocked heavy import {fullname}')\n        return None\nsys.meta_path.insert(0, HeavyImportProbe())\nfrom benchmarks.oc20neb_tace_mace import analyze_rtece_projection_error as mod\nsys.argv = ['analyze_rtece_projection_error.py', '--help']\ntry:\n    mod.main()\nexcept SystemExit as exc:\n    raise SystemExit(exc.code)"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "--auto-candidate-strategy" in result.stdout
    assert "--atomic-cross-radial-sketch-channels" in result.stdout
    assert "--energy-target-key" in result.stdout


def test_rtece_projection_candidate_helpers_do_not_import_torch_or_tace():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    script = "import importlib.abc\nimport json\nimport sys\nblocked_roots = ('torch', 'tace')\nattempts = []\nclass HeavyImportProbe(importlib.abc.MetaPathFinder):\n    def find_spec(self, fullname, path=None, target=None):\n        if fullname in blocked_roots or any(fullname.startswith(root + '.') for root in blocked_roots):\n            attempts.append(fullname)\n            raise ImportError(f'blocked heavy import {fullname}')\n        return None\nsys.meta_path.insert(0, HeavyImportProbe())\nfrom benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import generate_projection_candidate_specs, rank_projection_rows\ngenerate_projection_candidate_specs(('atomic.radial_density', 'edge.direct.radial'))\nrank_projection_rows([{'candidate': 'x', 'relative_residual': 0.0, 'candidate_dim': 1}])\nprint(json.dumps(attempts))"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == []


def test_rtece_projection_candidate_generator_builds_single_delete_path_sets():
    from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import generate_projection_candidate_specs

    reference_paths = (
        "atomic.radial_density",
        "edge.full_moment.vector_dot",
        "edge.full_moment.vector_shell_dot",
        "edge.direct.radial",
    )

    candidates = generate_projection_candidate_specs(reference_paths, strategies=("single_delete",))

    assert candidates == [
        (
            "single_delete_edge_full_moment_vector_dot",
            ("atomic.radial_density", "edge.full_moment.vector_shell_dot", "edge.direct.radial"),
        ),
        (
            "single_delete_edge_full_moment_vector_shell_dot",
            ("atomic.radial_density", "edge.full_moment.vector_dot", "edge.direct.radial"),
        ),
        (
            "single_delete_edge_direct_radial",
            ("atomic.radial_density", "edge.full_moment.vector_dot", "edge.full_moment.vector_shell_dot"),
        ),
    ]


def test_rtece_projection_candidate_generator_builds_prefix_path_sets():
    from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import generate_projection_candidate_specs

    reference_paths = (
        "atomic.radial_density",
        "edge.full_moment.vector_dot",
        "edge.full_moment.vector_shell_dot",
        "edge.direct.radial",
    )

    candidates = generate_projection_candidate_specs(reference_paths, strategies=("prefix",))

    assert candidates == [
        ("prefix_001", ("atomic.radial_density",)),
        ("prefix_002", ("atomic.radial_density", "edge.full_moment.vector_dot")),
        (
            "prefix_003",
            ("atomic.radial_density", "edge.full_moment.vector_dot", "edge.full_moment.vector_shell_dot"),
        ),
    ]


def test_rtece_projection_rows_are_ranked_by_residual_then_descriptor_dim():
    from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import rank_projection_rows

    rows = [
        {"candidate": "wide_worse", "relative_residual": 0.20, "candidate_dim": 8},
        {"candidate": "small_best", "relative_residual": 0.10, "candidate_dim": 3},
        {"candidate": "wide_best_tie", "relative_residual": 0.10, "candidate_dim": 5},
    ]

    ranked = rank_projection_rows(rows)

    assert [row["candidate"] for row in ranked] == ["small_best", "wide_best_tie", "wide_worse"]
    assert [row["projection_rank"] for row in ranked] == [1, 2, 3]


def test_rtece_projection_rows_add_energy_force_ranks_and_pareto_flags():
    from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import rank_projection_rows

    rows = [
        {"candidate": "energy_best", "relative_residual": 0.4, "candidate_dim": 16, "energy_per_atom_rmse": 0.10, "force_rmse": 0.50},
        {"candidate": "balanced", "relative_residual": 0.2, "candidate_dim": 12, "energy_per_atom_rmse": 0.20, "force_rmse": 0.20},
        {"candidate": "dominated", "relative_residual": 0.1, "candidate_dim": 14, "energy_per_atom_rmse": 0.30, "force_rmse": 0.30},
        {"candidate": "force_best", "relative_residual": 0.3, "candidate_dim": 20, "energy_per_atom_rmse": 0.50, "force_rmse": 0.10},
    ]

    ranked = rank_projection_rows(rows)
    by_candidate = {row["candidate"]: row for row in ranked}

    assert [row["candidate"] for row in ranked] == ["dominated", "balanced", "force_best", "energy_best"]
    assert by_candidate["energy_best"]["energy_rank"] == 1
    assert by_candidate["force_best"]["force_rank"] == 1
    assert by_candidate["balanced"]["ef_combined_rank"] == 1
    assert by_candidate["balanced"]["ef_rank_max"] == 2
    assert by_candidate["balanced"]["ef_rank_sum"] == 4
    assert by_candidate["dominated"]["ef_pareto_dominated"] is True
    assert by_candidate["balanced"]["ef_pareto_dominated"] is False
    assert by_candidate["energy_best"]["ef_energy_metric"] == "energy_per_atom_rmse"
    assert by_candidate["energy_best"]["ef_force_metric"] == "force_rmse"


def test_rtece_projection_rows_use_total_energy_rank_when_per_atom_metric_missing():
    from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import rank_projection_rows

    ranked = rank_projection_rows([
        {"candidate": "a", "relative_residual": 0.0, "candidate_dim": 2, "energy_rmse": 2.0},
        {"candidate": "b", "relative_residual": 0.1, "candidate_dim": 1, "energy_rmse": 1.0},
    ])
    by_candidate = {row["candidate"]: row for row in ranked}

    assert by_candidate["b"]["energy_rank"] == 1
    assert by_candidate["b"]["ef_energy_metric"] == "energy_rmse"
    assert "force_rank" not in by_candidate["b"]
    assert "ef_combined_rank" not in by_candidate["b"]


def test_rtece_active_set_ranking_uses_marginal_gain_per_cost():
    from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import rank_active_set_candidate_rows

    rows = [
        {
            "candidate": "baseline",
            "path_ids": ["atomic.radial_density"],
            "candidate_dim": 8,
            "energy_per_atom_rmse": 50.0,
            "force_rmse": 100.0,
            "relative_residual": 0.30,
        },
        {
            "candidate": "cheap_energy_gain",
            "path_ids": ["atomic.radial_density", "atomic.species_basis_density"],
            "candidate_dim": 10,
            "energy_per_atom_rmse": 44.0,
            "force_rmse": 98.0,
            "relative_residual": 0.28,
            "marginal_cost_proxy": 2.0,
        },
        {
            "candidate": "expensive_bigger_gain",
            "path_ids": ["atomic.radial_density", "edge.cavity.quadrupole_frobenius"],
            "candidate_dim": 28,
            "energy_per_atom_rmse": 40.0,
            "force_rmse": 94.0,
            "relative_residual": 0.25,
            "marginal_cost_proxy": 20.0,
        },
        {
            "candidate": "worse_energy",
            "path_ids": ["atomic.radial_density", "edge.cavity.target_vector_projection"],
            "candidate_dim": 9,
            "energy_per_atom_rmse": 55.0,
            "force_rmse": 101.0,
            "relative_residual": 0.35,
            "marginal_cost_proxy": 1.0,
        },
    ]

    ranked = rank_active_set_candidate_rows(
        rows,
        baseline_candidate="baseline",
        energy_weight=1.0,
        force_weight=0.5,
        projection_weight=10.0,
    )

    assert [row["candidate"] for row in ranked] == [
        "cheap_energy_gain",
        "expensive_bigger_gain",
        "worse_energy",
    ]
    cheap = ranked[0]
    worse = ranked[-1]
    assert cheap["baseline_candidate"] == "baseline"
    assert cheap["marginal_paths"] == ["atomic.species_basis_density"]
    assert cheap["energy_marginal_gain"] == pytest.approx(6.0)
    assert cheap["force_marginal_gain"] == pytest.approx(2.0)
    assert cheap["projection_marginal_gain"] == pytest.approx(0.02)
    assert cheap["active_set_promoted"] is True
    assert worse["active_set_promoted"] is False
    assert worse["marginal_gain_per_cost"] < 0.0


def test_rtece_active_set_ranking_can_use_relative_metric_gains():
    from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import rank_active_set_candidate_rows

    rows = [
        {
            "candidate": "baseline",
            "path_ids": ["atomic.radial_density"],
            "energy_per_atom_rmse": 40.0,
            "force_rmse": 80.0,
            "candidate_dim": 4,
        },
        {
            "candidate": "energy_only",
            "path_ids": ["atomic.radial_density", "edge.cavity.target_vector_projection"],
            "energy_per_atom_rmse": 20.0,
            "force_rmse": 120.0,
            "candidate_dim": 6,
        },
        {
            "candidate": "balanced",
            "path_ids": ["atomic.radial_density", "edge.direct.radial"],
            "energy_per_atom_rmse": 32.0,
            "force_rmse": 72.0,
            "candidate_dim": 6,
        },
    ]

    ranked = rank_active_set_candidate_rows(
        rows,
        baseline_candidate="baseline",
        energy_weight=1.0,
        force_weight=1.0,
        gain_mode="relative",
    )

    assert [row["candidate"] for row in ranked] == ["balanced", "energy_only"]
    assert ranked[0]["energy_marginal_gain"] == pytest.approx(0.2)
    assert ranked[0]["force_marginal_gain"] == pytest.approx(0.1)
    assert ranked[0]["active_set_gain_mode"] == "relative"
    assert ranked[0]["active_set_promoted"] is True
    assert ranked[1]["energy_marginal_gain"] == pytest.approx(0.5)
    assert ranked[1]["force_marginal_gain"] == pytest.approx(-0.5)
    assert ranked[1]["active_set_promoted"] is False


def test_rtece_active_set_can_reject_excessive_force_regression():
    from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import rank_active_set_candidate_rows

    rows = [
        {
            "candidate": "baseline",
            "path_ids": ["atomic.radial_density"],
            "energy_per_atom_rmse": 40.0,
            "force_rmse": 80.0,
            "candidate_dim": 4,
        },
        {
            "candidate": "energy_repair_force_bad",
            "path_ids": ["atomic.radial_density", "edge.cavity.target_vector_projection"],
            "energy_per_atom_rmse": 10.0,
            "force_rmse": 104.0,
            "candidate_dim": 6,
        },
        {
            "candidate": "energy_repair_force_guarded",
            "path_ids": ["atomic.radial_density", "edge.cavity.vector_dot"],
            "energy_per_atom_rmse": 14.0,
            "force_rmse": 84.0,
            "candidate_dim": 6,
        },
    ]

    ranked = rank_active_set_candidate_rows(
        rows,
        baseline_candidate="baseline",
        energy_weight=1.0,
        force_weight=1.0,
        gain_mode="relative",
        max_force_regression_fraction=0.10,
        require_energy_gain=True,
    )

    by_candidate = {row["candidate"]: row for row in ranked}
    guarded = by_candidate["energy_repair_force_guarded"]
    force_bad = by_candidate["energy_repair_force_bad"]
    assert guarded["active_set_promoted"] is True
    assert guarded["active_set_constraint_passed"] is True
    assert guarded["force_regression_fraction"] == pytest.approx(0.05)
    assert force_bad["active_set_promoted"] is False
    assert force_bad["active_set_constraint_passed"] is False
    assert force_bad["force_regression_fraction"] == pytest.approx(0.30)
    assert "force_regression_fraction" in force_bad["active_set_rejection_reasons"]


def test_rtece_active_set_can_require_energy_gain():
    from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import rank_active_set_candidate_rows

    rows = [
        {
            "candidate": "baseline",
            "path_ids": ["atomic.radial_density"],
            "energy_per_atom_rmse": 40.0,
            "force_rmse": 80.0,
            "candidate_dim": 4,
        },
        {
            "candidate": "force_only",
            "path_ids": ["atomic.radial_density", "edge.direct.radial"],
            "energy_per_atom_rmse": 44.0,
            "force_rmse": 40.0,
            "candidate_dim": 6,
        },
    ]

    ranked = rank_active_set_candidate_rows(
        rows,
        baseline_candidate="baseline",
        energy_weight=1.0,
        force_weight=2.0,
        gain_mode="relative",
        require_energy_gain=True,
    )

    assert ranked[0]["candidate"] == "force_only"
    assert ranked[0]["weighted_marginal_gain"] > 0.0
    assert ranked[0]["active_set_promoted"] is False
    assert "energy_gain_required" in ranked[0]["active_set_rejection_reasons"]



def test_rtece_projection_rows_cache_reference_descriptors(monkeypatch):
    from benchmarks.oc20neb_tace_mace import analyze_rtece_projection_error as mod

    reference = RTECEScalarConfig(variant="reference", num_radial=1)
    candidate_a = RTECEScalarConfig(variant="candidate_a", num_radial=1)
    candidate_b = RTECEScalarConfig(variant="candidate_b", num_radial=1)
    calls = {"reference": 0, "candidate_a": 0, "candidate_b": 0}

    def fake_descriptor_matrices(graphs, config):
        calls[config.variant] += 1
        if config.variant == "reference":
            matrix = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
            return matrix, matrix
        if config.variant == "candidate_a":
            matrix = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
            return matrix, matrix
        matrix = torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float64)
        return matrix, matrix

    monkeypatch.setattr(mod, "_descriptor_matrices", fake_descriptor_matrices)

    rows = mod.make_projection_diagnostic_rows(
        [
            ("candidate_a", candidate_a),
            ("candidate_b", candidate_b),
        ],
        reference_config=reference,
        graphs=[object()],
    )

    assert [row["candidate"] for row in rows] == ["candidate_a", "candidate_b"]
    assert calls == {"reference": 1, "candidate_a": 1, "candidate_b": 1}


def test_rtece_projection_subset_candidates_slice_reference_descriptors(monkeypatch):
    from benchmarks.oc20neb_tace_mace import analyze_rtece_projection_error as mod

    reference = mod.build_projection_config(
        "reference",
        ("atomic.radial_density", "atomic.vector_norm"),
        num_radial=1,
    )
    candidate = mod.build_projection_config(
        "candidate",
        ("atomic.radial_density",),
        num_radial=1,
    )
    calls = {"reference": 0, "candidate": 0}

    def fake_descriptor_matrices(graphs, config):
        calls[config.variant] += 1
        if config.variant == "reference":
            matrix = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=torch.float64)
            return matrix, matrix
        raise AssertionError("subset candidate should be sliced from the reference descriptor matrix")

    monkeypatch.setattr(mod, "_descriptor_matrices", fake_descriptor_matrices)

    rows = mod.make_projection_diagnostic_rows(
        [("radial_only", candidate)],
        reference_config=reference,
        graphs=[object()],
    )

    assert rows[0]["candidate"] == "radial_only"
    assert rows[0]["candidate_dim"] == 1
    assert calls == {"reference": 1, "candidate": 0}


def test_rtece_projection_reference_slice_matches_explicit_candidate_descriptors():
    from benchmarks.oc20neb_tace_mace import analyze_rtece_projection_error as mod

    graph = RTECEGraph(
        z=torch.tensor([1, 6, 8], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.1, 0.0], [0.2, 0.9, 0.0]], dtype=torch.float64),
        edge_index=complete_directed_edges(3),
        batch=torch.zeros(3, dtype=torch.long),
    )
    reference = mod.build_projection_config(
        "reference",
        (
            "atomic.radial_density",
            "atomic.vector_norm",
            "atomic.vector_cross_radial_dot",
            "edge.cavity.vector_dot",
            "edge.direct.radial",
        ),
        num_radial=3,
        atomic_cross_radial_sketch_channels=2,
    )
    candidate = mod.build_projection_config(
        "candidate",
        ("atomic.vector_cross_radial_dot", "atomic.radial_density", "edge.direct.radial"),
        num_radial=3,
        atomic_cross_radial_sketch_channels=2,
    )

    reference_descriptors = mod._descriptor_matrix([graph], reference)
    sliced = mod.candidate_descriptors_from_reference(
        reference_descriptors,
        candidate_config=candidate,
        reference_config=reference,
    )
    explicit = mod._descriptor_matrix([graph], candidate)

    assert sliced is not None
    assert torch.allclose(sliced, explicit, atol=1e-12, rtol=1e-12)


def test_rtece_energy_label_projection_metrics_report_error_statistics():
    from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import energy_label_projection_metrics

    source = torch.tensor([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]], dtype=torch.float64)
    target = torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64)

    metrics = energy_label_projection_metrics(source, target, ridge=0.0)

    assert metrics["energy_num_samples"] == 3
    assert metrics["energy_source_dim"] == 2
    assert metrics["energy_target_dim"] == 1
    assert metrics["energy_rmse"] == pytest.approx((1.0 / 18.0) ** 0.5)
    assert metrics["energy_mae"] == pytest.approx(2.0 / 9.0)
    assert metrics["energy_bias"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["energy_max_abs"] == pytest.approx(1.0 / 3.0)
    assert metrics["energy_per_atom_rmse"] is None


def test_rtece_energy_label_projection_metrics_reports_holdout_error():
    from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import energy_label_projection_metrics

    source = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float64)
    target = torch.tensor([0.0, 1.0, 2.0, 10.0], dtype=torch.float64)

    metrics = energy_label_projection_metrics(
        source,
        target,
        ridge=0.0,
        fit_indices=torch.tensor([0, 1, 2], dtype=torch.long),
        eval_indices=torch.tensor([3], dtype=torch.long),
    )

    assert metrics["energy_fit_num_samples"] == 3
    assert metrics["energy_eval_num_samples"] == 1
    assert metrics["energy_degrees_of_freedom"] == 2
    assert metrics["energy_underdetermined"] is False
    assert metrics["energy_rmse"] == pytest.approx(7.0)
    assert metrics["energy_mae"] == pytest.approx(7.0)
    assert metrics["energy_bias"] == pytest.approx(7.0)
    assert metrics["energy_max_abs"] == pytest.approx(7.0)


def test_rtece_sampled_force_descriptor_rows_match_full_jacobian():
    from benchmarks.oc20neb_tace_mace import analyze_rtece_projection_error as mod

    graph = RTECEGraph(
        z=torch.tensor([1, 1], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.1, 0.0]], dtype=torch.float64),
        edge_index=complete_directed_edges(2),
        batch=torch.zeros(2, dtype=torch.long),
    )
    config = mod.build_projection_config(
        "sampled_force",
        ("atomic.radial_density", "atomic.vector_norm", "edge.direct.radial"),
        num_radial=2,
    )
    sampled_rows = torch.tensor([0, 2, 5], dtype=torch.long)

    full = mod._force_descriptor_matrix([graph], config)
    sampled = mod._force_descriptor_matrix([graph], config, force_row_indices=sampled_rows)

    assert sampled.shape == (3, full.shape[1])
    assert torch.allclose(sampled, full[sampled_rows], atol=1e-8, rtol=1e-6)


def test_rtece_force_label_projection_metrics_reports_holdout_error():
    from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import force_label_projection_metrics

    source = torch.tensor([[0.0], [1.0], [2.0], [3.0]], dtype=torch.float64)
    target = torch.tensor([0.0, 1.0, 2.0, 10.0], dtype=torch.float64)

    metrics = force_label_projection_metrics(
        source,
        target,
        ridge=0.0,
        fit_indices=torch.tensor([0, 1, 2], dtype=torch.long),
        eval_indices=torch.tensor([3], dtype=torch.long),
    )

    assert metrics["force_fit_num_samples"] == 3
    assert metrics["force_eval_num_samples"] == 1
    assert metrics["force_degrees_of_freedom"] == 2
    assert metrics["force_underdetermined"] is False
    assert metrics["force_rmse"] == pytest.approx(7.0)
    assert metrics["force_mae"] == pytest.approx(7.0)
    assert metrics["force_bias"] == pytest.approx(7.0)
    assert metrics["force_max_abs"] == pytest.approx(7.0)


def test_rtece_energy_label_projection_metrics_can_fit_element_count_baseline():
    from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import energy_label_projection_metrics

    source = torch.zeros((2, 1), dtype=torch.float64)
    baseline = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    target = torch.tensor([1.25, -2.5], dtype=torch.float64)

    metrics = energy_label_projection_metrics(
        source,
        target,
        ridge=0.0,
        baseline_features=baseline,
    )

    assert metrics["energy_source_dim"] == 1
    assert metrics["energy_baseline_dim"] == 2
    assert metrics["energy_fit_dim"] == 3
    assert metrics["energy_degrees_of_freedom"] == -1
    assert metrics["energy_underdetermined"] is True
    assert metrics["energy_rmse"] == pytest.approx(0.0, abs=1e-12)


def test_rtece_projection_rows_reuse_single_descriptor_pass_for_energy_reference(monkeypatch):
    from benchmarks.oc20neb_tace_mace import analyze_rtece_projection_error as mod

    reference = mod.build_projection_config(
        "reference",
        ("atomic.radial_density", "atomic.vector_norm"),
        num_radial=1,
    )
    candidate = mod.build_projection_config(
        "candidate",
        ("atomic.radial_density",),
        num_radial=1,
    )
    calls = {"reference": 0}

    def fake_descriptor_matrices(graphs, config):
        calls[config.variant] += 1
        if config.variant == "reference":
            matrix = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=torch.float64)
            return matrix, matrix
        raise AssertionError("subset candidate should be sliced from the reference descriptor matrices")

    monkeypatch.setattr(mod, "_descriptor_matrices", fake_descriptor_matrices)
    monkeypatch.setattr(mod, "_descriptor_matrix", lambda graphs, config: (_ for _ in ()).throw(AssertionError("separate node descriptor pass should not run")))
    monkeypatch.setattr(mod, "_graph_descriptor_matrix", lambda graphs, config: (_ for _ in ()).throw(AssertionError("separate graph descriptor pass should not run")))

    rows = mod.make_projection_diagnostic_rows(
        [("radial_only", candidate)],
        reference_config=reference,
        graphs=[object(), object(), object()],
        energy_targets=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
    )

    assert rows[0]["energy_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert calls == {"reference": 1}


def test_rtece_projection_rows_add_energy_label_metrics_from_sliced_reference_graph_descriptors(monkeypatch):
    from benchmarks.oc20neb_tace_mace import analyze_rtece_projection_error as mod

    reference = mod.build_projection_config(
        "reference",
        ("atomic.radial_density", "atomic.vector_norm"),
        num_radial=1,
    )
    candidate = mod.build_projection_config(
        "candidate",
        ("atomic.radial_density",),
        num_radial=1,
    )
    calls = {"reference": 0, "candidate": 0}

    def fake_descriptor_matrices(graphs, config):
        calls[config.variant] += 1
        if config.variant == "reference":
            matrix = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=torch.float64)
            return matrix, matrix
        raise AssertionError("subset candidate graph descriptors should be sliced from the reference graph matrix")

    monkeypatch.setattr(mod, "_descriptor_matrices", fake_descriptor_matrices)

    rows = mod.make_projection_diagnostic_rows(
        [("radial_only", candidate)],
        reference_config=reference,
        graphs=[object(), object(), object()],
        energy_targets=torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
        atom_counts=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
    )

    assert rows[0]["candidate"] == "radial_only"
    assert rows[0]["energy_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert rows[0]["energy_per_atom_rmse"] == pytest.approx(0.0, abs=1e-12)
    assert calls == {"reference": 1, "candidate": 0}


def test_rtece_projection_element_focus_weights_are_atom_aligned_and_mean_normalized():
    from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import element_focus_sample_weights

    graph = RTECEGraph(
        z=torch.tensor([8, 14, 29, 1], dtype=torch.long),
        pos=torch.zeros((4, 3), dtype=torch.float64),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=torch.zeros(4, dtype=torch.long),
    )

    weights = element_focus_sample_weights([graph], focus_atomic_numbers=(8, 29), focus_weight=4.0)

    assert weights.shape == (4,)
    assert weights.mean().item() == pytest.approx(1.0)
    assert weights[[0, 2]].tolist() == pytest.approx([1.6, 1.6])
    assert weights[[1, 3]].tolist() == pytest.approx([0.4, 0.4])


def test_rtece_projection_cli_generates_focus_element_weights(tmp_path):
    import numpy as np
    import ase.io
    from ase import Atoms

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    configs = tmp_path / "hocnonspecific.xyz"
    output = tmp_path / "projection_focus.json"
    atoms = Atoms("HOCu", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0], [1.9, 0.0, 0.0]])
    atoms.info["energy"] = 0.0
    atoms.arrays["forces"] = np.zeros((3, 3), dtype=np.float64)
    ase.io.write(configs, [atoms], format="extxyz")

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py",
            "--configs",
            str(configs),
            "--output-json",
            str(output),
            "--reference-path-ids",
            "atomic.radial_density,edge.direct.radial",
            "--auto-candidate-strategy",
            "single_delete",
            "--num-radial",
            "3",
            "--limit-configs",
            "1",
            "--neighborlist-backend",
            "ase",
            "--focus-elements",
            "O,Cu",
            "--focus-weight",
            "4.0",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["weighted"] is True
    assert payload["sample_weight_source"] == "element_focus:O,Cu:weight=4"
    assert payload["focus_elements"] == ["O", "Cu"]
    assert payload["focus_atomic_numbers"] == [8, 29]
    assert payload["rows"][0]["weighted"] is True
    assert payload["rows"][0]["weight_sum"] == pytest.approx(3.0)


def test_rtece_projection_cli_adds_energy_label_projection_metrics(tmp_path):
    import numpy as np
    import ase.io
    from ase import Atoms

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    configs = tmp_path / "h2_energy.xyz"
    output = tmp_path / "projection_energy.json"
    atoms_a = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    atoms_b = Atoms("H2", positions=[[0.0, 0.0, 0.0], [1.10, 0.0, 0.0]])
    for idx, atoms in enumerate([atoms_a, atoms_b], start=1):
        atoms.info["teacher_energy"] = float(idx)
        atoms.info["energy"] = -10.0
        atoms.arrays["forces"] = np.zeros((2, 3), dtype=np.float64)
    ase.io.write(configs, [atoms_a, atoms_b], format="extxyz")

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py",
            "--configs",
            str(configs),
            "--output-json",
            str(output),
            "--reference-path-ids",
            "atomic.radial_density,edge.direct.radial",
            "--auto-candidate-strategy",
            "single_delete",
            "--num-radial",
            "3",
            "--limit-configs",
            "2",
            "--neighborlist-backend",
            "ase",
            "--energy-target-key",
            "teacher_energy",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["energy_target_key"] == "teacher_energy"
    assert payload["energy_target_num_configs"] == 2
    assert "energy_rmse" in payload["rows"][0]
    assert "energy_per_atom_rmse" in payload["rows"][0]


def test_rtece_projection_cli_adds_energy_holdout_split_metadata(tmp_path):
    import numpy as np
    import ase.io
    from ase import Atoms

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    configs = tmp_path / "h2_energy_split.xyz"
    output = tmp_path / "projection_energy_split.json"
    atoms_list = []
    for idx, distance in enumerate([0.70, 0.80, 0.90, 1.00]):
        atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
        atoms.info["teacher_energy"] = float(idx)
        atoms.arrays["forces"] = np.zeros((2, 3), dtype=np.float64)
        atoms_list.append(atoms)
    ase.io.write(configs, atoms_list, format="extxyz")

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py",
            "--configs",
            str(configs),
            "--output-json",
            str(output),
            "--reference-path-ids",
            "atomic.radial_density,edge.direct.radial",
            "--auto-candidate-strategy",
            "single_delete",
            "--num-radial",
            "3",
            "--limit-configs",
            "4",
            "--neighborlist-backend",
            "ase",
            "--energy-target-key",
            "teacher_energy",
            "--energy-eval-stride",
            "2",
            "--energy-eval-offset",
            "1",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["energy_eval_stride"] == 2
    assert payload["energy_eval_offset"] == 1
    assert payload["energy_fit_num_configs"] == 2
    assert payload["energy_eval_num_configs"] == 2
    assert payload["rows"][0]["energy_fit_num_samples"] == 2
    assert payload["rows"][0]["energy_eval_num_samples"] == 2


def test_rtece_projection_cli_adds_force_label_projection_metrics(tmp_path):
    import numpy as np
    import ase.io
    from ase import Atoms

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    configs = tmp_path / "h2_force_split.xyz"
    output = tmp_path / "projection_force_split.json"
    atoms_list = []
    for idx, distance in enumerate([0.70, 0.80, 0.90, 1.00]):
        atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
        atoms.info["energy"] = 0.0
        atoms.arrays["forces"] = np.zeros((2, 3), dtype=np.float64)
        atoms.arrays["teacher_forces"] = np.full((2, 3), float(idx), dtype=np.float64)
        atoms_list.append(atoms)
    ase.io.write(configs, atoms_list, format="extxyz")

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py",
            "--configs",
            str(configs),
            "--output-json",
            str(output),
            "--reference-path-ids",
            "atomic.radial_density,edge.direct.radial",
            "--auto-candidate-strategy",
            "single_delete",
            "--num-radial",
            "3",
            "--limit-configs",
            "4",
            "--neighborlist-backend",
            "ase",
            "--force-target-key",
            "teacher_forces",
            "--force-eval-stride",
            "2",
            "--force-eval-offset",
            "1",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["force_target_key"] == "teacher_forces"
    assert payload["force_target_num_configs"] == 4
    assert payload["force_eval_stride"] == 2
    assert payload["force_eval_offset"] == 1
    assert payload["force_fit_num_configs"] == 2
    assert payload["force_eval_num_configs"] == 2
    assert payload["rows"][0]["force_fit_num_samples"] == 12
    assert payload["rows"][0]["force_eval_num_samples"] == 12
    assert "force_rmse" in payload["rows"][0]
    assert "force_max_abs" in payload["rows"][0]


def test_rtece_projection_cli_samples_force_components(tmp_path):
    import numpy as np
    import ase.io
    from ase import Atoms

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    configs = tmp_path / "h2_force_sample.xyz"
    output = tmp_path / "projection_force_sample.json"
    atoms_list = []
    for idx, distance in enumerate([0.70, 0.80, 0.90, 1.00]):
        atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
        atoms.info["energy"] = 0.0
        atoms.arrays["forces"] = np.zeros((2, 3), dtype=np.float64)
        atoms.arrays["teacher_forces"] = np.full((2, 3), float(idx), dtype=np.float64)
        atoms_list.append(atoms)
    ase.io.write(configs, atoms_list, format="extxyz")

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py",
            "--configs",
            str(configs),
            "--output-json",
            str(output),
            "--reference-path-ids",
            "atomic.radial_density,edge.direct.radial",
            "--candidate",
            "baseline:atomic.radial_density",
            "--candidate",
            "full_reference:atomic.radial_density,edge.direct.radial",
            "--num-radial",
            "3",
            "--limit-configs",
            "4",
            "--neighborlist-backend",
            "ase",
            "--force-target-key",
            "teacher_forces",
            "--force-eval-stride",
            "2",
            "--force-eval-offset",
            "1",
            "--force-component-sample-count",
            "6",
            "--active-set-baseline-candidate",
            "baseline",
            "--active-set-energy-weight",
            "0.0",
            "--active-set-force-weight",
            "1.0",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["force_component_sample_count"] == 6
    assert payload["force_component_sampled"] is True
    assert payload["force_target_num_components"] == 24
    assert payload["force_sampled_num_components"] == 6
    assert payload["force_fit_num_components"] == 3
    assert payload["force_eval_num_components"] == 3
    assert all(row["force_fit_num_samples"] == 3 for row in payload["rows"])
    assert all(row["force_eval_num_samples"] == 3 for row in payload["rows"])
    assert payload["active_set_rows"][0]["active_set_force_metric"] == "force_rmse"


def test_rtece_projection_cli_adds_combined_energy_force_ranking_fields(tmp_path):
    import numpy as np
    import ase.io
    from ase import Atoms

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    configs = tmp_path / "h2_energy_force_split.xyz"
    output = tmp_path / "projection_energy_force_split.json"
    atoms_list = []
    for idx, distance in enumerate([0.70, 0.80, 0.90, 1.00]):
        atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
        atoms.info["teacher_energy"] = float(idx)
        atoms.arrays["forces"] = np.zeros((2, 3), dtype=np.float64)
        atoms.arrays["teacher_forces"] = np.full((2, 3), float(idx), dtype=np.float64)
        atoms_list.append(atoms)
    ase.io.write(configs, atoms_list, format="extxyz")

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py",
            "--configs",
            str(configs),
            "--output-json",
            str(output),
            "--reference-path-ids",
            "atomic.radial_density,edge.direct.radial",
            "--auto-candidate-strategy",
            "single_delete",
            "--candidate",
            "full_reference:atomic.radial_density,edge.direct.radial",
            "--num-radial",
            "3",
            "--limit-configs",
            "4",
            "--neighborlist-backend",
            "ase",
            "--energy-target-key",
            "teacher_energy",
            "--energy-eval-stride",
            "2",
            "--force-target-key",
            "teacher_forces",
            "--force-eval-stride",
            "2",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["rank_metrics"]["energy"] == "energy_per_atom_rmse"
    assert payload["rank_metrics"]["force"] == "force_rmse"
    assert payload["rank_metrics"]["combined"] == "minimize max(energy_rank, force_rank), then rank sum, then descriptor dim"
    assert all("energy_rank" in row for row in payload["rows"])
    assert all("force_rank" in row for row in payload["rows"])
    assert all("ef_combined_rank" in row for row in payload["rows"])
    assert all("ef_pareto_dominated" in row for row in payload["rows"])


def test_rtece_projection_cli_emits_active_set_candidate_rows(tmp_path):
    import numpy as np
    import ase.io
    from ase import Atoms

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    configs = tmp_path / "h2_active_set.xyz"
    output = tmp_path / "projection_active_set.json"
    atoms_list = []
    for idx, distance in enumerate([0.70, 0.80, 0.90, 1.00]):
        atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
        atoms.info["teacher_energy"] = float(idx)
        atoms.arrays["forces"] = np.zeros((2, 3), dtype=np.float64)
        atoms.arrays["teacher_forces"] = np.full((2, 3), float(idx), dtype=np.float64)
        atoms_list.append(atoms)
    ase.io.write(configs, atoms_list, format="extxyz")

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py",
            "--configs",
            str(configs),
            "--output-json",
            str(output),
            "--reference-path-ids",
            "atomic.radial_density,edge.direct.radial",
            "--candidate",
            "baseline:atomic.radial_density",
            "--candidate",
            "full_reference:atomic.radial_density,edge.direct.radial",
            "--num-radial",
            "3",
            "--limit-configs",
            "4",
            "--neighborlist-backend",
            "ase",
            "--energy-target-key",
            "teacher_energy",
            "--force-target-key",
            "teacher_forces",
            "--active-set-baseline-candidate",
            "baseline",
            "--active-set-energy-weight",
            "1.0",
            "--active-set-force-weight",
            "0.5",
            "--active-set-projection-weight",
            "10.0",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["active_set"] == {
        "baseline_candidate": "baseline",
        "energy_weight": 1.0,
        "force_weight": 0.5,
        "projection_weight": 10.0,
        "gain_mode": "absolute",
        "max_force_regression_fraction": None,
        "require_energy_gain": False,
    }
    assert [row["candidate"] for row in payload["active_set_rows"]] == ["full_reference"]
    active_row = payload["active_set_rows"][0]
    assert active_row["baseline_candidate"] == "baseline"
    assert active_row["marginal_paths"] == ["edge.direct.radial"]
    assert "marginal_cost_proxy" in active_row
    assert "marginal_gain_per_cost" in active_row
    assert "active_set_promoted" in active_row


def test_rtece_projection_cli_generates_auto_candidates_and_ranks_rows(tmp_path):
    import numpy as np
    import ase.io
    from ase import Atoms

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    configs = tmp_path / "h2.xyz"
    output = tmp_path / "projection.json"
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    atoms.info["energy"] = 0.0
    atoms.arrays["forces"] = np.zeros((2, 3), dtype=np.float64)
    ase.io.write(configs, [atoms], format="extxyz")

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py",
            "--configs",
            str(configs),
            "--output-json",
            str(output),
            "--reference-path-ids",
            "atomic.radial_density,edge.direct.radial",
            "--auto-candidate-strategy",
            "single_delete",
            "--num-radial",
            "3",
            "--limit-configs",
            "1",
            "--neighborlist-backend",
            "ase",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["auto_candidate_strategies"] == ["single_delete"]
    assert payload["auto_candidate_count"] == 1
    assert len(payload["rows"]) == 1
    row = payload["rows"][0]
    assert row["candidate"] == "single_delete_edge_direct_radial"
    assert row["candidate_scalar_path_ids"] == ["atomic.radial_density"]
    assert row["deleted_scalar_path_ids"] == ["edge.direct.radial"]
    assert row["projection_rank"] == 1
    assert row["relative_residual"] >= 0.0


def test_rtece_projection_diagnostic_row_reports_deleted_path_residual():
    from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import make_projection_diagnostic_row
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import build_rtece_config_from_path_ids

    candidate = build_rtece_config_from_path_ids(
        "rtece_path_keep_vector",
        ("atomic.radial_density", "edge.cavity.vector_dot"),
        num_radial=3,
    )
    reference = build_rtece_config_from_path_ids(
        "rtece_path_keep_vector_and_radial",
        ("atomic.radial_density", "edge.cavity.vector_dot", "edge.direct.radial"),
        num_radial=3,
    )
    graph = RTECEGraph(
        z=torch.tensor([6, 1, 8], dtype=torch.long),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [0.7, 0.1, 0.0], [0.2, 0.9, 0.1]],
            dtype=torch.float64,
        ),
        edge_index=complete_directed_edges(3),
        batch=torch.zeros(3, dtype=torch.long),
    )

    row = make_projection_diagnostic_row(
        "keep_vector",
        candidate_config=candidate,
        reference_config=reference,
        graphs=[graph],
    )

    assert row["candidate"] == "keep_vector"
    assert row["num_graphs"] == 1
    assert row["num_samples"] == 3
    assert row["candidate_dim"] == 4
    assert row["reference_dim"] == 6
    assert row["deleted_scalar_path_ids"] == ["edge.direct.radial"]
    assert row["deleted_cost_groups"] == ["direct_pair_radial"]
    assert row["deleted_descriptor_dim"] == 2
    assert row["descriptor_dim_reduction"] == 2
    assert row["retained_cost_groups"] == ["atomic_scalar_density", "edge_cavity_relations"]
    assert row["relative_residual"] >= 0.0
    assert row["candidate_manifest_hash"] == rtece_path_manifest(candidate)["manifest_hash"]
    assert row["reference_manifest_hash"] == rtece_path_manifest(reference)["manifest_hash"]


def test_rtece_projection_diagnostic_row_accepts_sample_weights():
    from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import make_projection_diagnostic_row
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import build_rtece_config_from_path_ids

    candidate = build_rtece_config_from_path_ids(
        "rtece_path_keep_radial",
        ("atomic.radial_density",),
        num_radial=3,
    )
    reference = build_rtece_config_from_path_ids(
        "rtece_path_keep_vector_and_radial",
        ("atomic.radial_density", "edge.cavity.vector_dot"),
        num_radial=3,
    )
    graph = RTECEGraph(
        z=torch.tensor([6, 1, 8], dtype=torch.long),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [0.7, 0.1, 0.0], [0.2, 0.9, 0.1]],
            dtype=torch.float64,
        ),
        edge_index=complete_directed_edges(3),
        batch=torch.zeros(3, dtype=torch.long),
    )

    row = make_projection_diagnostic_row(
        "weighted_keep_radial",
        candidate_config=candidate,
        reference_config=reference,
        graphs=[graph],
        sample_weights=torch.tensor([3.0, 1.0, 2.0]),
    )

    assert row["weighted"] is True
    assert row["weight_sum"] == pytest.approx(6.0)
    assert row["num_samples"] == 3


def test_rtece_species_basis_variant_has_route_contract():
    config = build_rtece_config("rtece_species_basis4")
    route = rtece_route_contract(config, force_mode="autograd")

    assert config.species_basis_channels == 4
    assert descriptor_dim(config) == config.num_radial * 5
    assert route["semantic_tier"] == "T3_low_rank_species_density"
    assert route["descriptor_family"] == "species_basis_density"
    assert "low_rank_neighbor_species_basis" in route["retained_tece_groups"]


def test_learnable_species_basis_initializes_as_fixed_z_power_and_is_trainable():
    fixed = build_rtece_config_from_path_ids(
        "fixed_species_basis",
        ("atomic.radial_density", "atomic.species_basis_density"),
        num_radial=3,
        species_basis_channels=4,
    )
    learnable = build_rtece_config_from_path_ids(
        "learnable_species_basis",
        ("atomic.radial_density", "atomic.species_basis_density"),
        num_radial=3,
        species_basis_channels=4,
        species_basis_mode="learnable_embedding",
    )
    graph = RTECEGraph(
        z=torch.tensor([6, 8, 1, 7], dtype=torch.long),
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.1, 0.0],
                [0.2, 0.9, 0.1],
                [-0.3, 0.4, 0.7],
            ],
            dtype=torch.float64,
        ),
        edge_index=complete_directed_edges(4),
        batch=torch.zeros(4, dtype=torch.long),
    )
    model = RTECEScalarModel(learnable).double()

    assert learnable.species_basis_mode == "learnable_embedding"
    assert "species_basis_embedding.weight" in dict(model.named_parameters())
    powers = torch.arange(1, 5, dtype=torch.float64)
    expected_c = (torch.tensor(6.0, dtype=torch.float64) / float(learnable.max_atomic_number)).pow(powers)
    assert torch.allclose(model.species_basis_embedding.weight[6], expected_c)

    fixed_descriptors = rtece_descriptors(graph, fixed)
    learnable_descriptors = rtece_descriptors(
        graph,
        learnable,
        species_basis_embedding=model.species_basis_embedding.weight,
    )

    assert torch.allclose(learnable_descriptors, fixed_descriptors, atol=1e-12, rtol=1e-12)
    route = rtece_route_contract(learnable)
    manifest = rtece_path_manifest(learnable)
    assert "learnable_low_rank_species_basis" in route["retained_tece_groups"]
    assert "trainable_species_basis" in route["pareto_axes"]
    assert manifest["config"]["species_basis_mode"] == "learnable_embedding"
    assert manifest["moments"][1]["chemistry_basis"] == "learnable_embedding_4"


def test_learnable_species_basis_forwards_to_cavity_edge_moments():
    config = build_rtece_config_from_path_ids(
        "learnable_species_cavity",
        ("atomic.radial_density", "atomic.species_basis_density", "edge.cavity.vector_dot"),
        num_radial=3,
        species_basis_channels=4,
        species_basis_mode="learnable_embedding",
        moment_l_max=1,
    )
    graph = RTECEGraph(
        z=torch.tensor([6, 8, 1, 7], dtype=torch.long),
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.8, 0.1, 0.0],
                [0.2, 0.9, 0.1],
                [-0.3, 0.4, 0.7],
            ],
            dtype=torch.float64,
        ),
        edge_index=complete_directed_edges(4),
        batch=torch.zeros(4, dtype=torch.long),
    )
    model = RTECEScalarModel(config).double()

    descriptors = rtece_descriptors(
        graph,
        config,
        species_basis_embedding=model.species_basis_embedding.weight,
    )
    output = model(graph)

    assert descriptors.shape == (4, descriptor_dim(config))
    assert torch.isfinite(output["energy"]).all()


def test_rtece_species_cavity_path_reports_both_retained_groups():
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import build_rtece_config_from_path_ids

    config = build_rtece_config_from_path_ids(
        "species_cavity_vec",
        ("atomic.radial_density", "atomic.species_basis_density", "edge.cavity.vector_dot"),
        species_basis_channels=4,
    )

    route = rtece_route_contract(config, force_mode="autograd")

    assert "low_rank_neighbor_species_basis" in route["retained_tece_groups"]
    assert "low_order_atomic_moments" in route["retained_tece_groups"]
    assert "cavity_edge_relational_scalar_sketches" in route["retained_tece_groups"]
    assert route["semantic_tier"] == "T3_species_cavity_edge_scalar_sketch"
    assert route["descriptor_family"] == "species_basis_density_plus_cavity_edge_sketch"


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
    configs = [
        build_rtece_config("rtece_edge_sketch8"),
        build_rtece_config("rtece_cavity_edge_sketch8"),
        build_rtece_config("rtece_cavity_radial_edge_sketch14"),
    ]
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

    for config in configs:
        sketches = edge_relational_sketches(graph, config)
        sketches_rot = edge_relational_sketches(rotated, config)
        full = rtece_descriptors(graph, config)
        full_rot = rtece_descriptors(rotated, config)

        assert sketches.shape == (4, config.num_edge_sketches)
        assert torch.allclose(sketches, sketches_rot, atol=1e-10, rtol=1e-10)
        assert torch.allclose(full, full_rot, atol=1e-10, rtol=1e-10)


def test_edge_sketch16_adds_non_repeated_relational_paths():
    config = build_rtece_config("rtece_edge_sketch16")
    z = torch.tensor([6, 8, 1, 7], dtype=torch.long)
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

    sketches = edge_relational_sketches(graph, config)

    assert sketches.shape == (4, 16)
    assert not torch.allclose(sketches[:, :8], sketches[:, 8:], atol=1e-12, rtol=1e-12)



def test_radial_edge_projection_preserves_shell_information_lost_by_mean():
    channels_a = torch.tensor([[[1.0], [1.0], [0.0], [0.0]]], dtype=torch.float64)
    channels_b = torch.tensor([[[0.0], [0.0], [1.0], [1.0]]], dtype=torch.float64)

    assert torch.allclose(channels_a.mean(dim=1), channels_b.mean(dim=1))
    projected_a = _project_radial_edge_channels(channels_a, 2)
    projected_b = _project_radial_edge_channels(channels_b, 2)

    assert projected_a.shape == (1, 2, 1)
    assert not torch.allclose(projected_a, projected_b)


def test_radial_edge_projection_accepts_explicit_projection_matrix():
    channels = torch.tensor(
        [
            [[1.0], [2.0], [4.0], [8.0]],
            [[0.5], [1.5], [2.5], [3.5]],
        ],
        dtype=torch.float64,
    )
    projection = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5, 0.0],
        ],
        dtype=torch.float64,
    )

    projected = _project_radial_edge_channels(channels, 2, projection)

    expected = torch.tensor([[[1.0], [3.0]], [[0.5], [2.0]]], dtype=torch.float64)
    assert projected.shape == (2, 2, 1)
    assert torch.allclose(projected, expected)


def test_learnable_atomic_cross_radial_projection_is_manifested_and_trainable():
    config = build_rtece_config_from_path_ids(
        "learnable_cross_projection",
        ("atomic.radial_density", "atomic.vector_norm", "atomic.vector_cross_radial_dot"),
        moment_l_max=1,
        atomic_cross_radial_sketch_channels=2,
        atomic_cross_radial_projection="learnable",
    )
    model = RTECEScalarModel(config)
    manifest = rtece_path_manifest(config)

    assert model.atomic_cross_radial_projection is not None
    assert tuple(model.atomic_cross_radial_projection.weight.shape) == (2, config.num_radial)
    assert any(
        path["id"] == "atomic.vector_cross_radial_dot"
        and path["radial_projection"] == "learnable_2x8_cross_radial_projection"
        for path in manifest["scalar_paths"]
    )
    assert "learnable_atomic_cross_radial_projection" in manifest["retained_tece_groups"]


def test_pod_fixed_atomic_cross_radial_projection_is_manifested_and_not_trainable():
    projection = ((1.0, 0.0, 0.0, 0.0), (0.0, 0.25, 0.25, 0.5))
    config = build_rtece_config_from_path_ids(
        "pod_fixed_cross_projection",
        ("atomic.radial_density", "atomic.vector_norm", "atomic.vector_cross_radial_dot"),
        num_radial=4,
        moment_l_max=1,
        atomic_cross_radial_sketch_channels=2,
        atomic_cross_radial_projection="pod_fixed",
        atomic_cross_radial_projection_matrix=projection,
    )
    model = RTECEScalarModel(config)
    manifest = rtece_path_manifest(config)

    assert model.atomic_cross_radial_projection is None
    assert tuple(model.atomic_cross_radial_projection_buffer.shape) == (2, 4)
    assert not any(name == "atomic_cross_radial_projection_buffer" for name, _ in model.named_parameters())
    assert any(
        path["id"] == "atomic.vector_cross_radial_dot"
        and path["radial_projection"] == "pod_fixed_2x4_cross_radial_projection"
        for path in manifest["scalar_paths"]
    )
    assert "pod_fixed_atomic_cross_radial_projection" in manifest["retained_tece_groups"]


def test_pod_fixed_atomic_cross_radial_projection_requires_matrix_shape():
    with pytest.raises(ValueError, match="atomic_cross_radial_projection_matrix must have shape"):
        build_rtece_config_from_path_ids(
            "bad_pod_fixed_cross_projection",
            ("atomic.radial_density", "atomic.vector_norm", "atomic.vector_cross_radial_dot"),
            num_radial=4,
            moment_l_max=1,
            atomic_cross_radial_sketch_channels=2,
            atomic_cross_radial_projection="pod_fixed",
            atomic_cross_radial_projection_matrix=((1.0, 0.0), (0.0, 1.0)),
        )


def test_cavity_edge_sketches_remove_self_edge_leakage_for_isolated_pair():
    full_config = build_rtece_config("rtece_edge_sketch8")
    cavity_config = build_rtece_config("rtece_cavity_edge_sketch8")
    z = torch.tensor([6, 8], dtype=torch.long)
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]], dtype=torch.float64)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    graph = RTECEGraph(
        z=z,
        pos=pos,
        edge_index=edge_index,
        batch=torch.zeros(2, dtype=torch.long),
    )

    full = edge_relational_sketches(graph, full_config)
    cavity = edge_relational_sketches(graph, cavity_config)

    assert not torch.allclose(full[:, :6], torch.zeros_like(full[:, :6]))
    assert torch.allclose(cavity[:, :6], torch.zeros_like(cavity[:, :6]), atol=1e-12, rtol=1e-12)
    assert torch.allclose(cavity[:, 6:], full[:, 6:], atol=1e-12, rtol=1e-12)


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
    assert metadata["tece_path_manifest"]["schema_version"] == "rtece_path_manifest.v1"
    assert metadata["tece_path_manifest"]["manifest_hash"] == rtece_path_manifest(
        config,
        force_mode="analytic_density",
        graph_construction_backend="torch_radius_nopbc",
    )["manifest_hash"]
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


def test_rtece_package_exports_workflow_alias_for_import_smoke():
    from tace.models import RTECEWorkflow, RTECEScalarModel, rtece_workflow

    assert RTECEScalarModel is not None
    assert RTECEWorkflow is rtece_workflow
    assert RTECEWorkflow.load_checkpoint is rtece_workflow.load_checkpoint


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


def test_rtece_summary_uses_benchmark_architecture_manifest_for_path_id_label():
    from benchmarks.oc20neb_tace_mace.summarize_tece_distill import make_student_row
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import build_rtece_config_from_path_ids

    config = build_rtece_config_from_path_ids(
        "radial",
        ("atomic.radial_density",),
        num_radial=3,
        hidden_channels=(4,),
    )
    manifest = rtece_path_manifest(config)
    dft = {
        "model": "rtece_scalar.pt",
        "variant": "radial",
        "force_mode": "autograd",
        "hidden_channels": [4],
        "num_radial": 3,
        "atoms_per_second": 10.0,
        "mae_e_mev_atom": 1.0,
        "mae_f_mev_a": 2.0,
        "tece_architecture_path_manifest": manifest,
        "tece_architecture_path_manifest_hash": manifest["manifest_hash"],
    }
    teacher = dict(dft)
    teacher["mae_f_mev_a"] = 3.0

    row = make_student_row("radial", dft_benchmark=dft, teacher_benchmark=teacher)

    assert row["tece_path_manifest_hash"] == manifest["manifest_hash"]
    assert row["tece_path_manifest"]["config"]["scalar_path_ids"] == ["atomic.radial_density"]


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
        "bias_e_mev_atom": -3.0,
        "max_abs_e_mev_atom": 55.0,
        "mae_f_mev_a": 40.0,
        "rmse_f_mev_a": 80.0,
        "max_abs_f_mev_a": 700.0,
    }
    teacher = dict(dft)
    teacher["mae_f_mev_a"] = 39.0
    teacher["rmse_f_mev_a"] = 79.0
    teacher["max_abs_f_mev_a"] = 690.0

    row = make_student_row("rtece_edge_sketch8", dft_benchmark=dft, teacher_benchmark=teacher)

    assert row["variant"] == "rtece_edge_sketch8"
    assert row["atoms_per_second"] == 100000.0
    assert row["dft_f_mae_mev_a"] == 40.0
    assert row["dft_f_rmse_mev_a"] == 80.0
    assert row["dft_f_max_abs_mev_a"] == 700.0
    assert row["dft_e_rmse_mev_atom"] == 20.0
    assert row["dft_e_bias_mev_atom"] == -3.0
    assert row["dft_e_max_abs_mev_atom"] == 55.0
    assert row["teacher_f_mae_mev_a"] == 39.0
    assert row["teacher_f_rmse_mev_a"] == 79.0
    assert row["teacher_f_max_abs_mev_a"] == 690.0


def test_train_rtece_scalar_builds_config_from_scalar_path_ids():
    from types import SimpleNamespace

    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import build_training_config

    args = SimpleNamespace(
        variant="radial_cavity_vec",
        scalar_path_ids="atomic.radial_density,edge.cavity.vector_dot",
        hidden_channels="8",
        num_radial=3,
    )

    config = build_training_config(args)

    assert config.variant == "radial_cavity_vec"
    assert config.scalar_path_ids == ("atomic.radial_density", "edge.cavity.vector_dot")
    assert config.hidden_channels == (8,)
    assert config.num_radial == 3
    assert config.num_edge_sketches == 1


def test_train_rtece_scalar_builds_species_cavity_path_id_config():
    from types import SimpleNamespace

    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import build_training_config

    args = SimpleNamespace(
        variant="species_cavity_vec",
        scalar_path_ids="atomic.radial_density,atomic.species_basis_density,edge.cavity.vector_dot",
        species_basis_channels=4,
        hidden_channels="16,16",
        num_radial=8,
    )

    config = build_training_config(args)

    assert config.scalar_path_ids == (
        "atomic.radial_density",
        "atomic.species_basis_density",
        "edge.cavity.vector_dot",
    )
    assert config.species_basis_channels == 4
    assert config.num_edge_sketches == 1
    assert config.use_cavity_edge_sketches is True


def test_training_config_builders_forward_species_basis_mode():
    from types import SimpleNamespace

    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import build_training_config as script_build
    from tace.lightning.rtece import build_training_config as lightning_build

    args = SimpleNamespace(
        variant="species_learnable",
        scalar_path_ids="atomic.radial_density,atomic.species_basis_density",
        species_basis_channels=8,
        species_basis_mode="learnable_embedding",
        hidden_channels="32,32",
        num_radial=4,
    )

    script_config = script_build(args)
    lightning_config = lightning_build(
        variant="species_learnable",
        scalar_path_ids="atomic.radial_density,atomic.species_basis_density",
        species_basis_channels=8,
        species_basis_mode="learnable_embedding",
        hidden_channels="32,32",
        num_radial=4,
    )

    assert script_config.species_basis_mode == "learnable_embedding"
    assert lightning_config.species_basis_mode == "learnable_embedding"


def test_train_rtece_scalar_builds_config_with_moment_l_max():
    import argparse
    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import build_training_config

    args = argparse.Namespace(
        variant="rtece_pair",
        scalar_path_ids=None,
        hidden_channels="4",
        num_radial=3,
        species_basis_channels=0,
        use_short_range_repulsion=False,
        short_range_repulsion_potential="softplus_overlap",
        short_range_repulsion_strength=0.0,
        short_range_repulsion_beta=10.0,
        short_range_repulsion_radius_scale=0.75,
        learnable_radial_mixing=False,
        moment_l_max=2,
    )

    config = build_training_config(args)

    assert config.moment_l_max == 2
    assert config.use_atomic_moments is True
    assert descriptor_dim(config) == 9


def test_train_rtece_scalar_builds_config_with_learnable_radial_mixing():
    import argparse
    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import build_training_config

    args = argparse.Namespace(
        variant="radial_learnable",
        scalar_path_ids="atomic.radial_density",
        hidden_channels="4",
        num_radial=3,
        species_basis_channels=0,
        use_short_range_repulsion=False,
        short_range_repulsion_strength=0.0,
        short_range_repulsion_beta=10.0,
        short_range_repulsion_radius_scale=0.75,
        learnable_radial_mixing=True,
    )

    config = build_training_config(args)

    assert config.learnable_radial_mixing is True
    assert config.scalar_path_ids == ("atomic.radial_density",)


def test_train_rtece_scalar_builds_config_with_atomic_cross_radial_sketch_channels():
    import argparse
    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import build_training_config

    args = argparse.Namespace(
        variant="l1_cross_k4",
        scalar_path_ids="atomic.radial_density,atomic.vector_norm,atomic.vector_cross_radial_dot",
        hidden_channels="4",
        num_radial=8,
        species_basis_channels=0,
        moment_l_max=1,
        atomic_cross_radial_sketch_channels=4,
        atomic_cross_radial_projection="learnable",
        use_short_range_repulsion=False,
        short_range_repulsion_potential="softplus_overlap",
        short_range_repulsion_strength=0.0,
        short_range_repulsion_beta=10.0,
        short_range_repulsion_radius_scale=0.75,
        learnable_radial_mixing=False,
    )

    config = build_training_config(args)

    assert config.atomic_cross_radial_sketch_channels == 4
    assert config.atomic_cross_radial_projection == "learnable"
    assert descriptor_dim(config) == 8 + 8 + 6


def test_lightning_build_training_config_accepts_atomic_cross_radial_projection():
    from tace.lightning.rtece import build_training_config

    config = build_training_config(
        variant="l1_cross_learnproj",
        scalar_path_ids="atomic.radial_density,atomic.vector_norm,atomic.vector_cross_radial_dot",
        moment_l_max=1,
        atomic_cross_radial_sketch_channels=2,
        atomic_cross_radial_projection="learnable",
    )

    assert config.atomic_cross_radial_projection == "learnable"


def test_train_rtece_scalar_loads_pod_fixed_atomic_cross_radial_projection_json(tmp_path):
    from tace.scripts.rtece_train_scalar import load_atomic_cross_radial_projection_matrix

    path = tmp_path / "radial_pod.json"
    path.write_text(json.dumps({"projection_matrix": [[1.0, 0.0, 0.0, 0.0], [0.0, 0.25, 0.25, 0.5]]}))

    matrix = load_atomic_cross_radial_projection_matrix(path)

    assert matrix == ((1.0, 0.0, 0.0, 0.0), (0.0, 0.25, 0.25, 0.5))


def test_lightning_build_training_config_accepts_pod_fixed_atomic_cross_radial_projection_matrix():
    from tace.lightning.rtece import build_training_config

    config = build_training_config(
        variant="l1_cross_podproj",
        scalar_path_ids="atomic.radial_density,atomic.vector_norm,atomic.vector_cross_radial_dot",
        num_radial=4,
        moment_l_max=1,
        atomic_cross_radial_sketch_channels=2,
        atomic_cross_radial_projection="pod_fixed",
        atomic_cross_radial_projection_matrix=((1.0, 0.0, 0.0, 0.0), (0.0, 0.25, 0.25, 0.5)),
    )

    assert config.atomic_cross_radial_projection == "pod_fixed"
    assert config.atomic_cross_radial_projection_matrix == ((1.0, 0.0, 0.0, 0.0), (0.0, 0.25, 0.25, 0.5))


def test_descriptor_residual_conditioner_is_identity_initialized_and_manifested():
    base = build_rtece_config_from_path_ids(
        "l1_cross_base",
        ("atomic.radial_density", "atomic.vector_norm", "atomic.vector_cross_radial_dot"),
        num_radial=4,
        hidden_channels=(8,),
        moment_l_max=1,
        atomic_cross_radial_sketch_channels=2,
    )
    conditioned_config = build_rtece_config_from_path_ids(
        "l1_cross_conditioned",
        ("atomic.radial_density", "atomic.vector_norm", "atomic.vector_cross_radial_dot"),
        num_radial=4,
        hidden_channels=(8,),
        moment_l_max=1,
        atomic_cross_radial_sketch_channels=2,
        descriptor_conditioner="residual_mlp",
        descriptor_conditioner_hidden_channels=5,
    )
    base_model = RTECEScalarModel(base).double()
    conditioned_model = RTECEScalarModel(conditioned_config).double()
    conditioned_model.energy_head.load_state_dict(base_model.energy_head.state_dict())
    graph = RTECEGraph(
        z=torch.tensor([6, 1, 8], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.1, 0.0], [0.2, 0.9, 0.1]], dtype=torch.float64),
        edge_index=complete_directed_edges(3),
        batch=torch.zeros(3, dtype=torch.long),
    )

    base_out = base_model(graph)
    conditioned_out = conditioned_model(graph)
    manifest = rtece_path_manifest(conditioned_config)
    route = rtece_route_contract(conditioned_config)

    assert conditioned_config.descriptor_conditioner == "residual_mlp"
    assert conditioned_config.descriptor_conditioner_hidden_channels == 5
    assert conditioned_model.descriptor_conditioner is not None
    assert tuple(conditioned_model.descriptor_conditioner[0].weight.shape) == (5, descriptor_dim(conditioned_config))
    assert torch.allclose(conditioned_model.descriptor_conditioner[-1].weight, torch.zeros_like(conditioned_model.descriptor_conditioner[-1].weight))
    assert torch.allclose(conditioned_out["energy"], base_out["energy"], atol=1e-12)
    assert torch.allclose(conditioned_out["forces"], base_out["forces"], atol=1e-12)
    assert manifest["config"]["descriptor_conditioner"] == "residual_mlp"
    assert manifest["config"]["descriptor_conditioner_hidden_channels"] == 5
    assert "trainable_scalar_descriptor_conditioner" in manifest["retained_tece_groups"]
    assert "scalar_descriptor_conditioning" in route["pareto_axes"]


def test_build_config_accepts_radial_species_adapter_channels():
    import argparse

    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import build_training_config as benchmark_build
    from tace.lightning.rtece import build_training_config as lightning_build

    args = argparse.Namespace(
        variant="radial_species_adapter_train",
        hidden_channels="8",
        num_radial=4,
        scalar_path_ids="atomic.radial_density",
        learnable_radial_mixing=False,
        radial_species_adapter_channels=6,
        radial_species_adapter_scope="atomic",
        moment_l_max=0,
        species_basis_channels=0,
        species_basis_mode="fixed_z_power",
        atomic_cross_radial_sketch_channels=2,
        atomic_cross_radial_projection="fixed_shell_mean",
        atomic_cross_radial_projection_matrix=None,
        descriptor_conditioner="none",
        descriptor_conditioner_hidden_channels=0,
        descriptor_bottleneck_dim=0,
        use_short_range_repulsion=False,
        short_range_repulsion_potential="softplus_overlap",
        short_range_repulsion_strength=0.0,
        short_range_repulsion_beta=10.0,
        short_range_repulsion_radius_scale=0.75,
    )

    benchmark_config = benchmark_build(args)
    lightning_config = lightning_build(
        variant="radial_species_adapter_train",
        hidden_channels="8",
        num_radial=4,
        scalar_path_ids="atomic.radial_density",
        moment_l_max=0,
        radial_species_adapter_channels=6,
        radial_species_adapter_scope="atomic",
    )

    assert benchmark_config.radial_species_adapter_channels == 6
    assert benchmark_config.radial_species_adapter_scope == "atomic"
    assert lightning_config.radial_species_adapter_channels == 6
    assert lightning_config.radial_species_adapter_scope == "atomic"


def test_descriptor_bottleneck_reduces_head_input_and_is_manifested():
    config = build_rtece_config_from_path_ids(
        "l2_bottleneck",
        (
            "atomic.radial_density",
            "atomic.species_basis_density",
            "atomic.vector_norm",
            "atomic.vector_cross_radial_dot",
            "atomic.quadrupole_norm",
            "atomic.quadrupole_cross_radial_frobenius",
        ),
        num_radial=4,
        hidden_channels=(8,),
        moment_l_max=2,
        species_basis_channels=6,
        descriptor_bottleneck_dim=5,
        atomic_cross_radial_sketch_channels=2,
    )
    model = RTECEScalarModel(config).double()
    graph = RTECEGraph(
        z=torch.tensor([6, 1, 8], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.1, 0.0], [0.2, 0.9, 0.1]], dtype=torch.float64),
        edge_index=complete_directed_edges(3),
        batch=torch.zeros(3, dtype=torch.long),
    )

    output = model(graph)
    manifest = rtece_path_manifest(config)
    route = rtece_route_contract(config)

    assert config.descriptor_bottleneck_dim == 5
    assert model.descriptor_bottleneck is not None
    assert tuple(model.descriptor_bottleneck[0].weight.shape) == (5, descriptor_dim(config))
    assert tuple(model.energy_head[0].weight.shape) == (8, 6)
    assert torch.isfinite(output["energy"]).all()
    assert torch.isfinite(output["forces"]).all()
    assert manifest["config"]["descriptor_bottleneck_dim"] == 5
    assert route["descriptor_readout_dim"] == 5
    assert "trainable_low_rank_descriptor_mixer" in manifest["retained_tece_groups"]
    assert "descriptor_bottleneck" in route["pareto_axes"]


def test_radial_species_adapter_rejects_inference_only_analytic_backends():
    config = build_rtece_config_from_path_ids(
        "radial_species_adapter_pair",
        ("atomic.radial_density",),
        num_radial=4,
        hidden_channels=(8,),
        radial_species_adapter_channels=3,
    )
    model = RTECEScalarModel(config).double().eval()
    graph = RTECEGraph(
        z=torch.tensor([1, 6], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]], dtype=torch.float64),
        edge_index=complete_directed_edges(2),
        batch=torch.zeros(2, dtype=torch.long),
    )

    with pytest.raises(ValueError, match="radial species adapter"):
        model.forward_pair_analytic_forces(graph)


def test_descriptor_bottleneck_rejects_inference_only_analytic_backends():
    config = build_rtece_config_from_path_ids(
        "bottleneck_pair",
        ("atomic.radial_density",),
        num_radial=4,
        hidden_channels=(8,),
        descriptor_bottleneck_dim=3,
    )
    model = RTECEScalarModel(config).double().eval()
    graph = RTECEGraph(
        z=torch.tensor([1, 1], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]], dtype=torch.float64),
        edge_index=complete_directed_edges(2),
        batch=torch.zeros(2, dtype=torch.long),
    )

    with pytest.raises(ValueError, match="descriptor bottleneck"):
        model.forward_pair_analytic_forces(graph)


def test_descriptor_conditioner_rejects_inference_only_analytic_backends():
    config = build_rtece_config_from_path_ids(
        "conditioned_pair",
        ("atomic.radial_density",),
        num_radial=4,
        hidden_channels=(8,),
        descriptor_conditioner="residual_mlp",
        descriptor_conditioner_hidden_channels=4,
    )
    model = RTECEScalarModel(config).double().eval()
    graph = RTECEGraph(
        z=torch.tensor([1, 1], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.75, 0.0, 0.0]], dtype=torch.float64),
        edge_index=complete_directed_edges(2),
        batch=torch.zeros(2, dtype=torch.long),
    )

    with pytest.raises(ValueError, match="descriptor conditioner"):
        model.forward_pair_analytic_forces(graph)


def test_train_and_lightning_build_config_accept_descriptor_conditioner():
    import argparse
    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import build_training_config as benchmark_build_training_config
    from tace.lightning.rtece import build_training_config as lightning_build_training_config

    args = argparse.Namespace(
        variant="l1_conditioned",
        scalar_path_ids="atomic.radial_density,atomic.vector_norm",
        hidden_channels="8",
        num_radial=4,
        species_basis_channels=0,
        moment_l_max=1,
        atomic_cross_radial_sketch_channels=2,
        atomic_cross_radial_projection="fixed_shell_mean",
        atomic_cross_radial_projection_matrix=None,
        use_short_range_repulsion=False,
        short_range_repulsion_potential="softplus_overlap",
        short_range_repulsion_strength=0.0,
        short_range_repulsion_beta=10.0,
        short_range_repulsion_radius_scale=0.75,
        learnable_radial_mixing=True,
        descriptor_conditioner="residual_mlp",
        descriptor_conditioner_hidden_channels=6,
        descriptor_bottleneck_dim=5,
    )

    benchmark_config = benchmark_build_training_config(args)
    lightning_config = lightning_build_training_config(
        variant="l1_conditioned",
        scalar_path_ids="atomic.radial_density,atomic.vector_norm",
        hidden_channels="8",
        num_radial=4,
        moment_l_max=1,
        learnable_radial_mixing=True,
        descriptor_conditioner="residual_mlp",
        descriptor_conditioner_hidden_channels=6,
        descriptor_bottleneck_dim=5,
    )

    assert benchmark_config.descriptor_conditioner == "residual_mlp"
    assert benchmark_config.descriptor_conditioner_hidden_channels == 6
    assert benchmark_config.descriptor_bottleneck_dim == 5
    assert lightning_config.descriptor_conditioner == "residual_mlp"
    assert lightning_config.descriptor_conditioner_hidden_channels == 6
    assert lightning_config.descriptor_bottleneck_dim == 5


def test_train_rtece_scalar_builds_config_with_short_range_repulsive_core():
    from types import SimpleNamespace

    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import build_training_config

    args = SimpleNamespace(
        variant="radial_core",
        scalar_path_ids="atomic.radial_density",
        hidden_channels="8",
        num_radial=3,
        use_short_range_repulsion=True,
        short_range_repulsion_strength=4.0,
        short_range_repulsion_beta=12.0,
        short_range_repulsion_radius_scale=0.8,
    )

    config = build_training_config(args)

    assert config.scalar_path_ids == ("atomic.radial_density",)
    assert config.use_short_range_repulsion is True
    assert config.short_range_repulsion_strength == pytest.approx(4.0)
    assert config.short_range_repulsion_beta == pytest.approx(12.0)
    assert config.short_range_repulsion_radius_scale == pytest.approx(0.8)



def test_parse_force_focus_elements_accepts_symbols_and_atomic_numbers():
    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import parse_force_focus_elements

    assert parse_force_focus_elements(None) == ()
    assert parse_force_focus_elements("") == ()
    assert parse_force_focus_elements("C,N") == (6, 7)
    assert parse_force_focus_elements("6, 7") == (6, 7)

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


def test_loss_for_batch_supports_normalized_force_focus_weighting():
    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import loss_for_batch

    config = build_rtece_config("rtece_pair")
    model = RTECEScalarModel(config).double()
    for param in model.parameters():
        param.data.zero_()
    graph = RTECEGraph(
        z=torch.tensor([6, 29], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]], dtype=torch.float64),
        edge_index=complete_directed_edges(2),
        batch=torch.zeros(2, dtype=torch.long),
    )
    ref_energy = torch.zeros(1, dtype=torch.float64)
    ref_forces = torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]], dtype=torch.float64)

    assert torch.allclose(
        loss_for_batch(model, graph, ref_energy, ref_forces, force_weight=1.0),
        torch.tensor(0.5, dtype=torch.float64),
    )
    assert torch.allclose(
        loss_for_batch(
            model,
            graph,
            ref_energy,
            ref_forces,
            force_weight=1.0,
            force_focus_atomic_numbers=(6,),
            force_focus_weight=3.0,
        ),
        torch.tensor(0.75, dtype=torch.float64),
    )

def test_loss_for_batch_supports_per_config_sobolev_weights():
    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import loss_for_batch

    config = build_rtece_config("rtece_pair")
    model = RTECEScalarModel(config).double()
    for param in model.parameters():
        param.data.zero_()
    graph = RTECEGraph(
        z=torch.tensor([1, 1, 1, 1], dtype=torch.long),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [2.0, 0.0, 0.0], [2.7, 0.0, 0.0]],
            dtype=torch.float64,
        ),
        edge_index=complete_directed_edges(4),
        batch=torch.tensor([0, 0, 1, 1], dtype=torch.long),
    )
    ref_energy = torch.tensor([2.0, 4.0], dtype=torch.float64)
    ref_forces = torch.tensor(
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]],
        dtype=torch.float64,
    )

    weighted = loss_for_batch(
        model,
        graph,
        ref_energy,
        ref_forces,
        energy_weight=1.0,
        force_weight=1.0,
        energy_sample_weights=torch.tensor([1.0, 0.0], dtype=torch.float64),
        force_sample_weights=torch.tensor([1.0, 0.0], dtype=torch.float64),
    )

    assert torch.allclose(weighted, torch.tensor(1.0, dtype=torch.float64))


def test_relative_energy_group_loss_ignores_per_group_energy_gauge():
    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import relative_energy_group_loss

    pred_energy = torch.tensor([11.0, 13.0, 15.0], dtype=torch.float64)
    ref_energy = torch.tensor([1.0, 3.0, 5.0], dtype=torch.float64)
    natoms = torch.tensor([2.0, 2.0, 2.0], dtype=torch.float64)

    loss = relative_energy_group_loss(
        pred_energy,
        ref_energy,
        natoms,
        group_ids=("path-a", "path-a", "path-a"),
        image_indices=torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64),
    )

    assert torch.allclose(loss, torch.tensor(0.0, dtype=torch.float64))


def test_loss_for_batch_can_use_relative_neb_energy_without_absolute_gauge():
    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import loss_for_batch

    config = build_rtece_config_from_path_ids(
        "relative_loss",
        ("atomic.radial_density",),
        cutoff=1.0,
        num_radial=2,
        hidden_channels=(4,),
    )
    model = RTECEScalarModel(config).double()
    for param in model.parameters():
        param.data.zero_()
    graph = RTECEGraph(
        z=torch.tensor([1, 1], dtype=torch.long),
        pos=torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=torch.float64),
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=torch.tensor([0, 1], dtype=torch.long),
    )
    ref_energy = torch.tensor([0.0, 2.0], dtype=torch.float64)
    ref_forces = torch.zeros((2, 3), dtype=torch.float64)

    loss = loss_for_batch(
        model,
        graph,
        ref_energy,
        ref_forces,
        energy_weight=0.0,
        force_weight=0.0,
        relative_energy_weight=3.0,
        relative_group_ids=("path-a", "path-a"),
        relative_image_indices=torch.tensor([0.0, 1.0], dtype=torch.float64),
    )

    assert torch.allclose(loss, torch.tensor(6.0, dtype=torch.float64))


def test_loss_for_batch_normalizes_batched_energy_by_each_config_natoms():
    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import loss_for_batch

    config = build_rtece_config("rtece_pair")
    model = RTECEScalarModel(config).double()
    for param in model.parameters():
        param.data.zero_()
    graph = RTECEGraph(
        z=torch.tensor([1, 1, 1, 1, 1], dtype=torch.long),
        pos=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [0.7, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [2.7, 0.0, 0.0],
                [3.4, 0.0, 0.0],
            ],
            dtype=torch.float64,
        ),
        edge_index=complete_directed_edges(5),
        batch=torch.tensor([0, 0, 1, 1, 1], dtype=torch.long),
    )
    ref_energy = torch.tensor([2.0, 6.0], dtype=torch.float64)
    ref_forces = torch.zeros((5, 3), dtype=torch.float64)

    loss = loss_for_batch(model, graph, ref_energy, ref_forces, energy_weight=1.0, force_weight=0.0)

    assert torch.allclose(loss, torch.tensor(2.5, dtype=torch.float64))


def test_lightning_load_samples_reads_extxyz_property_weights(tmp_path):
    import ase.io
    from ase import Atoms
    from tace.lightning.rtece import load_samples

    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    atoms.info["energy"] = -0.5
    atoms.info["energy_weight"] = 2.5
    atoms.info["forces_weight"] = 3.5
    atoms.arrays["forces"] = torch.zeros((2, 3), dtype=torch.float64).numpy()
    path = tmp_path / "weighted.extxyz"
    ase.io.write(path, [atoms])

    sample = load_samples(
        path,
        cutoff=5.0,
        dtype=torch.float64,
        include_sample_weights=True,
    )[0]

    assert len(sample) == 5
    assert torch.allclose(sample[3], torch.tensor([2.5], dtype=torch.float64))
    assert torch.allclose(sample[4], torch.tensor([3.5], dtype=torch.float64))


def test_lightning_load_samples_reads_relative_neb_metadata(tmp_path):
    import ase.io
    from ase import Atoms
    from tace.lightning.rtece import load_samples

    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    atoms.info["energy"] = -0.5
    atoms.info["energy_weight"] = 2.5
    atoms.info["forces_weight"] = 3.5
    atoms.info["case_id"] = "neb-a"
    atoms.info["source_frame"] = 7
    atoms.arrays["forces"] = torch.zeros((2, 3), dtype=torch.float64).numpy()
    path = tmp_path / "relative.extxyz"
    ase.io.write(path, [atoms])

    sample = load_samples(
        path,
        cutoff=5.0,
        dtype=torch.float64,
        include_sample_weights=True,
        include_relative_metadata=True,
    )[0]

    assert len(sample) == 7
    assert sample[5] == "neb-a"
    assert torch.allclose(sample[6], torch.tensor([7.0], dtype=torch.float64))


def test_lightning_shared_step_accepts_relative_neb_metadata():
    from tace.lightning.rtece import RTECELightningModule, _collate_rtece_samples

    config = build_rtece_config_from_path_ids(
        "relative_lightning",
        ("atomic.radial_density",),
        cutoff=1.0,
        num_radial=2,
        hidden_channels=(4,),
    )
    model = RTECEScalarModel(config).double()
    for param in model.parameters():
        param.data.zero_()
    samples = []
    for image, energy in enumerate([0.0, 2.0]):
        graph = RTECEGraph(
            z=torch.tensor([1], dtype=torch.long),
            pos=torch.tensor([[float(image) * 2.0, 0.0, 0.0]], dtype=torch.float64),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            batch=torch.zeros(1, dtype=torch.long),
        )
        samples.append(
            (
                graph,
                torch.tensor([energy], dtype=torch.float64),
                torch.zeros((1, 3), dtype=torch.float64),
                torch.tensor([1.0], dtype=torch.float64),
                torch.tensor([1.0], dtype=torch.float64),
                "path-a",
                torch.tensor([float(image)], dtype=torch.float64),
            )
        )
    batch = _collate_rtece_samples(samples)
    lit = RTECELightningModule(
        model,
        config,
        energy_weight=0.0,
        force_weight=0.0,
        relative_energy_weight=3.0,
    ).double()

    loss = lit._shared_step(batch, "train")

    assert torch.allclose(loss, torch.tensor(6.0, dtype=torch.float64))


def test_apply_extxyz_sample_weights_supports_source_energy_and_force_multipliers(tmp_path):
    import ase.io
    from ase import Atoms
    import numpy as np

    from benchmarks.oc20neb_tace_mace.apply_extxyz_sample_weights import apply_extxyz_sample_weights

    base = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    base.info["energy"] = -1.0
    base.info["rtece_concat_source"] = "base_mixed_train_tw0p75"
    base.arrays["forces"] = np.zeros((1, 3))
    teacher = Atoms("H", positions=[[0.1, 0.0, 0.0]])
    teacher.info["energy"] = -0.8
    teacher.info["rtece_concat_source"] = "teacher_relax_trajectory320"
    teacher.arrays["forces"] = np.ones((1, 3))
    source = tmp_path / "input.extxyz"
    out = tmp_path / "weighted.extxyz"
    ase.io.write(source, [base, teacher], format="extxyz")

    summary = apply_extxyz_sample_weights(
        input_path=source,
        output_path=out,
        source_energy_multipliers={"base_mixed_train_tw0p75": 2.0, "teacher_relax_trajectory320": 0.5},
        source_force_multipliers={"teacher_relax_trajectory320": 3.0},
        normalize_energy_mean=True,
        normalize_force_mean=True,
    )
    frames = ase.io.read(out, index=":")

    assert summary["schema_version"] == "rtece_extxyz_sample_weights.v1"
    assert summary["source_energy_multipliers"] == {
        "base_mixed_train_tw0p75": 2.0,
        "teacher_relax_trajectory320": 0.5,
    }
    assert summary["energy_weight_mean"] == pytest.approx(1.0)
    assert summary["force_weight_mean"] == pytest.approx(1.0)
    assert [frame.info["energy_weight"] for frame in frames] == pytest.approx([1.6, 0.4])
    assert [frame.info["forces_weight"] for frame in frames] == pytest.approx([0.5, 1.5])
    assert [frame.get_potential_energy() for frame in frames] == pytest.approx([-1.0, -0.8])
    assert np.allclose(frames[1].get_forces(), np.ones((1, 3)))


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
    assert summary["learnable_radial_mixing"] is False
    assert summary["short_range_repulsion_potential"] == "softplus_overlap"


def test_train_rtece_scalar_cli_trains_scalar_path_id_route(tmp_path):
    from ase import Atoms
    import ase.io

    train_file = tmp_path / "train.extxyz"
    out_dir = tmp_path / "rtece_path"
    h2 = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    h2.info["energy"] = -1.0
    h2.arrays["forces"] = torch.zeros((2, 3), dtype=torch.float64).numpy()
    ase.io.write(str(train_file), [h2], format="extxyz")

    subprocess.run(
        [
            sys.executable,
            "benchmarks/oc20neb_tace_mace/train_rtece_scalar.py",
            "--variant",
            "radial",
            "--scalar-path-ids",
            "atomic.radial_density",
            "--train-file",
            str(train_file),
            "--valid-file",
            str(train_file),
            "--output-dir",
            str(out_dir),
            "--limit-configs",
            "1",
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

    summary = json.loads((out_dir / "train_summary.json").read_text())
    _model, config = __import__(
        "benchmarks.oc20neb_tace_mace.train_rtece_scalar",
        fromlist=["load_checkpoint"],
    ).load_checkpoint(out_dir / "rtece_scalar.pt", dtype=torch.float64)

    assert summary["variant"] == "radial"
    assert summary["scalar_path_ids"] == ["atomic.radial_density"]
    assert config.scalar_path_ids == ("atomic.radial_density",)
    assert config.variant == "radial"


def test_benchmark_rtece_scalar_writes_architecture_manifest_for_path_id_checkpoint(tmp_path):
    from ase import Atoms
    import ase.io
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import build_rtece_config_from_path_ids
    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import save_checkpoint

    config = build_rtece_config_from_path_ids(
        "radial",
        ("atomic.radial_density",),
        num_radial=2,
        hidden_channels=(4,),
    )
    model = RTECEScalarModel(config).double()
    checkpoint = tmp_path / "rtece_scalar.pt"
    configs = tmp_path / "valid.extxyz"
    output = tmp_path / "benchmark.json"
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    atoms.info["energy"] = -1.0
    atoms.arrays["forces"] = torch.zeros((2, 3), dtype=torch.float64).numpy()
    ase.io.write(str(configs), [atoms], format="extxyz")
    save_checkpoint(checkpoint, model, config)

    subprocess.run(
        [
            sys.executable,
            "benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py",
            "--model",
            str(checkpoint),
            "--configs",
            str(configs),
            "--output",
            str(output),
            "--variant",
            "radial",
            "--device",
            "cpu",
            "--default-dtype",
            "float64",
            "--limit-configs",
            "1",
            "--measure-passes",
            "1",
            "--force-mode",
            "autograd",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output.read_text())
    manifest = rtece_path_manifest(config)

    assert payload["tece_architecture_path_manifest_hash"] == manifest["manifest_hash"]
    assert payload["tece_architecture_path_manifest"]["config"]["scalar_path_ids"] == ["atomic.radial_density"]
    assert payload["tece_path_manifest_hash"] is not None


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


def test_train_steps_respects_min_eval_step_for_best_checkpoint(tmp_path):
    from ase import Atoms
    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import atoms_to_graph, train_steps

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
        max_steps=3,
        lr=1e-3,
        valid_samples=[sample],
        eval_interval=1,
        min_eval_step=3,
        best_checkpoint_path=best_path,
        config=config,
    )

    assert best_path.exists()
    assert summary["best_step"] == 3
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



def test_benchmark_build_atom_graph_supports_tace_matscipy_pbc_backend():
    from ase import Atoms
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import build_atom_graph

    atoms = Atoms(
        "H2",
        positions=[[0.1, 0.0, 0.0], [4.9, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )

    graph = build_atom_graph(
        atoms,
        graph_construction_backend="matscipy_neighborlist",
        cutoff=0.5,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    _vectors, distances, _unit = compute_pair_geometry(graph)

    assert graph.cell is not None
    assert graph.edge_shifts is not None
    assert torch.allclose(distances, torch.tensor([0.2, 0.2], dtype=torch.float64), atol=1e-12)


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




def test_prediction_error_payload_includes_relative_energy_metrics_when_groups_are_given():
    import numpy as np
    from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import prediction_error_payload

    ref_e = np.array([0.0, 1.0, 3.0])
    pred_e = np.array([5.0, 6.2, 8.0])
    ref_f = np.zeros((3, 3))
    pred_f = np.zeros((3, 3))
    natoms = np.array([2.0, 2.0, 2.0])
    outputs = [{"energy": pred_e.copy(), "forces": pred_f.copy()}]

    payload = prediction_error_payload(
        outputs,
        ref_e,
        ref_f,
        natoms,
        group_ids=["path-a", "path-a", "path-a"],
        image_indices=[0, 1, 2],
    )

    assert payload["relative_energy_errors_available"] is True
    assert payload["relative_image_rmse_mev_atom"] == pytest.approx((10000.0 / 3) ** 0.5)
    assert payload["barrier_rmse_mev_atom"] == pytest.approx(0.0)
    assert payload["energy_decomposition_metric_schema_version"] == "rtece_energy_error_decomposition.v1"
    assert payload["first_image_anchor_rmse_mev_atom"] == pytest.approx((10000.0 / 3) ** 0.5)

def test_benchmark_error_summary_reports_signed_energy_bias_and_absolute_max_errors():
    import numpy as np
    from benchmarks.oc20neb_tace_mace.benchmark_models import summarize_errors

    pred_e = np.array([2.0, 0.0])
    ref_e = np.array([1.0, 2.0])
    natoms = np.array([2.0, 1.0])
    pred_f = np.array([[0.0, -3.0, 1.0], [4.0, 0.0, -1.0]])
    ref_f = np.zeros((2, 3))

    summary = summarize_errors(pred_e, pred_f, ref_e, ref_f, natoms)

    assert summary["bias_e_mev_atom"] == pytest.approx(-750.0)
    assert summary["mean_signed_e_mev_atom"] == pytest.approx(-750.0)
    assert summary["max_abs_e_mev_atom"] == pytest.approx(2000.0)
    assert summary["max_abs_f_mev_a"] == pytest.approx(4000.0)


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
    assert "matscipy_neighborlist" in result.stdout
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
    assert "--moment-l-max" in result.stdout
    assert "--learnable-radial-mixing" in result.stdout
    assert "--seed" in result.stdout




def test_stage_sweep_summary_collects_benchmarks_and_marks_missing_rows(tmp_path):
    from benchmarks.oc20neb_tace_mace.summarize_rtece_stage_sweep import collect_stage_sweep_results

    index = tmp_path / "rtece_pareto_sweep_index.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": "rtece_pareto_sweep.v1",
                "row_set": "unit-stage",
                "rows": [
                    {
                        "name": "fast",
                        "hidden_channels": "64,64",
                        "moment_l_max": 1,
                        "scalar_path_ids": "atomic.radial_density",
                        "tece_axes": ["scalar_head_capacity"],
                    },
                    {
                        "name": "missing",
                        "hidden_channels": "128,128",
                        "moment_l_max": 2,
                        "scalar_path_ids": "atomic.radial_density,atomic.quadrupole_norm",
                        "tece_axes": ["atomic_l2_scalar_paths"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    result_dir = tmp_path / "results" / "fast"
    result_dir.mkdir(parents=True)
    (result_dir / "fast_limit1024_benchmark.json").write_text(
        json.dumps(
            {
                "variant": "fast",
                "configs": 1024,
                "atoms": 59193,
                "num_parameters": 5505,
                "hidden_channels": [64, 64],
                "moment_l_max": 1,
                "atoms_per_second": 4.2e6,
                "rmse_f_mev_a": 90.0,
                "mae_f_mev_a": 35.0,
                "max_abs_f_mev_a": 700.0,
                "rmse_e_mev_atom": 120.0,
                "mae_e_mev_atom": 80.0,
                "bias_e_mev_atom": -3.0,
                "max_abs_e_mev_atom": 260.0,
                "tece_path_manifest": {"manifest_hash": "abc", "scalar_paths": [{"id": "atomic.radial_density"}]},
            }
        ),
        encoding="utf-8",
    )

    payload = collect_stage_sweep_results(index, benchmark_roots=[tmp_path / "results"])

    assert payload["schema_version"] == "rtece_stage_sweep_summary.v1"
    assert payload["row_set"] == "unit-stage"
    assert payload["primary_error_metric"] == "dft_f_rmse_mev_a"
    assert payload["missing_benchmark_rows"] == ["missing"]
    by_name = {row["name"]: row for row in payload["rows"]}
    assert by_name["fast"]["status"] == "benchmark_found"
    assert by_name["fast"]["dft_f_rmse_mev_a"] == 90.0
    assert by_name["fast"]["dft_f_max_abs_mev_a"] == 700.0
    assert by_name["fast"]["dft_e_bias_mev_atom"] == -3.0
    assert by_name["fast"]["atoms_per_second"] == 4.2e6
    assert by_name["missing"]["status"] == "missing_benchmark"
    assert by_name["missing"]["dft_f_rmse_mev_a"] is None
    assert [row["name"] for row in payload["dft_force_rmse_pareto_front"]] == ["fast"]


def test_stage_sweep_summary_preserves_relative_neb_energy_metrics(tmp_path):
    from benchmarks.oc20neb_tace_mace.summarize_rtece_stage_sweep import collect_stage_sweep_results, format_markdown

    index = tmp_path / "rtece_pareto_sweep_index.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": "rtece_pareto_sweep.v1",
                "row_set": "unit-stage",
                "rows": [{"name": "energy_shape", "hidden_channels": "16,16", "moment_l_max": 2}],
            }
        ),
        encoding="utf-8",
    )
    result_dir = tmp_path / "results" / "energy_shape"
    result_dir.mkdir(parents=True)
    (result_dir / "energy_shape_limit128_benchmark.json").write_text(
        json.dumps(
            {
                "variant": "energy_shape",
                "rmse_f_mev_a": 109.0,
                "rmse_e_mev_atom": 296.0,
                "relative_energy_errors_available": True,
                "energy_decomposition_metric_schema_version": "rtece_energy_error_decomposition.v1",
                "relative_image_rmse_mev_atom": 8.4,
                "relative_image_mae_mev_atom": 6.3,
                "relative_image_max_abs_mev_atom": 31.0,
                "barrier_rmse_mev_atom": 18.5,
                "barrier_mae_mev_atom": 15.0,
                "barrier_max_abs_mev_atom": 42.0,
                "group_mean_offset_rmse_mev_atom": 6.6,
                "first_image_anchor_rmse_mev_atom": 17.2,
            }
        ),
        encoding="utf-8",
    )

    payload = collect_stage_sweep_results(index, benchmark_roots=[tmp_path / "results"])
    row = payload["rows"][0]
    markdown = format_markdown(payload)

    assert row["dft_relative_energy_errors_available"] is True
    assert row["dft_energy_decomposition_metric_schema_version"] == "rtece_energy_error_decomposition.v1"
    assert row["dft_relative_image_rmse_mev_atom"] == pytest.approx(8.4)
    assert row["dft_relative_image_mae_mev_atom"] == pytest.approx(6.3)
    assert row["dft_relative_image_max_abs_mev_atom"] == pytest.approx(31.0)
    assert row["dft_barrier_rmse_mev_atom"] == pytest.approx(18.5)
    assert row["dft_barrier_mae_mev_atom"] == pytest.approx(15.0)
    assert row["dft_barrier_max_abs_mev_atom"] == pytest.approx(42.0)
    assert row["dft_group_mean_offset_rmse_mev_atom"] == pytest.approx(6.6)
    assert row["dft_first_image_anchor_rmse_mev_atom"] == pytest.approx(17.2)
    assert "relative image/barrier RMSE" in markdown
    assert "rel image RMSE" in markdown
    assert "case offset RMSE" in markdown
    assert "8.400" in markdown
    assert "18.500" in markdown
    assert "6.600" in markdown
    assert "17.200" in markdown


def test_stage_sweep_summary_preserves_design_metadata_for_pending_rows(tmp_path):
    from benchmarks.oc20neb_tace_mace.summarize_rtece_stage_sweep import collect_stage_sweep_results, format_markdown

    index = tmp_path / "rtece_pareto_sweep_index.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": "rtece_pareto_sweep.v1",
                "row_set": "representation-ladder-stage118",
                "rows": [
                    {
                        "name": "l2_species_cavity_edge_h128",
                        "train_variant": "l2_species_cavity_edge_h128",
                        "hidden_channels": "128,128",
                        "moment_l_max": 2,
                        "scalar_path_ids": "atomic.radial_density,atomic.species_basis_density,edge.cavity.vector_dot",
                        "tece_axes": ["representation_ladder", "low_rank_neighbor_species_basis"],
                        "stage_basis": "stage118_representation_ladder",
                        "num_parameters_estimate": 22593,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = collect_stage_sweep_results(index, benchmark_roots=[tmp_path / "missing-results"])
    row = payload["rows"][0]
    markdown = format_markdown(payload)

    assert row["status"] == "missing_benchmark"
    assert row["train_variant"] == "l2_species_cavity_edge_h128"
    assert row["num_parameters_estimate"] == 22593
    assert row["num_parameters_source"] == "index_estimate"
    assert row["tece_axes"] == ["representation_ladder", "low_rank_neighbor_species_basis"]
    assert "l2_species_cavity_edge_h128" in markdown
    assert "22593" in markdown
    assert "representation_ladder, low_rank_neighbor_species_basis" in markdown


def test_stage_sweep_summary_collects_force_stratification_and_physical_diagnostics(tmp_path):
    from benchmarks.oc20neb_tace_mace.summarize_rtece_stage_sweep import collect_stage_sweep_results

    index = tmp_path / "rtece_pareto_sweep_index.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": "rtece_pareto_sweep.v1",
                "row_set": "unit-stage",
                "rows": [{"name": "l1_cross_k2", "hidden_channels": "16,16", "moment_l_max": 1}],
            }
        ),
        encoding="utf-8",
    )
    benchmark_dir = tmp_path / "benchmarks" / "l1_cross_k2"
    benchmark_dir.mkdir(parents=True)
    (benchmark_dir / "l1_cross_k2_limit1024_benchmark.json").write_text(
        json.dumps(
            {
                "variant": "l1_cross_k2",
                "atoms_per_second": 4.9e6,
                "rmse_f_mev_a": 119.0,
                "mae_f_mev_a": 44.0,
                "max_abs_f_mev_a": 5075.0,
                "rmse_e_mev_atom": 284.0,
                "mae_e_mev_atom": 224.0,
            }
        ),
        encoding="utf-8",
    )
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    (diagnostics / "l1_cross_k2_teacher_valid256_force_stratification.json").write_text(
        json.dumps(
            {
                "schema_version": "rtece_force_error_stratification.v1",
                "selection_focus_label": "C_or_N",
                "selection_score_mev_a": 571.0,
                "focus_groups": [
                    {
                        "label": "C_or_N",
                        "atom_fraction": 0.031,
                        "mae_f_mev_a": 305.7,
                        "rmse_f_mev_a": 494.3,
                        "max_abs_f_mev_a": 5106.5,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (diagnostics / "l1_cross_k2_physical_pareto_placeholder.json").write_text(
        json.dumps(
            {
                "schema_version": "rtece_physical_pareto_summary.v1",
                "rows": [
                    {
                        "variant": "l1_cross_k2",
                        "physical_score": 2.0e6,
                        "physical_gate_pass": False,
                        "benchmark_gate_pass": False,
                        "dimer_gate_pass": True,
                        "rattle_gate_pass": False,
                        "dimer_short_repulsive_fraction": 1.0,
                        "rattle_focus_label": "C_or_N",
                        "focus_rattle_final_rmsd_a": None,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = collect_stage_sweep_results(
        index,
        benchmark_roots=[tmp_path / "benchmarks"],
        diagnostic_roots=[diagnostics],
    )

    row = payload["rows"][0]
    assert row["status"] == "benchmark_found"
    assert row["force_stratification_path"].endswith("l1_cross_k2_teacher_valid256_force_stratification.json")
    assert row["focus_force_label"] == "C_or_N"
    assert row["focus_f_rmse_mev_a"] == pytest.approx(494.3)
    assert row["focus_f_max_abs_mev_a"] == pytest.approx(5106.5)
    assert row["force_selection_score_mev_a"] == pytest.approx(571.0)
    assert row["physical_pareto_path"].endswith("l1_cross_k2_physical_pareto_placeholder.json")
    assert row["physical_score"] == pytest.approx(2.0e6)
    assert row["physical_gate_pass"] is False
    assert row["dimer_gate_pass"] is True
    assert row["rattle_gate_pass"] is False


def test_stage_sweep_summary_cli_writes_json_and_markdown(tmp_path):
    from pathlib import Path

    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": "rtece_pareto_sweep.v1",
                "row_set": "cli-stage",
                "rows": [{"name": "row", "hidden_channels": "64,64", "moment_l_max": 1, "scalar_path_ids": "atomic.radial_density"}],
            }
        ),
        encoding="utf-8",
    )
    result_dir = tmp_path / "bench" / "row"
    result_dir.mkdir(parents=True)
    (result_dir / "row_limit1024_benchmark.json").write_text(
        json.dumps(
            {
                "variant": "row",
                "atoms_per_second": 1.5e6,
                "rmse_f_mev_a": 77.0,
                "mae_f_mev_a": 30.0,
                "max_abs_f_mev_a": 500.0,
                "rmse_e_mev_atom": 100.0,
                "mae_e_mev_atom": 70.0,
                "bias_e_mev_atom": 4.0,
                "max_abs_e_mev_atom": 220.0,
            }
        ),
        encoding="utf-8",
    )

    script = "benchmarks/oc20neb_tace_mace/summarize_rtece_stage_sweep.py"
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--index",
            str(index),
            "--benchmark-root",
            str(tmp_path / "bench"),
            "--output-json",
            str(tmp_path / "summary.json"),
            "--output-md",
            str(tmp_path / "summary.md"),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads((tmp_path / "summary.json").read_text())
    markdown = (tmp_path / "summary.md").read_text()
    assert payload["row_set"] == "cli-stage"
    assert payload["rows"][0]["dft_f_rmse_mev_a"] == 77.0
    assert "| row | benchmark_found | NA | atomic.radial_density | 1.500e+06 | 77.000 | 30.000 | 500.000 | 100.000 | 70.000 | 4.000 | 220.000 |" in markdown
    assert str(tmp_path / "summary.json") in result.stdout


def test_summary_payload_includes_rmse_first_pareto_fronts():
    from benchmarks.oc20neb_tace_mace.summarize_tece_distill import make_summary_payload

    rows = [
        {
            "variant": "fast_mae_bad_rmse",
            "atoms_per_second": 20.0,
            "dft_f_mae_mev_a": 35.0,
            "dft_f_rmse_mev_a": 130.0,
            "teacher_f_mae_mev_a": 34.0,
            "teacher_f_rmse_mev_a": 128.0,
        },
        {
            "variant": "slower_rmse_good",
            "atoms_per_second": 18.0,
            "dft_f_mae_mev_a": 37.0,
            "dft_f_rmse_mev_a": 118.0,
            "teacher_f_mae_mev_a": 36.0,
            "teacher_f_rmse_mev_a": 117.0,
        },
        {
            "variant": "dominated",
            "atoms_per_second": 12.0,
            "dft_f_mae_mev_a": 39.0,
            "dft_f_rmse_mev_a": 150.0,
            "teacher_f_mae_mev_a": 38.0,
            "teacher_f_rmse_mev_a": 149.0,
        },
    ]

    payload = make_summary_payload(rows, baselines=[])

    assert [row["variant"] for row in payload["dft_force_rmse_pareto_front"]] == [
        "fast_mae_bad_rmse",
        "slower_rmse_good",
    ]
    assert [row["variant"] for row in payload["teacher_force_rmse_pareto_front"]] == [
        "fast_mae_bad_rmse",
        "slower_rmse_good",
    ]
    assert payload["primary_error_metric"] == "dft_f_rmse_mev_a"

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
    assert row["tece_path_manifest"]["schema_version"] == "rtece_path_manifest.v1"
    assert row["tece_path_manifest_hash"] == row["tece_path_manifest"]["manifest_hash"]
    assert row["tece_route"]["force_realization"] == "triton_fused_descriptor_force"
    assert row["tece_route"]["edge_state_lifetime"] == "counted_exact_edge_buffer"
    assert "| variant | TECE route | manifest | graph backend | force mode |" in markdown
    assert f"| radial8h24 | T3_element_conditioned_scalar_density | {row['tece_path_manifest_hash']}" in markdown


def test_rtece_summary_reconstructs_species_and_cavity_routes():
    from benchmarks.oc20neb_tace_mace.summarize_tece_distill import make_student_row

    dft = {
        "model": "rtece_scalar.pt",
        "atoms_per_second": 1.0,
        "configs_per_second": 1.0,
        "seconds_per_pass": 1.0,
        "peak_allocated_mb": 1.0,
        "peak_reserved_mb": 1.0,
        "num_parameters": 1,
        "mae_e_mev_atom": 1.0,
        "rmse_e_mev_atom": 1.0,
        "mae_f_mev_a": 1.0,
        "rmse_f_mev_a": 1.0,
    }
    teacher = dict(dft)

    species = make_student_row("rtece_species_basis4", dft_benchmark=dft, teacher_benchmark=teacher)
    cavity = make_student_row("rtece_cavity_edge_sketch8", dft_benchmark=dft, teacher_benchmark=teacher)
    radial_cavity = make_student_row(
        "rtece_cavity_radial_edge_sketch14",
        dft_benchmark=dft,
        teacher_benchmark=teacher,
    )

    assert species["tece_route"]["semantic_tier"] == "T3_low_rank_species_density"
    assert "low_rank_neighbor_species_basis" in species["tece_route"]["retained_tece_groups"]
    assert cavity["tece_route"]["semantic_tier"] == "T3_cavity_edge_scalar_sketch"
    assert "cavity_edge_relational_scalar_sketches" in cavity["tece_route"]["retained_tece_groups"]
    assert radial_cavity["tece_route"]["semantic_tier"] == "T3_cavity_radial_edge_scalar_sketch"
    assert "cross_radial_edge_invariants" in radial_cavity["tece_route"]["retained_tece_groups"]


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
    assert "| variant | TECE route | manifest | graph backend | force mode |" in markdown
    assert f"| radial4h16 | T3_element_conditioned_scalar_density | {row['tece_path_manifest_hash']} | torch_radius_nopbc | analytic_element_triton_descriptor_force |" in markdown


def test_rtece_summary_groups_rows_by_path_manifest():
    from benchmarks.oc20neb_tace_mace.summarize_tece_distill import (
        make_student_row,
        manifest_group_rows,
    )

    base = {
        "model": "rtece_scalar.pt",
        "variant": "rtece_cavity_radial_edge_sketch14",
        "force_mode": "autograd",
        "graph_construction_backend": "matscipy_pbc",
        "hidden_channels": [16, 16],
        "num_radial": 4,
        "atoms_per_second": 100.0,
        "configs_per_second": 2.0,
        "seconds_per_pass": 0.1,
        "peak_allocated_mb": 101.0,
        "peak_reserved_mb": 120.0,
        "num_parameters": 737,
        "mae_e_mev_atom": 80.0,
        "rmse_e_mev_atom": 100.0,
        "mae_f_mev_a": 40.0,
        "rmse_f_mev_a": 80.0,
    }
    faster = dict(base)
    faster["atoms_per_second"] = 160.0
    faster["mae_f_mev_a"] = 45.0
    better = dict(base)
    better["atoms_per_second"] = 90.0
    better["mae_f_mev_a"] = 35.0
    teacher_faster = dict(faster)
    teacher_faster["mae_f_mev_a"] = 43.0
    teacher_better = dict(better)
    teacher_better["mae_f_mev_a"] = 33.0

    rows = [
        make_student_row("radial_cavity_smoke_fast", dft_benchmark=faster, teacher_benchmark=teacher_faster),
        make_student_row("radial_cavity_smoke_better", dft_benchmark=better, teacher_benchmark=teacher_better),
    ]
    groups = manifest_group_rows(rows)

    assert len(groups) == 1
    group = groups[0]
    assert group["manifest_hash"] == rows[0]["tece_path_manifest_hash"]
    assert group["semantic_tier"] == "T3_cavity_radial_edge_scalar_sketch"
    assert group["variants"] == ["radial_cavity_smoke_better", "radial_cavity_smoke_fast"]
    assert group["row_count"] == 2
    assert group["best_atoms_per_second"] == 160.0
    assert group["best_dft_f_mae_mev_a"] == 35.0
    assert group["best_teacher_f_mae_mev_a"] == 33.0
    assert "low_rank_radial_edge_moment_sketches" in group["retained_tece_groups"]
    assert "edge.cavity.vector_cross_radial_dot" in group["scalar_path_ids"]


def test_rtece_summary_markdown_includes_manifest_groups_section():
    from benchmarks.oc20neb_tace_mace.summarize_tece_distill import (
        format_markdown,
        make_student_row,
    )

    dft = {
        "model": "rtece_scalar.pt",
        "variant": "rtece_species_basis4",
        "force_mode": "autograd",
        "hidden_channels": [16, 16],
        "num_radial": 4,
        "atoms_per_second": 220.0,
        "configs_per_second": 3.0,
        "seconds_per_pass": 0.1,
        "peak_allocated_mb": 36.0,
        "peak_reserved_mb": 48.0,
        "num_parameters": 641,
        "mae_e_mev_atom": 60.0,
        "rmse_e_mev_atom": 90.0,
        "mae_f_mev_a": 38.0,
        "rmse_f_mev_a": 70.0,
    }
    teacher = dict(dft)
    teacher["mae_f_mev_a"] = 37.0

    row = make_student_row("species_smoke", dft_benchmark=dft, teacher_benchmark=teacher)
    markdown = format_markdown([row], baselines=[])

    assert "## Manifest Groups" in markdown
    assert "| manifest | TECE route | variants | retained groups | scalar paths | projection residual | deleted projection paths | projection samples | best atoms/s | best DFT F MAE | best teacher F MAE | rows |" in markdown
    assert f"| {row['tece_path_manifest_hash']} | T3_low_rank_species_density | species_smoke |" in markdown


def test_rtece_summary_load_projection_rows_preserves_weight_metadata(tmp_path):
    from benchmarks.oc20neb_tace_mace.summarize_tece_distill import load_projection_diagnostic_rows

    path = tmp_path / "projection.json"
    path.write_text(
        json.dumps({
            "weighted": True,
            "sample_weight_source": "teacher_force_norm_per_atom_mean1",
            "rows": [{"candidate": "radial", "relative_residual": 0.2}],
        }),
        encoding="utf-8",
    )

    rows = load_projection_diagnostic_rows([path])

    assert rows[0]["weighted"] is True
    assert rows[0]["sample_weight_source"] == "teacher_force_norm_per_atom_mean1"
    assert rows[0]["projection_diagnostic"] == str(path)


def test_rtece_summary_manifest_groups_attach_projection_residuals():
    from benchmarks.oc20neb_tace_mace.summarize_tece_distill import (
        make_student_row,
        manifest_group_rows,
    )

    dft = {
        "model": "rtece_scalar.pt",
        "variant": "rtece_species_basis4",
        "force_mode": "autograd",
        "hidden_channels": [16, 16],
        "num_radial": 4,
        "atoms_per_second": 220.0,
        "configs_per_second": 3.0,
        "seconds_per_pass": 0.1,
        "peak_allocated_mb": 36.0,
        "peak_reserved_mb": 48.0,
        "num_parameters": 641,
        "mae_e_mev_atom": 60.0,
        "rmse_e_mev_atom": 90.0,
        "mae_f_mev_a": 38.0,
        "rmse_f_mev_a": 70.0,
    }
    teacher = dict(dft)
    teacher["mae_f_mev_a"] = 37.0
    row = make_student_row("species_smoke", dft_benchmark=dft, teacher_benchmark=teacher)
    projection_rows = [
        {
            "candidate": "species_projection",
            "candidate_manifest_hash": row["tece_path_manifest_hash"],
            "relative_residual": 0.0125,
            "deleted_scalar_path_ids": ["edge.direct.radial"],
            "num_samples": 128,
            "weighted": True,
            "sample_weight_source": "teacher_force_norm_per_atom_mean1",
        }
    ]

    groups = manifest_group_rows([row], projection_rows=projection_rows)

    assert groups[0]["projection_relative_residual"] == 0.0125
    assert groups[0]["projection_deleted_scalar_path_ids"] == ["edge.direct.radial"]
    assert groups[0]["projection_num_samples"] == 128
    assert groups[0]["projection_weighted"] is True
    assert groups[0]["projection_sample_weight_source"] == "teacher_force_norm_per_atom_mean1"


def test_rtece_summary_manifest_groups_keep_zero_projection_residual_as_best():
    from benchmarks.oc20neb_tace_mace.summarize_tece_distill import (
        make_student_row,
        manifest_group_rows,
    )

    dft = {
        "model": "rtece_scalar.pt",
        "variant": "rtece_species_basis4",
        "force_mode": "autograd",
        "hidden_channels": [16, 16],
        "num_radial": 4,
        "atoms_per_second": 220.0,
        "configs_per_second": 3.0,
        "seconds_per_pass": 0.1,
        "peak_allocated_mb": 36.0,
        "peak_reserved_mb": 48.0,
        "num_parameters": 641,
        "mae_e_mev_atom": 60.0,
        "rmse_e_mev_atom": 90.0,
        "mae_f_mev_a": 38.0,
        "rmse_f_mev_a": 70.0,
    }
    teacher = dict(dft)
    teacher["mae_f_mev_a"] = 37.0
    row = make_student_row("species_smoke", dft_benchmark=dft, teacher_benchmark=teacher)
    projection_rows = [
        {
            "candidate_manifest_hash": row["tece_path_manifest_hash"],
            "relative_residual": 0.0,
            "deleted_scalar_path_ids": [],
        },
        {
            "candidate_manifest_hash": row["tece_path_manifest_hash"],
            "relative_residual": 0.2,
            "deleted_scalar_path_ids": ["edge.direct.radial"],
        },
    ]

    groups = manifest_group_rows([row], projection_rows=projection_rows)

    assert groups[0]["projection_relative_residual"] == 0.0
    assert groups[0]["projection_deleted_scalar_path_ids"] == []


def test_rtece_summary_manifest_groups_attach_projection_by_scalar_path_ids_when_hash_differs():
    from benchmarks.oc20neb_tace_mace.summarize_tece_distill import (
        make_student_row,
        manifest_group_rows,
    )
    from benchmarks.oc20neb_tace_mace.rtece_scalar_model import build_rtece_config_from_path_ids

    config = build_rtece_config_from_path_ids(
        "radial",
        ("atomic.radial_density",),
        num_radial=3,
        hidden_channels=(4,),
    )
    manifest = rtece_path_manifest(config)
    dft = {
        "model": "rtece_scalar.pt",
        "variant": "radial",
        "force_mode": "autograd",
        "hidden_channels": [4],
        "num_radial": 3,
        "atoms_per_second": 10.0,
        "mae_e_mev_atom": 1.0,
        "mae_f_mev_a": 2.0,
        "tece_architecture_path_manifest": manifest,
        "tece_architecture_path_manifest_hash": manifest["manifest_hash"],
    }
    teacher = dict(dft)
    projection_rows = [
        {
            "candidate_manifest_hash": "descriptor-hash-from-different-head",
            "candidate_scalar_path_ids": ["atomic.radial_density"],
            "relative_residual": 0.125,
            "deleted_scalar_path_ids": ["edge.cavity.vector_dot"],
        }
    ]
    row = make_student_row("radial", dft_benchmark=dft, teacher_benchmark=teacher)

    groups = manifest_group_rows([row], projection_rows=projection_rows)

    assert groups[0]["projection_relative_residual"] == 0.125
    assert groups[0]["projection_deleted_scalar_path_ids"] == ["edge.cavity.vector_dot"]


def test_rtece_summary_markdown_includes_projection_residuals_in_manifest_groups():
    from benchmarks.oc20neb_tace_mace.summarize_tece_distill import (
        format_markdown,
        make_student_row,
    )

    dft = {
        "model": "rtece_scalar.pt",
        "variant": "rtece_species_basis4",
        "force_mode": "autograd",
        "hidden_channels": [16, 16],
        "num_radial": 4,
        "atoms_per_second": 220.0,
        "configs_per_second": 3.0,
        "seconds_per_pass": 0.1,
        "peak_allocated_mb": 36.0,
        "peak_reserved_mb": 48.0,
        "num_parameters": 641,
        "mae_e_mev_atom": 60.0,
        "rmse_e_mev_atom": 90.0,
        "mae_f_mev_a": 38.0,
        "rmse_f_mev_a": 70.0,
    }
    teacher = dict(dft)
    teacher["mae_f_mev_a"] = 37.0
    row = make_student_row("species_smoke", dft_benchmark=dft, teacher_benchmark=teacher)
    projection_rows = [
        {
            "candidate": "species_projection",
            "candidate_manifest_hash": row["tece_path_manifest_hash"],
            "relative_residual": 0.0125,
            "deleted_scalar_path_ids": ["edge.direct.radial"],
            "num_samples": 128,
        }
    ]

    markdown = format_markdown([row], baselines=[], projection_rows=projection_rows)

    assert "projection residual" in markdown
    assert "deleted projection paths" in markdown
    assert "0.013" in markdown
    assert "edge.direct.radial" in markdown


def test_rtece_matrix_sbatch_separates_training_and_benchmark_validation_files():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    script = (root / "benchmarks/oc20neb_tace_mace/rtece_scalar_matrix.sbatch").read_text()

    assert "TRAIN_VALID_FILE=${TRAIN_VALID_FILE:-${DFT_VALID_FILE}}" in script
    assert '--valid-file "${TRAIN_VALID_FILE}"' in script
    assert '--configs "${DFT_VALID_FILE}"' in script


def test_rtece_matrix_sbatch_forwards_species_basis_mode():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    script = (root / "benchmarks/oc20neb_tace_mace/rtece_scalar_matrix.sbatch").read_text()

    assert "SPECIES_BASIS_MODE=${SPECIES_BASIS_MODE:-fixed_z_power}" in script
    assert 'echo "species_basis_mode=${SPECIES_BASIS_MODE}"' in script
    assert '--species-basis-mode "${SPECIES_BASIS_MODE}"' in script


def test_rtece_matrix_sbatch_forwards_radial_species_adapter_channels():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    script = (root / "benchmarks/oc20neb_tace_mace/rtece_scalar_matrix.sbatch").read_text()

    assert "RADIAL_SPECIES_ADAPTER_CHANNELS=${RADIAL_SPECIES_ADAPTER_CHANNELS:-0}" in script
    assert "radial_species_adapter_channels=${RADIAL_SPECIES_ADAPTER_CHANNELS}" in script
    assert "--radial-species-adapter-channels" in script


def test_rtece_matrix_sbatch_forwards_descriptor_bottleneck_dim():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    script = (root / "benchmarks/oc20neb_tace_mace/rtece_scalar_matrix.sbatch").read_text()

    assert "DESCRIPTOR_BOTTLENECK_DIM=${DESCRIPTOR_BOTTLENECK_DIM:-0}" in script
    assert "descriptor_bottleneck_dim=${DESCRIPTOR_BOTTLENECK_DIM}" in script
    assert "--descriptor-bottleneck-dim" in script
    assert "${DESCRIPTOR_BOTTLENECK_DIM}" in script


def test_rtece_matrix_sbatch_forwards_atomic_cross_radial_projection_file_without_sbatch_export():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    script = (root / "benchmarks/oc20neb_tace_mace/rtece_scalar_matrix.sbatch").read_text()

    assert "ATOMIC_CROSS_RADIAL_PROJECTION_FILE=${ATOMIC_CROSS_RADIAL_PROJECTION_FILE:-}" in script
    assert "atomic_cross_radial_projection_file_args=()" in script
    assert '--atomic-cross-radial-projection-file "${ATOMIC_CROSS_RADIAL_PROJECTION_FILE}"' in script
    assert "--export" not in script
    assert "--mem" not in script
    assert "--cpus-per-task" not in script


def test_rtece_matrix_sbatch_forwards_scalar_path_ids_without_sbatch_export():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    script = (root / "benchmarks/oc20neb_tace_mace/rtece_scalar_matrix.sbatch").read_text()

    assert "SCALAR_PATH_IDS=${SCALAR_PATH_IDS:-}" in script
    assert "ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS=${ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS:-2}" in script
    assert "ATOMIC_CROSS_RADIAL_PROJECTION=${ATOMIC_CROSS_RADIAL_PROJECTION:-fixed_shell_mean}" in script
    assert "scalar_path_args=()" in script
    assert "atomic_cross_radial_args=()" in script
    assert "--scalar-path-ids" in script
    assert "--atomic-cross-radial-sketch-channels" in script
    assert "--atomic-cross-radial-projection" in script
    assert "--export" not in script


def test_rtece_matrix_sbatch_forwards_short_range_repulsive_core_without_sbatch_export():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    script = (root / "benchmarks/oc20neb_tace_mace/rtece_scalar_matrix.sbatch").read_text()

    assert "USE_SHORT_RANGE_REPULSION=${USE_SHORT_RANGE_REPULSION:-0}" in script
    assert "SHORT_RANGE_REPULSION_POTENTIAL=${SHORT_RANGE_REPULSION_POTENTIAL:-softplus_overlap}" in script
    assert "LEARNABLE_RADIAL_MIXING=${LEARNABLE_RADIAL_MIXING:-0}" in script
    assert "learnable_radial_args=()" in script
    assert "short_range_args=()" in script
    assert "--learnable-radial-mixing" in script
    assert "--use-short-range-repulsion" in script
    assert '--short-range-repulsion-potential "${SHORT_RANGE_REPULSION_POTENTIAL}"' in script
    assert '--short-range-repulsion-strength "${SHORT_RANGE_REPULSION_STRENGTH}"' in script
    assert '--short-range-repulsion-beta "${SHORT_RANGE_REPULSION_BETA}"' in script
    assert '--short-range-repulsion-radius-scale "${SHORT_RANGE_REPULSION_RADIUS_SCALE}"' in script
    assert "--export" not in script


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
        configs="/tmp/valid.extxyz",
        force_mode="analytic_element_direct_padded_descriptor_force",
        limit_configs_list="64",
        measure_passes=1,
        out_dir="/tmp/rtece-bench",
    )
    command = build_sbatch_command(wrapper)
    text = wrapper.read_text()

    assert "--export" not in command
    assert "--export" not in text
    assert "#SBATCH --gpus-per-node=1" in text
    assert "#SBATCH --qos=flood-1o2gpu" in text
    assert "--mem" not in text
    assert "--cpus-per-task" not in text
    assert "MODEL=/tmp/rtece.pt" in text
    assert "CONFIGS=/tmp/valid.extxyz" in text
    assert "FORCE_MODE=analytic_element_direct_padded_descriptor_force" in text
    assert "exec /bin/bash" in text
    assert "rtece_scalar_benchmark.sbatch" in text


def test_atomic_cross_radial_pod_projection_returns_orthonormal_rows():
    from benchmarks.oc20neb_tace_mace.make_atomic_cross_radial_pod_projection import (
        compute_atomic_cross_radial_pod_projection,
    )

    config = RTECEScalarConfig(variant="pod", cutoff=4.0, num_radial=4)
    graph = RTECEGraph(
        z=torch.tensor([6, 1, 8], dtype=torch.long),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.1, 1.1, 0.2]],
            dtype=torch.float64,
        ),
        edge_index=torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long),
        batch=torch.zeros(3, dtype=torch.long),
    )

    matrix = compute_atomic_cross_radial_pod_projection([graph], config, num_sketches=2, moment="vector")
    gram = matrix @ matrix.T

    assert matrix.shape == (2, 4)
    assert torch.allclose(gram, torch.eye(2, dtype=torch.float64), atol=1e-10)


def test_atomic_cross_radial_pod_covariance_accepts_atom_weights():
    from benchmarks.oc20neb_tace_mace.make_atomic_cross_radial_pod_projection import (
        _accumulate_radial_covariance,
    )

    moment = torch.tensor(
        [
            [[2.0], [0.0]],
            [[0.0], [5.0]],
        ],
        dtype=torch.float64,
    )
    covariance = torch.zeros((2, 2), dtype=torch.float64)

    unweighted = _accumulate_radial_covariance(covariance, moment)
    weighted = _accumulate_radial_covariance(
        covariance,
        moment,
        atom_weights=torch.tensor([10.0, 1.0], dtype=torch.float64),
    )

    assert torch.allclose(unweighted, torch.tensor([[4.0, 0.0], [0.0, 25.0]], dtype=torch.float64))
    assert torch.allclose(weighted, torch.tensor([[40.0, 0.0], [0.0, 25.0]], dtype=torch.float64))


def test_atomic_cross_radial_pod_projection_validates_graph_atom_weights():
    from benchmarks.oc20neb_tace_mace.make_atomic_cross_radial_pod_projection import (
        compute_atomic_cross_radial_pod_projection,
    )

    config = RTECEScalarConfig(variant="pod", cutoff=4.0, num_radial=4)
    graph = RTECEGraph(
        z=torch.tensor([6, 1, 8], dtype=torch.long),
        pos=torch.tensor(
            [[0.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.1, 1.1, 0.2]],
            dtype=torch.float64,
        ),
        edge_index=torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]], dtype=torch.long),
        batch=torch.zeros(3, dtype=torch.long),
    )

    with pytest.raises(ValueError, match="atom_weights length"):
        compute_atomic_cross_radial_pod_projection(
            [graph],
            config,
            num_sketches=2,
            moment="vector",
            atom_weights=[torch.ones(2, dtype=torch.float64)],
        )


def test_atomic_cross_radial_pod_force_magnitude_weights_are_mean_normalized():
    from ase import Atoms
    from benchmarks.oc20neb_tace_mace.make_atomic_cross_radial_pod_projection import (
        force_magnitude_atom_weights,
    )

    first = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]])
    first.arrays["teacher_forces"] = torch.tensor(
        [[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]],
        dtype=torch.float64,
    ).numpy()
    second = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    second.arrays["teacher_forces"] = torch.tensor([[0.0, 12.0, 0.0]], dtype=torch.float64).numpy()

    weights, source = force_magnitude_atom_weights([first, second], force_key="teacher_forces")

    assert source == "force_magnitude:teacher_forces:mean1"
    assert [item.shape for item in weights] == [(2,), (1,)]
    assert torch.cat(weights).mean().item() == pytest.approx(1.0)
    assert torch.cat(weights).tolist() == pytest.approx([15.0 / 17.0, 0.0, 36.0 / 17.0])


def test_rtece_pareto_sweep_default_rows_cover_documented_design_axes():
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import default_pareto_rows

    rows = default_pareto_rows()
    names = [row["name"] for row in rows]

    assert names == [
        "l0_scalar_fixed",
        "l0_scalar_learnable_zbl",
        "l1_vector_learnable_zbl",
        "l2_atomic_learnable_zbl",
        "l1_vector_cross_radial_learnable_zbl",
        "l2_atomic_cross_radial_learnable_zbl",
        "l1_vector_cross_radial_k3_learnable_zbl",
        "l2_atomic_cross_radial_k3_learnable_zbl",
        "l1_cavity_vector_learnable_zbl",
        "l2_cavity_vector_quad_learnable_zbl",
    ]
    assert {row["moment_l_max"] for row in rows} == {0, 1, 2}
    assert any(row["learnable_radial_mixing"] for row in rows)
    assert any(row["short_range_repulsion_potential"] == "zbl" for row in rows)
    assert any("edge.cavity.vector_dot" in row["scalar_path_ids"] for row in rows)
    assert any("atomic.vector_cross_radial_dot" in row["scalar_path_ids"] for row in rows)
    assert any(row["atomic_cross_radial_sketch_channels"] == 3 for row in rows)
    assert any("cross_radial_invariants" in row["tece_axes"] for row in rows)
    assert all("tece_axes" in row for row in rows)
    assert "angular_bandwidth_l_max" in rows[0]["tece_axes"]


def test_rtece_stage115_ef_active_rows_follow_stage114_architecture_decision():
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import stage115_ef_active_rows

    rows = stage115_ef_active_rows()
    by_name = {row["name"]: row for row in rows}

    assert list(by_name) == ["l0_radial", "l1_cross_k2", "l2_atomic_no_edge_k2"]
    assert by_name["l0_radial"]["scalar_path_ids"] == "atomic.radial_density"
    assert by_name["l0_radial"]["moment_l_max"] == 0
    assert by_name["l1_cross_k2"]["scalar_path_ids"] == (
        "atomic.radial_density,atomic.vector_norm,atomic.vector_cross_radial_dot"
    )
    assert by_name["l1_cross_k2"]["moment_l_max"] == 1
    assert by_name["l2_atomic_no_edge_k2"]["scalar_path_ids"] == (
        "atomic.radial_density,atomic.vector_norm,atomic.vector_cross_radial_dot,"
        "atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius"
    )
    assert by_name["l2_atomic_no_edge_k2"]["moment_l_max"] == 2
    assert not any("edge." in row["scalar_path_ids"] for row in rows)
    assert all(row["stage_basis"] == "stage114_ef_active_rank" for row in rows)
    assert all("stage114_ef_active_selection" in row["tece_axes"] for row in rows)


def test_rtece_stage116_capacity_ladder_keeps_paths_fixed_and_scales_head():
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import stage116_capacity_ladder_rows

    rows = stage116_capacity_ladder_rows()
    by_name = {row["name"]: row for row in rows}

    assert list(by_name) == [
        "l1_cross_k2_h64",
        "l1_cross_k2_h128",
        "l1_cross_k2_h128x3",
        "l2_atomic_no_edge_k2_h64",
        "l2_atomic_no_edge_k2_h128",
        "l2_atomic_no_edge_k2_h128x3",
    ]
    assert {row["hidden_channels"] for row in rows} == {"64,64", "128,128", "128,128,128"}
    assert by_name["l1_cross_k2_h64"]["scalar_path_ids"] == (
        "atomic.radial_density,atomic.vector_norm,atomic.vector_cross_radial_dot"
    )
    assert by_name["l1_cross_k2_h128x3"]["scalar_path_ids"] == by_name["l1_cross_k2_h64"]["scalar_path_ids"]
    assert by_name["l2_atomic_no_edge_k2_h64"]["scalar_path_ids"] == (
        "atomic.radial_density,atomic.vector_norm,atomic.vector_cross_radial_dot,"
        "atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius"
    )
    assert by_name["l2_atomic_no_edge_k2_h128x3"]["scalar_path_ids"] == by_name["l2_atomic_no_edge_k2_h64"]["scalar_path_ids"]
    assert {row["moment_l_max"] for row in rows if row["name"].startswith("l1_")} == {1}
    assert {row["moment_l_max"] for row in rows if row["name"].startswith("l2_")} == {2}
    assert all(row["stage_basis"] == "stage116_capacity_ladder" for row in rows)
    assert all("scalar_head_capacity_ladder" in row["tece_axes"] for row in rows)
    assert all(row["short_range_repulsion_potential"] == "zbl" for row in rows)
    assert all(row["learnable_radial_mixing"] for row in rows)


def test_rtece_stage118_representation_ladder_rows_follow_tece_review_axes():
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import stage118_representation_ladder_rows

    rows = stage118_representation_ladder_rows()
    by_name = {row["name"]: row for row in rows}

    assert list(by_name) == [
        "l2_atomic_cross_h128",
        "l2_cavity_edge_h128",
        "l2_cavity_radial_edge_h128",
        "l2_species_cavity_edge_h128",
        "l2_conditioned_cavity_edge_h128",
        "l2_cavity_edge_h128x3",
    ]
    assert {row["hidden_channels"] for row in rows} == {"128,128", "128,128,128"}
    assert all(row["stage_basis"] == "stage118_representation_ladder" for row in rows)
    assert all(row["short_range_repulsion_potential"] == "zbl" for row in rows)
    assert all(row["learnable_radial_mixing"] for row in rows)
    assert all("representation_ladder" in row["tece_axes"] for row in rows)
    assert "cross_radial_invariants" in by_name["l2_atomic_cross_h128"]["tece_axes"]
    assert "cavity_edge_relational_scalar_sketches" in by_name["l2_cavity_edge_h128"]["tece_axes"]
    assert "low_rank_radial_edge_moment_sketches" in by_name["l2_cavity_radial_edge_h128"]["tece_axes"]
    assert by_name["l2_cavity_radial_edge_h128"]["train_variant"] == "rtece_cavity_radial_edge_sketch14"
    assert by_name["l2_cavity_radial_edge_h128"]["radial_edge_sketch_channels"] == 2
    assert "atomic.species_basis_density" in by_name["l2_species_cavity_edge_h128"]["scalar_path_ids"]
    assert by_name["l2_species_cavity_edge_h128"]["species_basis_channels"] == 4
    assert by_name["l2_conditioned_cavity_edge_h128"]["descriptor_conditioner"] == "residual_mlp"
    assert by_name["l2_conditioned_cavity_edge_h128"]["descriptor_conditioner_hidden_channels"] == 64


def test_rtece_stage119_frontloaded_representation_rows_keep_head_fixed_and_grow_front_features():
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import stage119_frontloaded_representation_rows

    rows = stage119_frontloaded_representation_rows()
    by_name = {row["name"]: row for row in rows}

    assert list(by_name) == [
        "l0_species8_learnembed_h64",
        "l1_species16_cross_learnembed_h64",
        "l2_species16_atomic_cross_learnembed_h64",
        "l2_species16_cavity_edge_learnembed_h64",
        "l2_species32_cavity_edge_learnembed_h64",
        "l2_species32_cavity_atomic_cross_learnembed_h64",
    ]
    assert {row["hidden_channels"] for row in rows} == {"64,64"}
    assert all(row["stage_basis"] == "stage119_frontloaded_representation_ladder" for row in rows)
    assert all(row["species_basis_mode"] == "learnable_embedding" for row in rows)
    assert all("frontloaded_representation_capacity" in row["tece_axes"] for row in rows)
    assert all("trainable_species_basis" in row["tece_axes"] for row in rows)
    assert "trainable_cross_radial_projection" in by_name["l2_species16_atomic_cross_learnembed_h64"]["tece_axes"]
    assert "cavity_edge_relational_scalar_sketches" in by_name["l2_species16_cavity_edge_learnembed_h64"]["tece_axes"]
    assert "direct_edge_radial_path" in by_name["l2_species32_cavity_atomic_cross_learnembed_h64"]["tece_axes"]
    assert by_name["l2_species32_cavity_atomic_cross_learnembed_h64"]["species_basis_channels"] == 32
    assert by_name["l2_species32_cavity_atomic_cross_learnembed_h64"]["num_parameters_estimate"] > by_name["l0_species8_learnembed_h64"]["num_parameters_estimate"]
    assert by_name["l2_species32_cavity_atomic_cross_learnembed_h64"]["representation_parameters_estimate"] > by_name["l0_species8_learnembed_h64"]["representation_parameters_estimate"]
    assert all(row["capacity_allocation"] == "frontloaded_representation_not_readout" for row in rows)


def test_rtece_stage120_descriptor_bottleneck_rows_keep_head_fixed_and_mix_front_paths():
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import stage120_descriptor_bottleneck_rows

    rows = stage120_descriptor_bottleneck_rows()
    by_name = {row["name"]: row for row in rows}

    assert list(by_name) == [
        "l0_species8_bneck16_h64",
        "l0_species8_bneck32_h64",
        "l2_species32_cavity_atomic_bneck16_h64",
        "l2_species32_cavity_atomic_bneck32_h64",
    ]
    assert {row["hidden_channels"] for row in rows} == {"64,64"}
    assert {row["descriptor_bottleneck_dim"] for row in rows} == {16, 32}
    assert all(row["stage_basis"] == "stage120_descriptor_bottleneck_ladder" for row in rows)
    assert all("descriptor_bottleneck" in row["tece_axes"] for row in rows)
    assert all("front_low_rank_path_mixer" in row["tece_axes"] for row in rows)
    assert by_name["l0_species8_bneck16_h64"]["readout_parameters_estimate"] < by_name["l0_species8_bneck32_h64"]["readout_parameters_estimate"]
    assert by_name["l2_species32_cavity_atomic_bneck16_h64"]["representation_parameters_estimate"] > by_name["l0_species8_bneck16_h64"]["representation_parameters_estimate"]


def test_rtece_stage122_radial_species_adapter_rows_keep_head_fixed_and_adapt_front_radial_basis():
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import stage122_radial_species_adapter_rows

    rows = stage122_radial_species_adapter_rows()
    by_name = {row["name"]: row for row in rows}

    assert list(by_name) == [
        "l0_pair_radial_species8_h64",
        "l0_species8_radial_species8_h64",
        "l1_active_radial_species8_h64",
        "l1_active_radial_species16_h64",
    ]
    assert {row["hidden_channels"] for row in rows} == {"64,64"}
    assert all(row["descriptor_bottleneck_dim"] == 0 for row in rows)
    assert all(row["stage_basis"] == "stage122_radial_species_adapter" for row in rows)
    assert all(row["radial_species_adapter_channels"] in {8, 16} for row in rows)
    assert all("trainable_edge_species_radial_basis" in row["tece_axes"] for row in rows)
    assert all(row["capacity_allocation"] == "front_edge_species_radial_basis_not_wider_head" for row in rows)
    assert by_name["l1_active_radial_species16_h64"]["num_parameters_estimate"] > by_name["l1_active_radial_species8_h64"]["num_parameters_estimate"]
    assert by_name["l1_active_radial_species8_h64"]["representation_parameters_estimate"] > by_name["l0_pair_radial_species8_h64"]["representation_parameters_estimate"]


def test_rtece_stage123_path_scoped_adapter_rows_select_front_capacity_by_path_group():
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import stage123_path_scoped_adapter_rows

    rows = stage123_path_scoped_adapter_rows()
    by_name = {row["name"]: row for row in rows}

    assert list(by_name) == [
        "l1_active_atomic_radial_species8_h64",
        "l1_active_all_radial_species8_h64",
        "l2_cavity_edge_radial_species8_h64",
    ]
    assert {row["hidden_channels"] for row in rows} == {"64,64"}
    assert all(row["descriptor_bottleneck_dim"] == 0 for row in rows)
    assert all(row["stage_basis"] == "stage123_residual_projected_path_scoped_adapter" for row in rows)
    assert all(row["capacity_allocation"] == "residual_projected_path_scoped_front_adapter_not_wider_head" for row in rows)
    assert by_name["l1_active_atomic_radial_species8_h64"]["radial_species_adapter_scope"] == "atomic"
    assert by_name["l1_active_all_radial_species8_h64"]["radial_species_adapter_scope"] == "all"
    assert by_name["l2_cavity_edge_radial_species8_h64"]["radial_species_adapter_scope"] == "edge"
    assert "edge.cavity.vector_dot" in by_name["l2_cavity_edge_radial_species8_h64"]["scalar_path_ids"]
    assert all("stage123_residual_projected_active_path_scope" in row["tece_axes"] for row in rows)
    assert all("stage122_source" in row for row in rows)


def test_rtece_stage124_residual_edge_ladder_rows_keep_l1_active_backbone():
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import stage124_residual_edge_ladder_rows

    rows = stage124_residual_edge_ladder_rows()
    by_name = {row["name"]: row for row in rows}

    assert list(by_name) == [
        "l1_active_edge_direct_radial_species8_h64",
        "l1_active_edge_cavity_vec_radial_species8_h64",
        "l1_active_edge_cavity_vecq_radial_species8_h64",
        "l1_active_all_cavity_vecq_radial_species8_h64",
    ]
    assert {row["hidden_channels"] for row in rows} == {"64,64"}
    assert all(row["descriptor_bottleneck_dim"] == 0 for row in rows)
    assert all(row["stage_basis"] == "stage124_residual_edge_ladder" for row in rows)
    assert all(row["capacity_allocation"] == "l1_active_atomic_backbone_plus_edge_residual_paths" for row in rows)
    assert all(row["radial_species_adapter_channels"] == 8 for row in rows)
    assert all(row["species_basis_channels"] == 16 for row in rows)
    assert all(row["species_basis_mode"] == "learnable_embedding" for row in rows)
    assert all(row["atomic_cross_radial_projection"] == "learnable" for row in rows)
    assert all("stage124_residual_edge_ladder" in row["tece_axes"] for row in rows)
    assert all("stage123_source" in row for row in rows)

    base_paths = (
        "atomic.radial_density",
        "atomic.species_basis_density",
        "atomic.vector_norm",
        "atomic.vector_cross_radial_dot",
    )
    for row in rows:
        paths = tuple(part.strip() for part in row["scalar_path_ids"].split(",") if part.strip())
        assert paths[:4] == base_paths

    assert by_name["l1_active_edge_direct_radial_species8_h64"]["radial_species_adapter_scope"] == "edge"
    assert "edge.direct.radial" in by_name["l1_active_edge_direct_radial_species8_h64"]["scalar_path_ids"]
    assert "edge.cavity.vector_dot" in by_name["l1_active_edge_cavity_vec_radial_species8_h64"]["scalar_path_ids"]
    assert "edge.cavity.quadrupole_frobenius" in by_name["l1_active_edge_cavity_vecq_radial_species8_h64"]["scalar_path_ids"]
    assert by_name["l1_active_all_cavity_vecq_radial_species8_h64"]["radial_species_adapter_scope"] == "all"
    assert by_name["l1_active_edge_direct_radial_species8_h64"]["num_parameters_estimate"] >= by_name["l1_active_edge_cavity_vec_radial_species8_h64"]["num_parameters_estimate"] - 500


def test_rtece_stage125_front_capacity_ladder_rows_expand_representation_not_head():
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import stage125_front_capacity_ladder_rows

    rows = stage125_front_capacity_ladder_rows()
    by_name = {row["name"]: row for row in rows}

    assert list(by_name) == [
        "l1_active_species24_radial_species12_cross4_h64",
        "l1_active_species32_radial_species16_cross4_h64",
        "l1_active_species32_radial_species16_cross4_bneck32_h64",
        "l1_active_species48_radial_species24_cross5_bneck48_h64",
    ]
    assert {row["hidden_channels"] for row in rows} == {"64,64"}
    assert all(row["stage_basis"] == "stage125_front_capacity_ladder" for row in rows)
    assert all(row["capacity_allocation"] == "frontloaded_l1_atomic_representation_ladder_not_wider_head" for row in rows)
    assert all(row["moment_l_max"] == 1 for row in rows)
    assert all(row["radial_species_adapter_scope"] == "all" for row in rows)
    assert all(row["species_basis_mode"] == "learnable_embedding" for row in rows)
    assert all(row["atomic_cross_radial_projection"] == "learnable" for row in rows)
    assert all("stage125_front_capacity_ladder" in row["tece_axes"] for row in rows)
    assert all("stage124_source" in row for row in rows)

    base_paths = (
        "atomic.radial_density",
        "atomic.species_basis_density",
        "atomic.vector_norm",
        "atomic.vector_cross_radial_dot",
    )
    for row in rows:
        paths = tuple(part.strip() for part in row["scalar_path_ids"].split(",") if part.strip())
        assert paths == base_paths
        assert not any(path.startswith("edge.") for path in paths)

    assert by_name["l1_active_species32_radial_species16_cross4_h64"]["num_parameters_estimate"] > by_name["l1_active_species24_radial_species12_cross4_h64"]["num_parameters_estimate"]
    assert by_name["l1_active_species48_radial_species24_cross5_bneck48_h64"]["representation_parameters_estimate"] > by_name["l1_active_species32_radial_species16_cross4_h64"]["representation_parameters_estimate"]
    assert by_name["l1_active_species48_radial_species24_cross5_bneck48_h64"]["num_parameters_estimate"] >= 30_000


def test_rtece_stage126_rank_neighborhood_rows_downfold_stage125_anchor():
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import stage126_rank_neighborhood_rows

    rows = stage126_rank_neighborhood_rows()
    by_name = {row["name"]: row for row in rows}

    assert list(by_name) == [
        "l1_active_nrad10_species16_radial_species8_cross3_h64",
        "l1_active_nrad12_species16_radial_species8_cross3_h64",
        "l1_active_nrad12_species24_radial_species8_cross3_h64",
        "l1_active_nrad12_species16_radial_species12_cross3_h64",
        "l1_active_nrad12_species24_radial_species12_cross3_h64",
    ]
    assert {row["hidden_channels"] for row in rows} == {"64,64"}
    assert all(row["stage_basis"] == "stage126_rank_neighborhood" for row in rows)
    assert all(row["capacity_allocation"] == "rank_neighborhood_downfolding_around_stage125_anchor" for row in rows)
    assert all(row["moment_l_max"] == 1 for row in rows)
    assert all(row["radial_species_adapter_scope"] == "all" for row in rows)
    assert all(row["species_basis_mode"] == "learnable_embedding" for row in rows)
    assert all(row["atomic_cross_radial_projection"] == "learnable" for row in rows)
    assert all(row["atomic_cross_radial_sketch_channels"] == 3 for row in rows)
    assert all(row["descriptor_bottleneck_dim"] == 0 for row in rows)
    assert all("stage126_rank_neighborhood" in row["tece_axes"] for row in rows)
    assert all("stage125_source" in row for row in rows)

    base_paths = (
        "atomic.radial_density",
        "atomic.species_basis_density",
        "atomic.vector_norm",
        "atomic.vector_cross_radial_dot",
    )
    for row in rows:
        paths = tuple(part.strip() for part in row["scalar_path_ids"].split(",") if part.strip())
        assert paths == base_paths
        assert not any(path.startswith("edge.") for path in paths)
        assert row["num_parameters_estimate"] < 30_033

    assert by_name["l1_active_nrad12_species16_radial_species8_cross3_h64"]["num_parameters_estimate"] > by_name["l1_active_nrad10_species16_radial_species8_cross3_h64"]["num_parameters_estimate"]
    assert by_name["l1_active_nrad12_species24_radial_species8_cross3_h64"]["representation_parameters_estimate"] > by_name["l1_active_nrad12_species16_radial_species8_cross3_h64"]["representation_parameters_estimate"]
    assert by_name["l1_active_nrad12_species16_radial_species12_cross3_h64"]["representation_parameters_estimate"] > by_name["l1_active_nrad12_species16_radial_species8_cross3_h64"]["representation_parameters_estimate"]


def test_rtece_stage127_local_cross_species_rows_sweep_active_set_neighborhood():
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import stage127_local_cross_species_rows

    rows = stage127_local_cross_species_rows()
    by_name = {row["name"]: row for row in rows}

    assert list(by_name) == [
        "l1_active_nrad12_species20_radial_species8_cross3_h64",
        "l1_active_nrad12_species24_radial_species8_cross2_h64",
        "l1_active_nrad12_species24_radial_species8_cross3_h64",
        "l1_active_nrad12_species24_radial_species8_cross4_h64",
    ]
    assert {row["hidden_channels"] for row in rows} == {"64,64"}
    assert all(row["stage_basis"] == "stage127_local_cross_species" for row in rows)
    assert all(row["capacity_allocation"] == "local_cross_species_active_set_around_stage126_winner" for row in rows)
    assert all(row["moment_l_max"] == 1 for row in rows)
    assert all(row["num_radial"] == 12 for row in rows)
    assert all(row["radial_species_adapter_channels"] == 8 for row in rows)
    assert all(row["radial_species_adapter_scope"] == "all" for row in rows)
    assert all(row["species_basis_mode"] == "learnable_embedding" for row in rows)
    assert all(row["atomic_cross_radial_projection"] == "learnable" for row in rows)
    assert all(row["descriptor_bottleneck_dim"] == 0 for row in rows)
    assert all("stage127_local_cross_species" in row["tece_axes"] for row in rows)
    assert all("stage126_source" in row for row in rows)

    base_paths = (
        "atomic.radial_density",
        "atomic.species_basis_density",
        "atomic.vector_norm",
        "atomic.vector_cross_radial_dot",
    )
    for row in rows:
        paths = tuple(part.strip() for part in row["scalar_path_ids"].split(",") if part.strip())
        assert paths == base_paths
        assert not any(path.startswith("edge.") for path in paths)

    species24_rows = [row for row in rows if row["species_basis_channels"] == 24]
    assert {row["atomic_cross_radial_sketch_channels"] for row in species24_rows} == {2, 3, 4}
    assert by_name["l1_active_nrad12_species20_radial_species8_cross3_h64"]["species_basis_channels"] == 20
    assert by_name["l1_active_nrad12_species24_radial_species8_cross3_h64"]["stage126_source"].endswith("stage126_interpretation.md")


def test_rtece_stage131_residual_edge_current_pareto_rows_add_minimal_cavity_paths_after_weighting_negative():
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import stage131_residual_edge_current_pareto_rows

    rows = stage131_residual_edge_current_pareto_rows()
    by_name = {row["name"]: row for row in rows}

    assert list(by_name) == [
        "l1_active_species24_cavity_vec_residual_h64",
        "l2_active_species24_cavity_vecq_residual_h64",
    ]
    assert {row["hidden_channels"] for row in rows} == {"64,64"}
    assert all(row["stage_basis"] == "stage131_residual_edge_current_pareto" for row in rows)
    assert all(row["capacity_allocation"] == "current_pareto_atomic_backbone_plus_minimal_edge_relational_residual" for row in rows)
    assert all(row["num_radial"] == 12 for row in rows)
    assert all(row["radial_species_adapter_channels"] == 8 for row in rows)
    assert all(row["radial_species_adapter_scope"] == "all" for row in rows)
    assert all(row["species_basis_channels"] == 24 for row in rows)
    assert all(row["species_basis_mode"] == "learnable_embedding" for row in rows)
    assert all(row["atomic_cross_radial_sketch_channels"] == 3 for row in rows)
    assert all(row["atomic_cross_radial_projection"] == "learnable" for row in rows)
    assert all(row["short_range_repulsion_potential"] == "zbl" for row in rows)
    assert all(row["descriptor_bottleneck_dim"] == 0 for row in rows)
    assert all(row["stage130_source"].endswith("stage130_physical_triage_summary.md") for row in rows)
    assert all(row["stage129_source"].endswith("stage129_interpretation.md") for row in rows)
    assert all(row["stage124_source"].endswith("stage124_interpretation.md") for row in rows)

    base_paths = (
        "atomic.radial_density",
        "atomic.species_basis_density",
        "atomic.vector_norm",
        "atomic.vector_cross_radial_dot",
    )
    for row in rows:
        paths = tuple(part.strip() for part in row["scalar_path_ids"].split(",") if part.strip())
        assert paths[:4] == base_paths
        assert "edge.cavity.vector_dot" in paths
        assert "edge.direct.radial" in paths
        assert "stage131_residual_edge_current_pareto" in row["tece_axes"]
        assert "stage130_weighting_negative_control" in row["tece_axes"]
        assert "stage129_current_pareto_anchor" in row["tece_axes"]
        assert "stage127_local_cross_species" in row["tece_axes"]
        assert "cavity_edge_relational_scalar_sketches" in row["tece_axes"]
        assert "direct_edge_radial_path" in row["tece_axes"]

    assert by_name["l1_active_species24_cavity_vec_residual_h64"]["moment_l_max"] == 1
    assert "edge.cavity.quadrupole_frobenius" not in by_name["l1_active_species24_cavity_vec_residual_h64"]["scalar_path_ids"]
    assert by_name["l2_active_species24_cavity_vecq_residual_h64"]["moment_l_max"] == 2
    assert "edge.cavity.quadrupole_frobenius" in by_name["l2_active_species24_cavity_vecq_residual_h64"]["scalar_path_ids"]
    assert by_name["l2_active_species24_cavity_vecq_residual_h64"]["num_parameters_estimate"] >= by_name["l1_active_species24_cavity_vec_residual_h64"]["num_parameters_estimate"]


def test_rtece_stage121_active_frontloaded_rows_keep_active_atomic_paths_and_move_capacity_front():
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import stage121_active_frontloaded_rows

    rows = stage121_active_frontloaded_rows()
    by_name = {row["name"]: row for row in rows}

    assert list(by_name) == [
        "l1_active_species16_bneck16_h64",
        "l1_active_species16_bneck32_h64",
        "l2_active_species16_bneck16_h64",
        "l2_active_species16_bneck32_h64",
    ]
    assert {row["hidden_channels"] for row in rows} == {"64,64"}
    assert {row["descriptor_bottleneck_dim"] for row in rows} == {16, 32}
    assert all(row["species_basis_channels"] == 16 for row in rows)
    assert all(row["species_basis_mode"] == "learnable_embedding" for row in rows)
    assert all(row["learnable_radial_mixing"] for row in rows)
    assert all(row["atomic_cross_radial_projection"] == "learnable" for row in rows)
    assert all("edge." not in row["scalar_path_ids"] for row in rows)
    assert all(row["stage_basis"] == "stage121_active_frontloaded_representation" for row in rows)
    assert all("stage114_ef_active_selection" in row["tece_axes"] for row in rows)
    assert all("front_low_rank_path_mixer" in row["tece_axes"] for row in rows)
    assert by_name["l1_active_species16_bneck16_h64"]["moment_l_max"] == 1
    assert by_name["l2_active_species16_bneck16_h64"]["moment_l_max"] == 2
    assert by_name["l2_active_species16_bneck16_h64"]["representation_parameters_estimate"] > by_name["l1_active_species16_bneck16_h64"]["representation_parameters_estimate"]



def test_rtece_stage132_broad_teacher_distill_manifest_materializes_no_export_wrappers(tmp_path):
    from pathlib import Path

    from benchmarks.oc20neb_tace_mace.make_rtece_stage132_broad_teacher_distill import (
        audit_stage132_manifest,
        make_stage132_manifest,
        materialize_stage132,
    )

    payload = make_stage132_manifest(
        output_root=tmp_path / "stage132",
        base_train="base.extxyz",
        source_configs="source.extxyz",
        teacher_model="teacher.ckpt",
        train_valid_file="train_valid.extxyz",
        dft_valid_file="dft_valid.extxyz",
        teacher_valid_file="teacher_valid.extxyz",
        source_start_config=0,
        source_limit_configs=32,
        copies_per_config=16,
        base_limit_configs=2048,
        valid_limit_configs=256,
        bench_limit_configs=1024,
        max_steps=20000,
        lr_warmup_steps=500,
        early_stopping_patience=400,
    )

    assert payload["schema_version"] == "rtece_stage132_broad_teacher_distill.v1"
    assert payload["stage"] == "stage132_broad_teacher_distill_projection_check"
    assert payload["distillation_semantics"] == "broader_teacher_fake_labels_on_deployment_rattle_window"
    assert payload["source_start_config"] == 0
    assert payload["source_limit_configs"] == 32
    assert payload["copies_per_config"] == 16
    assert payload["augmented_limit_configs"] == 2560
    assert payload["row_set"] == "stage132-broad-teacher-distill"
    assert [row["variant"] for row in payload["rows"]] == [
        "l1_active_nrad12_species24_radial_species8_cross3_h64",
        "l1_active_species24_cavity_vec_residual_h64",
    ]
    assert "projection error" in payload["comparison_question"]
    assert "distillation error" in payload["comparison_question"]

    audit = audit_stage132_manifest(payload)
    assert audit["contract_pass"] is True
    assert audit["failed_checks"] == []

    materialized = materialize_stage132(payload)
    assert len(materialized["train_wrappers"]) == 2
    for wrapper in materialized["train_wrappers"]:
        text = Path(wrapper).read_text()
        assert "--export" not in text
        assert "--mem" not in text
        assert "--cpus-per-task" not in text
        assert "MAX_STEPS=20000" in text
        assert "LR_WARMUP_STEPS=500" in text
        assert "EARLY_STOPPING_PATIENCE=400" in text
        assert "TRAIN_FILE=" in text
        assert "augmented_train_base2048_plus_teacher_rattle512.extxyz" in text
    wrapper_text = "\\n".join(Path(wrapper).read_text() for wrapper in materialized["train_wrappers"])
    assert "edge.cavity.vector_dot" in wrapper_text
    assert "edge.cavity.quadrupole_frobenius" not in wrapper_text


def test_rtece_stage133_teacher_relax_distill_manifest_materializes_single_atomic_no_export_row(tmp_path):
    from pathlib import Path

    from benchmarks.oc20neb_tace_mace.make_rtece_stage133_teacher_relax_distill import (
        audit_stage133_manifest,
        make_stage133_manifest,
        materialize_stage133,
    )

    payload = make_stage133_manifest(
        output_root=tmp_path / "stage133",
        base_train="base.extxyz",
        source_configs="source.extxyz",
        teacher_model="teacher.ckpt",
        train_valid_file="train_valid.extxyz",
        dft_valid_file="dft_valid.extxyz",
        teacher_valid_file="teacher_valid.extxyz",
        source_start_config=0,
        source_limit_configs=32,
        copies_per_config=2,
        relax_max_steps=4,
        base_limit_configs=2048,
        valid_limit_configs=256,
        bench_limit_configs=1024,
        max_steps=20000,
        lr_warmup_steps=500,
        early_stopping_patience=400,
    )

    assert payload["schema_version"] == "rtece_stage133_teacher_relax_distill.v1"
    assert payload["stage"] == "stage133_teacher_relax_trajectory_distill"
    assert payload["distillation_semantics"] == "teacher_energy_force_labels_on_teacher_lbfgs_relaxation_trajectory"
    assert payload["trajectory_frame_count"] == 320
    assert payload["augmented_limit_configs"] == 2368
    assert payload["row_set"] == "stage133-teacher-relax-distill"
    assert [row["variant"] for row in payload["rows"]] == [
        "l1_active_nrad12_species24_radial_species8_cross3_h64",
    ]
    assert "projection error" in payload["comparison_question"]
    assert "distillation error" in payload["comparison_question"]
    assert "teacher PES relaxation manifold" in payload["comparison_question"]

    audit = audit_stage133_manifest(payload)
    assert audit["contract_pass"] is True
    assert audit["failed_checks"] == []

    materialized = materialize_stage133(payload)
    assert len(materialized["train_wrappers"]) == 1
    train_text = Path(materialized["train_wrappers"][0]).read_text()
    prep_text = Path(materialized["artifacts"]["prep_wrapper"]).read_text()
    combined = prep_text + "\n" + train_text
    assert "--export" not in combined
    assert "--mem" not in combined
    assert "--cpus-per-task" not in combined
    assert "make_teacher_relax_distill_configs.py" in prep_text
    assert "RELAX_MAX_STEPS" in prep_text
    assert "teacher_relax_valid_start0_limit32_copies2_std0p05_steps4.extxyz" in prep_text
    assert "MAX_STEPS=20000" in train_text
    assert "LR_WARMUP_STEPS=500" in train_text
    assert "EARLY_STOPPING_PATIENCE=400" in train_text
    assert "augmented_train_base2048_plus_teacher_relax320.extxyz" in train_text
    assert "edge.cavity.vector_dot" not in train_text


def test_teacher_relax_distill_configs_records_fixed_length_trajectory_with_teacher_labels():
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    import numpy as np

    from benchmarks.oc20neb_tace_mace.make_teacher_relax_distill_configs import (
        make_teacher_relax_distill_configs,
    )

    class HarmonicCalculator(Calculator):
        implemented_properties = ["energy", "forces", "free_energy"]

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            pos = atoms.get_positions()
            energy = float(0.5 * np.sum(pos * pos))
            self.results["energy"] = energy
            self.results["free_energy"] = energy
            self.results["forces"] = -pos

    atoms = Atoms("CH", positions=[[0.2, 0.0, 0.0], [0.0, 0.3, 0.0]], cell=[8, 8, 8], pbc=True)
    atoms.info["energy"] = 123.0
    atoms.arrays["forces"] = np.ones((2, 3))

    frames, summary = make_teacher_relax_distill_configs(
        [atoms],
        calculator=HarmonicCalculator(),
        source_start_config=7,
        copies_per_config=2,
        rattle_std_a=0.01,
        seed=13,
        relax_max_steps=3,
        reference_prefix="source_",
    )

    assert len(frames) == 8
    assert summary["configs"] == 8
    assert summary["atoms"] == 16
    assert summary["source_configs"] == 1
    assert summary["copies_per_config"] == 2
    assert summary["relax_max_steps"] == 3
    assert {frame.info["teacher_relax_step"] for frame in frames} == {0, 1, 2, 3}
    assert {frame.info["teacher_relax_source_config_index"] for frame in frames} == {7}
    assert {frame.info["teacher_relax_copy_index"] for frame in frames} == {0, 1}
    assert all("energy" in frame.info for frame in frames)
    assert all("forces" in frame.arrays for frame in frames)
    assert all("source_energy" in frame.info for frame in frames)
    assert all("source_forces" in frame.arrays for frame in frames)


def test_rtece_stage134_balanced_teacher_relax_manifest_materializes_weighted_atomic_no_export_row(tmp_path):
    from pathlib import Path

    from benchmarks.oc20neb_tace_mace.make_rtece_stage134_balanced_teacher_relax import (
        audit_stage134_manifest,
        make_stage134_manifest,
        materialize_stage134,
    )

    payload = make_stage134_manifest(
        output_root=tmp_path / "stage134",
        base_train="base.extxyz",
        source_configs="source.extxyz",
        teacher_model="teacher.ckpt",
        train_valid_file="train_valid.extxyz",
        dft_valid_file="dft_valid.extxyz",
        teacher_valid_file="teacher_valid.extxyz",
        source_start_config=0,
        source_limit_configs=32,
        copies_per_config=2,
        relax_max_steps=4,
        base_limit_configs=2048,
        base_energy_multiplier=1.25,
        teacher_energy_multiplier=0.25,
        teacher_force_multiplier=2.0,
        valid_limit_configs=256,
        bench_limit_configs=1024,
        max_steps=20000,
        lr_warmup_steps=500,
        early_stopping_patience=400,
    )

    assert payload["schema_version"] == "rtece_stage134_balanced_teacher_relax_distill.v1"
    assert payload["stage"] == "stage134_balanced_teacher_relax_distill"
    assert payload["distillation_semantics"] == "dft_energy_anchor_plus_teacher_relax_force_weighted_distillation"
    assert payload["trajectory_frame_count"] == 320
    assert payload["augmented_limit_configs"] == 2368
    assert payload["row_set"] == "stage134-balanced-teacher-relax-distill"
    assert payload["weight_policy"]["source_energy_multipliers"] == {
        "base_mixed_train_tw0p75": 1.25,
        "teacher_relax_trajectory320": 0.25,
    }
    assert payload["weight_policy"]["source_force_multipliers"] == {"teacher_relax_trajectory320": 2.0}
    assert payload["weight_policy"]["normalize_energy_mean"] is True
    assert payload["weight_policy"]["normalize_force_mean"] is True
    assert [row["variant"] for row in payload["rows"]] == [
        "l1_active_nrad12_species24_radial_species8_cross3_h64",
    ]
    assert "energy drift" in payload["comparison_question"]
    assert "projection error" in payload["comparison_question"]

    audit = audit_stage134_manifest(payload)
    assert audit["contract_pass"] is True
    assert audit["failed_checks"] == []

    materialized = materialize_stage134(payload)
    assert len(materialized["train_wrappers"]) == 1
    train_text = Path(materialized["train_wrappers"][0]).read_text()
    prep_text = Path(materialized["artifacts"]["prep_wrapper"]).read_text()
    combined = prep_text + "\n" + train_text
    assert "--export" not in combined
    assert "--mem" not in combined
    assert "--cpus-per-task" not in combined
    assert "#SBATCH --time=03:55:00" in prep_text
    assert "make_teacher_relax_distill_configs.py" in prep_text
    assert "apply_extxyz_sample_weights.py" in prep_text
    assert "--source-energy-multiplier base_mixed_train_tw0p75:1.25" in prep_text
    assert "--source-energy-multiplier teacher_relax_trajectory320:0.25" in prep_text
    assert "--source-force-multiplier teacher_relax_trajectory320:2.0" in prep_text
    assert "--normalize-energy-mean" in prep_text
    assert "weighted_train_base2048_plus_teacher_relax320_eanchor.extxyz" in train_text
    assert "MAX_STEPS=20000" in train_text
    assert "LR_WARMUP_STEPS=500" in train_text
    assert "EARLY_STOPPING_PATIENCE=400" in train_text
    assert "edge.cavity.vector_dot" not in train_text


def test_rtece_stage142_teacher_relax_coverage_manifest_materializes_force_only_cond32_row(tmp_path):
    from pathlib import Path

    from benchmarks.oc20neb_tace_mace.make_rtece_stage142_teacher_relax_coverage import (
        audit_stage142_manifest,
        make_stage142_manifest,
        materialize_stage142,
    )

    payload = make_stage142_manifest(
        output_root=tmp_path / "stage142",
        base_train="base.extxyz",
        source_configs="source.extxyz",
        teacher_model="teacher.ckpt",
        train_valid_file="train_valid.extxyz",
        dft_valid_file="dft_valid.extxyz",
        teacher_valid_file="teacher_valid.extxyz",
        source_start_config=0,
        source_limit_configs=64,
        copies_per_config=2,
        relax_max_steps=4,
        base_limit_configs=2048,
        base_energy_multiplier=1.25,
        teacher_force_multiplier=2.0,
        valid_limit_configs=256,
        bench_limit_configs=1024,
        max_steps=20000,
        lr_warmup_steps=500,
        early_stopping_patience=400,
    )

    assert payload["schema_version"] == "rtece_stage142_teacher_relax_coverage.v1"
    assert payload["stage"] == "stage142_teacher_relax_coverage"
    assert payload["distillation_semantics"] == "dft_energy_anchor_plus_force_only_teacher_relax_coverage"
    assert payload["trajectory_frame_count"] == 640
    assert payload["augmented_limit_configs"] == 2688
    assert payload["row_set"] == "stage142-teacher-relax-coverage"
    assert payload["weight_policy"]["source_energy_multipliers"] == {
        "base_mixed_train_tw0p75": 1.25,
        "teacher_relax_trajectory640": 0.0,
    }
    assert payload["weight_policy"]["source_force_multipliers"] == {"teacher_relax_trajectory640": 2.0}
    assert payload["weight_policy"]["teacher_energy_multiplier_is_fixed_zero"] is True
    assert "deployment-measure coverage" in payload["comparison_question"]
    assert "architecture fixed" in payload["comparison_question"]
    assert [row["variant"] for row in payload["rows"]] == [
        "l2_active_nrad12_species24_radial_species8_cross3_cond32_h64",
    ]
    row = payload["rows"][0]
    assert row["moment_l_max"] == 2
    assert row["descriptor_conditioner"] == "residual_mlp"
    assert row["descriptor_conditioner_hidden_channels"] == 32
    assert row["descriptor_bottleneck_dim"] == 0
    assert "atomic.quadrupole_norm" in row["scalar_path_ids"]
    assert "atomic.quadrupole_cross_radial_frobenius" in row["scalar_path_ids"]
    assert "edge.cavity.vector_dot" not in row["scalar_path_ids"]

    audit = audit_stage142_manifest(payload)
    assert audit["contract_pass"] is True
    assert audit["failed_checks"] == []

    materialized = materialize_stage142(payload)
    assert len(materialized["train_wrappers"]) == 1
    assert len(materialized["physical_wrappers"]) == 1
    train_text = Path(materialized["train_wrappers"][0]).read_text()
    prep_text = Path(materialized["artifacts"]["prep_wrapper"]).read_text()
    physical_text = Path(materialized["physical_wrappers"][0]).read_text()
    combined = prep_text + '\n' + train_text + '\n' + physical_text
    assert "--export" not in combined
    assert "--mem" not in combined
    assert "--cpus-per-task" not in combined
    assert "#SBATCH --time=03:55:00" in prep_text
    assert "make_teacher_relax_distill_configs.py" in prep_text
    assert "apply_extxyz_sample_weights.py" in prep_text
    assert "--source-energy-multiplier base_mixed_train_tw0p75:1.25" in prep_text
    assert "--source-energy-multiplier teacher_relax_trajectory640:0.0" in prep_text
    assert "--source-force-multiplier teacher_relax_trajectory640:2.0" in prep_text
    assert "--normalize-energy-mean" in prep_text
    assert "weighted_train_base2048_plus_teacher_relax640_forceonly_eanchor.extxyz" in train_text
    assert "MOMENT_L_MAX=2" in train_text
    assert "DESCRIPTOR_CONDITIONER=residual_mlp" in train_text
    assert "DESCRIPTOR_CONDITIONER_HIDDEN_CHANNELS=32" in train_text
    assert "MAX_STEPS=20000" in train_text
    assert "LR_WARMUP_STEPS=500" in train_text
    assert "EARLY_STOPPING_PATIENCE=400" in train_text
    assert "rattle_relax_rtece.py" in physical_text


def test_rtece_stage143_semantic_active_set_manifest_materializes_projection_wrapper(tmp_path):
    from pathlib import Path

    from benchmarks.oc20neb_tace_mace.make_rtece_stage143_semantic_active_set import (
        audit_stage143_manifest,
        make_stage143_manifest,
        materialize_stage143,
    )

    payload = make_stage143_manifest(
        output_root=tmp_path / "stage143",
        train_configs="stage142_weighted.extxyz",
        valid_configs="dft_valid.extxyz",
        energy_eval_stride=4,
    )

    assert payload["schema_version"] == "rtece_stage143_semantic_active_set.v1"
    assert payload["stage"] == "stage143_semantic_active_set"
    assert payload["diagnostic_semantics"] == "semantic_path_active_set_projection_after_stage142"
    assert payload["deployment_measure_source"] == "stage142_force_only_teacher_relax_distribution"
    assert payload["force_projection_status"] == "deferred_force_descriptor_jacobian_cost"
    assert payload["force_target_key"] is None
    assert "Schur" in payload["comparison_question"]
    assert "not another same-window teacher-relax expansion" in payload["comparison_question"]
    assert payload["train_configs"] == "stage142_weighted.extxyz"
    assert payload["valid_configs"] == "dft_valid.extxyz"
    assert payload["limit_configs"] == 512

    assert payload["reference_path_ids"] == [
        "atomic.radial_density",
        "atomic.species_basis_density",
        "atomic.vector_norm",
        "atomic.vector_cross_radial_dot",
        "atomic.quadrupole_norm",
        "atomic.quadrupole_cross_radial_frobenius",
        "edge.cavity.vector_dot",
        "edge.cavity.quadrupole_frobenius",
        "edge.direct.radial",
    ]

    candidates = payload["candidates"]
    assert [candidate["name"] for candidate in candidates] == [
        "t2_l0_species_radial",
        "t2_l1_atomic_cross",
        "t2_l2_atomic_cross",
        "t3_l2_cavity_vector",
        "t3_l2_cavity_vector_quadrupole_direct",
    ]
    assert candidates[0]["tece_tier"] == "T2_scalar_endpoint"
    assert candidates[-1]["tece_tier"] == "T3_rtece_edge_relational"
    assert candidates[0]["cost_proxy"]["edge_scalar_paths"] == 0
    assert candidates[-1]["cost_proxy"]["edge_scalar_paths"] == 4
    assert "edge.cavity.vector_dot" in candidates[-2]["path_ids"]
    assert "edge.cavity.quadrupole_frobenius" in candidates[-1]["path_ids"]
    assert "edge.direct.radial" in candidates[-1]["path_ids"]
    assert all(candidate["moment_l_max"] in {0, 1, 2} for candidate in candidates)

    audit = audit_stage143_manifest(payload)
    assert audit["contract_pass"] is True
    assert audit["failed_checks"] == []

    materialized = materialize_stage143(payload)
    wrapper_text = Path(materialized["artifacts"]["wrapper"]).read_text()
    combined = wrapper_text + "\n" + Path(materialized["artifacts"]["stage_plan"]).read_text()
    assert "--export" not in combined
    assert "--mem" not in combined
    assert "--cpus-per-task" not in combined
    assert "#SBATCH --time=03:55:00" in wrapper_text
    assert "analyze_rtece_projection_error.py" in wrapper_text
    assert "--reference-path-ids" in wrapper_text
    assert "--candidate t2_l0_species_radial:" in wrapper_text
    assert "--candidate t3_l2_cavity_vector_quadrupole_direct:" in wrapper_text
    assert "--force-target-key" not in wrapper_text
    assert "TRAIN_CONFIGS=stage142_weighted.extxyz" in wrapper_text
    assert "VALID_CONFIGS=dft_valid.extxyz" in wrapper_text
    assert "LIMIT_CONFIGS=512" in wrapper_text
    assert "NUM_RADIAL=12" in wrapper_text
    assert "SPECIES_BASIS_CHANNELS=24" in wrapper_text
    assert "ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS=3" in wrapper_text


def test_rtece_stage144_t3_cavity_vector_smoke_contract_allows_short_run(tmp_path):
    from pathlib import Path

    from benchmarks.oc20neb_tace_mace.make_rtece_stage144_t3_cavity_vector import (
        audit_stage144_manifest,
        make_stage144_manifest,
        materialize_stage144,
    )

    payload = make_stage144_manifest(
        output_root=tmp_path / "stage144-smoke",
        train_file="stage142_weighted.extxyz",
        train_valid_file="train_valid.extxyz",
        dft_valid_file="dft_valid.extxyz",
        teacher_valid_file="teacher_valid.extxyz",
        limit_configs=512,
        valid_limit_configs=64,
        bench_limit_configs=128,
        max_steps=2000,
        lr_warmup_steps=500,
        early_stopping_patience=400,
    )

    audit = audit_stage144_manifest(payload)

    assert audit["contract_pass"] is True
    assert audit["failed_checks"] == []
    materialized = materialize_stage144(payload)
    train_text = Path(materialized["train_wrappers"][0]).read_text()
    assert "LIMIT_CONFIGS=512" in train_text
    assert "VALID_LIMIT_CONFIGS=64" in train_text
    assert "BENCH_LIMIT_CONFIGS=128" in train_text
    assert "MAX_STEPS=2000" in train_text


def test_rtece_stage144_t3_cavity_vector_manifest_materializes_minimal_edge_row(tmp_path):
    from pathlib import Path

    from benchmarks.oc20neb_tace_mace.make_rtece_stage144_t3_cavity_vector import (
        audit_stage144_manifest,
        make_stage144_manifest,
        materialize_stage144,
    )

    payload = make_stage144_manifest(
        output_root=tmp_path / "stage144",
        train_file="stage142_weighted.extxyz",
        train_valid_file="train_valid.extxyz",
        dft_valid_file="dft_valid.extxyz",
        teacher_valid_file="teacher_valid.extxyz",
        valid_limit_configs=256,
        bench_limit_configs=1024,
        max_steps=20000,
        lr_warmup_steps=500,
        early_stopping_patience=400,
    )

    assert payload["schema_version"] == "rtece_stage144_t3_cavity_vector.v1"
    assert payload["stage"] == "stage144_t3_cavity_vector"
    assert payload["row_set"] == "stage144-t3-cavity-vector"
    assert payload["distillation_semantics"] == "stage143_active_set_guided_minimal_t3_edge_relational_training"
    assert payload["train_file"] == "stage142_weighted.extxyz"
    assert "one edge.cavity.vector_dot" in payload["comparison_question"]
    assert "architecture increment" in payload["comparison_question"]
    assert payload["stage143_active_set_source"].endswith("stage143_results_summary.json")
    assert [row["variant"] for row in payload["rows"]] == [
        "t3_l2_cavity_vector_cond32_h64",
    ]

    row = payload["rows"][0]
    path_ids = tuple(part.strip() for part in row["scalar_path_ids"].split(",") if part.strip())
    assert row["moment_l_max"] == 2
    assert row["hidden_channels"] == "64,64"
    assert row["num_radial"] == 12
    assert row["species_basis_channels"] == 24
    assert row["species_basis_mode"] == "learnable_embedding"
    assert row["radial_species_adapter_channels"] == 8
    assert row["atomic_cross_radial_sketch_channels"] == 3
    assert row["atomic_cross_radial_projection"] == "learnable"
    assert row["descriptor_conditioner"] == "residual_mlp"
    assert row["descriptor_conditioner_hidden_channels"] == 32
    assert row["descriptor_bottleneck_dim"] == 0
    assert row["short_range_repulsion_potential"] == "zbl"
    assert "edge.cavity.vector_dot" in path_ids
    assert "edge.cavity.quadrupole_frobenius" not in path_ids
    assert "edge.direct.radial" not in path_ids
    assert row["tece_tier"] == "T3_rtece_edge_relational_minimal"
    assert row["stage144_isolated_increment"] == "edge.cavity.vector_dot"

    audit = audit_stage144_manifest(payload)
    assert audit["contract_pass"] is True
    assert audit["failed_checks"] == []

    materialized = materialize_stage144(payload)
    assert len(materialized["train_wrappers"]) == 1
    assert len(materialized["physical_wrappers"]) == 1
    train_text = Path(materialized["train_wrappers"][0]).read_text()
    physical_text = Path(materialized["physical_wrappers"][0]).read_text()
    combined = train_text + "\n" + physical_text + "\n" + Path(materialized["artifacts"]["stage_plan"]).read_text()
    assert "--export" not in combined
    assert "--mem" not in combined
    assert "--cpus-per-task" not in combined
    assert "set -euo pipefail" not in combined
    assert "set -eo pipefail" in train_text
    assert "set -eo pipefail" in physical_text
    assert "TRAIN_FILE=stage142_weighted.extxyz" in train_text
    assert "MAX_STEPS=20000" in train_text
    assert "LR_WARMUP_STEPS=500" in train_text
    assert "EARLY_STOPPING_PATIENCE=400" in train_text
    assert "MOMENT_L_MAX=2" in train_text
    assert "SCALAR_PATH_IDS=atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot" in train_text
    assert "DESCRIPTOR_CONDITIONER=residual_mlp" in train_text
    assert "DESCRIPTOR_CONDITIONER_HIDDEN_CHANNELS=32" in train_text
    assert "rattle_relax_rtece.py" in physical_text
    assert "dimer_scan_rtece.py" in physical_text


def test_community_baselines_stage145_manifest_materializes_no_export_wrappers(tmp_path):
    from pathlib import Path

    from benchmarks.oc20neb_tace_mace.make_community_baselines_stage145 import (
        audit_stage145_manifest,
        make_stage145_manifest,
        materialize_stage145,
    )

    payload = make_stage145_manifest(
        output_root=tmp_path / "stage145",
        limit_configs=16,
        valid_limit_configs=4,
        bench_limit_configs=8,
        nep_generations=20,
        deepmd_stop_batch=20,
    )

    assert payload["schema_version"] == "community_baselines_stage145.v1"
    assert payload["stage"] == "community_baselines_stage145"
    assert payload["train_contract"]["label_target"] == "mixed_energy_forces"
    assert payload["train_contract"]["energy_key"] == "energy"
    assert payload["train_contract"]["forces_key"] == "forces"
    assert payload["train_contract"]["dft_energy_key"] == "dft_energy"
    assert payload["train_contract"]["teacher_forces_key"] == "teacher_forces"

    rows = {row["name"]: row for row in payload["rows"]}
    assert set(rows) == {"nep4_mixed_smoke", "deepmd_dpa_like_mixed_smoke"}
    assert rows["nep4_mixed_smoke"]["engine"] == "nep"
    assert rows["nep4_mixed_smoke"]["module"] == "gpumd/4.8-cuda12.4"
    assert rows["deepmd_dpa_like_mixed_smoke"]["engine"] == "deepmd"
    assert rows["deepmd_dpa_like_mixed_smoke"]["module"] == "deepmd-kit/3.1.2"
    assert rows["deepmd_dpa_like_mixed_smoke"]["descriptor_label"] in {
        "dpa1_zero_attention",
        "dpa_like_low_attention",
    }

    audit = audit_stage145_manifest(payload)
    assert audit["contract_pass"], audit["failed_checks"]

    materialized = materialize_stage145(payload)
    assert Path(materialized["manifest"]).exists()
    assert Path(materialized["audit"]).exists()
    assert Path(materialized["stage_plan"]).exists()
    assert set(materialized["wrappers"]) == {"nep4_mixed_smoke", "deepmd_dpa_like_mixed_smoke"}

    for wrapper_path in materialized["wrappers"].values():
        text = Path(wrapper_path).read_text()
        assert "#SBATCH --nodes=1" in text
        assert "#SBATCH --ntasks=1" in text
        assert "#SBATCH --gpus-per-node=1" in text
        assert "#SBATCH --partition=16V100" in text
        assert "#SBATCH --qos=flood-1o2gpu" in text
        assert "--export" not in text
        assert "--mem" not in text
        assert "--cpus-per-task" not in text
        assert "community-baselines-stage145" in text
        assert "/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python benchmarks/oc20neb_tace_mace/convert_stage145_" in text
        assert "set -eo pipefail" in text
        assert "set -euo pipefail" not in text


def test_community_baselines_stage145_materializes_benchmark_and_physical_wrappers(tmp_path):
    from pathlib import Path

    from benchmarks.oc20neb_tace_mace.make_community_baselines_stage145 import (
        audit_stage145_manifest,
        make_stage145_manifest,
        materialize_stage145,
    )

    payload = make_stage145_manifest(
        output_root=tmp_path / "stage145",
        limit_configs=16,
        valid_limit_configs=4,
        bench_limit_configs=8,
        nep_generations=20,
        deepmd_stop_batch=20,
    )
    audit = audit_stage145_manifest(payload)

    assert audit["contract_pass"], audit["failed_checks"]
    for row in payload["rows"]:
        assert row["benchmark_wrapper"].endswith("_benchmark_no_export.sbatch")
        assert row["physical_wrapper"].endswith("_physical_no_export.sbatch")
        assert row["dft_benchmark"].endswith(f"{row['name']}_dft_benchmark.json")
        assert row["physical_pareto"].endswith(f"{row['name']}_physical_pareto.json")

    materialized = materialize_stage145(payload)

    assert set(materialized["benchmark_wrappers"]) == {"nep4_mixed_smoke", "deepmd_dpa_like_mixed_smoke"}
    assert set(materialized["physical_wrappers"]) == {"nep4_mixed_smoke", "deepmd_dpa_like_mixed_smoke"}
    for wrappers in (materialized["benchmark_wrappers"], materialized["physical_wrappers"]):
        for wrapper_path in wrappers.values():
            text = Path(wrapper_path).read_text()
            assert "#SBATCH --partition=16V100" in text
            assert "#SBATCH --qos=flood-1o2gpu" in text
            assert "--export" not in text
            assert "--mem" not in text
            assert "--cpus-per-task" not in text
            assert "set -eo pipefail" in text
            assert "set -euo pipefail" not in text
            assert "community-baselines-stage145" in text

    benchmark_text = Path(materialized["benchmark_wrappers"]["deepmd_dpa_like_mixed_smoke"]).read_text()
    physical_text = Path(materialized["physical_wrappers"]["deepmd_dpa_like_mixed_smoke"]).read_text()
    assert "benchmark_stage145_community.py" in benchmark_text
    assert "physical_stage145_community.py" in physical_text
    assert "deepmd_dpa_like_mixed_smoke_dft_benchmark.json" in benchmark_text
    assert "deepmd_dpa_like_mixed_smoke_physical_pareto.json" in physical_text


def test_stage145_nep_converter_writes_gpumd_train_xyz(tmp_path):
    import numpy as np
    import ase.io
    from ase import Atoms
    from ase.constraints import FixAtoms

    from benchmarks.oc20neb_tace_mace.convert_stage145_nep import convert_extxyz_to_nep

    source = tmp_path / "input.extxyz"
    atoms = Atoms(
        "CN",
        positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    atoms.info["energy"] = -3.0
    atoms.info["dft_energy"] = -2.9
    atoms.info["teacher_energy"] = -3.1
    atoms.set_constraint(FixAtoms(indices=[1]))
    atoms.arrays["forces"] = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=float)
    atoms.arrays["dft_forces"] = atoms.arrays["forces"]
    atoms.arrays["teacher_forces"] = atoms.arrays["forces"]
    ase.io.write(source, [atoms], format="extxyz")

    summary = convert_extxyz_to_nep(source, tmp_path / "nep", limit_configs=1)

    assert summary["engine"] == "nep"
    assert summary["num_configs"] == 1
    assert summary["type_map"] == ["C", "N"]
    text = (tmp_path / "nep" / "train.xyz").read_text()
    assert "energy=-3.0" in text
    assert "Properties=species:S:1:pos:R:3:force:R:3" in text
    assert "move_mask" not in text
    assert "dft_energy" not in text
    assert "teacher_energy" not in text
    assert "C " in text and "N " in text
    nep_in = (tmp_path / "nep" / "nep.in").read_text()
    assert "type         2 C N" in nep_in
    assert "generation   20000" in nep_in


def test_stage145_deepmd_converter_writes_system_and_input(tmp_path):
    import json
    import numpy as np
    import ase.io
    from ase import Atoms

    from benchmarks.oc20neb_tace_mace.convert_stage145_deepmd import convert_extxyz_to_deepmd

    source = tmp_path / "input.extxyz"
    atoms = Atoms(
        "CN",
        positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    atoms.info["energy"] = -3.0
    atoms.arrays["forces"] = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=float)
    ase.io.write(source, [atoms], format="extxyz")

    summary = convert_extxyz_to_deepmd(source, tmp_path / "dp", limit_configs=1, stop_batch=20)

    assert summary["engine"] == "deepmd"
    assert summary["num_configs"] == 1
    assert summary["type_map"] == ["C", "N"]
    assert (tmp_path / "dp" / "type_map.raw").read_text().splitlines() == ["C", "N"]
    assert np.load(tmp_path / "dp" / "mixed" / "set.000" / "coord.npy").shape == (1, 6)
    assert np.load(tmp_path / "dp" / "mixed" / "set.000" / "force.npy").shape == (1, 6)
    assert (tmp_path / "dp" / "mixed" / "type_map.raw").read_text().splitlines() == ["C", "N"]
    payload = json.loads((tmp_path / "dp" / "input.json").read_text())
    assert payload["training"]["numb_steps"] == 20
    assert payload["model"]["type_map"] == ["C", "N"]
    assert payload["model"]["descriptor"]["type"] in {"dpa2", "se_atten_v2", "se_atten"}
    assert payload["model"]["descriptor"]["sel"] == 128
    assert payload["model"]["descriptor"]["attn_layer"] == 0


def test_stage145_deepmd_converter_splits_mixed_atom_orders(tmp_path):
    import json
    import numpy as np
    import ase.io
    from ase import Atoms

    from benchmarks.oc20neb_tace_mace.convert_stage145_deepmd import convert_extxyz_to_deepmd

    source = tmp_path / "input.extxyz"
    cn = Atoms("CN", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], cell=[8.0, 8.0, 8.0], pbc=True)
    h2o = Atoms("H2O", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0], [0.0, 0.7, 0.0]], cell=[8.0, 8.0, 8.0], pbc=True)
    for atoms, energy in [(cn, -3.0), (h2o, -1.0)]:
        atoms.info["energy"] = energy
        atoms.arrays["forces"] = np.zeros((len(atoms), 3), dtype=float)
    ase.io.write(source, [cn, h2o], format="extxyz")

    summary = convert_extxyz_to_deepmd(source, tmp_path / "dp", limit_configs=2, stop_batch=20)

    assert summary["split_by_atom_order"] is True
    assert summary["num_systems"] == 2
    assert summary["systems"] == ["mixed_000", "mixed_001"]
    assert np.load(tmp_path / "dp" / "mixed_000" / "set.000" / "coord.npy").shape == (1, 6)
    assert np.load(tmp_path / "dp" / "mixed_001" / "set.000" / "coord.npy").shape == (1, 9)
    assert (tmp_path / "dp" / "mixed_000" / "type_map.raw").read_text().splitlines() == ["C", "H", "N", "O"]
    assert (tmp_path / "dp" / "mixed_001" / "type_map.raw").read_text().splitlines() == ["C", "H", "N", "O"]
    payload = json.loads((tmp_path / "dp" / "input.json").read_text())
    assert payload["training"]["training_data"]["systems"] == ["mixed_000", "mixed_001"]


def test_rtece_stage164_force_protected_uv_manifest_materializes_single_uv_row(tmp_path):
    from pathlib import Path

    from benchmarks.oc20neb_tace_mace.make_rtece_stage164_force_protected_uv import (
        audit_stage164_manifest,
        make_stage164_manifest,
        materialize_stage164,
    )

    payload = make_stage164_manifest(
        output_root=tmp_path / "stage164",
        train_file="mixed_train.extxyz",
        train_valid_file="train_valid.extxyz",
        dft_valid_file="dft_valid.extxyz",
        teacher_valid_file="teacher_valid.extxyz",
        limit_configs=2048,
        valid_limit_configs=256,
        bench_limit_configs=1024,
        max_steps=20000,
        lr_warmup_steps=500,
        early_stopping_patience=400,
    )

    assert payload["schema_version"] == "rtece_stage164_force_protected_uv.v1"
    assert payload["stage"] == "stage164_force_protected_uv_training"
    assert payload["row_set"] == "stage164-force-protected-uv"
    assert payload["distillation_semantics"] == "stage163_force_protected_uv_training"
    assert payload["active_set_gate"]["max_force_regression_fraction"] == pytest.approx(0.10)
    assert payload["active_set_gate"]["require_energy_gain"] is True
    assert payload["relative_energy_weight"] == pytest.approx(0.25)
    assert [row["variant"] for row in payload["rows"]] == ["stage164_uv_force_gate_rel0p25_b32"]

    row = payload["rows"][0]
    path_ids = tuple(part.strip() for part in row["scalar_path_ids"].split(",") if part.strip())
    assert "edge.cavity.target_vector_projection" in path_ids
    assert "edge.cavity.source_vector_projection" in path_ids
    assert "edge.cavity.target_quadrupole_projection" not in path_ids
    assert "edge.cavity.source_quadrupole_projection" not in path_ids
    assert row["stage163_promoted_gate"] == "force_regression<=0.10_and_energy_gain"
    assert row["stage164_isolated_increment"] == "edge.cavity.target/source_vector_projection"
    assert row["hidden_channels"] == "64,64"
    assert row["moment_l_max"] == 2
    assert row["short_range_repulsion_potential"] == "zbl"

    audit = audit_stage164_manifest(payload)
    assert audit["contract_pass"] is True
    assert audit["failed_checks"] == []

    materialized = materialize_stage164(payload)
    assert len(materialized["train_wrappers"]) == 1
    train_text = Path(materialized["train_wrappers"][0]).read_text()
    stage_plan = Path(materialized["artifacts"]["stage_plan"]).read_text()
    combined = train_text + "\n" + stage_plan
    assert "--export" not in combined
    assert "--mem" not in combined
    assert "--cpus-per-task" not in combined
    assert "set -euo pipefail" not in train_text
    assert "set -eo pipefail" in train_text
    assert "TRAIN_FILE=mixed_train.extxyz" in train_text
    assert "MAX_STEPS=20000" in train_text
    assert "LR_WARMUP_STEPS=500" in train_text
    assert "EARLY_STOPPING_PATIENCE=400" in train_text
    assert "BATCH_SIZE=32" in train_text
    assert "VALID_BATCH_SIZE=32" in train_text
    assert "RELATIVE_ENERGY_WEIGHT=0.25" in train_text
    assert "RELATIVE_ENERGY_GROUP_KEY=case_id" in train_text
    assert "RELATIVE_ENERGY_IMAGE_KEY=source_frame" in train_text
    assert "SCALAR_PATH_IDS=atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot,edge.direct.radial,edge.cavity.target_vector_projection,edge.cavity.source_vector_projection" in train_text
    assert "Stage163" in stage_plan
    assert "force-protected" in stage_plan



def test_stage164_results_summary_reports_pending_and_metric_deltas(tmp_path):
    import json

    from benchmarks.oc20neb_tace_mace.summarize_rtece_stage164_results import (
        render_stage164_results_markdown,
        summarize_stage164_results,
    )

    baseline = tmp_path / "baseline_dft.json"
    baseline.write_text(json.dumps({
        "rmse_e_mev_atom": 52.0,
        "mae_e_mev_atom": 35.0,
        "max_abs_e_mev_atom": 205.0,
        "group_mean_offset_rmse_mev_atom": 4.8,
        "first_image_anchor_rmse_mev_atom": 11.4,
        "relative_image_rmse_mev_atom": 7.1,
        "barrier_rmse_mev_atom": 11.4,
        "rmse_f_mev_a": 94.0,
        "mae_f_mev_a": 42.0,
        "max_abs_f_mev_a": 2093.0,
        "atoms_per_second": 500000.0,
    }))
    run_dir = tmp_path / "run"
    variant = "stage164_uv_force_gate_rel0p25_b32"
    manifest = {
        "stage": "stage164_force_protected_uv_training",
        "rows": [{"variant": variant}],
        "artifacts": {"run_root": str(run_dir)},
        "active_set_gate": {"max_force_regression_fraction": 0.10},
    }

    pending = summarize_stage164_results(manifest, baseline_dft_benchmark=baseline)
    assert pending["rows"][0]["artifact_status"] == "pending"
    assert pending["rows"][0]["dft_benchmark_status"] == "missing"
    assert pending["rows"][0]["baseline_dft_f_rmse_mev_a"] == pytest.approx(94.0)

    out_dir = run_dir / variant
    out_dir.mkdir(parents=True)
    (out_dir / f"{variant}_dft_benchmark.json").write_text(json.dumps({
        "rmse_e_mev_atom": 40.0,
        "mae_e_mev_atom": 28.0,
        "max_abs_e_mev_atom": 150.0,
        "group_mean_offset_rmse_mev_atom": 5.5,
        "first_image_anchor_rmse_mev_atom": 10.0,
        "relative_image_rmse_mev_atom": 6.5,
        "barrier_rmse_mev_atom": 10.5,
        "rmse_f_mev_a": 100.0,
        "mae_f_mev_a": 45.0,
        "max_abs_f_mev_a": 2200.0,
        "atoms_per_second": 470000.0,
    }))

    summary = summarize_stage164_results(manifest, baseline_dft_benchmark=baseline)
    row = summary["rows"][0]
    assert row["artifact_status"] == "complete"
    assert row["dft_e_rmse_delta_mev_atom"] == pytest.approx(-12.0)
    assert row["dft_f_rmse_delta_mev_a"] == pytest.approx(6.0)
    assert row["force_rmse_regression_fraction"] == pytest.approx(6.0 / 94.0)
    assert row["force_gate_passed"] is True
    assert row["energy_rmse_improved"] is True
    markdown = render_stage164_results_markdown(summary)
    assert "force gate" in markdown
    assert "stage164_uv_force_gate_rel0p25_b32" in markdown
    assert "complete" in markdown



def test_stage165_energy_baseline_triage_summarizes_case_offsets(tmp_path):
    import json

    from benchmarks.oc20neb_tace_mace.summarize_rtece_stage165_energy_baseline_triage import (
        render_stage165_energy_baseline_markdown,
        summarize_stage165_energy_baseline_triage,
    )

    first = tmp_path / "rel0_dft.json"
    first.write_text(json.dumps({
        "variant": "rel0",
        "rmse_e_mev_atom": 50.0,
        "group_mean_offset_rmse_mev_atom": 10.0,
        "group_mean_offsets_eV_per_atom": {
            "dissociation_id_1_neb1.0": 0.020,
            "dissociation_id_2_neb1.0": -0.010,
            "adsorption_id_3_neb1.0": 0.005,
        },
    }))
    second = tmp_path / "rel025_dft.json"
    second.write_text(json.dumps({
        "variant": "rel0p25",
        "rmse_e_mev_atom": 40.0,
        "group_mean_offset_rmse_mev_atom": 8.0,
        "group_mean_offsets_eV_per_atom": {
            "dissociation_id_1_neb1.0": 0.016,
            "adsorption_id_3_neb1.0": -0.004,
        },
    }))

    summary = summarize_stage165_energy_baseline_triage([first, second], top_k=2)
    assert summary["schema_version"] == "rtece_stage165_energy_baseline_triage.v1"
    assert summary["rows"][0]["case_offset_explained_raw_rmse_fraction"] == pytest.approx(0.96)
    assert summary["rows"][0]["max_case_offset_abs_mev_atom"] == pytest.approx(20.0)
    assert summary["rows"][0]["top_case_offsets"][0]["case_id"] == "dissociation_id_1_neb1.0"
    assert summary["rows"][0]["family_offset_summary"][0]["family"] == "dissociation"
    assert summary["rows"][1]["case_offset_explained_raw_rmse_fraction"] == pytest.approx(0.96)

    markdown = render_stage165_energy_baseline_markdown(summary)
    assert "Stage165 Energy Baseline Triage" in markdown
    assert "not a deployable correction" in markdown
    assert "dissociation_id_1_neb1.0" in markdown



def test_stage166_deployable_proxy_baseline_scores_feature_sets():
    from benchmarks.oc20neb_tace_mace.summarize_rtece_stage166_deployable_proxy_baseline import (
        fit_proxy_feature_sets,
        render_stage166_proxy_baseline_markdown,
    )

    case_features = {
        "case-a": {"natoms": 50.0, "frac_z6": 0.02, "volume_per_atom": 7.0},
        "case-b": {"natoms": 50.0, "frac_z6": 0.04, "volume_per_atom": 7.2},
        "case-c": {"natoms": 60.0, "frac_z6": 0.06, "volume_per_atom": 8.1},
        "case-d": {"natoms": 60.0, "frac_z6": 0.08, "volume_per_atom": 8.4},
    }
    offsets = {
        case_id: 100.0 * row["frac_z6"] + 2.0 * row["volume_per_atom"]
        for case_id, row in case_features.items()
    }
    summary = fit_proxy_feature_sets(
        case_offsets_mev_atom=offsets,
        case_features=case_features,
        feature_sets={
            "intercept_only": [],
            "composition": ["frac_z6"],
            "composition_cell": ["frac_z6", "volume_per_atom"],
        },
        ridge=1.0e-8,
    )

    assert summary["schema_version"] == "rtece_stage166_deployable_proxy_baseline.v1"
    assert summary["best_train_rmse_feature_set"] == "composition_cell"
    assert summary["rows"][0]["uses_case_id_as_feature"] is False
    best = {row["feature_set"]: row for row in summary["rows"]}["composition_cell"]
    assert best["train_rmse_mev_atom"] < 1.0e-3
    assert best["feature_count"] == 2

    markdown = render_stage166_proxy_baseline_markdown(summary)
    assert "Stage166 Deployable Proxy Baseline" in markdown
    assert "case_id is used only for grouping" in markdown
    assert "composition_cell" in markdown



def test_stage145_summary_keeps_rmse_first_and_missing_outputs_explicit(tmp_path):
    import json

    from benchmarks.oc20neb_tace_mace.summarize_community_baselines_stage145 import (
        render_stage145_markdown,
        summarize_stage145,
    )

    manifest = {
        "schema_version": "community_baselines_stage145.v1",
        "rows": [
            {"name": "nep4_mixed_smoke", "engine": "nep", "train_dir": str(tmp_path / "nep")},
            {"name": "deepmd_dpa_like_mixed_smoke", "engine": "deepmd", "train_dir": str(tmp_path / "dp")},
        ],
    }
    (tmp_path / "nep").mkdir()
    (tmp_path / "nep" / "conversion_summary.json").write_text(
        json.dumps({"engine": "nep", "num_configs": 2, "type_map": ["C", "N"]})
    )
    (tmp_path / "stage145_training_status.json").write_text(
        json.dumps(
            {
                "rows": [
                    {"name": "nep4_mixed_smoke", "training_status": "running", "latest_step": 200},
                    {"name": "deepmd_dpa_like_mixed_smoke", "training_status": "completed", "latest_step": 20},
                ]
            }
        )
    )

    summary = summarize_stage145(manifest, tmp_path)

    assert summary["schema_version"] == "community_baselines_stage145_results.v1"
    assert summary["primary_ranking_metric"] == "dft_f_rmse_mev_a"
    rows = {row["name"]: row for row in summary["rows"]}
    assert rows["nep4_mixed_smoke"]["conversion_status"] == "found"
    assert rows["deepmd_dpa_like_mixed_smoke"]["conversion_status"] == "missing"
    assert rows["deepmd_dpa_like_mixed_smoke"]["dft_benchmark_status"] == "missing"
    assert rows["nep4_mixed_smoke"]["training_status"] == "running"
    assert rows["deepmd_dpa_like_mixed_smoke"]["training_status"] == "completed"

    markdown = render_stage145_markdown(summary)
    assert "Primary ranking metric: DFT force RMSE" in markdown
    assert "| row | engine | conversion | training | DFT F RMSE | DFT E RMSE | rel image RMSE | barrier RMSE | case offset RMSE | first anchor RMSE | atoms/s | physical |" in markdown
    assert "deepmd_dpa_like_mixed_smoke" in markdown




def test_stage145_summary_preserves_relative_neb_energy_metrics(tmp_path):
    import json

    from benchmarks.oc20neb_tace_mace.summarize_community_baselines_stage145 import (
        render_stage145_markdown,
        summarize_stage145,
    )

    train_dir = tmp_path / "nep"
    train_dir.mkdir()
    (train_dir / "conversion_summary.json").write_text(json.dumps({"engine": "nep", "num_configs": 4}))
    (train_dir / "nep4_mixed_smoke_dft_benchmark.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "rmse_f_mev_a": 111.0,
                "rmse_e_mev_atom": 222.0,
                "relative_energy_errors_available": True,
                "relative_image_rmse_mev_atom": 8.5,
                "barrier_rmse_mev_atom": 19.25,
                "group_mean_offset_rmse_mev_atom": 6.75,
                "first_image_anchor_rmse_mev_atom": 17.5,
                "atoms_per_second": 12345.0,
            }
        )
    )
    manifest = {
        "schema_version": "community_baselines_stage145.v1",
        "rows": [{"name": "nep4_mixed_smoke", "engine": "nep", "train_dir": str(train_dir)}],
    }

    summary = summarize_stage145(manifest, tmp_path)
    row = summary["rows"][0]

    assert row["dft_relative_image_rmse_mev_atom"] == pytest.approx(8.5)
    assert row["dft_barrier_rmse_mev_atom"] == pytest.approx(19.25)
    assert row["dft_group_mean_offset_rmse_mev_atom"] == pytest.approx(6.75)
    assert row["dft_first_image_anchor_rmse_mev_atom"] == pytest.approx(17.5)
    markdown = render_stage145_markdown(summary)
    assert "rel image RMSE" in markdown
    assert "barrier RMSE" in markdown
    assert "case offset RMSE" in markdown
    assert "first anchor RMSE" in markdown
    assert "8.500" in markdown
    assert "19.250" in markdown
    assert "6.750" in markdown
    assert "17.500" in markdown



def test_stage162_energy_offset_summary_ranks_case_offsets(tmp_path):
    import json

    from benchmarks.oc20neb_tace_mace.summarize_rtece_energy_offsets import (
        render_energy_offset_markdown,
        summarize_energy_offset_benchmarks,
    )

    benchmark = tmp_path / "stage157_dft_benchmark.json"
    benchmark.write_text(
        json.dumps(
            {
                "rmse_e_mev_atom": 50.0,
                "mae_e_mev_atom": 35.0,
                "max_abs_e_mev_atom": 205.0,
                "group_mean_offset_rmse_mev_atom": 5.0,
                "first_image_anchor_rmse_mev_atom": 11.0,
                "relative_image_rmse_mev_atom": 7.0,
                "barrier_rmse_mev_atom": 12.0,
                "rmse_f_mev_a": 94.0,
                "mae_f_mev_a": 42.0,
                "max_abs_f_mev_a": 2090.0,
                "atoms_per_second": 500000.0,
                "group_mean_offsets_eV_per_atom": {
                    "case_a": -0.012,
                    "case_b": 0.105,
                    "case_c": -0.050,
                },
            }
        )
    )

    summary = summarize_energy_offset_benchmarks(
        [{"name": "stage157_rel0p25", "path": str(benchmark)}],
        top_k=2,
    )

    assert summary["schema_version"] == "rtece_energy_offset_summary.v1"
    row = summary["rows"][0]
    assert row["raw_e_rmse_mev_atom"] == pytest.approx(50.0)
    assert row["group_offset_e_rmse_mev_atom"] == pytest.approx(5.0)
    assert row["raw_rmse_removed_by_group_offset_fraction"] == pytest.approx(0.9)
    assert row["top_group_offsets_mev_atom"][0] == {
        "group": "case_b",
        "offset_mev_atom": pytest.approx(105.0),
        "abs_offset_mev_atom": pytest.approx(105.0),
    }
    assert row["top_group_offsets_mev_atom"][1]["group"] == "case_c"

    markdown = render_energy_offset_markdown(summary)
    assert "raw E RMSE" in markdown
    assert "group-offset E RMSE" in markdown
    assert "case_b:+105.000" in markdown
    assert "stage157_rel0p25" in markdown


def test_stage145_nep_calculator_adds_vacuum_cell_for_nonperiodic_dimer(tmp_path, monkeypatch):
    import numpy as np
    from ase import Atoms

    from benchmarks.oc20neb_tace_mace import physical_stage145_community as physical

    captured = {}

    def fake_run_nep_prediction(*, model_artifact, atoms_list, energy_key, forces_key, work_dir):
        captured["atoms"] = atoms_list[0].copy()
        return np.array([0.0], dtype=np.float64), np.zeros((len(atoms_list[0]), 3), dtype=np.float64), {}

    monkeypatch.setattr(physical, "run_nep_prediction", fake_run_nep_prediction)
    calc = physical.NEPPredictionCalculator(tmp_path / "nep.txt", run_dir=tmp_path / "work").calculator
    atoms = Atoms("CN", positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], pbc=False)
    atoms.calc = calc

    assert atoms.get_potential_energy() == pytest.approx(0.0)
    passed = captured["atoms"]
    assert passed.cell.volume > 0.0
    assert passed.pbc.all()


def test_stage145_nep_prediction_writer_reads_singlepoint_reference_keys(tmp_path):
    import numpy as np
    from ase import Atoms
    from ase.calculators.singlepoint import SinglePointCalculator

    from benchmarks.oc20neb_tace_mace.benchmark_stage145_community import _write_nep_prediction_xyz

    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-1.25,
        forces=np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=np.float64),
    )

    output = tmp_path / "train.xyz"
    _write_nep_prediction_xyz([atoms], output, energy_key="energy", forces_key="forces")

    text = output.read_text()
    assert "energy=-1.25" in text
    assert "Properties=species:S:1:pos:R:3:force:R:3" in text


def test_stage145_community_benchmark_emits_relative_energy_metrics(tmp_path, monkeypatch):
    import argparse
    import json
    import numpy as np
    import ase.io
    from ase import Atoms

    from benchmarks.oc20neb_tace_mace import benchmark_stage145_community as bench

    configs = tmp_path / "valid.extxyz"
    frames = []
    for image, energy in enumerate([0.0, 1.0, 3.0]):
        atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7 + 0.01 * image, 0.0, 0.0]])
        atoms.info["energy"] = energy
        atoms.info["case_id"] = "path-a"
        atoms.info["source_frame"] = image
        atoms.arrays["forces"] = np.zeros((2, 3), dtype=np.float64)
        frames.append(atoms)
    ase.io.write(str(configs), frames, format="extxyz")

    def fake_deepmd_ase(**kwargs):
        pred_e = np.array([5.0, 6.2, 8.0], dtype=np.float64)
        pred_f = np.zeros((6, 3), dtype=np.float64)
        return pred_e, pred_f, [0.01], {"engine_protocol": "fake_deepmd"}

    monkeypatch.setattr(bench, "run_deepmd_ase", fake_deepmd_ase)
    output = tmp_path / "benchmark.json"
    args = argparse.Namespace(
        engine="deepmd",
        model_artifact=tmp_path / "model.pb",
        configs=configs,
        output=output,
        row_name="deepmd_fake",
        energy_key="energy",
        forces_key="forces",
        start_config=0,
        limit_configs=3,
        warmup_passes=0,
        measure_passes=1,
        device="cpu",
    )

    payload = bench.run_benchmark(args)

    assert payload["schema_version"] == "community_baseline_dft_benchmark.v1"
    assert payload["relative_energy_metric_schema_version"] == "rtece_relative_neb_energy_metrics.v1"
    assert payload["relative_energy_errors_available"] is True
    assert payload["relative_image_rmse_mev_atom"] == pytest.approx((10000.0 / 3) ** 0.5)
    assert payload["barrier_rmse_mev_atom"] == pytest.approx(0.0)
    assert payload["energy_decomposition_metric_schema_version"] == "rtece_energy_error_decomposition.v1"
    assert payload["first_image_anchor_rmse_mev_atom"] == pytest.approx((10000.0 / 3) ** 0.5)
    saved = json.loads(output.read_text())
    assert saved["schema_version"] == "community_baseline_dft_benchmark.v1"
    assert saved["relative_energy_errors_available"] is True

def test_stage145_training_status_collects_deepmd_and_nep_progress(tmp_path):
    import json

    from benchmarks.oc20neb_tace_mace.collect_community_baselines_stage145 import (
        collect_stage145_training_status,
        render_training_status_markdown,
    )

    nep_dir = tmp_path / "nep"
    dp_dir = tmp_path / "dp"
    nep_dir.mkdir()
    dp_dir.mkdir()
    (nep_dir / "loss.out").write_text(
        "100 0.7 0.03 0.04 0.20 0.50 0 0 0 0\n"
        "200 0.6 0.04 0.05 0.18 0.40 0 0 0 0\n"
    )
    (dp_dir / "lcurve.out").write_text(
        "# step rmse_val rmse_trn rmse_e_val rmse_e_trn rmse_f_val rmse_f_trn lr\n"
        "1 1.0 2.0 0.3 0.4 0.5 0.6 1.0e-3\n"
        "20 0.2 0.3 0.04 0.05 0.06 0.07 1.0e-8\n"
    )
    (dp_dir / "frozen_model.pth").write_text("fake")

    manifest = {
        "rows": [
            {"name": "nep4_mixed_smoke", "engine": "nep", "train_dir": str(nep_dir), "generation": 20000},
            {"name": "deepmd_dpa_like_mixed_smoke", "engine": "deepmd", "train_dir": str(dp_dir), "stop_batch": 20},
        ]
    }

    status = collect_stage145_training_status(
        manifest,
        tmp_path,
        job_states={
            "nep4_mixed_smoke": {"state": "RUNNING", "job_id": "1"},
            "deepmd_dpa_like_mixed_smoke": {"state": "COMPLETED", "job_id": "2"},
        },
    )

    rows = {row["name"]: row for row in status["rows"]}
    assert rows["nep4_mixed_smoke"]["training_status"] == "running"
    assert rows["nep4_mixed_smoke"]["latest_step"] == 200
    assert rows["nep4_mixed_smoke"]["latest_rmse_f_train"] == 0.40
    assert rows["deepmd_dpa_like_mixed_smoke"]["training_status"] == "completed"
    assert rows["deepmd_dpa_like_mixed_smoke"]["latest_step"] == 20
    assert rows["deepmd_dpa_like_mixed_smoke"]["latest_rmse_f_val"] == 0.06
    assert (tmp_path / "stage145_training_status.json").exists()
    markdown = render_training_status_markdown(status)
    assert "| row | engine | training | job | latest step | F RMSE val | F RMSE train |" in markdown
    assert "deepmd_dpa_like_mixed_smoke" in markdown


def test_rtece_stage136_l2_projection_diagnostic_manifest_materializes_no_export_wrapper(tmp_path):
    from pathlib import Path

    from benchmarks.oc20neb_tace_mace.make_rtece_stage136_l2_projection_diagnostic import (
        audit_stage136_manifest,
        make_stage136_manifest,
        materialize_stage136,
    )

    payload = make_stage136_manifest(
        output_root=tmp_path / "stage136",
        train_configs="stage132_train.extxyz",
        valid_configs="dft_valid.extxyz",
        limit_configs=64,
        energy_eval_stride=4,
        force_eval_stride=4,
    )

    assert payload["schema_version"] == "rtece_stage136_l2_projection_diagnostic.v1"
    assert payload["stage"] == "stage136_l2_projection_diagnostic"
    assert payload["diagnostic_semantics"] == "stage135_l2_reference_vs_stage132_l1_projection_residual"
    assert payload["reference_path_ids"] == [
        "atomic.radial_density",
        "atomic.species_basis_density",
        "atomic.vector_norm",
        "atomic.vector_cross_radial_dot",
        "atomic.quadrupole_norm",
        "atomic.quadrupole_cross_radial_frobenius",
    ]
    assert [candidate["name"] for candidate in payload["candidates"]] == [
        "full_l2_reference",
        "stage132_l1_atomic",
        "drop_quadrupole_cross",
        "drop_quadrupole_norm",
    ]
    l1 = next(candidate for candidate in payload["candidates"] if candidate["name"] == "stage132_l1_atomic")
    assert "atomic.quadrupole_norm" not in l1["path_ids"]
    assert "atomic.quadrupole_cross_radial_frobenius" not in l1["path_ids"]
    assert payload["num_radial"] == 12
    assert payload["species_basis_channels"] == 24
    assert payload["atomic_cross_radial_sketch_channels"] == 3
    assert payload["limit_configs"] == 64
    assert payload["energy_target_key"] == "energy"
    assert payload["force_target_key"] == "forces"
    assert "projection error" in payload["comparison_question"]
    assert "Stage135" in payload["comparison_question"]

    audit = audit_stage136_manifest(payload)
    assert audit["contract_pass"] is True
    assert audit["failed_checks"] == []

    materialized = materialize_stage136(payload)
    wrapper = Path(materialized["wrapper"])
    text = wrapper.read_text()
    assert "--export" not in text
    assert "--mem" not in text
    assert "--cpus-per-task" not in text
    assert "analyze_rtece_projection_error.py" in text
    assert "TRAIN_CONFIGS=stage132_train.extxyz" in text
    assert "VALID_CONFIGS=dft_valid.extxyz" in text
    assert "LIMIT_CONFIGS=64" in text
    assert "--reference-path-ids" in text
    assert "atomic.quadrupole_norm" in text
    assert "atomic.quadrupole_cross_radial_frobenius" in text
    assert "--candidate stage132_l1_atomic:atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,atomic.vector_cross_radial_dot" in text
    assert "--energy-target-key energy" in text
    assert "--force-target-key forces" in text
    assert "--energy-eval-stride 4" in text
    assert "--force-eval-stride 4" in text


def test_rtece_stage135_l2_atomic_projection_manifest_materializes_no_export_row(tmp_path):
    from pathlib import Path

    from benchmarks.oc20neb_tace_mace.make_rtece_stage135_l2_atomic_projection import (
        audit_stage135_manifest,
        make_stage135_manifest,
        materialize_stage135,
    )

    payload = make_stage135_manifest(
        output_root=tmp_path / "stage135",
        train_file="stage132_augmented.extxyz",
        train_valid_file="train_valid.extxyz",
        dft_valid_file="dft_valid.extxyz",
        teacher_valid_file="teacher_valid.extxyz",
        valid_limit_configs=256,
        bench_limit_configs=1024,
        max_steps=20000,
        lr_warmup_steps=500,
        early_stopping_patience=400,
    )

    assert payload["schema_version"] == "rtece_stage135_l2_atomic_projection.v1"
    assert payload["stage"] == "stage135_l2_atomic_projection"
    assert payload["row_set"] == "stage135-l2-atomic-projection"
    assert payload["distillation_semantics"] == "fixed_stage132_broad_teacher_rattle_train_l2_atomic_projection"
    assert "L_A=2" in payload["comparison_question"]
    assert "projection error" in payload["comparison_question"]
    assert payload["train_file"] == "stage132_augmented.extxyz"
    assert [row["variant"] for row in payload["rows"]] == [
        "l2_active_nrad12_species24_radial_species8_cross3_h64",
    ]
    row = payload["rows"][0]
    path_ids = tuple(part.strip() for part in row["scalar_path_ids"].split(",") if part.strip())
    assert row["moment_l_max"] == 2
    assert row["hidden_channels"] == "64,64"
    assert row["num_radial"] == 12
    assert row["species_basis_channels"] == 24
    assert row["radial_species_adapter_channels"] == 8
    assert row["atomic_cross_radial_sketch_channels"] == 3
    assert row["short_range_repulsion_potential"] == "zbl"
    assert "atomic.quadrupole_norm" in path_ids
    assert "atomic.quadrupole_cross_radial_frobenius" in path_ids
    assert not any(path_id.startswith("edge.") for path_id in path_ids)

    audit = audit_stage135_manifest(payload)
    assert audit["contract_pass"] is True
    assert audit["failed_checks"] == []

    materialized = materialize_stage135(payload)
    assert len(materialized["train_wrappers"]) == 1
    assert len(materialized["physical_wrappers"]) == 1
    train_text = Path(materialized["train_wrappers"][0]).read_text()
    physical_text = Path(materialized["physical_wrappers"][0]).read_text()
    combined = train_text + "\n" + physical_text
    assert "--export" not in combined
    assert "--mem" not in combined
    assert "--cpus-per-task" not in combined
    assert "TRAIN_FILE=stage132_augmented.extxyz" in train_text
    assert "MAX_STEPS=20000" in train_text
    assert "LR_WARMUP_STEPS=500" in train_text
    assert "EARLY_STOPPING_PATIENCE=400" in train_text
    assert "MOMENT_L_MAX=2" in train_text
    assert "atomic.quadrupole_norm" in train_text
    assert "atomic.quadrupole_cross_radial_frobenius" in train_text
    assert "edge.cavity.vector_dot" not in train_text
    assert "rattle_relax_rtece.py" in physical_text


def test_rtece_stage137_l2_conditioned_front_manifest_materializes_no_export_rows(tmp_path):
    from pathlib import Path

    from benchmarks.oc20neb_tace_mace.make_rtece_stage137_l2_conditioned_front import (
        audit_stage137_manifest,
        make_stage137_manifest,
        materialize_stage137,
    )

    payload = make_stage137_manifest(
        output_root=tmp_path / "stage137",
        train_file="stage132_augmented.extxyz",
        train_valid_file="train_valid.extxyz",
        dft_valid_file="dft_valid.extxyz",
        teacher_valid_file="teacher_valid.extxyz",
        valid_limit_configs=256,
        bench_limit_configs=1024,
        max_steps=20000,
        lr_warmup_steps=500,
        early_stopping_patience=400,
    )

    assert payload["schema_version"] == "rtece_stage137_l2_conditioned_front.v1"
    assert payload["stage"] == "stage137_l2_conditioned_front"
    assert payload["row_set"] == "stage137-l2-conditioned-front"
    assert payload["distillation_semantics"] == "stage136_force_projection_guided_l2_low_rank_conditioning"
    assert "Stage136" in payload["comparison_question"]
    assert "force projection" in payload["comparison_question"]
    assert payload["train_file"] == "stage132_augmented.extxyz"
    assert [row["variant"] for row in payload["rows"]] == [
        "l2_active_nrad12_species24_radial_species8_cross3_bneck32_h64",
        "l2_active_nrad12_species24_radial_species8_cross3_cond32_h64",
    ]
    for row in payload["rows"]:
        path_ids = tuple(part.strip() for part in row["scalar_path_ids"].split(",") if part.strip())
        assert row["moment_l_max"] == 2
        assert row["hidden_channels"] == "64,64"
        assert row["num_radial"] == 12
        assert row["species_basis_channels"] == 24
        assert row["radial_species_adapter_channels"] == 8
        assert row["atomic_cross_radial_sketch_channels"] == 3
        assert row["short_range_repulsion_potential"] == "zbl"
        assert "atomic.quadrupole_norm" in path_ids
        assert "atomic.quadrupole_cross_radial_frobenius" in path_ids
        assert not any(path_id.startswith("edge.") for path_id in path_ids)
    bneck = payload["rows"][0]
    cond = payload["rows"][1]
    assert bneck["descriptor_bottleneck_dim"] == 32
    assert bneck["descriptor_conditioner"] == "none"
    assert cond["descriptor_bottleneck_dim"] == 0
    assert cond["descriptor_conditioner"] == "residual_mlp"
    assert cond["descriptor_conditioner_hidden_channels"] == 32

    audit = audit_stage137_manifest(payload)
    assert audit["contract_pass"] is True
    assert audit["failed_checks"] == []

    materialized = materialize_stage137(payload)
    assert len(materialized["train_wrappers"]) == 2
    assert len(materialized["physical_wrappers"]) == 2
    combined = "\n".join(Path(path).read_text() for path in [*materialized["train_wrappers"], *materialized["physical_wrappers"]])
    assert "--export" not in combined
    assert "--mem" not in combined
    assert "--cpus-per-task" not in combined
    assert "TRAIN_FILE=stage132_augmented.extxyz" in combined
    assert "MAX_STEPS=20000" in combined
    assert "LR_WARMUP_STEPS=500" in combined
    assert "EARLY_STOPPING_PATIENCE=400" in combined
    assert "MOMENT_L_MAX=2" in combined
    assert "DESCRIPTOR_BOTTLENECK_DIM=32" in combined
    assert "DESCRIPTOR_CONDITIONER=residual_mlp" in combined
    assert "DESCRIPTOR_CONDITIONER_HIDDEN_CHANNELS=32" in combined
    assert "atomic.quadrupole_norm" in combined
    assert "atomic.quadrupole_cross_radial_frobenius" in combined
    assert "edge.cavity.vector_dot" not in combined
    assert "rattle_relax_rtece.py" in combined


def test_rtece_stage128_physical_triage_writes_no_export_wrappers(tmp_path):
    from pathlib import Path

    from benchmarks.oc20neb_tace_mace.make_rtece_physical_triage import (
        stage128_physical_triage_cases,
        write_physical_triage_wrappers,
    )

    cases = stage128_physical_triage_cases(stage127_root="/tmp/stage127")

    assert [case["variant"] for case in cases] == [
        "l1_active_nrad12_species20_radial_species8_cross3_h64",
        "l1_active_nrad12_species24_radial_species8_cross3_h64",
        "l1_active_nrad12_species24_radial_species8_cross4_h64",
    ]
    assert not any("cross2" in case["variant"] for case in cases)
    assert all(case["stage127_source"].endswith("stage127_interpretation.md") for case in cases)

    index = write_physical_triage_wrappers(
        tmp_path / "wrappers",
        run_root="/tmp/stage128",
        configs="/tmp/valid.extxyz",
        cases=cases,
        dimer_pairs=("C-N", "C-O", "C-H", "N-H", "O-H", "C-C", "N-N"),
        dimer_num_points=24,
        rattle_start_config=58,
        rattle_limit_configs=8,
        rattle_max_steps=10,
    )

    assert index["schema_version"] == "rtece_physical_triage.v1"
    assert index["stage"] == "stage128_physical_triage"
    assert index["rattle_focus_label"] == "C_or_N"
    assert index["stage127_source"].endswith("stage127_interpretation.md")
    assert len(index["cases"]) == 3

    for case in index["cases"]:
        wrapper = Path(case["wrapper"])
        text = wrapper.read_text(encoding="utf-8")
        assert "--export" not in text
        assert "--mem" not in text
        assert "--cpus-per-task" not in text
        assert "dimer_scan_rtece.py" in text
        assert "rattle_relax_rtece.py" in text
        assert "summarize_rtece_physical_pareto.py" in text
        assert "RATTLE_START_CONFIG=58" in text
        assert "RATTLE_LIMIT_CONFIGS=8" in text
        assert "RATTLE_MAX_STEPS=10" in text
        assert "C-N C-O C-H N-H O-H C-C N-N" in text


def test_rtece_physical_triage_accepts_custom_stage_cases(tmp_path):
    from pathlib import Path

    from benchmarks.oc20neb_tace_mace.make_rtece_physical_triage import (
        parse_case_spec,
        write_physical_triage_wrappers,
    )

    case = parse_case_spec(
        "stage129_variant:/tmp/model.pt:/tmp/dft.json:/tmp/teacher.json"
    )
    index = write_physical_triage_wrappers(
        tmp_path / "wrappers",
        run_root=str(tmp_path / "stage129-phys"),
        configs="/tmp/valid.extxyz",
        cases=[case],
        stage="stage129_physical_triage",
        source="runs/oc20neb_tace_mace/rtece-stage129-teacher-rattle-distill/stage129_interpretation.md",
        design_basis="stage129 teacher-rattle physical robustness triage",
        job_name="rtece-phys129",
    )

    assert index["stage"] == "stage129_physical_triage"
    assert index["source"].endswith("stage129_interpretation.md")
    assert index["job_name"] == "rtece-phys129"
    assert index["cases"][0]["checkpoint"] == "/tmp/model.pt"

    wrapper = Path(index["cases"][0]["wrapper"])
    text = wrapper.read_text(encoding="utf-8")
    assert "#SBATCH --job-name=rtece-phys129" in text
    assert "/home/gengjianrui/bin/logs/rtece-phys129-%j.out" in text
    assert "CHECKPOINT=/tmp/model.pt" in text
    assert "DFT_BENCHMARK=/tmp/dft.json" in text
    assert "TEACHER_BENCHMARK=/tmp/teacher.json" in text
    assert "--export" not in text
    assert "--mem" not in text
    assert "--cpus-per-task" not in text


def test_rtece_stage121_active_frontloaded_sweep_cli_generates_four_rows(tmp_path):
    import json
    import subprocess
    import sys

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    train = tmp_path / "train.extxyz"
    valid = tmp_path / "valid.extxyz"
    teacher = tmp_path / "teacher.extxyz"
    dft = tmp_path / "dft.extxyz"
    for path in (train, valid, teacher, dft):
        path.write_text("", encoding="utf-8")
    out = tmp_path / "wrappers"
    subprocess.run(
        [
            sys.executable,
            str(root / "benchmarks/oc20neb_tace_mace/make_rtece_pareto_sweep.py"),
            "--output-dir",
            str(out),
            "--run-root",
            str(tmp_path / "runs"),
            "--train-file",
            str(train),
            "--train-valid-file",
            str(valid),
            "--dft-valid-file",
            str(dft),
            "--teacher-valid-file",
            str(teacher),
            "--row-set",
            "active-frontloaded-stage121",
        ],
        check=True,
        cwd=root,
    )
    index = json.loads((out / "rtece_pareto_sweep_index.json").read_text(encoding="utf-8"))
    assert index["row_set"] == "active-frontloaded-stage121"
    assert [row["name"] for row in index["rows"]] == [
        "l1_active_species16_bneck16_h64",
        "l1_active_species16_bneck32_h64",
        "l2_active_species16_bneck16_h64",
        "l2_active_species16_bneck32_h64",
    ]
    for row in index["rows"]:
        text = __import__("pathlib").Path(row["wrapper"]).read_text(encoding="utf-8")
        assert "DESCRIPTOR_BOTTLENECK_DIM=" in text
        assert "SPECIES_BASIS_MODE=learnable_embedding" in text
        assert "--export" not in text
        assert "--mem" not in text
        assert "--cpus-per-task" not in text


def test_rtece_pareto_sweep_preflight_reports_malformed_extxyz(tmp_path):
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import preflight_extxyz_file

    malformed = tmp_path / "malformed.extxyz"
    malformed.write_text(
        "2\nProperties=species:S:1:pos:R:3\nH 0 0 0\nH 0 0 1\n"
        "3\nProperties=species:S:1:pos:R:3\nH 0 0 0\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="malformed.extxyz.*readable frames=1"):
        preflight_extxyz_file(malformed, limit_configs=2)


def test_rtece_stage115_ef_active_sweep_writes_selected_wrappers_without_sbatch_export(tmp_path):
    from pathlib import Path
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import stage115_ef_active_rows, write_pareto_sweep

    index = write_pareto_sweep(
        tmp_path,
        run_root="/tmp/rtece-stage115",
        train_file="/tmp/train.extxyz",
        train_valid_file="/tmp/train_valid.extxyz",
        dft_valid_file="/tmp/dft_valid.extxyz",
        teacher_valid_file="/tmp/teacher_valid.extxyz",
        rows=stage115_ef_active_rows(),
        limit_configs=2048,
        valid_limit_configs=256,
        bench_limit_configs=1024,
        max_steps=20000,
        batch_size=8,
        valid_batch_size=16,
        early_stopping_patience=400,
        lr_warmup_steps=500,
    )

    assert (tmp_path / "rtece_pareto_sweep_index.json").exists()
    assert index["row_set"] == "custom"
    assert [row["name"] for row in index["rows"]] == ["l0_radial", "l1_cross_k2", "l2_atomic_no_edge_k2"]
    l2 = next(row for row in index["rows"] if row["name"] == "l2_atomic_no_edge_k2")
    text = Path(l2["wrapper"]).read_text()
    assert "--export" not in text
    assert "--mem" not in text
    assert "--cpus-per-task" not in text
    assert "RUN_ROOT=/tmp/rtece-stage115/l2_atomic_no_edge_k2" in text
    assert (
        "SCALAR_PATH_IDS=atomic.radial_density,atomic.vector_norm,atomic.vector_cross_radial_dot,"
        "atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius"
    ) in text
    assert "MOMENT_L_MAX=2" in text
    assert "ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS=2" in text
    assert "SHORT_RANGE_REPULSION_POTENTIAL=zbl" in text
    assert "TRAINER_BACKEND=lightning" in text
    assert "LR_WARMUP_STEPS=500" in text


def test_rtece_pareto_sweep_writes_wrappers_without_sbatch_export(tmp_path):
    from pathlib import Path
    from benchmarks.oc20neb_tace_mace.make_rtece_pareto_sweep import write_pareto_sweep

    index = write_pareto_sweep(
        tmp_path,
        run_root="/tmp/rtece-stage96",
        train_file="/tmp/train.extxyz",
        train_valid_file="/tmp/train_valid.extxyz",
        dft_valid_file="/tmp/dft_valid.extxyz",
        teacher_valid_file="/tmp/teacher_valid.extxyz",
        limit_configs=2048,
        valid_limit_configs=256,
        bench_limit_configs=1024,
        max_steps=20000,
        batch_size=8,
        valid_batch_size=16,
        early_stopping_patience=400,
    )

    assert (tmp_path / "rtece_pareto_sweep_index.json").exists()
    assert len(index["rows"]) == 10
    first = index["rows"][0]
    wrapper = Path(first["wrapper"])
    text = wrapper.read_text()
    assert "--export" not in text
    assert "--mem" not in text
    assert "--cpus-per-task" not in text
    assert "RUN_ROOT=/tmp/rtece-stage96/l0_scalar_fixed" in text
    assert "MOMENT_L_MAX=0" in text
    assert "MAX_STEPS=20000" in text
    zbl_row = next(row for row in index["rows"] if row["name"] == "l2_cavity_vector_quad_learnable_zbl")
    zbl_text = Path(zbl_row["wrapper"]).read_text()
    assert "SCALAR_PATH_IDS=atomic.radial_density,edge.cavity.vector_dot,edge.cavity.quadrupole_frobenius,edge.direct.radial" in zbl_text
    assert "MOMENT_L_MAX=2" in zbl_text
    assert "LEARNABLE_RADIAL_MIXING=1" in zbl_text
    assert "SHORT_RANGE_REPULSION_POTENTIAL=zbl" in zbl_text


def test_rtece_pareto_sweep_cli_runs_from_repo_script_path(tmp_path):
    from pathlib import Path

    script = "benchmarks/oc20neb_tace_mace/make_rtece_pareto_sweep.py"
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--output-dir",
            str(tmp_path / "wrappers"),
            "--run-root",
            "/tmp/rtece-stage96",
            "--train-file",
            "train.extxyz",
            "--train-valid-file",
            "teacher-valid.extxyz",
            "--dft-valid-file",
            "dft-valid.extxyz",
            "--teacher-valid-file",
            "teacher-valid.extxyz",
            "--limit-configs",
            "8",
            "--valid-limit-configs",
            "4",
            "--bench-limit-configs",
            "4",
            "--max-steps",
            "10",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    stdout = json.loads(result.stdout)
    assert stdout["rows"] == 10
    assert stdout["row_set"] == "design-space-default"
    assert (tmp_path / "wrappers" / "rtece_pareto_sweep_index.json").exists()


def test_rtece_stage115_ef_active_sweep_cli_generates_three_rows(tmp_path):
    from pathlib import Path

    script = "benchmarks/oc20neb_tace_mace/make_rtece_pareto_sweep.py"
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--output-dir",
            str(tmp_path / "wrappers"),
            "--run-root",
            "/tmp/rtece-stage115",
            "--train-file",
            "train.extxyz",
            "--train-valid-file",
            "teacher-valid.extxyz",
            "--dft-valid-file",
            "dft-valid.extxyz",
            "--teacher-valid-file",
            "teacher-valid.extxyz",
            "--row-set",
            "ef-active-stage115",
            "--limit-configs",
            "8",
            "--valid-limit-configs",
            "4",
            "--bench-limit-configs",
            "4",
            "--max-steps",
            "10",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    stdout = json.loads(result.stdout)
    index = json.loads((tmp_path / "wrappers" / "rtece_pareto_sweep_index.json").read_text())
    assert stdout["rows"] == 3
    assert stdout["row_set"] == "ef-active-stage115"
    assert index["row_set"] == "ef-active-stage115"
    assert [row["name"] for row in index["rows"]] == ["l0_radial", "l1_cross_k2", "l2_atomic_no_edge_k2"]


def test_rtece_wrapper_contract_audit_reports_training_and_slurm_readiness(tmp_path):
    from benchmarks.oc20neb_tace_mace.audit_rtece_wrapper_contract import audit_wrapper_index

    wrapper = tmp_path / "row" / "rtece_scalar_matrix_no_export.sbatch"
    wrapper.parent.mkdir()
    wrapper.write_text(
        """#!/bin/bash
#SBATCH --job-name=rtece
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
export VARIANTS=row
export MAX_STEPS=20000
export BATCH_SIZE=8
export VALID_BATCH_SIZE=16
export LR_SCHEDULER=plateau
export EARLY_STOPPING_PATIENCE=400
export DFT_VALID_FILE=runs/oc20neb_tace_mace/tece-distill-20260717/mixed_valid_tw0.75_regen.extxyz
export SHORT_RANGE_REPULSION_POTENTIAL=zbl
exec /bin/bash rtece_scalar_matrix.sbatch
""",
        encoding="utf-8",
    )
    bad = tmp_path / "bad" / "rtece_scalar_matrix_no_export.sbatch"
    bad.parent.mkdir()
    bad.write_text(
        """#!/bin/bash
#SBATCH --job-name=rtece
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
export VARIANTS=bad
export MAX_STEPS=1000
export BATCH_SIZE=1
exec /bin/bash rtece_scalar_matrix.sbatch
""",
        encoding="utf-8",
    )
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": "rtece_pareto_sweep.v1",
                "row_set": "contract-stage",
                "rows": [
                    {"name": "row", "wrapper": str(wrapper)},
                    {"name": "bad", "wrapper": str(bad)},
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = audit_wrapper_index(
        index,
        expected_exports={
            "MAX_STEPS": "20000",
            "BATCH_SIZE": "8",
            "VALID_BATCH_SIZE": "16",
            "LR_SCHEDULER": "plateau",
            "EARLY_STOPPING_PATIENCE": "400",
            "DFT_VALID_FILE": "runs/oc20neb_tace_mace/tece-distill-20260717/mixed_valid_tw0.75_regen.extxyz",
            "SHORT_RANGE_REPULSION_POTENTIAL": "zbl",
        },
    )

    assert payload["schema_version"] == "rtece_wrapper_contract_audit.v1"
    assert payload["row_set"] == "contract-stage"
    assert payload["contract_pass"] is False
    rows = {row["name"]: row for row in payload["rows"]}
    assert rows["row"]["contract_pass"] is True
    assert rows["row"]["forbidden_sbatch_options"] == []
    assert rows["bad"]["contract_pass"] is False
    assert "--mem" in rows["bad"]["forbidden_sbatch_options"]
    assert rows["bad"]["mismatched_exports"]["MAX_STEPS"] == {"expected": "20000", "actual": "1000"}
    assert rows["bad"]["missing_exports"] == [
        "DFT_VALID_FILE",
        "EARLY_STOPPING_PATIENCE",
        "LR_SCHEDULER",
        "SHORT_RANGE_REPULSION_POTENTIAL",
        "VALID_BATCH_SIZE",
    ]
    assert payload["failed_rows"] == ["bad"]


def test_rtece_stage116_capacity_ladder_sweep_cli_generates_six_rows(tmp_path):
    from pathlib import Path

    script = "benchmarks/oc20neb_tace_mace/make_rtece_pareto_sweep.py"
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--output-dir",
            str(tmp_path / "wrappers"),
            "--run-root",
            "/tmp/rtece-stage116",
            "--train-file",
            "train.extxyz",
            "--train-valid-file",
            "teacher-valid.extxyz",
            "--dft-valid-file",
            "dft-valid.extxyz",
            "--teacher-valid-file",
            "teacher-valid.extxyz",
            "--row-set",
            "capacity-ladder-stage116",
            "--limit-configs",
            "8",
            "--valid-limit-configs",
            "4",
            "--bench-limit-configs",
            "4",
            "--max-steps",
            "10",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    stdout = json.loads(result.stdout)
    index = json.loads((tmp_path / "wrappers" / "rtece_pareto_sweep_index.json").read_text())
    assert stdout["rows"] == 6
    assert stdout["row_set"] == "capacity-ladder-stage116"
    assert index["row_set"] == "capacity-ladder-stage116"
    assert [row["hidden_channels"] for row in index["rows"]] == [
        "64,64",
        "128,128",
        "128,128,128",
        "64,64",
        "128,128",
        "128,128,128",
    ]


def test_rtece_stage118_representation_ladder_sweep_cli_generates_six_contract_rows(tmp_path):
    from pathlib import Path

    script = "benchmarks/oc20neb_tace_mace/make_rtece_pareto_sweep.py"
    result = subprocess.run(
        [
            sys.executable,
            script,
            "--output-dir",
            str(tmp_path / "wrappers"),
            "--run-root",
            "/tmp/rtece-stage118",
            "--train-file",
            "train.extxyz",
            "--train-valid-file",
            "teacher-valid.extxyz",
            "--dft-valid-file",
            "dft-valid.extxyz",
            "--teacher-valid-file",
            "teacher-valid.extxyz",
            "--row-set",
            "representation-ladder-stage118",
            "--limit-configs",
            "8",
            "--valid-limit-configs",
            "4",
            "--bench-limit-configs",
            "4",
            "--max-steps",
            "10",
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
    )

    stdout = json.loads(result.stdout)
    index = json.loads((tmp_path / "wrappers" / "rtece_pareto_sweep_index.json").read_text())
    assert stdout["rows"] == 6
    assert stdout["row_set"] == "representation-ladder-stage118"
    assert index["row_set"] == "representation-ladder-stage118"
    by_name = {row["name"]: row for row in index["rows"]}
    assert by_name["l2_cavity_radial_edge_h128"]["train_variant"] == "rtece_cavity_radial_edge_sketch14"
    radial_text = Path(by_name["l2_cavity_radial_edge_h128"]["wrapper"]).read_text()
    assert "VARIANTS=rtece_cavity_radial_edge_sketch14" in radial_text
    assert "SCALAR_PATH_IDS=" not in radial_text
    conditioned_text = Path(by_name["l2_conditioned_cavity_edge_h128"]["wrapper"]).read_text()
    species_text = Path(by_name["l2_species_cavity_edge_h128"]["wrapper"]).read_text()
    atomic_text = Path(by_name["l2_atomic_cross_h128"]["wrapper"]).read_text()
    for text in (radial_text, conditioned_text, species_text, atomic_text):
        assert "--export" not in text
        assert "--mem" not in text
        assert "--cpus-per-task" not in text
        assert "TRAINER_BACKEND=lightning" in text
        assert "BATCH_SIZE=8" in text
        assert "VALID_BATCH_SIZE=16" in text
        assert "LR_WARMUP_STEPS=500" in text
        assert "SHORT_RANGE_REPULSION_POTENTIAL=zbl" in text
        assert "LEARNABLE_RADIAL_MIXING=1" in text
        assert "HIDDEN_CHANNELS=128,128" in text
    assert "DESCRIPTOR_CONDITIONER=residual_mlp" in conditioned_text
    assert "DESCRIPTOR_CONDITIONER_HIDDEN_CHANNELS=64" in conditioned_text
    assert "SPECIES_BASIS_CHANNELS=4" in species_text
    assert "ATOMIC_CROSS_RADIAL_PROJECTION=learnable" in atomic_text


def test_rtece_matrix_submit_helper_generates_wrapper_without_sbatch_export(tmp_path):
    from benchmarks.oc20neb_tace_mace.submit_rtece_scalar_matrix import (
        build_sbatch_command,
        write_rtece_matrix_wrapper,
    )

    wrapper = write_rtece_matrix_wrapper(
        tmp_path,
        variants="rtece_species_basis4 rtece_cavity_radial_edge_sketch14",
        run_root="/tmp/rtece-stage63",
        train_file="/tmp/train.extxyz",
        train_valid_file="/tmp/valid.extxyz",
        dft_valid_file="/tmp/dft.extxyz",
        teacher_valid_file="/tmp/teacher.extxyz",
        limit_configs=16,
        valid_limit_configs=8,
        bench_limit_configs=32,
        max_steps=4,
        hidden_channels="16,16",
        num_radial=4,
        moment_l_max=2,
        scalar_path_ids="atomic.radial_density,edge.cavity.vector_dot",
        species_basis_channels=4,
        species_basis_mode="learnable_embedding",
        atomic_cross_radial_sketch_channels=3,
        atomic_cross_radial_projection="learnable",
        force_weight=30.0,
        force_focus_elements="C,N",
        force_focus_weight=4.0,
        force_mode="autograd",
        measure_passes=1,
        default_dtype="float32",
        trainer_backend="lightning",
        batch_size=2,
        valid_batch_size=4,
        lr_scheduler="plateau",
        lr_patience=5,
        lr_factor=0.25,
        early_stopping_patience=8,
        lr_warmup_steps=3,
        gradient_clip_val=1.0,
    )
    command = build_sbatch_command(wrapper)
    text = wrapper.read_text()

    assert "--export" not in command
    assert "--export" not in text
    assert "#SBATCH --gpus-per-node=1" in text
    assert "#SBATCH --qos=flood-1o2gpu" in text
    assert "--mem" not in text
    assert "--cpus-per-task" not in text
    assert "VARIANTS='rtece_species_basis4 rtece_cavity_radial_edge_sketch14'" in text
    assert "TRAIN_FILE=/tmp/train.extxyz" in text
    assert "BENCH_LIMIT_CONFIGS=32" in text
    assert "SCALAR_PATH_IDS=atomic.radial_density,edge.cavity.vector_dot" in text
    assert "MOMENT_L_MAX=2" in text
    assert "SPECIES_BASIS_CHANNELS=4" in text
    assert "SPECIES_BASIS_MODE=learnable_embedding" in text
    assert "ATOMIC_CROSS_RADIAL_SKETCH_CHANNELS=3" in text
    assert "ATOMIC_CROSS_RADIAL_PROJECTION=learnable" in text
    assert "FORCE_WEIGHT=30.0" in text
    assert "FORCE_FOCUS_ELEMENTS=C,N" in text
    assert "FORCE_FOCUS_WEIGHT=4.0" in text
    assert "TRAINER_BACKEND=lightning" in text
    assert "BATCH_SIZE=2" in text
    assert "VALID_BATCH_SIZE=4" in text
    assert "LR_SCHEDULER=plateau" in text
    assert "LR_PATIENCE=5" in text
    assert "LR_FACTOR=0.25" in text
    assert "EARLY_STOPPING_PATIENCE=8" in text
    assert "LR_WARMUP_STEPS=3" in text
    assert "GRADIENT_CLIP_VAL=1.0" in text
    assert "exec /bin/bash" in text
    assert "rtece_scalar_matrix.sbatch" in text


def test_rtece_matrix_submit_helper_forwards_atomic_cross_radial_projection_file_without_sbatch_export(tmp_path):
    from benchmarks.oc20neb_tace_mace.submit_rtece_scalar_matrix import write_rtece_matrix_wrapper

    wrapper = write_rtece_matrix_wrapper(
        tmp_path,
        variants="l1_cross_podproj",
        run_root="/tmp/rtece-stage104",
        atomic_cross_radial_projection="pod_fixed",
        atomic_cross_radial_projection_file="/tmp/radial_pod.json",
    )
    text = wrapper.read_text()

    assert "--export" not in text
    assert "ATOMIC_CROSS_RADIAL_PROJECTION=pod_fixed" in text
    assert "ATOMIC_CROSS_RADIAL_PROJECTION_FILE=/tmp/radial_pod.json" in text


def test_rtece_matrix_submit_helper_forwards_min_eval_step_without_sbatch_export(tmp_path):
    from benchmarks.oc20neb_tace_mace.submit_rtece_scalar_matrix import write_rtece_matrix_wrapper

    wrapper = write_rtece_matrix_wrapper(
        tmp_path,
        variants="element_core_train2048",
        run_root="/tmp/rtece-stage94",
        eval_interval=256,
        min_eval_step=2048,
        checkpoint_name="rtece_scalar_best.pt",
    )
    text = wrapper.read_text()

    assert "--export" not in text
    assert "EVAL_INTERVAL=256" in text
    assert "MIN_EVAL_STEP=2048" in text
    assert "CHECKPOINT_NAME=rtece_scalar_best.pt" in text


def test_rtece_matrix_submit_helper_forwards_short_range_core_without_sbatch_export(tmp_path):
    from benchmarks.oc20neb_tace_mace.submit_rtece_scalar_matrix import write_rtece_matrix_wrapper

    wrapper = write_rtece_matrix_wrapper(
        tmp_path,
        variants="radial_core_s0p3",
        run_root="/tmp/rtece-stage82",
        scalar_path_ids="atomic.radial_density",
        use_short_range_repulsion=True,
        short_range_repulsion_potential="zbl",
        short_range_repulsion_strength=0.3,
        short_range_repulsion_beta=20.0,
        short_range_repulsion_radius_scale=0.9,
        force_mode="autograd",
        learnable_radial_mixing=True,
    )
    text = wrapper.read_text()

    assert "--export" not in text
    assert "USE_SHORT_RANGE_REPULSION=1" in text
    assert "SHORT_RANGE_REPULSION_POTENTIAL=zbl" in text
    assert "LEARNABLE_RADIAL_MIXING=1" in text
    assert "SHORT_RANGE_REPULSION_STRENGTH=0.3" in text
    assert "SHORT_RANGE_REPULSION_BETA=20.0" in text
    assert "SHORT_RANGE_REPULSION_RADIUS_SCALE=0.9" in text
    assert "SCALAR_PATH_IDS=atomic.radial_density" in text


def test_rtece_matrix_sbatch_forwards_benchmark_force_mode():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    script = (root / "benchmarks/oc20neb_tace_mace/rtece_scalar_matrix.sbatch").read_text()

    assert "FORCE_MODE=${FORCE_MODE:-autograd}" in script
    assert "FORCE_FOCUS_ELEMENTS=${FORCE_FOCUS_ELEMENTS:-}" in script
    assert "--force-focus-elements" in script
    assert "--force-focus-weight" in script
    assert '--force-mode "${FORCE_MODE}"' in script


def test_rtece_matrix_sbatch_forwards_num_radial():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    script = (root / "benchmarks/oc20neb_tace_mace/rtece_scalar_matrix.sbatch").read_text()

    assert "NUM_RADIAL=${NUM_RADIAL:-8}" in script
    assert "MOMENT_L_MAX=${MOMENT_L_MAX:-}" in script
    assert '--num-radial "${NUM_RADIAL}"' in script
    assert '--moment-l-max "${MOMENT_L_MAX}"' in script


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


def test_rtece_ase_calculator_is_formal_package_entrypoint(tmp_path):
    from ase import Atoms
    from tace.interface.ase import RTECEAseCalc
    from tace.models import RTECEScalarModel, save_rtece_checkpoint

    config = RTECEScalarConfig(variant="rtece_pair", cutoff=2.0, num_radial=4, hidden_channels=(4,))
    model = RTECEScalarModel(config).float().eval()
    checkpoint = tmp_path / "rtece.pt"
    save_rtece_checkpoint(checkpoint, model, config)

    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    atoms.calc = RTECEAseCalc(str(checkpoint), device="cpu", dtype="float32", neighborlist_backend="ase")

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()

    assert isinstance(energy, float)
    assert forces.shape == (2, 3)
    assert torch.isfinite(torch.tensor(energy))
    assert torch.isfinite(torch.tensor(forces)).all()


def test_rtece_ase_graph_builder_preserves_periodic_edge_shifts():
    from ase import Atoms
    from tace.interface.ase import atoms_to_rtece_graph

    atoms = Atoms(
        "H2",
        positions=[[0.1, 0.0, 0.0], [4.9, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )

    graph = atoms_to_rtece_graph(
        atoms,
        cutoff=0.5,
        device=torch.device("cpu"),
        dtype=torch.float64,
        neighborlist_backend="matscipy",
    )
    _vectors, distances, _unit = compute_pair_geometry(graph)

    assert graph.cell is not None
    assert graph.edge_shifts is not None
    assert graph.edge_batch is not None
    assert graph.edge_index.shape[1] == 2
    assert torch.allclose(distances, torch.tensor([0.2, 0.2], dtype=torch.float64), atol=1e-12)


def test_rattle_relax_rtece_calculator_uses_formal_package_entrypoint():
    from benchmarks.oc20neb_tace_mace.rattle_relax_rtece import make_rtece_calculator
    from tace.interface.ase import RTECEAseCalc

    config = RTECEScalarConfig(variant="rtece_pair", cutoff=0.5, num_radial=4, hidden_channels=(4,))
    model = RTECEScalarModel(config).double().eval()

    calc = make_rtece_calculator(
        model,
        config,
        device=torch.device("cpu"),
        dtype=torch.float64,
        force_mode="autograd",
        neighborlist_backend="matscipy",
    )

    assert isinstance(calc, RTECEAseCalc)
    assert calc.neighborlist_backend == "matscipy"


def test_relax_lbfgs_compare_rtece_calculator_preserves_periodic_edge_shifts():
    from ase import Atoms
    from benchmarks.oc20neb_tace_mace.relax_lbfgs_compare import RTECECalculator

    config = RTECEScalarConfig(variant="rtece_pair", cutoff=0.5, num_radial=4, hidden_channels=(4,))
    model = RTECEScalarModel(config).double().eval()
    calc = RTECECalculator(model, config, device="cpu", dtype=torch.float64, neighborlist_backend="matscipy")
    atoms = Atoms(
        "H2",
        positions=[[0.1, 0.0, 0.0], [4.9, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )

    graph = calc.atoms_to_graph(atoms, cutoff=config.cutoff, device=torch.device("cpu"), dtype=torch.float64)
    _vectors, distances, _unit = compute_pair_geometry(graph)

    assert graph.cell is not None
    assert graph.edge_shifts is not None
    assert graph.edge_batch is not None
    assert graph.edge_index.shape[1] == 2
    assert torch.allclose(distances, torch.tensor([0.2, 0.2], dtype=torch.float64), atol=1e-12)


def test_rtece_ase_calculator_stress_matches_ase_finite_strain(tmp_path):
    import numpy as np
    from ase import Atoms
    from ase.calculators.fd import calculate_numerical_stress
    from tace.interface.ase import RTECEAseCalc
    from tace.models import RTECEScalarModel, save_rtece_checkpoint

    config = RTECEScalarConfig(variant="rtece_pair", cutoff=2.5, num_radial=4, hidden_channels=(4,))
    model = RTECEScalarModel(config).double().eval()
    checkpoint = tmp_path / "rtece.pt"
    save_rtece_checkpoint(checkpoint, model, config)

    atoms = Atoms(
        "H2",
        positions=[[0.2, 0.1, 0.0], [0.95, 0.2, 0.15]],
        cell=[5.0, 5.5, 6.0],
        pbc=True,
    )
    atoms.calc = RTECEAseCalc(str(checkpoint), device="cpu", dtype="float64", neighborlist_backend="matscipy")

    stress = atoms.get_stress()
    reference = calculate_numerical_stress(atoms, eps=1.0e-5, voigt=True, force_consistent=False)

    assert stress.shape == (6,)
    assert torch.isfinite(torch.as_tensor(stress)).all()
    assert np.allclose(stress, reference, rtol=2.0e-4, atol=2.0e-6)


def test_rtece_ase_calculator_rejects_stress_for_non_autograd_force_backend(tmp_path):
    from ase import Atoms
    from ase.calculators.calculator import PropertyNotImplementedError
    from tace.interface.ase import RTECEAseCalc
    from tace.models import RTECEScalarModel, save_rtece_checkpoint

    config = RTECEScalarConfig(variant="rtece_pair", cutoff=2.0, num_radial=4, hidden_channels=(4,))
    model = RTECEScalarModel(config).float().eval()
    checkpoint = tmp_path / "rtece.pt"
    save_rtece_checkpoint(checkpoint, model, config)

    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]], cell=[4.0, 4.0, 4.0], pbc=True)
    atoms.calc = RTECEAseCalc(
        str(checkpoint),
        device="cpu",
        dtype="float32",
        force_mode="analytic_pair",
        neighborlist_backend="matscipy",
    )

    with pytest.raises(PropertyNotImplementedError, match="autograd"):
        atoms.get_stress()


def test_rtece_eval_cli_is_registered_and_exposes_user_flags():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text()

    assert 'tace-rtece-eval = "tace.scripts.rtece_eval:main"' in pyproject

    result = subprocess.run(
        [sys.executable, "-m", "tace.scripts.rtece_eval", "--help"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "--model" in result.stdout
    assert "--input" in result.stdout
    assert "--force-mode" in result.stdout
    assert "--neighborlist-backend" in result.stdout


def test_rtece_train_cli_is_registered_and_exposes_user_flags():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text()

    assert 'tace-rtece-train-scalar = "tace.scripts.rtece_train_scalar:main"' in pyproject

    result = subprocess.run(
        [sys.executable, "-m", "tace.scripts.rtece_train_scalar", "--help"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "--train-file" in result.stdout
    assert "--valid-file" in result.stdout
    assert "--min-eval-step" in result.stdout
    assert "--force-focus-elements" in result.stdout


def test_rtece_matrix_sbatch_uses_package_train_entrypoint():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    script = (root / "benchmarks/oc20neb_tace_mace/rtece_scalar_matrix.sbatch").read_text()

    assert 'TRAINER_BACKEND=${TRAINER_BACKEND:-lightning}' in script
    assert 'BATCH_SIZE=${BATCH_SIZE:-1}' in script
    assert 'LR_SCHEDULER=${LR_SCHEDULER:-plateau}' in script
    assert '"${TACE_PYTHON}" -m tace.scripts.rtece_train_scalar' in script
    assert '"${lightning_args[@]}"' in script
    assert 'train_rtece_scalar.py"' not in script


def test_rtece_lightning_fit_smoke_saves_portable_checkpoint(tmp_path):
    import json
    import numpy as np
    import ase.io
    from ase import Atoms
    from tace.lightning.rtece import fit_rtece_lightning
    from tace.models.rtece_workflow import load_checkpoint

    train = tmp_path / "train.xyz"
    valid = tmp_path / "valid.xyz"
    output_dir = tmp_path / "lightning"
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    atoms.info["energy"] = 0.0
    atoms.arrays["forces"] = np.zeros((2, 3), dtype=np.float64)
    ase.io.write(train, [atoms], format="extxyz")
    ase.io.write(valid, [atoms], format="extxyz")

    summary = fit_rtece_lightning(
        variant="rtece_pair",
        train_file=train,
        valid_file=valid,
        output_dir=output_dir,
        limit_configs=1,
        valid_limit_configs=1,
        max_steps=1,
        batch_size=1,
        valid_batch_size=1,
        hidden_channels="4",
        num_radial=4,
        descriptor_bottleneck_dim=3,
        accelerator="cpu",
        devices=1,
        default_dtype="float32",
        neighborlist_backend="ase",
        lr_warmup_steps=2,
        enable_progress_bar=False,
        logger=False,
    )

    assert summary["trainer_backend"] == "lightning"
    assert summary["steps"] == 1
    assert summary["best_step"] == 1
    assert summary["batch_size"] == 1
    assert summary["lr_warmup_steps"] == 2
    assert summary["descriptor_bottleneck_dim"] == 3
    assert summary["best_checkpoint"] == str(output_dir / "rtece_scalar_best.pt")
    assert (output_dir / "rtece_scalar.pt").exists()
    assert (output_dir / "rtece_scalar_best.pt").exists()
    assert (output_dir / "train_summary.json").exists()
    saved_summary = json.loads((output_dir / "train_summary.json").read_text())
    assert saved_summary["trainer_backend"] == "lightning"
    _model, loaded_config, _metadata = load_checkpoint(output_dir / "rtece_scalar_best.pt", dtype=torch.float32)
    assert loaded_config.variant == "rtece_pair"
    assert loaded_config.descriptor_bottleneck_dim == 3


def test_rtece_lightning_default_logger_setting_survives_multi_epoch_fit(tmp_path, monkeypatch):
    import numpy as np
    import ase.io
    from ase import Atoms
    from tace.lightning.rtece import fit_rtece_lightning

    monkeypatch.chdir(tmp_path)
    train = tmp_path / "train.xyz"
    valid = tmp_path / "valid.xyz"
    output_dir = tmp_path / "lightning-default-logger"
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    atoms.info["energy"] = 0.0
    atoms.arrays["forces"] = np.zeros((2, 3), dtype=np.float64)
    ase.io.write(train, [atoms], format="extxyz")
    ase.io.write(valid, [atoms], format="extxyz")

    summary = fit_rtece_lightning(
        variant="rtece_pair",
        train_file=train,
        valid_file=valid,
        output_dir=output_dir,
        limit_configs=1,
        valid_limit_configs=1,
        max_steps=3,
        batch_size=1,
        valid_batch_size=1,
        hidden_channels="4",
        num_radial=4,
        accelerator="cpu",
        devices=1,
        default_dtype="float32",
        neighborlist_backend="ase",
        enable_progress_bar=False,
    )

    assert summary["steps"] == 3
    assert (output_dir / "rtece_scalar_best.pt").exists()
    assert not (tmp_path / "lightning_logs").exists()


def test_rtece_train_cli_disables_lightning_csv_logger_by_default(tmp_path):
    import numpy as np
    import ase.io
    from ase import Atoms

    train = tmp_path / "train.xyz"
    valid = tmp_path / "valid.xyz"
    output_dir = tmp_path / "cli-train"
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
    atoms.info["energy"] = 0.0
    atoms.arrays["forces"] = np.zeros((2, 3), dtype=np.float64)
    ase.io.write(train, [atoms], format="extxyz")
    ase.io.write(valid, [atoms], format="extxyz")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tace.scripts.rtece_train_scalar",
            "--variant",
            "rtece_pair",
            "--train-file",
            str(train),
            "--valid-file",
            str(valid),
            "--output-dir",
            str(output_dir),
            "--limit-configs",
            "1",
            "--valid-limit-configs",
            "1",
            "--max-steps",
            "3",
            "--hidden-channels",
            "4",
            "--num-radial",
            "4",
            "--device",
            "cpu",
            "--accelerator",
            "cpu",
            "--devices",
            "1",
            "--neighborlist-backend",
            "ase",
            "--no-progress-bar",
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "rtece_scalar_best.pt").exists()
    assert not (tmp_path / "lightning_logs").exists()


def test_rtece_train_cli_defaults_to_lightning_and_exposes_training_controls():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "tace.scripts.rtece_train_scalar", "--help"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    assert "--trainer-backend" in result.stdout
    assert "--batch-size" in result.stdout
    assert "--valid-batch-size" in result.stdout
    assert "--lr-scheduler" in result.stdout
    assert "--early-stopping-patience" in result.stdout
    assert "--lr-warmup-steps" in result.stdout

def test_stage146_energy_calibration_fits_per_element_residual_on_calibration_and_applies_to_eval():
    from benchmarks.oc20neb_tace_mace.analyze_rtece_energy_gauge import (
        apply_energy_calibration,
        fit_residual_energy_calibration,
    )

    elements = [1, 6]
    calib_counts = [
        {1: 2, 6: 1},
        {1: 0, 6: 2},
        {1: 2, 6: 0},
    ]
    calib_pred_e = torch.tensor([10.0, 8.0, 2.0], dtype=torch.float64).numpy()
    residual_e0 = {1: 0.25, 6: -0.75}
    calib_ref_e = calib_pred_e + torch.tensor([
        2 * residual_e0[1] + residual_e0[6],
        2 * residual_e0[6],
        2 * residual_e0[1],
    ], dtype=torch.float64).numpy()

    calibration = fit_residual_energy_calibration(
        calib_counts,
        calib_pred_e,
        calib_ref_e,
        elements=elements,
        mode="per_element",
        ridge=0.0,
    )

    assert calibration["kind"] == "per_element_residual_e0"
    assert calibration["residual_e0_by_z_eV"] == pytest.approx({"1": 0.25, "6": -0.75})

    eval_counts = [{1: 1, 6: 1}, {1: 4, 6: 0}]
    eval_pred_e = torch.tensor([3.0, -2.0], dtype=torch.float64).numpy()
    eval_ref_e = eval_pred_e + torch.tensor([-0.5, 1.0], dtype=torch.float64).numpy()

    corrected = apply_energy_calibration(eval_counts, eval_pred_e, calibration)

    assert corrected == pytest.approx(eval_ref_e)


def test_stage146_energy_calibration_keeps_global_shift_as_separate_control():
    from benchmarks.oc20neb_tace_mace.analyze_rtece_energy_gauge import (
        apply_energy_calibration,
        fit_residual_energy_calibration,
    )

    calibration = fit_residual_energy_calibration(
        [{1: 2}, {1: 3}],
        torch.tensor([1.0, 5.0], dtype=torch.float64).numpy(),
        torch.tensor([2.5, 6.5], dtype=torch.float64).numpy(),
        elements=[1],
        mode="global",
    )

    assert calibration["kind"] == "global_total_energy_shift"
    assert calibration["shift_eV"] == pytest.approx(1.5)
    assert apply_energy_calibration([{1: 1}], torch.tensor([10.0], dtype=torch.float64).numpy(), calibration) == pytest.approx([11.5])

def test_stage147_relative_energy_metrics_remove_group_endpoint_gauge():
    from benchmarks.oc20neb_tace_mace.analyze_rtece_energy_gauge import relative_energy_group_metrics

    ref_e = torch.tensor([0.0, 1.0, 3.0, 10.0, 12.0], dtype=torch.float64).numpy()
    pred_e = torch.tensor([5.0, 6.2, 8.0, -2.0, 0.0], dtype=torch.float64).numpy()
    natoms = torch.tensor([2, 2, 2, 4, 4], dtype=torch.float64).numpy()
    groups = ["path-a", "path-a", "path-a", "path-b", "path-b"]
    images = [0, 1, 2, 0, 1]

    metrics = relative_energy_group_metrics(pred_e, ref_e, natoms, groups, image_indices=images)

    assert metrics["num_groups"] == 2
    assert metrics["num_images"] == 5
    # path-a relative errors per atom: [0, 0.1, 0], path-b: [0, 0]
    assert metrics["relative_image_mae_mev_atom"] == pytest.approx(20.0)
    assert metrics["relative_image_rmse_mev_atom"] == pytest.approx((2000.0) ** 0.5)
    assert metrics["relative_image_max_abs_mev_atom"] == pytest.approx(100.0)


def test_stage147_relative_energy_metrics_report_barrier_errors_per_group():
    from benchmarks.oc20neb_tace_mace.analyze_rtece_energy_gauge import relative_energy_group_metrics

    ref_e = torch.tensor([0.0, 4.0, 1.0, 10.0, 15.0, 11.0], dtype=torch.float64).numpy()
    pred_e = torch.tensor([0.0, 3.0, 1.0, 10.0, 18.0, 11.0], dtype=torch.float64).numpy()
    natoms = torch.tensor([2, 2, 2, 4, 4, 4], dtype=torch.float64).numpy()
    groups = ["path-a", "path-a", "path-a", "path-b", "path-b", "path-b"]
    images = [0, 1, 2, 0, 1, 2]

    metrics = relative_energy_group_metrics(pred_e, ref_e, natoms, groups, image_indices=images)

    # Barrier errors: path-a = -0.5 eV/atom, path-b = +0.75 eV/atom.
    assert metrics["barrier_mae_mev_atom"] == pytest.approx(625.0)
    assert metrics["barrier_rmse_mev_atom"] == pytest.approx(((500.0**2 + 750.0**2) / 2) ** 0.5)
    assert metrics["barrier_max_abs_mev_atom"] == pytest.approx(750.0)


def test_stage148_energy_decomposition_identifies_case_offset_without_path_shape_error():
    from benchmarks.oc20neb_tace_mace.relative_energy_metrics import energy_error_decomposition_metrics

    ref_e = torch.tensor([0.0, 1.0, 2.0, 10.0, 12.0, 14.0], dtype=torch.float64).numpy()
    pred_e = torch.tensor([5.0, 6.0, 7.0, 6.0, 8.0, 10.0], dtype=torch.float64).numpy()
    natoms = torch.tensor([10, 10, 10, 20, 20, 20], dtype=torch.float64).numpy()
    groups = ["path-a", "path-a", "path-a", "path-b", "path-b", "path-b"]
    images = [0, 1, 2, 0, 1, 2]

    metrics = energy_error_decomposition_metrics(pred_e, ref_e, natoms, groups, image_indices=images)

    assert metrics["raw_rmse_mev_atom"] > 0.0
    assert metrics["global_offset_rmse_mev_atom"] > 0.0
    assert metrics["group_mean_offset_rmse_mev_atom"] == pytest.approx(0.0)
    assert metrics["first_image_anchor_rmse_mev_atom"] == pytest.approx(0.0)
    assert metrics["relative_image_rmse_mev_atom"] == pytest.approx(0.0)
    assert metrics["barrier_rmse_mev_atom"] == pytest.approx(0.0)


def test_stage148_energy_decomposition_preserves_along_path_shape_error():
    from benchmarks.oc20neb_tace_mace.relative_energy_metrics import energy_error_decomposition_metrics

    ref_e = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64).numpy()
    pred_e = torch.tensor([5.0, 7.0, 7.0], dtype=torch.float64).numpy()
    natoms = torch.tensor([10, 10, 10], dtype=torch.float64).numpy()
    groups = ["path-a", "path-a", "path-a"]
    images = [0, 1, 2]

    metrics = energy_error_decomposition_metrics(pred_e, ref_e, natoms, groups, image_indices=images)

    assert metrics["group_mean_offset_rmse_mev_atom"] > 0.0
    assert metrics["first_image_anchor_rmse_mev_atom"] == pytest.approx((10000.0 / 3) ** 0.5)
    assert metrics["relative_image_rmse_mev_atom"] == pytest.approx((10000.0 / 3) ** 0.5)
    assert metrics["barrier_rmse_mev_atom"] == pytest.approx(0.0)

