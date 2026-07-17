from __future__ import annotations

import math
import subprocess
import sys

import torch

from benchmarks.oc20neb_tace_mace.rtece_scalar_model import (
    RTECEGraph,
    RTECEScalarConfig,
    RTECEScalarModel,
    collate_graphs,
    atomic_scalar_descriptors,
    build_rtece_config,
    descriptor_dim,
    edge_relational_sketches,
    packed_element_density_descriptors,
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



def test_energy_per_atom_shift_adds_zeroth_order_energy():
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


def test_rtece_matrix_sbatch_separates_training_and_benchmark_validation_files():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    script = (root / "benchmarks/oc20neb_tace_mace/rtece_scalar_matrix.sbatch").read_text()

    assert "TRAIN_VALID_FILE=${TRAIN_VALID_FILE:-${DFT_VALID_FILE}}" in script
    assert '--valid-file "${TRAIN_VALID_FILE}"' in script
    assert '--configs "${DFT_VALID_FILE}"' in script

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
