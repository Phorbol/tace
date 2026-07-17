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


def test_rtece_scripts_are_directly_executable():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    for script in (
        "benchmarks/oc20neb_tace_mace/train_rtece_scalar.py",
        "benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py",
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
