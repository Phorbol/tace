# rTECE Scalar-Sketched Prototype Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first independent T3 `rtece_scalar` prototype that produces conservative scalar-sketched rTECE energies/forces, trains on the OC20NEB distillation labels, benchmarks with the existing Pareto metrics, and updates the TECE distillation summary.

**Architecture:** Implement a small standalone PyTorch model under `benchmarks/oc20neb_tace_mace/` rather than integrating into `e3nnTACE`. The model builds radial pair features, low-order Cartesian moments, atomic scalar contractions, and optional edge-relational scalar sketches, then predicts atomic energies with a scalar MLP and obtains conservative forces by autograd. Training and benchmarking scripts use the same extxyz data and JSON metric schema as the current TECE distillation matrix.

**Tech Stack:** Python 3.11, PyTorch, ASE, NumPy, pytest, existing OC20NEB extxyz files, existing `summarize_tece_distill.py` JSON schema.

## Global Constraints

- Keep the implementation independent from the full Hydra/TensorModel path for the first prototype.
- Preserve energy conservation by computing forces as `-grad(E, positions)`.
- Do not introduce full RRA, SO(2) edge tensor state, dense per-edge channel mixing, or softmax attention.
- Do not use parameter count as a success metric.
- Report DFT and teacher MAE/RMSE, atoms/s, configs/s, seconds/pass, and peak GPU memory.
- First prototype may time graph construction, but must mark this explicitly in benchmark JSON.
- Use TDD: every production behavior starts with a failing test.

---

## File Structure

- Create `benchmarks/oc20neb_tace_mace/rtece_scalar_model.py`
  - Owns model config dataclass, radial basis, graph container, descriptor construction, model forward, conservative force helper, and variant builder.
- Create `benchmarks/oc20neb_tace_mace/train_rtece_scalar.py`
  - Owns extxyz loading, graph construction, training loop, checkpoint save/load, and tiny/full dataset arguments.
- Create `benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py`
  - Owns checkpoint loading, benchmark timing, error metrics, and JSON output.
- Modify `benchmarks/oc20neb_tace_mace/summarize_tece_distill.py`
  - Accepts rTECE JSON rows without requiring TACE/MACE-only fields.
- Create `test/test_rtece_scalar.py`
  - Unit tests for descriptors, invariance, conservative forces, variants, and benchmark row compatibility.
- Modify `docs/superpowers/specs/2026-07-17-tece-renorm-distill-design.md` in Task 8
  - Add Stage-5 smoke results only after a benchmark JSON exists.

---

### Task 1: Descriptor Config And Variant Builder

**Files:**
- Create: `benchmarks/oc20neb_tace_mace/rtece_scalar_model.py`
- Test: `test/test_rtece_scalar.py`

**Interfaces:**
- Produces: `RTECEScalarConfig` dataclass with fields `variant: str`, `cutoff: float`, `num_radial: int`, `hidden_channels: tuple[int, ...]`, `max_atomic_number: int`, `use_atomic_moments: bool`, `num_edge_sketches: int`.
- Produces: `build_rtece_config(variant: str) -> RTECEScalarConfig`.
- Produces: `descriptor_dim(config: RTECEScalarConfig) -> int`.

- [ ] **Step 1: Write failing variant builder tests**

Add this to `test/test_rtece_scalar.py`:

```python
from benchmarks.oc20neb_tace_mace.rtece_scalar_model import (
    build_rtece_config,
    descriptor_dim,
)


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
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest -q test/test_rtece_scalar.py::test_build_rtece_config_defines_ordered_variants
```

Expected: FAIL with `ModuleNotFoundError` or missing symbol from `rtece_scalar_model.py`.

- [ ] **Step 3: Implement minimal config code**

Create `benchmarks/oc20neb_tace_mace/rtece_scalar_model.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RTECEScalarConfig:
    variant: str
    cutoff: float = 5.0
    num_radial: int = 8
    hidden_channels: tuple[int, ...] = (64, 64)
    max_atomic_number: int = 100
    use_atomic_moments: bool = False
    num_edge_sketches: int = 0


def build_rtece_config(variant: str) -> RTECEScalarConfig:
    if variant == "rtece_pair":
        return RTECEScalarConfig(variant=variant)
    if variant == "rtece_atomic_moments":
        return RTECEScalarConfig(variant=variant, use_atomic_moments=True)
    if variant == "rtece_edge_sketch8":
        return RTECEScalarConfig(variant=variant, use_atomic_moments=True, num_edge_sketches=8)
    if variant == "rtece_edge_sketch16":
        return RTECEScalarConfig(variant=variant, use_atomic_moments=True, num_edge_sketches=16)
    raise ValueError(f"unknown rTECE scalar variant {variant!r}")


def descriptor_dim(config: RTECEScalarConfig) -> int:
    dim = config.num_radial
    if config.use_atomic_moments:
        dim += 2 * config.num_radial
    dim += config.num_edge_sketches
    return dim
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest -q test/test_rtece_scalar.py::test_build_rtece_config_defines_ordered_variants
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/oc20neb_tace_mace/rtece_scalar_model.py test/test_rtece_scalar.py
git commit -m "feat: add rtece scalar variant config"
```

---

### Task 2: Graph Container And Radial/Moment Descriptors

**Files:**
- Modify: `benchmarks/oc20neb_tace_mace/rtece_scalar_model.py`
- Test: `test/test_rtece_scalar.py`

**Interfaces:**
- Consumes: `RTECEScalarConfig`.
- Produces: `RTECEGraph(z: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor)`.
- Produces: `compute_pair_geometry(graph: RTECEGraph) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]`.
- Produces: `compute_radial_features(distances: torch.Tensor, config: RTECEScalarConfig) -> torch.Tensor`.
- Produces: `compute_atomic_moments(graph: RTECEGraph, config: RTECEScalarConfig) -> dict[str, torch.Tensor]`.

- [ ] **Step 1: Write failing rotation-invariance test for atomic scalar contractions**

Add:

```python
import math
import torch

from benchmarks.oc20neb_tace_mace.rtece_scalar_model import (
    RTECEGraph,
    build_rtece_config,
    atomic_scalar_descriptors,
)


def rotation_z(theta: float) -> torch.Tensor:
    c = math.cos(theta)
    s = math.sin(theta)
    return torch.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=torch.float64)


def complete_directed_edges(num_nodes: int) -> torch.Tensor:
    edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def test_atomic_scalar_descriptors_are_rotation_invariant():
    config = build_rtece_config("rtece_atomic_moments")
    z = torch.tensor([6, 8, 1], dtype=torch.long)
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.2, 0.1], [-0.3, 0.6, -0.2]], dtype=torch.float64)
    edge_index = complete_directed_edges(3)
    batch = torch.zeros(3, dtype=torch.long)

    graph = RTECEGraph(z=z, pos=pos, edge_index=edge_index, batch=batch)
    rotated = RTECEGraph(z=z, pos=pos @ rotation_z(0.37).T, edge_index=edge_index, batch=batch)

    desc = atomic_scalar_descriptors(graph, config)
    desc_rot = atomic_scalar_descriptors(rotated, config)

    assert torch.allclose(desc, desc_rot, atol=1e-10, rtol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest -q test/test_rtece_scalar.py::test_atomic_scalar_descriptors_are_rotation_invariant
```

Expected: FAIL due missing `RTECEGraph` or `atomic_scalar_descriptors`.

- [ ] **Step 3: Implement graph, radial features, moments, atomic scalar descriptors**

Add to `rtece_scalar_model.py`:

```python
from dataclasses import dataclass
import math
import torch


@dataclass
class RTECEGraph:
    z: torch.Tensor
    pos: torch.Tensor
    edge_index: torch.Tensor
    batch: torch.Tensor


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


def compute_radial_features(distances: torch.Tensor, config: RTECEScalarConfig) -> torch.Tensor:
    centers = torch.linspace(
        0.0,
        config.cutoff,
        config.num_radial,
        device=distances.device,
        dtype=distances.dtype,
    )
    width = config.cutoff / max(config.num_radial - 1, 1)
    features = torch.exp(-0.5 * ((distances[:, None] - centers[None, :]) / width) ** 2)
    return features * cutoff_envelope(distances, config.cutoff)[:, None]


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
    vector = scatter_sum(radial[:, :, None] * unit[:, None, :], dst, num_nodes)
    eye = torch.eye(3, device=graph.pos.device, dtype=graph.pos.dtype)
    quad_unit = unit[:, :, None] * unit[:, None, :] - eye[None, :, :] / 3.0
    quadrupole = scatter_sum(radial[:, :, None, None] * quad_unit[:, None, :, :], dst, num_nodes)
    return {"density": density, "vector": vector, "quadrupole": quadrupole}


def atomic_scalar_descriptors(graph: RTECEGraph, config: RTECEScalarConfig) -> torch.Tensor:
    moments = compute_atomic_moments(graph, config)
    density = moments["density"]
    if not config.use_atomic_moments:
        return density
    vector_norm = (moments["vector"] ** 2).sum(dim=-1)
    quadrupole_norm = (moments["quadrupole"] ** 2).sum(dim=(-1, -2))
    return torch.cat([density, vector_norm, quadrupole_norm], dim=-1)
```

- [ ] **Step 4: Run descriptor tests**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest -q test/test_rtece_scalar.py::test_atomic_scalar_descriptors_are_rotation_invariant test/test_rtece_scalar.py::test_build_rtece_config_defines_ordered_variants
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/oc20neb_tace_mace/rtece_scalar_model.py test/test_rtece_scalar.py
git commit -m "feat: add rtece atomic moment descriptors"
```

---

### Task 3: Edge-Relational Scalar Sketches

**Files:**
- Modify: `benchmarks/oc20neb_tace_mace/rtece_scalar_model.py`
- Test: `test/test_rtece_scalar.py`

**Interfaces:**
- Consumes: `RTECEGraph`, `RTECEScalarConfig`, `compute_atomic_moments`.
- Produces: `edge_relational_sketches(graph: RTECEGraph, config: RTECEScalarConfig) -> torch.Tensor`.
- Produces: `rtece_descriptors(graph: RTECEGraph, config: RTECEScalarConfig) -> torch.Tensor`.

- [ ] **Step 1: Write failing rotation-invariance test for edge sketches**

Add:

```python
from benchmarks.oc20neb_tace_mace.rtece_scalar_model import edge_relational_sketches, rtece_descriptors


def test_edge_relational_sketches_are_rotation_invariant():
    config = build_rtece_config("rtece_edge_sketch8")
    z = torch.tensor([6, 8, 1, 1], dtype=torch.long)
    pos = torch.tensor(
        [[0.0, 0.0, 0.0], [0.7, 0.2, 0.1], [-0.3, 0.6, -0.2], [0.4, -0.5, 0.3]],
        dtype=torch.float64,
    )
    edge_index = complete_directed_edges(4)
    batch = torch.zeros(4, dtype=torch.long)
    graph = RTECEGraph(z=z, pos=pos, edge_index=edge_index, batch=batch)
    rotated = RTECEGraph(z=z, pos=pos @ rotation_z(1.11).T, edge_index=edge_index, batch=batch)

    sketches = edge_relational_sketches(graph, config)
    sketches_rot = edge_relational_sketches(rotated, config)
    full = rtece_descriptors(graph, config)
    full_rot = rtece_descriptors(rotated, config)

    assert sketches.shape == (4, config.num_edge_sketches)
    assert torch.allclose(sketches, sketches_rot, atol=1e-10, rtol=1e-10)
    assert torch.allclose(full, full_rot, atol=1e-10, rtol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest -q test/test_rtece_scalar.py::test_edge_relational_sketches_are_rotation_invariant
```

Expected: FAIL due missing `edge_relational_sketches`.

- [ ] **Step 3: Implement edge sketches**

Add:

```python
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
```

- [ ] **Step 4: Run edge sketch tests**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest -q test/test_rtece_scalar.py::test_edge_relational_sketches_are_rotation_invariant
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/oc20neb_tace_mace/rtece_scalar_model.py test/test_rtece_scalar.py
git commit -m "feat: add rtece edge scalar sketches"
```

---

### Task 4: Conservative Scalar Model

**Files:**
- Modify: `benchmarks/oc20neb_tace_mace/rtece_scalar_model.py`
- Test: `test/test_rtece_scalar.py`

**Interfaces:**
- Consumes: `rtece_descriptors`.
- Produces: `RTECEScalarModel(torch.nn.Module)`.
- Produces: `RTECEScalarModel.forward(graph: RTECEGraph) -> dict[str, torch.Tensor]` with keys `energy`, `atomic_energy`, `forces`.

- [ ] **Step 1: Write failing energy/force and permutation tests**

Add:

```python
from benchmarks.oc20neb_tace_mace.rtece_scalar_model import RTECEScalarModel


def test_rtece_scalar_model_returns_conservative_forces():
    config = build_rtece_config("rtece_edge_sketch8")
    model = RTECEScalarModel(config).double()
    z = torch.tensor([6, 8, 1], dtype=torch.long)
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.2, 0.1], [-0.3, 0.6, -0.2]], dtype=torch.float64)
    graph = RTECEGraph(z=z, pos=pos, edge_index=complete_directed_edges(3), batch=torch.zeros(3, dtype=torch.long))

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
    pos = torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.2, 0.1], [-0.3, 0.6, -0.2]], dtype=torch.float64)
    graph = RTECEGraph(z=z, pos=pos, edge_index=complete_directed_edges(3), batch=torch.zeros(3, dtype=torch.long))
    perm = torch.tensor([2, 0, 1], dtype=torch.long)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(3)
    edge_index_perm = inv[complete_directed_edges(3)]
    graph_perm = RTECEGraph(z=z[perm], pos=pos[perm], edge_index=edge_index_perm, batch=torch.zeros(3, dtype=torch.long))

    e = model(graph)["energy"]
    e_perm = model(graph_perm)["energy"]

    assert torch.allclose(e, e_perm, atol=1e-10, rtol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest -q test/test_rtece_scalar.py::test_rtece_scalar_model_returns_conservative_forces test/test_rtece_scalar.py::test_rtece_scalar_model_energy_is_permutation_invariant_for_complete_graph
```

Expected: FAIL due missing `RTECEScalarModel`.

- [ ] **Step 3: Implement model**

Add:

```python
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

    def forward(self, graph: RTECEGraph) -> dict[str, torch.Tensor]:
        pos = graph.pos
        if not pos.requires_grad:
            pos = pos.detach().clone().requires_grad_(True)
            graph = RTECEGraph(z=graph.z, pos=pos, edge_index=graph.edge_index, batch=graph.batch)
        z_scaled = graph.z.to(dtype=pos.dtype, device=pos.device).view(-1, 1) / float(self.config.max_atomic_number)
        descriptors = rtece_descriptors(graph, self.config)
        atomic_energy = self.energy_head(torch.cat([z_scaled, descriptors], dim=-1)).squeeze(-1)
        num_graphs = int(graph.batch.max().item()) + 1 if graph.batch.numel() else 1
        energy = scatter_sum(atomic_energy[:, None], graph.batch, num_graphs).squeeze(-1)
        forces = -torch.autograd.grad(energy.sum(), pos, create_graph=self.training, retain_graph=True)[0]
        return {"energy": energy, "atomic_energy": atomic_energy, "forces": forces}
```

- [ ] **Step 4: Run model tests**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest -q test/test_rtece_scalar.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/oc20neb_tace_mace/rtece_scalar_model.py test/test_rtece_scalar.py
git commit -m "feat: add conservative rtece scalar model"
```

---

### Task 5: Tiny Training Script

**Files:**
- Create: `benchmarks/oc20neb_tace_mace/train_rtece_scalar.py`
- Test: `test/test_rtece_scalar.py`

**Interfaces:**
- Consumes: `RTECEScalarModel`, `RTECEGraph`, `build_rtece_config`.
- Produces: `train_rtece_scalar.py --train-file ... --valid-file ... --variant ... --output-dir ... --max-steps ... --limit-configs ...`.
- Produces: checkpoint `rtece_scalar.pt` with `{"config": ..., "state_dict": ...}`.

- [ ] **Step 1: Write failing checkpoint roundtrip smoke test**

Add:

```python
def test_rtece_checkpoint_roundtrip(tmp_path):
    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import save_checkpoint, load_checkpoint

    config = build_rtece_config("rtece_pair")
    model = RTECEScalarModel(config).double()
    path = tmp_path / "rtece_scalar.pt"

    save_checkpoint(path, model, config)
    loaded_model, loaded_config = load_checkpoint(path, dtype=torch.float64)

    assert loaded_config.variant == "rtece_pair"
    assert isinstance(loaded_model, RTECEScalarModel)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest -q test/test_rtece_scalar.py::test_rtece_checkpoint_roundtrip
```

Expected: FAIL due missing `train_rtece_scalar.py`.

- [ ] **Step 3: Implement checkpoint helpers and CLI skeleton**

Create `train_rtece_scalar.py` with checkpoint helpers:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from benchmarks.oc20neb_tace_mace.rtece_scalar_model import (
    RTECEScalarConfig,
    RTECEScalarModel,
    build_rtece_config,
)


def save_checkpoint(path: Path, model: RTECEScalarModel, config: RTECEScalarConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(config), "state_dict": model.state_dict()}, path)


def load_checkpoint(path: Path, *, dtype: torch.dtype = torch.float32, device: str | torch.device = "cpu"):
    payload = torch.load(path, map_location=device)
    config = RTECEScalarConfig(**payload["config"])
    model = RTECEScalarModel(config).to(device=device, dtype=dtype)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, config
```

Also add this CLI skeleton; Task 6 replaces the stub with the training body:

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a scalar-sketched rTECE prototype.")
    parser.add_argument("--variant", required=True, choices=("rtece_pair", "rtece_atomic_moments", "rtece_edge_sketch8", "rtece_edge_sketch16"))
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--valid-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit-configs", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--default-dtype", choices=("float32", "float64"), default="float32")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raise SystemExit("training body is added in Task 6")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run checkpoint test**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest -q test/test_rtece_scalar.py::test_rtece_checkpoint_roundtrip
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/oc20neb_tace_mace/train_rtece_scalar.py test/test_rtece_scalar.py
git commit -m "feat: add rtece scalar checkpoint helpers"
```

---

### Task 6: Extxyz Graph Loading And Smoke Training

**Files:**
- Modify: `benchmarks/oc20neb_tace_mace/train_rtece_scalar.py`
- Test: `test/test_rtece_scalar.py`

**Interfaces:**
- Produces: `atoms_to_graph(atoms, cutoff: float, device: torch.device, dtype: torch.dtype) -> tuple[RTECEGraph, torch.Tensor, torch.Tensor]`.
- Produces: `train_one_epoch_or_steps(...)` used by CLI.
- CLI writes `train_summary.json`.

- [ ] **Step 1: Write failing synthetic training smoke test**

Add:

```python
def test_rtece_tiny_training_step_reduces_finite_loss(tmp_path):
    from ase import Atoms
    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import atoms_to_graph, train_steps

    atoms = Atoms("H2O", positions=[[0.0, 0.0, 0.0], [0.75, 0.0, 0.0], [-0.25, 0.65, 0.0]])
    atoms.info["energy"] = -1.0
    atoms.arrays["forces"] = torch.zeros((3, 3), dtype=torch.float64).numpy()
    config = build_rtece_config("rtece_pair")
    model = RTECEScalarModel(config).double()
    graph, energy, forces = atoms_to_graph(atoms, cutoff=config.cutoff, device=torch.device("cpu"), dtype=torch.float64)

    summary = train_steps(model, [(graph, energy, forces)], max_steps=2, lr=1e-3)

    assert summary["steps"] == 2
    assert torch.isfinite(torch.tensor(summary["final_loss"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest -q test/test_rtece_scalar.py::test_rtece_tiny_training_step_reduces_finite_loss
```

Expected: FAIL due missing `atoms_to_graph` or `train_steps`.

- [ ] **Step 3: Implement graph loading and tiny training**

Implement:

```python
def atoms_to_graph(atoms, *, cutoff: float, device: torch.device, dtype: torch.dtype):
    import numpy as np
    from ase.neighborlist import neighbor_list

    src, dst = neighbor_list("ij", atoms, cutoff)
    edge_index = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long, device=device)
    z = torch.tensor(atoms.numbers, dtype=torch.long, device=device)
    pos = torch.tensor(atoms.positions, dtype=dtype, device=device)
    batch = torch.zeros(len(atoms), dtype=torch.long, device=device)
    energy = torch.tensor([float(atoms.info["energy"])], dtype=dtype, device=device)
    forces = torch.tensor(atoms.arrays["forces"], dtype=dtype, device=device)
    return RTECEGraph(z=z, pos=pos, edge_index=edge_index, batch=batch), energy, forces


def loss_for_batch(model, graph, ref_energy, ref_forces):
    out = model(graph)
    natoms = graph.z.numel()
    e_loss = ((out["energy"] - ref_energy) / natoms).pow(2).mean()
    f_loss = (out["forces"] - ref_forces).pow(2).mean()
    return e_loss + 10.0 * f_loss


def train_steps(model, samples, *, max_steps: int, lr: float):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    final_loss = None
    for step in range(max_steps):
        graph, energy, forces = samples[step % len(samples)]
        opt.zero_grad(set_to_none=True)
        loss = loss_for_batch(model, graph, energy, forces)
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().cpu())
    return {"steps": max_steps, "final_loss": final_loss}
```

- [ ] **Step 4: Run smoke test**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest -q test/test_rtece_scalar.py::test_rtece_tiny_training_step_reduces_finite_loss
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/oc20neb_tace_mace/train_rtece_scalar.py test/test_rtece_scalar.py
git commit -m "feat: add rtece scalar smoke training"
```

---

### Task 7: Benchmark Script And Summary Compatibility

**Files:**
- Create: `benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py`
- Modify: `benchmarks/oc20neb_tace_mace/summarize_tece_distill.py`
- Test: `test/test_rtece_scalar.py`

**Interfaces:**
- Consumes: checkpoint from `train_rtece_scalar.py`.
- Produces: benchmark JSON with existing summary fields plus `backend: "rtece_scalar"` and `includes_graph_construction: true`.

- [ ] **Step 1: Write failing benchmark JSON compatibility test**

Add:

```python
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
```

- [ ] **Step 2: Run test to verify current compatibility**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest -q test/test_rtece_scalar.py::test_rtece_benchmark_row_is_summary_compatible
```

Expected: PASS if existing summary already accepts generic JSON. If it fails, update only the required field handling.

- [ ] **Step 3: Implement benchmark script**

Create `benchmark_rtece_scalar.py` with:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.oc20neb_tace_mace.benchmark_models import load_atoms, reference_arrays, summarize_errors, cuda_memory
from benchmarks.oc20neb_tace_mace.train_rtece_scalar import atoms_to_graph, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--default-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--limit-configs", type=int, default=128)
    parser.add_argument("--measure-passes", type=int, default=3)
    return parser.parse_args()
```

Implement `main()` with this complete structure:

```python
def main() -> None:
    args = parse_args()
    dtype = torch.float64 if args.default_dtype == "float64" else torch.float32
    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model, config = load_checkpoint(args.model, dtype=dtype, device=device)
    atoms_list = load_atoms(args.configs, args.limit_configs)
    ref_e, ref_f, natoms = reference_arrays(atoms_list, "energy", "forces")

    def forward_once(collect: bool):
        outputs = []
        for atoms in atoms_list:
            graph, _, _ = atoms_to_graph(atoms, cutoff=config.cutoff, device=device, dtype=dtype)
            out = model(graph)
            if collect:
                outputs.append({
                    "energy": out["energy"].detach().cpu().numpy(),
                    "forces": out["forces"].detach().cpu().numpy(),
                })
        return outputs

    for _ in range(1):
        forward_once(False)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    pass_times = []
    first_outputs = []
    for pass_idx in range(max(1, args.measure_passes)):
        start = time.perf_counter()
        outputs = forward_once(pass_idx == 0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        pass_times.append(time.perf_counter() - start)
        if pass_idx == 0:
            first_outputs = outputs

    pred_e = np.concatenate([item["energy"].reshape(-1) for item in first_outputs], axis=0)
    pred_f = np.concatenate([item["forces"].reshape(-1, 3) for item in first_outputs], axis=0)
    seconds_per_pass = float(np.mean(pass_times))
    atoms = int(natoms.sum())
    payload = {
        "backend": "rtece_scalar",
        "variant": args.variant,
        "model": str(args.model),
        "configs_path": str(args.configs),
        "configs": len(atoms_list),
        "atoms": atoms,
        "device": str(device),
        "default_dtype": args.default_dtype,
        "model_class": model.__class__.__name__,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "includes_graph_construction": True,
        "measure_passes": args.measure_passes,
        "pass_times_s": pass_times,
        "seconds_per_pass": seconds_per_pass,
        "atoms_per_second": atoms / seconds_per_pass,
        "configs_per_second": len(atoms_list) / seconds_per_pass,
        **summarize_errors(pred_e, pred_f, ref_e, ref_f, natoms),
        **cuda_memory(torch),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest -q test/test_rtece_scalar.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py benchmarks/oc20neb_tace_mace/summarize_tece_distill.py test/test_rtece_scalar.py
git commit -m "feat: add rtece scalar benchmark path"
```

---

### Task 8: Small OC20NEB Smoke Run

**Files:**
- Modify: `docs/superpowers/specs/2026-07-17-tece-renorm-distill-design.md`
- Runtime outputs: `runs/oc20neb_tace_mace/rtece-scalar-YYYYMMDD/`

**Interfaces:**
- Consumes: train and benchmark scripts from previous tasks.
- Produces: smoke checkpoint, benchmark JSON, and Stage-5 smoke result note.

- [ ] **Step 1: Run tiny training on mixed labels**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python benchmarks/oc20neb_tace_mace/train_rtece_scalar.py \
  --variant rtece_pair \
  --train-file runs/oc20neb_tace_mace/tece-distill-20260717/mixed_train_tw0.75.extxyz \
  --valid-file /home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz \
  --output-dir runs/oc20neb_tace_mace/rtece-scalar-smoke \
  --limit-configs 8 \
  --max-steps 4 \
  --device cuda
```

Expected: creates `runs/oc20neb_tace_mace/rtece-scalar-smoke/rtece_scalar.pt` and `train_summary.json`.

- [ ] **Step 2: Run tiny DFT benchmark**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python benchmarks/oc20neb_tace_mace/benchmark_rtece_scalar.py \
  --model runs/oc20neb_tace_mace/rtece-scalar-smoke/rtece_scalar.pt \
  --configs /home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz \
  --output runs/oc20neb_tace_mace/rtece-scalar-smoke/rtece_pair_dft_benchmark.json \
  --variant rtece_pair \
  --limit-configs 8 \
  --device cuda
```

Expected: JSON contains `backend: "rtece_scalar"`, `includes_graph_construction: true`, and finite `mae_f_mev_a`.

- [ ] **Step 3: Update Stage-5 smoke note**

Append to `docs/superpowers/specs/2026-07-17-tece-renorm-distill-design.md`:

```markdown
## Stage 5 Smoke: rTECE Scalar-Sketched Prototype

The independent T3 prototype has passed tiny training and benchmark smoke tests. This does not establish Pareto position yet; it only proves the conservative scalar-sketch path can train and emit the same benchmark schema. Full variants should be run only after the smoke benchmark JSON is inspected.
```

- [ ] **Step 4: Commit smoke integration**

```bash
git add docs/superpowers/specs/2026-07-17-tece-renorm-distill-design.md runs/oc20neb_tace_mace/rtece-scalar-smoke
git commit -m "test: smoke rtece scalar prototype"
```

---

## Self-Review

Spec coverage:

- Independent prototype rather than Hydra integration: Task 1-8.
- Pair, moments, edge sketches, scalar head: Task 1-4.
- Conservative forces: Task 4 and Task 6.
- Training on extxyz labels: Task 5-6.
- Benchmark JSON with Pareto metrics: Task 7-8.
- Summary/design update after smoke: Task 8.

No placeholders:

- No unspecified task remains.
- Full OC20NEB training is intentionally outside this first implementation plan; Stage-5 smoke decides whether to launch it.

Type consistency:

- `RTECEScalarConfig`, `RTECEGraph`, `RTECEScalarModel`, `build_rtece_config`, `descriptor_dim`, `rtece_descriptors`, `train_steps`, `atoms_to_graph`, `save_checkpoint`, and `load_checkpoint` are introduced before any downstream task consumes them.
