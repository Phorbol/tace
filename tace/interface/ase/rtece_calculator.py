from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from ase.calculators.calculator import Calculator, PropertyNotImplementedError, all_changes

from tace.dataset.neighbour_list import get_neighborhood
from tace.models.rtece_scalar import RTECEGraph, RTECEScalarConfig, RTECEScalarModel
from tace.models.rtece_workflow import load_checkpoint, predict


_DTYPE = {
    "float32": torch.float32,
    "float64": torch.float64,
    torch.float32: torch.float32,
    torch.float64: torch.float64,
}


def _resolve_dtype(dtype: str | torch.dtype | None) -> torch.dtype:
    if dtype is None:
        return torch.float32
    try:
        return _DTYPE[dtype]
    except KeyError as exc:
        raise ValueError("rTECE dtype must be 'float32' or 'float64'") from exc


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def atoms_to_rtece_graph(
    atoms,
    *,
    cutoff: float,
    device: torch.device | str = "cpu",
    dtype: torch.dtype | str = torch.float32,
    neighborlist_backend: str = "matscipy",
) -> RTECEGraph:
    """Build the rTECE graph used by package-level inference and ASE calculators.

    The graph follows the same periodic image convention as TACE's neighbor-list
    utility: edge shifts are integer cell offsets and are consumed by
    ``compute_pair_geometry`` in ``tace.models.rtece_scalar``.
    """
    resolved_device = _resolve_device(device)
    resolved_dtype = _resolve_dtype(dtype)
    pbc = tuple(bool(value) for value in atoms.pbc)
    lattice = np.asarray(atoms.cell.array, dtype=np.float64)
    backend = str(neighborlist_backend)
    if backend == "matscipy" and not any(pbc) and np.allclose(lattice, 0.0):
        backend = "ase"

    edge_index_np, shifts_np, _pbc, lattice_np = get_neighborhood(
        positions=np.asarray(atoms.positions, dtype=np.float64),
        cutoff=float(cutoff),
        pbc=pbc,
        lattice=lattice,
        backend=backend,
    )
    edge_index = torch.as_tensor(np.asarray(edge_index_np, dtype=np.int64), dtype=torch.long, device=resolved_device)
    edge_shifts = torch.as_tensor(np.asarray(shifts_np, dtype=np.int64), dtype=torch.long, device=resolved_device)
    pos = torch.as_tensor(np.asarray(atoms.positions, dtype=np.float64), dtype=resolved_dtype, device=resolved_device)
    z = torch.as_tensor(np.asarray(atoms.numbers, dtype=np.int64), dtype=torch.long, device=resolved_device)
    batch = torch.zeros(len(atoms), dtype=torch.long, device=resolved_device)
    cell = torch.as_tensor(np.asarray(lattice_np, dtype=np.float64), dtype=resolved_dtype, device=resolved_device).reshape(1, 3, 3)
    edge_batch = torch.zeros(edge_index.shape[1], dtype=torch.long, device=resolved_device)
    return RTECEGraph(
        z=z,
        pos=pos,
        edge_index=edge_index,
        batch=batch,
        cell=cell,
        edge_shifts=edge_shifts,
        edge_batch=edge_batch,
    )


def _voigt_6_from_stress_tensor(stress: torch.Tensor) -> np.ndarray:
    return np.asarray(
        [
            stress[0, 0].detach().cpu().item(),
            stress[1, 1].detach().cpu().item(),
            stress[2, 2].detach().cpu().item(),
            stress[1, 2].detach().cpu().item(),
            stress[0, 2].detach().cpu().item(),
            stress[0, 1].detach().cpu().item(),
        ],
        dtype=np.float64,
    )


def _strained_graph(graph: RTECEGraph, deformation: torch.Tensor) -> RTECEGraph:
    cell = None if graph.cell is None else graph.cell @ deformation
    return RTECEGraph(
        z=graph.z,
        pos=graph.pos @ deformation,
        edge_index=graph.edge_index,
        batch=graph.batch,
        cell=cell,
        edge_shifts=graph.edge_shifts,
        edge_batch=graph.edge_batch,
    )


class RTECEAseCalc(Calculator):
    """ASE calculator for scalar rTECE checkpoints.

    This is the package-level user inference path for the lightweight rTECE
    family. Energy and forces use the model's conservative autograd path by
    default. Stress is a validated finite-strain autograd inference path; fused
    edge-gradient virial remains a separate future backend.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model: str | Path | RTECEScalarModel,
        *,
        config: RTECEScalarConfig | None = None,
        dtype: str | torch.dtype | None = None,
        device: str | torch.device | None = None,
        cutoff: float | None = None,
        force_mode: str = "autograd",
        neighborlist_backend: str = "matscipy",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.dtype = _resolve_dtype(dtype)
        self.device = _resolve_device(device)
        self.force_mode = str(force_mode)
        self.neighborlist_backend = str(neighborlist_backend)
        self.metadata: dict[str, Any] = {}

        if isinstance(model, RTECEScalarModel):
            if config is None:
                config = model.config
            self.model = model.to(device=self.device, dtype=self.dtype)
            self.config = config
        else:
            self.model, self.config, self.metadata = load_checkpoint(model, dtype=self.dtype, device=self.device)
        self.cutoff = float(cutoff if cutoff is not None else self.config.cutoff)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def _finite_strain_stress(self, graph: RTECEGraph) -> np.ndarray:
        if self.force_mode != "autograd":
            raise PropertyNotImplementedError(
                "rTECE ASE stress currently requires force_mode='autograd'; "
                "analytic/triton virial backends are not implemented yet"
            )
        if graph.cell is None:
            raise PropertyNotImplementedError("rTECE ASE stress requires a cell tensor")
        if graph.cell.reshape(-1, 3, 3).shape[0] != 1:
            raise PropertyNotImplementedError("rTECE ASE stress currently supports one ASE Atoms object at a time")

        cell = graph.cell.reshape(1, 3, 3).to(device=self.device, dtype=self.dtype)
        volume = torch.linalg.det(cell[0]).abs()
        if not torch.isfinite(volume) or float(volume.detach().cpu()) <= 0.0:
            raise PropertyNotImplementedError("rTECE ASE stress requires a positive finite cell volume")

        strain = torch.zeros((3, 3), dtype=self.dtype, device=self.device, requires_grad=True)
        deformation = torch.eye(3, dtype=self.dtype, device=self.device) + strain
        strained = _strained_graph(graph, deformation)
        outputs = predict(
            self.model,
            strained,
            force_mode="autograd",
            graph_construction_backend="ase_neighborlist",
        )
        grad = torch.autograd.grad(outputs["energy"].sum(), strain, retain_graph=False, create_graph=False)[0]
        stress_tensor = 0.5 * (grad + grad.transpose(0, 1)) / volume
        return _voigt_6_from_stress_tensor(stress_tensor)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        requested = set(properties or self.implemented_properties)
        unsupported = requested.difference(self.implemented_properties)
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise PropertyNotImplementedError(f"rTECE ASE calculator does not implement: {names}")

        Calculator.calculate(self, atoms, properties, system_changes)
        graph = atoms_to_rtece_graph(
            atoms,
            cutoff=self.cutoff,
            device=self.device,
            dtype=self.dtype,
            neighborlist_backend=self.neighborlist_backend,
        )
        with torch.enable_grad():
            outputs = predict(
                self.model,
                graph,
                force_mode=self.force_mode,
                graph_construction_backend="ase_neighborlist",
            )
            stress = self._finite_strain_stress(graph) if "stress" in requested else None

        energy = outputs["energy"].detach().cpu().reshape(-1)[0].item()
        forces = outputs["forces"].detach().cpu().numpy()
        self.results = {
            "energy": float(energy),
            "free_energy": float(energy),
            "forces": np.asarray(forces, dtype=np.float64),
        }
        if stress is not None:
            self.results["stress"] = stress


__all__ = ["RTECEAseCalc", "atoms_to_rtece_graph"]
