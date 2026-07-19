from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .rtece_scalar import (
    RTECEGraph,
    RTECEScalarConfig,
    RTECEScalarModel,
    rtece_path_manifest,
    rtece_route_contract,
)


def save_checkpoint(
    path: str | Path,
    model: RTECEScalarModel,
    config: RTECEScalarConfig,
    *,
    force_mode: str = "autograd",
    graph_construction_backend: str | None = None,
    graph_update_backend: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    route = rtece_route_contract(
        config,
        force_mode=force_mode,
        graph_construction_backend=graph_construction_backend,
        graph_update_backend=graph_update_backend,
    )
    path_manifest = rtece_path_manifest(
        config,
        force_mode=force_mode,
        graph_construction_backend=graph_construction_backend,
        graph_update_backend=graph_update_backend,
    )
    payload = {
        "config": asdict(config),
        "state_dict": model.state_dict(),
        "tece_route": route,
        "tece_path_manifest": path_manifest,
        "metadata": dict(metadata or {}),
    }
    torch.save(payload, target)


def load_checkpoint(
    path: str | Path,
    *,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> tuple[RTECEScalarModel, RTECEScalarConfig, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=device)
    config_payload = dict(payload["config"])
    if config_payload.get("atomic_energies") is not None:
        config_payload["atomic_energies"] = {int(k): float(v) for k, v in config_payload["atomic_energies"].items()}
    config = RTECEScalarConfig(**config_payload)
    model = RTECEScalarModel(config).to(device=device, dtype=dtype)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    metadata = dict(payload.get("metadata") or {})
    metadata["tece_route"] = payload.get("tece_route") or rtece_route_contract(config)
    metadata["tece_path_manifest"] = payload.get("tece_path_manifest") or rtece_path_manifest(config)
    return model, config, metadata


def predict(
    model: RTECEScalarModel,
    graph: RTECEGraph,
    *,
    force_mode: str = "autograd",
    graph_construction_backend: str | None = None,
    graph_update_backend: str | None = None,
    include_route: bool = False,
) -> dict[str, torch.Tensor | dict[str, object]]:
    if force_mode == "autograd":
        out = model(graph)
    elif force_mode == "analytic_pair":
        out = model.forward_pair_analytic_forces(graph)
    elif force_mode == "analytic_pair_triton_force":
        out = model.forward_pair_triton_force_analytic_forces(graph)
    elif force_mode == "analytic_density":
        out = model.forward_density_analytic_forces(graph)
    elif force_mode == "analytic_element_packed":
        out = model.forward_element_density_packed_analytic_forces(graph)
    elif force_mode == "analytic_element_triton_force":
        out = model.forward_element_density_triton_force_analytic_forces(graph)
    elif force_mode == "analytic_element_triton_descriptor_force":
        out = model.forward_element_density_triton_descriptor_force_analytic_forces(graph)
    elif force_mode == "analytic_element_direct_padded_descriptor_force":
        out = model.forward_element_density_direct_padded_triton_descriptor_force_analytic_forces(graph)
    elif force_mode == "analytic_element_cell_list_descriptor_force":
        out = model.forward_element_density_cell_list_packed_analytic_forces(graph)
    else:
        raise ValueError(f"unknown rTECE force mode {force_mode!r}")
    if include_route:
        out = dict(out)
        out["tece_route"] = rtece_route_contract(
            model.config,
            force_mode=force_mode,
            graph_construction_backend=graph_construction_backend,
            graph_update_backend=graph_update_backend,
        )
    return out


def loss_for_batch(
    model: RTECEScalarModel,
    graph: RTECEGraph,
    ref_energy: torch.Tensor,
    ref_forces: torch.Tensor,
    *,
    energy_weight: float = 1.0,
    force_weight: float = 10.0,
    force_focus_atomic_numbers: tuple[int, ...] = (),
    force_focus_weight: float = 1.0,
) -> torch.Tensor:
    out = model(graph)
    natoms = graph.z.numel()
    e_loss = ((out["energy"] - ref_energy) / natoms).pow(2).mean()
    force_sq = (out["forces"] - ref_forces).pow(2)
    if force_focus_atomic_numbers and float(force_focus_weight) != 1.0:
        focus_numbers = torch.tensor(tuple(int(z) for z in force_focus_atomic_numbers), dtype=graph.z.dtype, device=graph.z.device)
        focus_mask = (graph.z.view(-1, 1) == focus_numbers.view(1, -1)).any(dim=1)
        atom_weights = torch.ones(graph.z.shape[0], dtype=force_sq.dtype, device=force_sq.device)
        atom_weights = torch.where(focus_mask, torch.full_like(atom_weights, float(force_focus_weight)), atom_weights)
        atom_weights = atom_weights / atom_weights.mean().clamp_min(torch.finfo(atom_weights.dtype).tiny)
        force_sq = force_sq * atom_weights.view(-1, 1)
    f_loss = force_sq.mean()
    return float(energy_weight) * e_loss + float(force_weight) * f_loss


def evaluate_loss(
    model: RTECEScalarModel,
    samples: list[tuple[RTECEGraph, torch.Tensor, torch.Tensor]],
    *,
    energy_weight: float = 1.0,
    force_weight: float = 10.0,
    force_focus_atomic_numbers: tuple[int, ...] = (),
    force_focus_weight: float = 1.0,
) -> float:
    if not samples:
        raise ValueError("evaluate_loss requires at least one sample")
    was_training = model.training
    model.eval()
    losses = []
    for graph, energy, forces in samples:
        losses.append(
            float(
                loss_for_batch(
                    model,
                    graph,
                    energy,
                    forces,
                    energy_weight=energy_weight,
                    force_weight=force_weight,
                    force_focus_atomic_numbers=force_focus_atomic_numbers,
                    force_focus_weight=force_focus_weight,
                )
                .detach()
                .cpu()
            )
        )
    if was_training:
        model.train()
    return float(sum(losses) / len(losses))


def train_steps(
    model: RTECEScalarModel,
    samples: list[tuple[RTECEGraph, torch.Tensor, torch.Tensor]],
    *,
    max_steps: int,
    lr: float,
    valid_samples: list[tuple[RTECEGraph, torch.Tensor, torch.Tensor]] | None = None,
    eval_interval: int = 0,
    min_eval_step: int = 0,
    best_checkpoint_path: str | Path | None = None,
    config: RTECEScalarConfig | None = None,
    energy_weight: float = 1.0,
    force_weight: float = 10.0,
    force_focus_atomic_numbers: tuple[int, ...] = (),
    force_focus_weight: float = 1.0,
) -> dict[str, float | int | None]:
    if not samples:
        raise ValueError("train_steps requires at least one sample")
    if best_checkpoint_path is not None and config is None:
        raise ValueError("config is required when best_checkpoint_path is set")
    min_eval_step = max(0, int(min_eval_step))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    final_loss: float | None = None
    best_valid_loss: float | None = None
    best_step: int | None = None
    for step in range(max_steps):
        graph, energy, forces = samples[step % len(samples)]
        opt.zero_grad(set_to_none=True)
        loss = loss_for_batch(
            model,
            graph,
            energy,
            forces,
            energy_weight=energy_weight,
            force_weight=force_weight,
            force_focus_atomic_numbers=force_focus_atomic_numbers,
            force_focus_weight=force_focus_weight,
        )
        loss.backward()
        opt.step()
        final_loss = float(loss.detach().cpu())
        step_num = step + 1
        if valid_samples is not None and eval_interval > 0 and step_num >= min_eval_step and step_num % eval_interval == 0:
            valid_loss = evaluate_loss(
                model,
                valid_samples,
                energy_weight=energy_weight,
                force_weight=force_weight,
                force_focus_atomic_numbers=force_focus_atomic_numbers,
                force_focus_weight=force_focus_weight,
            )
            if best_valid_loss is None or valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_step = step_num
                if best_checkpoint_path is not None:
                    save_checkpoint(best_checkpoint_path, model, config)
    if valid_samples is not None and best_valid_loss is None and max_steps >= min_eval_step:
        best_valid_loss = evaluate_loss(
            model,
            valid_samples,
            energy_weight=energy_weight,
            force_weight=force_weight,
            force_focus_atomic_numbers=force_focus_atomic_numbers,
            force_focus_weight=force_focus_weight,
        )
        best_step = max_steps
        if best_checkpoint_path is not None:
            save_checkpoint(best_checkpoint_path, model, config)
    return {
        "steps": max_steps,
        "final_loss": final_loss,
        "best_valid_loss": best_valid_loss,
        "best_step": best_step,
    }


__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "predict",
    "loss_for_batch",
    "evaluate_loss",
    "train_steps",
]
