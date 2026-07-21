#!/usr/bin/env python3
"""Measure descriptor-space projection error between rTECE path-id routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def _as_float64_matrix(values: torch.Tensor, name: str) -> torch.Tensor:
    import torch

    if values.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 matrix, got shape {tuple(values.shape)}")
    if values.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one sample")
    return values.detach().to(dtype=torch.float64, device="cpu")


def _as_sample_weights(sample_weights: torch.Tensor | None, num_samples: int) -> torch.Tensor | None:
    import torch

    if sample_weights is None:
        return None
    weights = sample_weights.detach().to(dtype=torch.float64, device="cpu").flatten()
    if weights.numel() != num_samples:
        raise ValueError(f"sample_weights must contain {num_samples} values, got {weights.numel()}")
    if not bool(torch.isfinite(weights).all().item()):
        raise ValueError("sample_weights must be finite")
    if bool((weights < 0.0).any().item()):
        raise ValueError("sample_weights must be non-negative")
    if float(weights.sum().item()) <= 0.0:
        raise ValueError("sample_weights must have positive total weight")
    return weights


def _ridge_projection_residual(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float = 1.0e-12,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, float]:
    import torch

    x = _as_float64_matrix(source, "source")
    y = _as_float64_matrix(target, "target")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"source/target sample counts differ: {x.shape[0]} vs {y.shape[0]}")
    if x.shape[1] < 1 or y.shape[1] < 1:
        raise ValueError("source and target must both have at least one descriptor column")

    weights = _as_sample_weights(sample_weights, int(x.shape[0]))
    if weights is None:
        fit_x = x
        fit_y = y
        residual_scale = None
        weight_sum = float(x.shape[0])
    else:
        residual_scale = torch.sqrt(weights).unsqueeze(-1)
        fit_x = x * residual_scale
        fit_y = y * residual_scale
        weight_sum = float(weights.sum().item())

    lhs = fit_x.T @ fit_x
    if ridge > 0.0:
        lhs = lhs + float(ridge) * torch.eye(lhs.shape[0], dtype=x.dtype)
    rhs = fit_x.T @ fit_y
    if ridge <= 0.0:
        coeff = torch.linalg.lstsq(fit_x, fit_y, driver="gelsd").solution
    else:
        try:
            coeff = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            coeff = torch.linalg.lstsq(fit_x, fit_y, driver="gelsd").solution
    residual = y - x @ coeff
    return coeff, residual, residual_scale, weight_sum


def projection_residual_metrics(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float = 1.0e-12,
    sample_weights: torch.Tensor | None = None,
) -> dict[str, float | int | bool]:
    """Project target descriptors onto source descriptors and report residual size."""
    import torch

    x = _as_float64_matrix(source, "source")
    y = _as_float64_matrix(target, "target")
    _coeff, residual, residual_scale, weight_sum = _ridge_projection_residual(
        x,
        y,
        ridge=ridge,
        sample_weights=sample_weights,
    )
    if residual_scale is None:
        residual_for_norm = residual
        target_for_norm = y
    else:
        residual_for_norm = residual * residual_scale
        target_for_norm = y * residual_scale
    residual_norm = torch.linalg.vector_norm(residual_for_norm)
    target_norm = torch.linalg.vector_norm(target_for_norm)
    relative = residual_norm / target_norm.clamp_min(torch.finfo(y.dtype).tiny)
    return {
        "num_samples": int(x.shape[0]),
        "source_dim": int(x.shape[1]),
        "target_dim": int(y.shape[1]),
        "residual_frobenius": float(residual_norm.item()),
        "target_frobenius": float(target_norm.item()),
        "relative_residual": float(relative.item()),
        "ridge": float(ridge),
        "weighted": sample_weights is not None,
        "weight_sum": weight_sum,
    }


def _normalize_row_indices(indices: torch.Tensor | list[int] | tuple[int, ...] | None, num_rows: int, name: str) -> torch.Tensor:
    import torch

    if indices is None:
        return torch.arange(int(num_rows), dtype=torch.long)
    values = torch.as_tensor(indices, dtype=torch.long, device="cpu").flatten()
    if values.numel() < 1:
        raise ValueError(f"{name} must contain at least one row index")
    if bool((values < 0).any().item()) or bool((values >= int(num_rows)).any().item()):
        raise ValueError(f"{name} contains an index outside [0, {int(num_rows)})")
    return values


def energy_label_projection_metrics(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float = 1.0e-12,
    sample_weights: torch.Tensor | None = None,
    atom_counts: torch.Tensor | None = None,
    baseline_features: torch.Tensor | None = None,
    fit_indices: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    eval_indices: torch.Tensor | list[int] | tuple[int, ...] | None = None,
) -> dict[str, float | int | bool | None]:
    import torch

    x = _as_float64_matrix(source, "source")
    y = target.detach().to(dtype=torch.float64, device="cpu")
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    y = _as_float64_matrix(y, "target")
    if baseline_features is None:
        fit_x = x
        baseline_dim = 0
    else:
        baseline = _as_float64_matrix(baseline_features, "baseline_features")
        if baseline.shape[0] != x.shape[0]:
            raise ValueError(f"baseline_features must contain {x.shape[0]} rows, got {baseline.shape[0]}")
        fit_x = torch.cat([x, baseline], dim=-1)
        baseline_dim = int(baseline.shape[1])
    fit_rows = _normalize_row_indices(fit_indices, int(x.shape[0]), "fit_indices")
    eval_rows = _normalize_row_indices(eval_indices, int(x.shape[0]), "eval_indices")
    fit_weights = None
    if sample_weights is not None:
        all_weights = _as_sample_weights(sample_weights, int(x.shape[0]))
        fit_weights = all_weights[fit_rows]
    coeff, _fit_residual, _fit_scale, weight_sum = _ridge_projection_residual(
        fit_x[fit_rows],
        y[fit_rows],
        ridge=ridge,
        sample_weights=fit_weights,
    )
    eval_x = fit_x[eval_rows]
    eval_y = y[eval_rows]
    residual = eval_y - eval_x @ coeff
    if sample_weights is not None:
        eval_weights = _as_sample_weights(sample_weights, int(x.shape[0]))[eval_rows]
        residual_scale = torch.sqrt(eval_weights).unsqueeze(-1)
    else:
        residual_scale = None
    if residual_scale is None:
        residual_for_norm = residual
        target_for_norm = eval_y
    else:
        residual_for_norm = residual * residual_scale
        target_for_norm = eval_y * residual_scale
    residual_norm = torch.linalg.vector_norm(residual_for_norm)
    target_norm = torch.linalg.vector_norm(target_for_norm)
    relative = residual_norm / target_norm.clamp_min(torch.finfo(y.dtype).tiny)
    flat_residual = residual.flatten()
    payload: dict[str, float | int | bool | None] = {
        "energy_num_samples": int(eval_rows.numel()),
        "energy_fit_num_samples": int(fit_rows.numel()),
        "energy_eval_num_samples": int(eval_rows.numel()),
        "energy_source_dim": int(x.shape[1]),
        "energy_baseline_dim": int(baseline_dim),
        "energy_fit_dim": int(fit_x.shape[1]),
        "energy_degrees_of_freedom": int(fit_rows.numel() - fit_x.shape[1]),
        "energy_underdetermined": bool(fit_x.shape[1] >= fit_rows.numel()),
        "energy_target_dim": int(y.shape[1]),
        "energy_residual_frobenius": float(residual_norm.item()),
        "energy_target_frobenius": float(target_norm.item()),
        "relative_energy_residual": float(relative.item()),
        "energy_rmse": float(torch.sqrt(torch.mean(flat_residual.square())).item()),
        "energy_mae": float(torch.mean(torch.abs(flat_residual)).item()),
        "energy_bias": float(torch.mean(flat_residual).item()),
        "energy_max_abs": float(torch.max(torch.abs(flat_residual)).item()),
        "energy_ridge": float(ridge),
        "energy_weighted": sample_weights is not None,
        "energy_weight_sum": weight_sum,
        "energy_per_atom_rmse": None,
        "energy_per_atom_mae": None,
        "energy_per_atom_bias": None,
        "energy_per_atom_max_abs": None,
    }
    if atom_counts is not None:
        counts = atom_counts.detach().to(dtype=torch.float64, device="cpu").flatten()
        if counts.numel() != int(x.shape[0]):
            raise ValueError(f"atom_counts must contain {x.shape[0]} values, got {counts.numel()}")
        if bool((counts <= 0.0).any().item()):
            raise ValueError("atom_counts must be positive")
        per_atom = residual.flatten() / counts[eval_rows]
        payload.update(
            {
                "energy_per_atom_rmse": float(torch.sqrt(torch.mean(per_atom.square())).item()),
                "energy_per_atom_mae": float(torch.mean(torch.abs(per_atom)).item()),
                "energy_per_atom_bias": float(torch.mean(per_atom).item()),
                "energy_per_atom_max_abs": float(torch.max(torch.abs(per_atom)).item()),
            }
        )
    return payload


def force_label_projection_metrics(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    ridge: float = 1.0e-12,
    sample_weights: torch.Tensor | None = None,
    fit_indices: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    eval_indices: torch.Tensor | list[int] | tuple[int, ...] | None = None,
) -> dict[str, float | int | bool]:
    import torch

    x = _as_float64_matrix(source, "source")
    y = target.detach().to(dtype=torch.float64, device="cpu")
    if y.ndim == 1:
        y = y.unsqueeze(-1)
    y = _as_float64_matrix(y, "target")
    if y.shape[0] != x.shape[0]:
        raise ValueError(f"force target rows must match source rows: {y.shape[0]} vs {x.shape[0]}")
    fit_rows = _normalize_row_indices(fit_indices, int(x.shape[0]), "fit_indices")
    eval_rows = _normalize_row_indices(eval_indices, int(x.shape[0]), "eval_indices")
    fit_weights = None
    if sample_weights is not None:
        all_weights = _as_sample_weights(sample_weights, int(x.shape[0]))
        fit_weights = all_weights[fit_rows]
    coeff, _fit_residual, _fit_scale, weight_sum = _ridge_projection_residual(
        x[fit_rows],
        y[fit_rows],
        ridge=ridge,
        sample_weights=fit_weights,
    )
    residual = y[eval_rows] - x[eval_rows] @ coeff
    if sample_weights is not None:
        eval_weights = _as_sample_weights(sample_weights, int(x.shape[0]))[eval_rows]
        residual_scale = torch.sqrt(eval_weights).unsqueeze(-1)
    else:
        residual_scale = None
    if residual_scale is None:
        residual_for_norm = residual
        target_for_norm = y[eval_rows]
    else:
        residual_for_norm = residual * residual_scale
        target_for_norm = y[eval_rows] * residual_scale
    residual_norm = torch.linalg.vector_norm(residual_for_norm)
    target_norm = torch.linalg.vector_norm(target_for_norm)
    relative = residual_norm / target_norm.clamp_min(torch.finfo(y.dtype).tiny)
    flat_residual = residual.flatten()
    return {
        "force_num_samples": int(eval_rows.numel()),
        "force_fit_num_samples": int(fit_rows.numel()),
        "force_eval_num_samples": int(eval_rows.numel()),
        "force_source_dim": int(x.shape[1]),
        "force_fit_dim": int(x.shape[1]),
        "force_degrees_of_freedom": int(fit_rows.numel() - x.shape[1]),
        "force_underdetermined": bool(x.shape[1] >= fit_rows.numel()),
        "force_target_dim": int(y.shape[1]),
        "force_residual_frobenius": float(residual_norm.item()),
        "force_target_frobenius": float(target_norm.item()),
        "relative_force_residual": float(relative.item()),
        "force_rmse": float(torch.sqrt(torch.mean(flat_residual.square())).item()),
        "force_mae": float(torch.mean(torch.abs(flat_residual)).item()),
        "force_bias": float(torch.mean(flat_residual).item()),
        "force_max_abs": float(torch.max(torch.abs(flat_residual)).item()),
        "force_ridge": float(ridge),
        "force_weighted": sample_weights is not None,
        "force_weight_sum": weight_sum,
    }


def _force_descriptor_matrix(
    graphs: list[RTECEGraph],
    config: RTECEScalarConfig,
    *,
    force_row_indices: torch.Tensor | list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    import torch
    from tace.models.rtece_scalar import RTECEGraph, rtece_descriptors

    if not graphs:
        raise ValueError("force projection diagnostic requires at least one graph")

    total_force_rows = sum(int(graph.pos.numel()) for graph in graphs)
    selected_rows = (
        None
        if force_row_indices is None
        else _normalize_row_indices(force_row_indices, total_force_rows, "force_row_indices")
    )

    def descriptor_sum(graph: RTECEGraph, pos: torch.Tensor) -> torch.Tensor:
        graph_with_pos = RTECEGraph(
            z=graph.z,
            pos=pos,
            edge_index=graph.edge_index,
            batch=graph.batch,
            cell=graph.cell,
            edge_shifts=graph.edge_shifts,
            edge_batch=graph.edge_batch,
        )
        return rtece_descriptors(graph_with_pos, config).sum(dim=0)

    if selected_rows is None:
        matrices = []
        for graph in graphs:
            pos = graph.pos.detach().clone().requires_grad_(True)
            phi = descriptor_sum(graph, pos)
            columns = []
            for column_idx in range(int(phi.numel())):
                if not phi[column_idx].requires_grad:
                    grad = torch.zeros_like(pos)
                else:
                    grad = torch.autograd.grad(
                        phi[column_idx],
                        pos,
                        retain_graph=True,
                        allow_unused=True,
                    )[0]
                    if grad is None:
                        grad = torch.zeros_like(pos)
                columns.append((-grad).reshape(-1))
            if columns:
                matrices.append(torch.stack(columns, dim=1).detach().cpu())
            else:
                matrices.append(pos.new_zeros((int(pos.numel()), 0)).detach().cpu())
        return torch.cat(matrices, dim=0)

    graph_offsets: list[int] = []
    offset = 0
    for graph in graphs:
        graph_offsets.append(offset)
        offset += int(graph.pos.numel())
    rows_by_graph: dict[int, list[int]] = {idx: [] for idx in range(len(graphs))}
    for row in selected_rows.tolist():
        row_int = int(row)
        for graph_idx, graph_offset in enumerate(graph_offsets):
            graph_end = graph_offset + int(graphs[graph_idx].pos.numel())
            if graph_offset <= row_int < graph_end:
                rows_by_graph[graph_idx].append(row_int - graph_offset)
                break

    sampled_rows = []
    for graph_idx, local_rows in rows_by_graph.items():
        if not local_rows:
            continue
        graph = graphs[graph_idx]
        base_pos = graph.pos.detach().clone()

        def fn(pos: torch.Tensor) -> torch.Tensor:
            return descriptor_sum(graph, pos)

        for local_row in local_rows:
            tangent = torch.zeros_like(base_pos)
            tangent.reshape(-1)[int(local_row)] = 1.0
            _value, jvp = torch.autograd.functional.jvp(fn, (base_pos,), (tangent,), create_graph=False, strict=False)
            sampled_rows.append((-jvp).detach().cpu())
    if not sampled_rows:
        raise ValueError("force_row_indices selected no force rows")
    return torch.stack(sampled_rows, dim=0)


def _force_row_indices_from_config_indices(
    graphs: list[RTECEGraph],
    config_indices: torch.Tensor | list[int] | tuple[int, ...] | None,
) -> torch.Tensor:
    import torch

    graph_rows = _normalize_row_indices(config_indices, len(graphs), "force_config_indices")
    offsets = []
    offset = 0
    for graph in graphs:
        offsets.append(offset)
        offset += int(graph.z.numel()) * 3
    rows = []
    for graph_idx in graph_rows.tolist():
        start = int(offsets[int(graph_idx)])
        end = start + int(graphs[int(graph_idx)].z.numel()) * 3
        rows.append(torch.arange(start, end, dtype=torch.long))
    return torch.cat(rows, dim=0)


def _take_evenly_spaced_rows(rows: torch.Tensor, count: int) -> torch.Tensor:
    import torch

    values = rows.detach().to(dtype=torch.long, device="cpu").flatten()
    if values.numel() < 1:
        raise ValueError("cannot sample from an empty row set")
    take = min(int(count), int(values.numel()))
    if take < 1:
        raise ValueError("sample count must be positive")
    if take == int(values.numel()):
        return values
    positions = torch.linspace(0, int(values.numel()) - 1, steps=take, dtype=torch.float64).round().to(dtype=torch.long)
    return values[positions]


def deterministic_force_component_sample(
    force_fit_indices: torch.Tensor | list[int] | tuple[int, ...],
    force_eval_indices: torch.Tensor | list[int] | tuple[int, ...],
    *,
    sample_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import torch

    fit = torch.as_tensor(force_fit_indices, dtype=torch.long, device="cpu").flatten()
    eval_rows = torch.as_tensor(force_eval_indices, dtype=torch.long, device="cpu").flatten()
    count = int(sample_count)
    if count < 1:
        raise ValueError("force component sample count must be positive")
    if fit.numel() < 1 or eval_rows.numel() < 1:
        raise ValueError("force component sampling requires non-empty fit and eval rows")
    if fit.numel() == eval_rows.numel() and bool(torch.equal(fit, eval_rows)):
        sampled = _take_evenly_spaced_rows(fit, count)
        remap = torch.arange(int(sampled.numel()), dtype=torch.long)
        return sampled, remap, remap

    fit_count = max(1, count // 2)
    eval_count = max(1, count - fit_count)
    sampled_fit = _take_evenly_spaced_rows(fit, fit_count)
    sampled_eval = _take_evenly_spaced_rows(eval_rows, eval_count)
    sampled_values: list[int] = []
    for value in sampled_fit.tolist() + sampled_eval.tolist():
        value_int = int(value)
        if value_int not in sampled_values:
            sampled_values.append(value_int)
    sampled = torch.tensor(sampled_values, dtype=torch.long)
    positions = {int(value): idx for idx, value in enumerate(sampled.tolist())}
    remapped_fit = torch.tensor([positions[int(value)] for value in sampled_fit.tolist()], dtype=torch.long)
    remapped_eval = torch.tensor([positions[int(value)] for value in sampled_eval.tolist()], dtype=torch.long)
    return sampled, remapped_fit, remapped_eval


def _force_component_weights_from_atom_weights(
    graphs: list[RTECEGraph],
    sample_weights: torch.Tensor | None,
) -> torch.Tensor | None:
    if sample_weights is None:
        return None
    atom_count = sum(int(graph.z.numel()) for graph in graphs)
    weights = _as_sample_weights(sample_weights, atom_count)
    return weights.repeat_interleave(3)


def _descriptor_matrices(graphs: list[RTECEGraph], config: RTECEScalarConfig) -> tuple[torch.Tensor, torch.Tensor]:
    import torch
    from tace.models.rtece_scalar import rtece_descriptors

    if not graphs:
        raise ValueError("projection diagnostic requires at least one graph")
    per_graph = [rtece_descriptors(graph, config).detach().cpu() for graph in graphs]
    return torch.cat(per_graph, dim=0), torch.stack([values.sum(dim=0) for values in per_graph], dim=0)


def _descriptor_matrix(graphs: list[RTECEGraph], config: RTECEScalarConfig) -> torch.Tensor:
    return _descriptor_matrices(graphs, config)[0]


def _graph_descriptor_matrix(graphs: list[RTECEGraph], config: RTECEScalarConfig) -> torch.Tensor:
    return _descriptor_matrices(graphs, config)[1]


def build_projection_config(
    variant: str,
    scalar_path_ids: tuple[str, ...] | list[str],
    *,
    cutoff: float = 5.0,
    num_radial: int = 8,
    species_basis_channels: int = 0,
    atomic_cross_radial_sketch_channels: int = 2,
) -> RTECEScalarConfig:
    from tace.models.rtece_scalar import build_rtece_config_from_path_ids

    return build_rtece_config_from_path_ids(
        variant,
        scalar_path_ids,
        cutoff=float(cutoff),
        num_radial=int(num_radial),
        species_basis_channels=int(species_basis_channels),
        atomic_cross_radial_sketch_channels=int(atomic_cross_radial_sketch_channels),
    )


def _deleted_path_ids(candidate: RTECEScalarConfig, reference: RTECEScalarConfig) -> list[str]:
    candidate_paths = set(candidate.scalar_path_ids or ())
    return [path_id for path_id in (reference.scalar_path_ids or ()) if path_id not in candidate_paths]


def _path_specs_by_id(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(path["id"]): dict(path) for path in manifest.get("scalar_paths", [])}


def _path_descriptor_slices(config: RTECEScalarConfig) -> dict[str, slice]:
    from tace.models.rtece_scalar import _scalar_path_descriptor_dim, descriptor_dim, rtece_path_manifest

    manifest = rtece_path_manifest(config)
    offset = 0
    slices: dict[str, slice] = {}
    for path in manifest.get("scalar_paths", []):
        path_id = str(path["id"])
        width = int(_scalar_path_descriptor_dim(path_id, config))
        slices[path_id] = slice(offset, offset + width)
        offset += width
    expected_dim = int(descriptor_dim(config))
    if offset != expected_dim:
        raise RuntimeError(f"path descriptor slices cover {offset} columns, expected {expected_dim}")
    return slices


def candidate_descriptors_from_reference(
    reference_descriptors: torch.Tensor,
    *,
    candidate_config: RTECEScalarConfig,
    reference_config: RTECEScalarConfig,
) -> torch.Tensor | None:
    if candidate_config.scalar_path_ids is None or reference_config.scalar_path_ids is None:
        return None
    reference_paths = set(reference_config.scalar_path_ids)
    candidate_paths = tuple(candidate_config.scalar_path_ids)
    if any(path_id not in reference_paths for path_id in candidate_paths):
        return None
    if int(candidate_config.num_radial) != int(reference_config.num_radial):
        return None
    if int(candidate_config.species_basis_channels) != int(reference_config.species_basis_channels):
        return None
    if int(candidate_config.atomic_cross_radial_sketch_channels) != int(reference_config.atomic_cross_radial_sketch_channels):
        return None
    if str(candidate_config.atomic_cross_radial_projection) != str(reference_config.atomic_cross_radial_projection):
        return None
    if candidate_config.atomic_cross_radial_projection_matrix != reference_config.atomic_cross_radial_projection_matrix:
        return None

    reference_slices = _path_descriptor_slices(reference_config)
    columns = [reference_descriptors[:, reference_slices[path_id]] for path_id in candidate_paths]
    if not columns:
        return None
    import torch

    return torch.cat(columns, dim=-1) if len(columns) > 1 else columns[0]


def _unique_in_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def make_projection_diagnostic_row(
    candidate_name: str,
    *,
    candidate_config: RTECEScalarConfig,
    reference_config: RTECEScalarConfig,
    graphs: list[RTECEGraph],
    ridge: float = 1.0e-12,
    sample_weights: torch.Tensor | None = None,
) -> dict[str, Any]:
    candidate_descriptors = _descriptor_matrix(graphs, candidate_config)
    reference_descriptors = _descriptor_matrix(graphs, reference_config)
    return make_projection_diagnostic_row_from_matrices(
        candidate_name,
        candidate_config=candidate_config,
        reference_config=reference_config,
        candidate_descriptors=candidate_descriptors,
        reference_descriptors=reference_descriptors,
        num_graphs=len(graphs),
        ridge=ridge,
        sample_weights=sample_weights,
    )


def make_projection_diagnostic_row_from_matrices(
    candidate_name: str,
    *,
    candidate_config: RTECEScalarConfig,
    reference_config: RTECEScalarConfig,
    candidate_descriptors: torch.Tensor,
    reference_descriptors: torch.Tensor,
    num_graphs: int,
    ridge: float = 1.0e-12,
    sample_weights: torch.Tensor | None = None,
) -> dict[str, Any]:
    from tace.models.rtece_scalar import descriptor_dim, rtece_path_manifest

    metrics = projection_residual_metrics(
        candidate_descriptors,
        reference_descriptors,
        ridge=ridge,
        sample_weights=sample_weights,
    )
    candidate_manifest = rtece_path_manifest(candidate_config)
    reference_manifest = rtece_path_manifest(reference_config)
    deleted_path_ids = _deleted_path_ids(candidate_config, reference_config)
    candidate_path_specs = _path_specs_by_id(candidate_manifest)
    reference_path_specs = _path_specs_by_id(reference_manifest)
    deleted_path_specs = [reference_path_specs[path_id] for path_id in deleted_path_ids if path_id in reference_path_specs]
    retained_path_specs = [candidate_path_specs[path_id] for path_id in (candidate_config.scalar_path_ids or ()) if path_id in candidate_path_specs]
    candidate_dim = int(descriptor_dim(candidate_config))
    reference_dim = int(descriptor_dim(reference_config))
    row: dict[str, Any] = {
        "candidate": str(candidate_name),
        "num_graphs": int(num_graphs),
        "candidate_variant": candidate_config.variant,
        "reference_variant": reference_config.variant,
        "candidate_manifest_hash": candidate_manifest["manifest_hash"],
        "reference_manifest_hash": reference_manifest["manifest_hash"],
        "candidate_scalar_path_ids": list(candidate_config.scalar_path_ids or []),
        "reference_scalar_path_ids": list(reference_config.scalar_path_ids or []),
        "deleted_scalar_path_ids": deleted_path_ids,
        "deleted_cost_groups": _unique_in_order([str(path.get("cost_group")) for path in deleted_path_specs]),
        "retained_cost_groups": _unique_in_order([str(path.get("cost_group")) for path in retained_path_specs]),
        "deleted_descriptor_dim": int(reference_dim - candidate_dim),
        "descriptor_dim_reduction": int(reference_dim - candidate_dim),
        "candidate_dim": candidate_dim,
        "reference_dim": reference_dim,
    }
    row.update(metrics)
    return row


def _graph_weights_from_atom_weights(
    graphs: list[RTECEGraph],
    sample_weights: torch.Tensor | None,
) -> torch.Tensor | None:
    import torch

    if sample_weights is None:
        return None
    weights = sample_weights.detach().to(dtype=torch.float64, device="cpu").flatten()
    graph_sizes = [int(graph.z.numel()) for graph in graphs]
    if sum(graph_sizes) != int(weights.numel()):
        return None
    graph_weights = []
    offset = 0
    for size in graph_sizes:
        graph_weights.append(weights[offset : offset + size].mean())
        offset += size
    values = torch.stack(graph_weights).to(dtype=torch.float64)
    return values / values.mean().clamp_min(torch.finfo(values.dtype).tiny)


def graph_element_count_matrix(graphs: list[RTECEGraph]) -> tuple[torch.Tensor, list[int]]:
    import torch

    if not graphs:
        raise ValueError("element count baseline requires at least one graph")
    species = sorted({int(z.item()) for graph in graphs for z in graph.z.detach().cpu().flatten()})
    columns = []
    for graph in graphs:
        z = graph.z.detach().to(device="cpu", dtype=torch.long).flatten()
        columns.append(torch.tensor([(z == atomic_number).sum().item() for atomic_number in species], dtype=torch.float64))
    return torch.stack(columns, dim=0), species


def deterministic_eval_split(num_rows: int, *, eval_stride: int, eval_offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    import torch

    total = int(num_rows)
    stride = int(eval_stride)
    offset = int(eval_offset)
    if total < 2:
        raise ValueError("energy eval split requires at least two rows")
    if stride < 2:
        all_rows = torch.arange(total, dtype=torch.long)
        return all_rows, all_rows
    if offset < 0 or offset >= stride:
        raise ValueError("energy_eval_offset must be in [0, energy_eval_stride)")
    rows = torch.arange(total, dtype=torch.long)
    eval_mask = (rows % stride) == offset
    fit_mask = ~eval_mask
    if int(eval_mask.sum().item()) < 1 or int(fit_mask.sum().item()) < 1:
        raise ValueError("energy eval split must leave at least one fit and one eval row")
    return rows[fit_mask], rows[eval_mask]


def make_projection_diagnostic_rows(
    candidate_configs: list[tuple[str, RTECEScalarConfig]] | tuple[tuple[str, RTECEScalarConfig], ...],
    *,
    reference_config: RTECEScalarConfig,
    graphs: list[RTECEGraph],
    ridge: float = 1.0e-12,
    sample_weights: torch.Tensor | None = None,
    energy_targets: torch.Tensor | None = None,
    atom_counts: torch.Tensor | None = None,
    energy_baseline_features: torch.Tensor | None = None,
    energy_fit_indices: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    energy_eval_indices: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    force_targets: torch.Tensor | None = None,
    force_fit_indices: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    force_eval_indices: torch.Tensor | list[int] | tuple[int, ...] | None = None,
    force_descriptor_indices: torch.Tensor | list[int] | tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    reference_descriptors, reference_graph_descriptors = _descriptor_matrices(graphs, reference_config)
    if energy_targets is None:
        reference_graph_descriptors = None
    reference_force_descriptors = (
        _force_descriptor_matrix(graphs, reference_config, force_row_indices=force_descriptor_indices)
        if force_targets is not None
        else None
    )
    graph_sample_weights = _graph_weights_from_atom_weights(graphs, sample_weights) if energy_targets is not None else None
    force_sample_weights = _force_component_weights_from_atom_weights(graphs, sample_weights) if force_targets is not None else None
    if force_sample_weights is not None and force_descriptor_indices is not None:
        force_sample_weights = force_sample_weights[_normalize_row_indices(force_descriptor_indices, int(force_sample_weights.numel()), "force_descriptor_indices")]
    rows = []
    for candidate_name, candidate_config in candidate_configs:
        candidate_descriptors = candidate_descriptors_from_reference(
            reference_descriptors,
            candidate_config=candidate_config,
            reference_config=reference_config,
        )
        candidate_graph_descriptors = None
        if candidate_descriptors is None:
            candidate_descriptors, candidate_graph_descriptors = _descriptor_matrices(graphs, candidate_config)
        row = make_projection_diagnostic_row_from_matrices(
            candidate_name,
            candidate_config=candidate_config,
            reference_config=reference_config,
            candidate_descriptors=candidate_descriptors,
            reference_descriptors=reference_descriptors,
            num_graphs=len(graphs),
            ridge=ridge,
            sample_weights=sample_weights,
        )
        if energy_targets is not None:
            candidate_graph_from_reference = candidate_descriptors_from_reference(
                reference_graph_descriptors,
                candidate_config=candidate_config,
                reference_config=reference_config,
            ) if reference_graph_descriptors is not None else None
            if candidate_graph_from_reference is not None:
                candidate_graph_descriptors = candidate_graph_from_reference
            if candidate_graph_descriptors is None:
                _candidate_descriptors, candidate_graph_descriptors = _descriptor_matrices(graphs, candidate_config)
            row.update(
                energy_label_projection_metrics(
                    candidate_graph_descriptors,
                    energy_targets,
                    ridge=ridge,
                    sample_weights=graph_sample_weights,
                    atom_counts=atom_counts,
                    baseline_features=energy_baseline_features,
                    fit_indices=energy_fit_indices,
                    eval_indices=energy_eval_indices,
                )
            )
        if force_targets is not None:
            candidate_force_descriptors = candidate_descriptors_from_reference(
                reference_force_descriptors,
                candidate_config=candidate_config,
                reference_config=reference_config,
            ) if reference_force_descriptors is not None else None
            if candidate_force_descriptors is None:
                candidate_force_descriptors = _force_descriptor_matrix(
                    graphs,
                    candidate_config,
                    force_row_indices=force_descriptor_indices,
                )
            row.update(
                force_label_projection_metrics(
                    candidate_force_descriptors,
                    force_targets,
                    ridge=ridge,
                    sample_weights=force_sample_weights,
                    fit_indices=force_fit_indices,
                    eval_indices=force_eval_indices,
                )
            )
        rows.append(row)
    return rows


def _candidate_name(prefix: str, path_id: str | None = None, index: int | None = None) -> str:
    if index is not None:
        return f"{prefix}_{int(index):03d}"
    if path_id is None:
        return prefix
    safe_path = path_id.replace(".", "_").replace("-", "_")
    return f"{prefix}_{safe_path}"


def generate_projection_candidate_specs(
    reference_path_ids: tuple[str, ...] | list[str],
    *,
    strategies: tuple[str, ...] | list[str] = ("single_delete",),
    required_path_ids: tuple[str, ...] | list[str] = ("atomic.radial_density",),
) -> list[tuple[str, tuple[str, ...]]]:
    reference_paths = tuple(str(path_id) for path_id in reference_path_ids)
    required = set(str(path_id) for path_id in required_path_ids)
    if not reference_paths:
        raise ValueError("reference path ids must not be empty")
    missing_required = [path_id for path_id in required if path_id not in reference_paths]
    if missing_required:
        raise ValueError(f"required path ids are not present in reference paths: {missing_required}")

    candidates: list[tuple[str, tuple[str, ...]]] = []
    seen: set[tuple[str, ...]] = set()

    def add_candidate(name: str, paths: tuple[str, ...]) -> None:
        if paths == reference_paths or paths in seen:
            return
        if not paths:
            return
        missing = [path_id for path_id in required if path_id not in paths]
        if missing:
            return
        seen.add(paths)
        candidates.append((name, paths))

    for strategy in tuple(str(value) for value in strategies):
        if strategy == "single_delete":
            for path_id in reference_paths:
                if path_id in required:
                    continue
                add_candidate(
                    _candidate_name("single_delete", path_id),
                    tuple(candidate_path for candidate_path in reference_paths if candidate_path != path_id),
                )
        elif strategy == "prefix":
            for end in range(1, len(reference_paths)):
                add_candidate(_candidate_name("prefix", index=end), reference_paths[:end])
        elif strategy == "cumulative":
            for end in range(1, len(reference_paths) + 1):
                add_candidate(_candidate_name("cumulative", index=end), reference_paths[:end])
        else:
            raise ValueError(f"unknown projection candidate strategy {strategy!r}")
    return candidates


def _safe_float_metric(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None:
        return float("inf")
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return float("inf")
    if metric != metric:
        return float("inf")
    return metric


def _first_available_metric(rows: list[dict[str, Any]], candidates: tuple[str, ...]) -> str | None:
    for key in candidates:
        if any(_safe_float_metric(row, key) < float("inf") for row in rows):
            return key
    return None


def _assign_metric_rank(rows: list[dict[str, Any]], metric_key: str, rank_key: str) -> None:
    ordered = sorted(
        rows,
        key=lambda row: (
            _safe_float_metric(row, metric_key),
            int(row.get("candidate_dim", 10**12)),
            str(row.get("candidate", "")),
        ),
    )
    for index, row in enumerate(ordered, start=1):
        row[rank_key] = int(index)


def _mark_energy_force_pareto(rows: list[dict[str, Any]], energy_metric: str, force_metric: str) -> None:
    objectives = []
    for row in rows:
        objectives.append(
            (
                _safe_float_metric(row, energy_metric),
                _safe_float_metric(row, force_metric),
                float(int(row.get("candidate_dim", 10**12))),
            )
        )
    for row_index, row in enumerate(rows):
        current = objectives[row_index]
        dominated = False
        for other_index, other in enumerate(objectives):
            if other_index == row_index:
                continue
            no_worse = all(other_value <= current_value for other_value, current_value in zip(other, current))
            strictly_better = any(other_value < current_value for other_value, current_value in zip(other, current))
            if no_worse and strictly_better:
                dominated = True
                break
        row["ef_pareto_dominated"] = bool(dominated)


def rank_projection_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            float(row.get("relative_residual", float("inf"))),
            int(row.get("candidate_dim", 10**12)),
            str(row.get("candidate", "")),
        ),
    )
    for index, row in enumerate(ranked, start=1):
        row["projection_rank"] = int(index)

    energy_metric = _first_available_metric(ranked, ("energy_per_atom_rmse", "energy_rmse"))
    force_metric = _first_available_metric(ranked, ("force_rmse",))
    if energy_metric is not None:
        _assign_metric_rank(ranked, energy_metric, "energy_rank")
        for row in ranked:
            row["ef_energy_metric"] = energy_metric
    if force_metric is not None:
        _assign_metric_rank(ranked, force_metric, "force_rank")
        for row in ranked:
            row["ef_force_metric"] = force_metric
    if energy_metric is not None and force_metric is not None:
        for row in ranked:
            row["ef_rank_sum"] = int(row["energy_rank"] + row["force_rank"])
            row["ef_rank_max"] = int(max(row["energy_rank"], row["force_rank"]))
        combined = sorted(
            ranked,
            key=lambda row: (
                int(row["ef_rank_max"]),
                int(row["ef_rank_sum"]),
                int(row.get("candidate_dim", 10**12)),
                str(row.get("candidate", "")),
            ),
        )
        for index, row in enumerate(combined, start=1):
            row["ef_combined_rank"] = int(index)
        _mark_energy_force_pareto(ranked, energy_metric, force_metric)
    return ranked


def _row_path_ids(row: dict[str, Any]) -> tuple[str, ...]:
    for key in ("path_ids", "candidate_scalar_path_ids", "scalar_path_ids"):
        value = row.get(key)
        if value is not None:
            return tuple(str(path_id) for path_id in value)
    return ()


def _metric_gain(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    keys: tuple[str, ...],
    *,
    gain_mode: str = "absolute",
) -> tuple[float, str | None]:
    mode = str(gain_mode)
    if mode not in {"absolute", "relative"}:
        raise ValueError(f"unsupported active-set gain mode {gain_mode!r}")
    for key in keys:
        baseline_value = _safe_float_metric(baseline, key)
        candidate_value = _safe_float_metric(candidate, key)
        if baseline_value < float("inf") and candidate_value < float("inf"):
            absolute_gain = baseline_value - candidate_value
            if mode == "relative":
                denom = max(abs(baseline_value), 1.0e-12)
                return absolute_gain / denom, key
            return absolute_gain, key
    return 0.0, None


def _candidate_cost_proxy(candidate: dict[str, Any], baseline: dict[str, Any]) -> float:
    for key in ("marginal_cost_proxy", "incremental_cost_proxy"):
        value = candidate.get(key)
        if value is not None:
            try:
                cost = float(value)
            except (TypeError, ValueError):
                break
            if cost > 0.0 and cost == cost:
                return cost
    candidate_dim = _safe_float_metric(candidate, "candidate_dim")
    baseline_dim = _safe_float_metric(baseline, "candidate_dim")
    if candidate_dim < float("inf") and baseline_dim < float("inf"):
        return max(candidate_dim - baseline_dim, 1.0)
    return 1.0


def rank_active_set_candidate_rows(
    rows: list[dict[str, Any]],
    *,
    baseline_candidate: str,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    projection_weight: float = 0.0,
    gain_mode: str = "absolute",
    max_force_regression_fraction: float | None = None,
    require_energy_gain: bool = False,
) -> list[dict[str, Any]]:
    baseline_rows = [row for row in rows if str(row.get("candidate")) == str(baseline_candidate)]
    if len(baseline_rows) != 1:
        raise ValueError(f"expected exactly one baseline candidate {baseline_candidate!r}, found {len(baseline_rows)}")
    baseline = baseline_rows[0]
    baseline_paths = set(_row_path_ids(baseline))
    ranked: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("candidate")) == str(baseline_candidate):
            continue
        candidate = dict(row)
        candidate_paths = _row_path_ids(candidate)
        marginal_paths = [path_id for path_id in candidate_paths if path_id not in baseline_paths]
        energy_gain, energy_metric = _metric_gain(
            candidate,
            baseline,
            ("energy_per_atom_rmse", "raw_rmse_mev_atom", "rmse_e_mev_atom", "energy_rmse"),
            gain_mode=gain_mode,
        )
        force_gain, force_metric = _metric_gain(
            candidate,
            baseline,
            ("force_rmse", "rmse_f_mev_a"),
            gain_mode=gain_mode,
        )
        projection_gain, projection_metric = _metric_gain(candidate, baseline, ("relative_residual",), gain_mode=gain_mode)
        weighted_gain = (
            float(energy_weight) * energy_gain
            + float(force_weight) * force_gain
            + float(projection_weight) * projection_gain
        )
        rejection_reasons: list[str] = []
        if bool(require_energy_gain) and energy_gain <= 0.0:
            rejection_reasons.append("energy_gain_required")
        force_regression_fraction = None
        if max_force_regression_fraction is not None:
            if force_metric is None:
                rejection_reasons.append("force_metric_missing")
            else:
                baseline_force_value = _safe_float_metric(baseline, force_metric)
                candidate_force_value = _safe_float_metric(candidate, force_metric)
                if baseline_force_value < float("inf") and candidate_force_value < float("inf"):
                    denom = max(abs(baseline_force_value), 1.0e-12)
                    force_regression_fraction = max(0.0, (candidate_force_value - baseline_force_value) / denom)
                    if force_regression_fraction > float(max_force_regression_fraction) + 1.0e-12:
                        rejection_reasons.append("force_regression_fraction")
                else:
                    rejection_reasons.append("force_metric_missing")
        constraints_passed = not rejection_reasons
        cost = _candidate_cost_proxy(candidate, baseline)
        candidate.update(
            {
                "baseline_candidate": str(baseline_candidate),
                "marginal_paths": marginal_paths,
                "marginal_cost_proxy": float(cost),
                "energy_marginal_gain": float(energy_gain),
                "force_marginal_gain": float(force_gain),
                "projection_marginal_gain": float(projection_gain),
                "active_set_energy_metric": energy_metric,
                "active_set_force_metric": force_metric,
                "active_set_projection_metric": projection_metric,
                "active_set_gain_mode": str(gain_mode),
                "active_set_require_energy_gain": bool(require_energy_gain),
                "active_set_max_force_regression_fraction": (
                    None if max_force_regression_fraction is None else float(max_force_regression_fraction)
                ),
                "force_regression_fraction": force_regression_fraction,
                "active_set_constraint_passed": bool(constraints_passed),
                "active_set_rejection_reasons": rejection_reasons,
                "weighted_marginal_gain": float(weighted_gain),
                "marginal_gain_per_cost": float(weighted_gain / max(cost, 1.0e-12)),
                "active_set_promoted": bool(weighted_gain > 0.0 and constraints_passed),
            }
        )
        ranked.append(candidate)
    ranked.sort(
        key=lambda row: (
            not bool(row.get("active_set_promoted", False)),
            -float(row.get("marginal_gain_per_cost", float("-inf"))),
            float(row.get("marginal_cost_proxy", float("inf"))),
            str(row.get("candidate", "")),
        )
    )
    for index, row in enumerate(ranked, start=1):
        row["active_set_rank"] = int(index)
    return ranked


def rank_metrics_summary(rows: list[dict[str, Any]]) -> dict[str, str | None]:
    energy_metric = _first_available_metric(rows, ("energy_per_atom_rmse", "energy_rmse"))
    force_metric = _first_available_metric(rows, ("force_rmse",))
    return {
        "descriptor": "relative_residual",
        "energy": energy_metric,
        "force": force_metric,
        "combined": "minimize max(energy_rank, force_rank), then rank sum, then descriptor dim"
        if energy_metric is not None and force_metric is not None
        else None,
        "pareto_objectives": "energy metric, force metric, descriptor dim"
        if energy_metric is not None and force_metric is not None
        else None,
    }


def _parse_path_ids(value: str) -> tuple[str, ...]:
    path_ids = tuple(part.strip() for part in value.split(",") if part.strip())
    if not path_ids:
        raise argparse.ArgumentTypeError("path id list must not be empty")
    return path_ids


def _parse_candidate(value: str) -> tuple[str, tuple[str, ...]]:
    parts = value.split(":", 1)
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("candidate must be name:path_id,path_id")
    return parts[0], _parse_path_ids(parts[1])


def parse_focus_elements(value: str | None) -> tuple[tuple[str, ...], tuple[int, ...]]:
    if value is None or not str(value).strip():
        return (), ()
    from ase.data import atomic_numbers, chemical_symbols

    symbols: list[str] = []
    numbers: list[int] = []
    for part in str(value).split(","):
        token = part.strip()
        if not token:
            continue
        if token.isdigit():
            number = int(token)
            if number < 1 or number >= len(chemical_symbols):
                raise ValueError(f"unknown focus atomic number {number}")
            symbol = str(chemical_symbols[number])
        else:
            if token not in atomic_numbers:
                raise ValueError(f"unknown focus element {token!r}")
            symbol = token
            number = int(atomic_numbers[token])
        if number not in numbers:
            numbers.append(number)
            symbols.append(symbol)
    return tuple(symbols), tuple(numbers)


def _format_focus_weight(value: float) -> str:
    return f"{float(value):g}"


def element_focus_sample_weights(
    graphs: list[RTECEGraph],
    *,
    focus_atomic_numbers: tuple[int, ...] | list[int],
    focus_weight: float,
) -> torch.Tensor:
    import torch

    if not graphs:
        raise ValueError("element focus sample weights require at least one graph")
    numbers = tuple(int(value) for value in focus_atomic_numbers)
    if not numbers:
        raise ValueError("focus_atomic_numbers must not be empty")
    if float(focus_weight) <= 0.0:
        raise ValueError("focus_weight must be positive")
    z = torch.cat([graph.z.detach().to(device="cpu", dtype=torch.long).flatten() for graph in graphs], dim=0)
    focus_numbers = torch.tensor(numbers, dtype=torch.long)
    focus_mask = (z.view(-1, 1) == focus_numbers.view(1, -1)).any(dim=1)
    weights = torch.ones(z.shape[0], dtype=torch.float64)
    weights = torch.where(focus_mask, torch.full_like(weights, float(focus_weight)), weights)
    return weights / weights.mean().clamp_min(torch.finfo(weights.dtype).tiny)


def _load_sample_weights_json(path: Path) -> tuple[torch.Tensor, str]:
    import torch

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        if "sample_weights" not in payload:
            raise ValueError("sample weight JSON object must contain sample_weights")
        values = payload["sample_weights"]
        source = str(payload.get("weight_source") or path)
    else:
        values = payload
        source = str(path)
    weights = torch.as_tensor(values, dtype=torch.float64).flatten()
    _as_sample_weights(weights, int(weights.numel()))
    return weights, source


def _read_atoms_energy(atoms: Any, key: str, config_idx: int) -> float:
    if key in atoms.info:
        return float(atoms.info[key])
    if atoms.calc is not None and key in getattr(atoms.calc, "results", {}):
        return float(atoms.calc.results[key])
    if key == "energy":
        try:
            return float(atoms.get_potential_energy())
        except Exception as exc:  # pragma: no cover - ASE calculator-specific path
            raise KeyError(f"configuration {config_idx} missing energy key {key!r}") from exc
    raise KeyError(f"configuration {config_idx} missing energy key {key!r}")


def _load_energy_targets(
    configs: Path,
    *,
    energy_key: str,
    limit_configs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    import ase.io
    import torch

    atoms_list = ase.io.read(str(configs), index=f":{int(limit_configs)}")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    energies = [_read_atoms_energy(atoms, energy_key, config_idx) for config_idx, atoms in enumerate(atoms_list)]
    atom_counts = [len(atoms) for atoms in atoms_list]
    return (
        torch.tensor(energies, dtype=torch.float64),
        torch.tensor(atom_counts, dtype=torch.float64),
    )


def _read_atoms_forces(atoms: Any, key: str, config_idx: int) -> Any:
    if key in atoms.arrays:
        return atoms.arrays[key]
    if atoms.calc is not None and key in getattr(atoms.calc, "results", {}):
        return atoms.calc.results[key]
    if key == "forces":
        try:
            return atoms.get_forces()
        except Exception as exc:  # pragma: no cover - ASE calculator-specific path
            raise KeyError(f"configuration {config_idx} missing force key {key!r}") from exc
    raise KeyError(f"configuration {config_idx} missing force key {key!r}")


def _load_force_targets(
    configs: Path,
    *,
    force_key: str,
    limit_configs: int,
) -> torch.Tensor:
    import ase.io
    import numpy as np
    import torch

    atoms_list = ase.io.read(str(configs), index=f":{int(limit_configs)}")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    rows = []
    for config_idx, atoms in enumerate(atoms_list):
        forces = np.asarray(_read_atoms_forces(atoms, force_key, config_idx), dtype=np.float64)
        expected_shape = (len(atoms), 3)
        if tuple(forces.shape) != expected_shape:
            raise ValueError(
                f"configuration {config_idx} force key {force_key!r} has shape {tuple(forces.shape)}, expected {expected_shape}"
            )
        rows.append(torch.as_tensor(forces.reshape(-1), dtype=torch.float64))
    return torch.cat(rows, dim=0)


def _load_graphs(
    configs: Path,
    *,
    cutoff: float,
    device: torch.device,
    dtype: torch.dtype,
    limit_configs: int,
    neighborlist_backend: str,
) -> list[RTECEGraph]:
    import ase.io
    from tace.interface.ase.rtece_calculator import atoms_to_rtece_graph

    atoms_list = ase.io.read(str(configs), index=f":{int(limit_configs)}")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    return [
        atoms_to_rtece_graph(
            atoms,
            cutoff=cutoff,
            device=device,
            dtype=dtype,
            neighborlist_backend=neighborlist_backend,
        )
        for atoms in atoms_list
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--reference-path-ids", type=_parse_path_ids, required=True)
    parser.add_argument("--candidate", action="append", type=_parse_candidate, default=[])
    parser.add_argument(
        "--auto-candidate-strategy",
        action="append",
        choices=("single_delete", "prefix", "cumulative"),
        default=[],
        help="Generate path-subset candidates from --reference-path-ids.",
    )
    parser.add_argument("--num-radial", type=int, default=8)
    parser.add_argument("--species-basis-channels", type=int, default=0)
    parser.add_argument("--atomic-cross-radial-sketch-channels", type=int, default=2)
    parser.add_argument("--reference-atomic-cross-radial-sketch-channels", type=int, default=None)
    parser.add_argument("--candidate-atomic-cross-radial-sketch-channels", type=int, default=None)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--limit-configs", type=int, default=32)
    parser.add_argument("--default-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--neighborlist-backend", choices=("ase", "vesin", "matscipy"), default="matscipy")
    parser.add_argument("--ridge", type=float, default=1.0e-12)
    parser.add_argument("--sample-weight-json", type=Path, default=None)
    parser.add_argument("--energy-target-key", default=None, help="Optional extxyz energy key for graph-summed label projection ranking.")
    parser.add_argument("--energy-baseline", choices=("element_counts", "none"), default="element_counts")
    parser.add_argument("--energy-eval-stride", type=int, default=0, help="Use every Nth config as held-out energy projection evaluation; 0 disables split.")
    parser.add_argument("--energy-eval-offset", type=int, default=0)
    parser.add_argument("--force-target-key", default=None, help="Optional extxyz force array key for force/Jacobian label projection ranking.")
    parser.add_argument("--force-eval-stride", type=int, default=0, help="Use every Nth config as held-out force projection evaluation; 0 disables split.")
    parser.add_argument("--force-eval-offset", type=int, default=0)
    parser.add_argument("--force-component-sample-count", type=int, default=0, help="Sample this many force components for force projection; 0 uses all components.")
    parser.add_argument("--focus-elements", default=None, help="Comma-separated element symbols or atomic numbers to upweight in projection rows.")
    parser.add_argument("--focus-weight", type=float, default=1.0)
    parser.add_argument("--active-set-baseline-candidate", default=None, help="Optional candidate name used as the active-set baseline for marginal gain/cost ranking.")
    parser.add_argument("--active-set-energy-weight", type=float, default=1.0)
    parser.add_argument("--active-set-force-weight", type=float, default=1.0)
    parser.add_argument("--active-set-projection-weight", type=float, default=0.0)
    parser.add_argument("--active-set-gain-mode", choices=("absolute", "relative"), default="absolute")
    parser.add_argument("--active-set-max-force-regression-fraction", type=float, default=None, help="Reject active-set candidates whose force RMSE regresses by more than this fraction relative to the baseline.")
    parser.add_argument("--active-set-require-energy-gain", action="store_true", help="Reject active-set candidates unless the selected energy metric improves over the baseline.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    import torch

    dtype = torch.float64 if args.default_dtype == "float64" else torch.float32
    device = torch.device("cpu")
    reference_cross_radial_sketch_channels = (
        int(args.reference_atomic_cross_radial_sketch_channels)
        if args.reference_atomic_cross_radial_sketch_channels is not None
        else int(args.atomic_cross_radial_sketch_channels)
    )
    candidate_cross_radial_sketch_channels = (
        int(args.candidate_atomic_cross_radial_sketch_channels)
        if args.candidate_atomic_cross_radial_sketch_channels is not None
        else int(args.atomic_cross_radial_sketch_channels)
    )
    reference_config = build_projection_config(
        "rtece_projection_reference",
        args.reference_path_ids,
        cutoff=float(args.cutoff),
        num_radial=int(args.num_radial),
        species_basis_channels=int(args.species_basis_channels),
        atomic_cross_radial_sketch_channels=reference_cross_radial_sketch_channels,
    )
    graphs = _load_graphs(
        args.configs,
        cutoff=float(args.cutoff),
        device=device,
        dtype=dtype,
        limit_configs=int(args.limit_configs),
        neighborlist_backend=args.neighborlist_backend,
    )
    focus_symbols, focus_atomic_numbers = parse_focus_elements(args.focus_elements)
    if args.sample_weight_json is not None and focus_atomic_numbers:
        raise ValueError("--sample-weight-json and --focus-elements are mutually exclusive")
    if args.sample_weight_json is not None:
        sample_weights, weight_source = _load_sample_weights_json(args.sample_weight_json)
    elif focus_atomic_numbers and float(args.focus_weight) != 1.0:
        sample_weights = element_focus_sample_weights(
            graphs,
            focus_atomic_numbers=focus_atomic_numbers,
            focus_weight=float(args.focus_weight),
        )
        weight_source = f"element_focus:{','.join(focus_symbols)}:weight={_format_focus_weight(float(args.focus_weight))}"
    else:
        sample_weights = None
        weight_source = None
    if args.energy_target_key:
        energy_targets, atom_counts = _load_energy_targets(
            args.configs,
            energy_key=str(args.energy_target_key),
            limit_configs=int(args.limit_configs),
        )
        if args.energy_baseline == "element_counts":
            energy_baseline_features, energy_baseline_atomic_numbers = graph_element_count_matrix(graphs)
        else:
            energy_baseline_features = None
            energy_baseline_atomic_numbers = []
        energy_fit_indices, energy_eval_indices = deterministic_eval_split(
            int(energy_targets.numel()),
            eval_stride=int(args.energy_eval_stride),
            eval_offset=int(args.energy_eval_offset),
        )
    else:
        energy_targets = None
        atom_counts = None
        energy_baseline_features = None
        energy_baseline_atomic_numbers = []
        energy_fit_indices = None
        energy_eval_indices = None
    force_descriptor_indices = None
    force_sampled_num_components = 0
    if args.force_target_key:
        force_targets = _load_force_targets(
            args.configs,
            force_key=str(args.force_target_key),
            limit_configs=int(args.limit_configs),
        )
        force_target_num_components = int(force_targets.numel())
        if int(args.force_eval_stride) > 0:
            force_fit_config_indices, force_eval_config_indices = deterministic_eval_split(
                len(graphs),
                eval_stride=int(args.force_eval_stride),
                eval_offset=int(args.force_eval_offset),
            )
        else:
            import torch

            force_fit_config_indices = torch.arange(len(graphs), dtype=torch.long)
            force_eval_config_indices = force_fit_config_indices
        force_fit_indices = _force_row_indices_from_config_indices(graphs, force_fit_config_indices)
        force_eval_indices = _force_row_indices_from_config_indices(graphs, force_eval_config_indices)
        if int(args.force_component_sample_count) > 0:
            force_descriptor_indices, force_fit_indices, force_eval_indices = deterministic_force_component_sample(
                force_fit_indices,
                force_eval_indices,
                sample_count=int(args.force_component_sample_count),
            )
            force_targets = force_targets[force_descriptor_indices]
        force_sampled_num_components = int(force_targets.numel())
    else:
        force_targets = None
        force_target_num_components = 0
        force_fit_config_indices = None
        force_eval_config_indices = None
        force_fit_indices = None
        force_eval_indices = None
    candidate_specs = list(args.candidate)
    auto_candidate_specs = generate_projection_candidate_specs(
        args.reference_path_ids,
        strategies=tuple(args.auto_candidate_strategy),
    ) if args.auto_candidate_strategy else []
    candidate_specs.extend(auto_candidate_specs)
    candidate_configs = []
    for candidate_name, candidate_path_ids in candidate_specs:
        candidate_config = build_projection_config(
            f"rtece_projection_{candidate_name}",
            candidate_path_ids,
            cutoff=float(args.cutoff),
            num_radial=int(args.num_radial),
            species_basis_channels=int(args.species_basis_channels),
            atomic_cross_radial_sketch_channels=candidate_cross_radial_sketch_channels,
        )
        candidate_configs.append((candidate_name, candidate_config))
    rows = make_projection_diagnostic_rows(
        candidate_configs,
        reference_config=reference_config,
        graphs=graphs,
        ridge=float(args.ridge),
        sample_weights=sample_weights,
        energy_targets=energy_targets,
        atom_counts=atom_counts,
        energy_baseline_features=energy_baseline_features,
        energy_fit_indices=energy_fit_indices,
        energy_eval_indices=energy_eval_indices,
        force_targets=force_targets,
        force_fit_indices=force_fit_indices,
        force_eval_indices=force_eval_indices,
        force_descriptor_indices=force_descriptor_indices,
    )
    ranked_rows = rank_projection_rows(rows)
    active_set_metadata = None
    active_set_rows = []
    if args.active_set_baseline_candidate:
        active_set_metadata = {
            "baseline_candidate": str(args.active_set_baseline_candidate),
            "energy_weight": float(args.active_set_energy_weight),
            "force_weight": float(args.active_set_force_weight),
            "projection_weight": float(args.active_set_projection_weight),
            "gain_mode": str(args.active_set_gain_mode),
            "max_force_regression_fraction": (
                None
                if args.active_set_max_force_regression_fraction is None
                else float(args.active_set_max_force_regression_fraction)
            ),
            "require_energy_gain": bool(args.active_set_require_energy_gain),
        }
        active_set_rows = rank_active_set_candidate_rows(
            ranked_rows,
            baseline_candidate=str(args.active_set_baseline_candidate),
            energy_weight=float(args.active_set_energy_weight),
            force_weight=float(args.active_set_force_weight),
            projection_weight=float(args.active_set_projection_weight),
            gain_mode=str(args.active_set_gain_mode),
            max_force_regression_fraction=args.active_set_max_force_regression_fraction,
            require_energy_gain=bool(args.active_set_require_energy_gain),
        )
    payload = {
        "schema_version": "rtece_projection_diagnostic.v1",
        "configs": str(args.configs),
        "limit_configs": int(args.limit_configs),
        "num_radial": int(args.num_radial),
        "species_basis_channels": int(args.species_basis_channels),
        "atomic_cross_radial_sketch_channels": int(args.atomic_cross_radial_sketch_channels),
        "reference_atomic_cross_radial_sketch_channels": reference_cross_radial_sketch_channels,
        "candidate_atomic_cross_radial_sketch_channels": candidate_cross_radial_sketch_channels,
        "cutoff": float(args.cutoff),
        "reference_path_ids": list(args.reference_path_ids),
        "sample_weight_json": str(args.sample_weight_json) if args.sample_weight_json else None,
        "sample_weight_source": weight_source,
        "weighted": sample_weights is not None,
        "energy_target_key": str(args.energy_target_key) if args.energy_target_key else None,
        "energy_target_num_configs": int(energy_targets.numel()) if energy_targets is not None else 0,
        "energy_fit_num_configs": int(energy_fit_indices.numel()) if energy_fit_indices is not None else 0,
        "energy_eval_num_configs": int(energy_eval_indices.numel()) if energy_eval_indices is not None else 0,
        "energy_eval_stride": int(args.energy_eval_stride) if args.energy_target_key else 0,
        "energy_eval_offset": int(args.energy_eval_offset) if args.energy_target_key else 0,
        "energy_baseline": str(args.energy_baseline) if args.energy_target_key else None,
        "energy_baseline_atomic_numbers": [int(value) for value in energy_baseline_atomic_numbers],
        "force_target_key": str(args.force_target_key) if args.force_target_key else None,
        "force_target_num_configs": int(len(graphs)) if force_targets is not None else 0,
        "force_target_num_components": force_target_num_components,
        "force_component_sample_count": int(args.force_component_sample_count) if args.force_target_key else 0,
        "force_component_sampled": force_descriptor_indices is not None,
        "force_sampled_num_components": force_sampled_num_components,
        "force_fit_num_components": int(force_fit_indices.numel()) if force_fit_indices is not None else 0,
        "force_eval_num_components": int(force_eval_indices.numel()) if force_eval_indices is not None else 0,
        "force_fit_num_configs": int(force_fit_config_indices.numel()) if force_fit_config_indices is not None else 0,
        "force_eval_num_configs": int(force_eval_config_indices.numel()) if force_eval_config_indices is not None else 0,
        "force_eval_stride": int(args.force_eval_stride) if args.force_target_key else 0,
        "force_eval_offset": int(args.force_eval_offset) if args.force_target_key else 0,
        "focus_elements": list(focus_symbols),
        "focus_atomic_numbers": [int(value) for value in focus_atomic_numbers],
        "focus_weight": float(args.focus_weight),
        "auto_candidate_strategies": list(args.auto_candidate_strategy),
        "auto_candidate_count": int(len(auto_candidate_specs)),
        "rank_metrics": rank_metrics_summary(rows),
        "active_set": active_set_metadata,
        "active_set_rows": active_set_rows,
        "rows": ranked_rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output_json)


if __name__ == "__main__":
    main()
