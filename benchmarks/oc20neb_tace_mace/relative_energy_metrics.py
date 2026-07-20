from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


def mean_abs_rmse_max(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mae_mev_atom": 0.0,
            f"{prefix}_rmse_mev_atom": 0.0,
            f"{prefix}_max_abs_mev_atom": 0.0,
        }
    abs_values = np.abs(values)
    return {
        f"{prefix}_mae_mev_atom": float(np.mean(abs_values)),
        f"{prefix}_rmse_mev_atom": float(math.sqrt(float(np.mean(values**2)))),
        f"{prefix}_max_abs_mev_atom": float(np.max(abs_values)),
    }


def _energy_arrays(
    pred_e: Sequence[float] | np.ndarray,
    ref_e: Sequence[float] | np.ndarray,
    natoms: Sequence[float] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = np.asarray(pred_e, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref_e, dtype=np.float64).reshape(-1)
    nat = np.asarray(natoms, dtype=np.float64).reshape(-1)
    if not (pred.shape == ref.shape == nat.shape):
        raise ValueError(f"pred/ref/natoms shape mismatch: {pred.shape}, {ref.shape}, {nat.shape}")
    if np.any(nat <= 0.0):
        raise ValueError("natoms must be positive for per-atom energy metrics")
    return pred, ref, nat


def _group_indices(group_ids: Sequence[object], size: int) -> dict[str, list[int]]:
    if len(group_ids) != size:
        raise ValueError(f"group_ids length {len(group_ids)} does not match energies length {size}")
    by_group: dict[str, list[int]] = {}
    for idx, group in enumerate(group_ids):
        by_group.setdefault(str(group), []).append(idx)
    return by_group


def energy_error_decomposition_metrics(
    pred_e: Sequence[float] | np.ndarray,
    ref_e: Sequence[float] | np.ndarray,
    natoms: Sequence[float] | np.ndarray,
    group_ids: Sequence[object],
    *,
    image_indices: Sequence[object] | None = None,
) -> dict[str, object]:
    pred, ref, nat = _energy_arrays(pred_e, ref_e, natoms)
    by_group = _group_indices(group_ids, pred.shape[0])
    if image_indices is not None and len(image_indices) != pred.shape[0]:
        raise ValueError(f"image_indices length {len(image_indices)} does not match energies length {pred.shape[0]}")

    raw_e_per_atom = (pred - ref) / nat
    global_offset = float(np.mean(raw_e_per_atom)) if raw_e_per_atom.size else 0.0
    global_corrected = raw_e_per_atom - global_offset

    group_mean_corrected: list[float] = []
    first_image_corrected: list[float] = []
    used_groups: list[str] = []
    skipped_groups: list[str] = []
    group_offsets: dict[str, float] = {}
    first_image_offsets: dict[str, float] = {}
    for group, indices in by_group.items():
        if len(indices) < 1:
            skipped_groups.append(group)
            continue
        ordered = list(indices)
        if image_indices is not None:
            ordered = sorted(ordered, key=lambda i: (float(image_indices[i]), i))
        group_errors = raw_e_per_atom[ordered]
        group_offset = float(np.mean(group_errors))
        first_offset = float(group_errors[0])
        group_mean_corrected.extend((group_errors - group_offset).tolist())
        first_image_corrected.extend((group_errors - first_offset).tolist())
        group_offsets[group] = group_offset
        first_image_offsets[group] = first_offset
        used_groups.append(group)

    group_mean = np.asarray(group_mean_corrected, dtype=np.float64) * 1000.0
    first_image = np.asarray(first_image_corrected, dtype=np.float64) * 1000.0
    relative = relative_energy_group_metrics(pred, ref, nat, group_ids, image_indices=image_indices)
    return {
        "schema_version": "rtece_energy_error_decomposition.v1",
        "baseline": "raw_vs_global_vs_group_mean_vs_first_image_anchor",
        "num_groups": len(used_groups),
        "num_skipped_groups": len(skipped_groups),
        "num_images": int(pred.shape[0]),
        "global_per_atom_residual_offset_eV": global_offset,
        "group_mean_offsets_eV_per_atom": group_offsets,
        "first_image_offsets_eV_per_atom": first_image_offsets,
        **mean_abs_rmse_max(raw_e_per_atom * 1000.0, "raw"),
        **mean_abs_rmse_max(global_corrected * 1000.0, "global_offset"),
        **mean_abs_rmse_max(group_mean, "group_mean_offset"),
        **mean_abs_rmse_max(first_image, "first_image_anchor"),
        "relative_energy_metric_schema_version": relative.get("schema_version"),
        "relative_energy_baseline": relative.get("baseline"),
        "relative_image_mae_mev_atom": relative["relative_image_mae_mev_atom"],
        "relative_image_rmse_mev_atom": relative["relative_image_rmse_mev_atom"],
        "relative_image_max_abs_mev_atom": relative["relative_image_max_abs_mev_atom"],
        "barrier_mae_mev_atom": relative["barrier_mae_mev_atom"],
        "barrier_rmse_mev_atom": relative["barrier_rmse_mev_atom"],
        "barrier_max_abs_mev_atom": relative["barrier_max_abs_mev_atom"],
    }


def relative_energy_group_metrics(
    pred_e: Sequence[float] | np.ndarray,
    ref_e: Sequence[float] | np.ndarray,
    natoms: Sequence[float] | np.ndarray,
    group_ids: Sequence[object],
    *,
    image_indices: Sequence[object] | None = None,
) -> dict[str, object]:
    pred = np.asarray(pred_e, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref_e, dtype=np.float64).reshape(-1)
    nat = np.asarray(natoms, dtype=np.float64).reshape(-1)
    if not (pred.shape == ref.shape == nat.shape):
        raise ValueError(f"pred/ref/natoms shape mismatch: {pred.shape}, {ref.shape}, {nat.shape}")
    if len(group_ids) != pred.shape[0]:
        raise ValueError(f"group_ids length {len(group_ids)} does not match energies length {pred.shape[0]}")
    if image_indices is not None and len(image_indices) != pred.shape[0]:
        raise ValueError(f"image_indices length {len(image_indices)} does not match energies length {pred.shape[0]}")

    by_group: dict[str, list[int]] = {}
    for idx, group in enumerate(group_ids):
        by_group.setdefault(str(group), []).append(idx)

    relative_errors: list[float] = []
    barrier_errors: list[float] = []
    used_groups: list[str] = []
    skipped_groups: list[str] = []
    for group, indices in by_group.items():
        if len(indices) < 2:
            skipped_groups.append(group)
            continue
        if image_indices is not None:
            indices = sorted(indices, key=lambda i: (float(image_indices[i]), i))
        p = pred[indices]
        r = ref[indices]
        n = nat[indices]
        denom = float(np.mean(n)) if np.all(n > 0.0) else 1.0
        pred_endpoint = min(float(p[0]), float(p[-1]))
        ref_endpoint = min(float(r[0]), float(r[-1]))
        rel_pred = p - pred_endpoint
        rel_ref = r - ref_endpoint
        relative_errors.extend(((rel_pred - rel_ref) / n * 1000.0).tolist())
        pred_barrier = float(np.max(p) - pred_endpoint)
        ref_barrier = float(np.max(r) - ref_endpoint)
        barrier_errors.append((pred_barrier - ref_barrier) / denom * 1000.0)
        used_groups.append(group)

    rel = np.asarray(relative_errors, dtype=np.float64)
    barrier = np.asarray(barrier_errors, dtype=np.float64)
    return {
        "schema_version": "rtece_relative_neb_energy_metrics.v1",
        "baseline": "observed_endpoint_min_per_group",
        "num_groups": len(used_groups),
        "num_skipped_groups": len(skipped_groups),
        "num_images": int(rel.size),
        "group_key_semantics": "case_id",
        "image_key_semantics": "source_frame",
        **mean_abs_rmse_max(rel, "relative_image"),
        **mean_abs_rmse_max(barrier, "barrier"),
    }


def atoms_group_values(atoms_list: Sequence[object], group_key: str, image_key: str) -> tuple[list[str], list[float]]:
    groups: list[str] = []
    images: list[float] = []
    for idx, atoms in enumerate(atoms_list):
        groups.append(str(atoms.info.get(group_key, f"config_{idx}")))
        raw_image = atoms.info.get(image_key, idx)
        try:
            images.append(float(raw_image))
        except (TypeError, ValueError):
            images.append(float(idx))
    return groups, images
