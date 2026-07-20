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
