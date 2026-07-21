#!/usr/bin/env python3
"""Stage174 deployable low-frequency feature probes for rTECE energy offsets."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import (
    energy_group_loocv_projection_metrics,
    load_config_group_labels,
    load_stage165_case_offset_residual_targets,
)

FORBIDDEN_METADATA_KEYS = ("case_id", "source_key", "source_frame", "group_id", "image_id")
DEFAULT_RIDGE_GRID = (1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0, 100.0)
DEFAULT_PAIR_BINS = (1.5, 2.5, 3.5, 5.0)
DEFAULT_FEATURE_FAMILIES = (
    "composition_fraction",
    "tag_composition",
    "geometry_z_profile",
    "composition_tag_geometry",
    "pair_histogram",
)


def _finite(value: float, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _tags_array(atoms: Any) -> np.ndarray | None:
    if hasattr(atoms, "has") and atoms.has("tags"):
        return np.asarray(atoms.get_tags(), dtype=np.int64)
    arrays = getattr(atoms, "arrays", {})
    if "tags" in arrays:
        return np.asarray(arrays["tags"], dtype=np.int64)
    return None


def _add_composition(row: dict[str, float], numbers: np.ndarray) -> None:
    natoms = float(len(numbers))
    row["natoms"] = natoms
    for z in sorted(set(int(value) for value in numbers.tolist())):
        count = float(np.count_nonzero(numbers == z))
        row[f"count_z{z}"] = count
        row[f"frac_z{z}"] = count / max(natoms, 1.0)


def _add_geometry(row: dict[str, float], atoms: Any, positions: np.ndarray) -> None:
    cell = np.asarray(atoms.cell.array, dtype=np.float64)
    lengths = np.asarray(atoms.cell.lengths(), dtype=np.float64)
    natoms = max(float(len(positions)), 1.0)
    volume = abs(_finite(atoms.cell.volume))
    xy_area = float(np.linalg.norm(np.cross(cell[0], cell[1]))) if cell.shape == (3, 3) else 0.0
    z_values = positions[:, 2] if positions.size else np.zeros((0,), dtype=np.float64)
    c_len = max(float(lengths[2]) if lengths.size >= 3 else 0.0, 1.0e-12)
    row.update(
        {
            "volume": volume,
            "volume_per_atom": volume / natoms,
            "xy_area": xy_area,
            "xy_area_per_atom": xy_area / natoms,
            "cell_a": float(lengths[0]) if lengths.size >= 1 else 0.0,
            "cell_b": float(lengths[1]) if lengths.size >= 2 else 0.0,
            "cell_c": float(lengths[2]) if lengths.size >= 3 else 0.0,
            "pos_z_span": float(np.max(z_values) - np.min(z_values)) if z_values.size else 0.0,
            "pos_z_mean_frac": float(np.mean(z_values) / c_len) if z_values.size else 0.0,
            "pos_z_std_frac": float(np.std(z_values) / c_len) if z_values.size else 0.0,
        }
    )


def _add_tag_features(
    row: dict[str, float],
    *,
    numbers: np.ndarray,
    positions: np.ndarray,
    tags: np.ndarray,
    include_geometry: bool,
) -> None:
    natoms = max(float(len(numbers)), 1.0)
    for tag in sorted(set(int(value) for value in tags.tolist())):
        mask = tags == tag
        tag_count = float(np.count_nonzero(mask))
        prefix = f"tag{tag}"
        row[f"{prefix}_count_atoms"] = tag_count
        row[f"{prefix}_frac_atoms"] = tag_count / natoms
        for z in sorted(set(int(value) for value in numbers.tolist())):
            count = float(np.count_nonzero(mask & (numbers == z)))
            row[f"{prefix}_count_z{z}"] = count
            row[f"{prefix}_frac_z{z}"] = count / natoms
        if include_geometry and tag_count > 0.0:
            z_values = positions[mask, 2]
            row[f"{prefix}_z_mean"] = float(np.mean(z_values))
            row[f"{prefix}_z_std"] = float(np.std(z_values))
            row[f"{prefix}_z_span"] = float(np.max(z_values) - np.min(z_values))


def _add_pair_histogram(
    row: dict[str, float],
    *,
    atoms: Any,
    numbers: np.ndarray,
    pair_bins: Sequence[float],
) -> None:
    if len(numbers) < 2:
        return
    try:
        distances = np.asarray(atoms.get_all_distances(mic=True), dtype=np.float64)
    except Exception:
        distances = np.asarray(atoms.get_all_distances(mic=False), dtype=np.float64)
    bins = tuple(float(value) for value in pair_bins)
    pair_total = max(float(len(numbers) * (len(numbers) - 1) // 2), 1.0)
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            z1, z2 = sorted((int(numbers[i]), int(numbers[j])))
            distance = float(distances[i, j])
            for cutoff in bins:
                if distance <= cutoff:
                    key = f"pair_z{z1}_z{z2}_rle{cutoff:g}_count"
                    row[key] = row.get(key, 0.0) + 1.0
                    break
    for key, value in list(row.items()):
        if key.startswith("pair_") and key.endswith("_count"):
            row[key.replace("_count", "_frac")] = float(value) / pair_total


def lowfreq_feature_row_from_atoms(
    atoms: Any,
    *,
    include_tags: bool,
    include_geometry: bool,
    include_pair_histogram: bool = False,
    pair_bins: tuple[float, ...] = DEFAULT_PAIR_BINS,
) -> tuple[dict[str, float], dict[str, object]]:
    """Build deployable scalar summaries from structure fields only."""
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    positions = np.asarray(atoms.positions, dtype=np.float64)
    row: dict[str, float] = {}
    _add_composition(row, numbers)
    if include_geometry:
        _add_geometry(row, atoms, positions)
    tags = _tags_array(atoms)
    if include_tags and tags is not None and tags.shape[0] == numbers.shape[0]:
        _add_tag_features(row, numbers=numbers, positions=positions, tags=tags, include_geometry=include_geometry)
    if include_pair_histogram:
        _add_pair_histogram(row, atoms=atoms, numbers=numbers, pair_bins=pair_bins)
    rejected = sorted(key for key in FORBIDDEN_METADATA_KEYS if key in getattr(atoms, "info", {}))
    metadata: dict[str, object] = {
        "requires_tags": bool(include_tags),
        "tags_available": tags is not None,
        "include_geometry": bool(include_geometry),
        "include_pair_histogram": bool(include_pair_histogram),
        "pair_bins": [float(value) for value in pair_bins],
        "rejected_metadata_keys": rejected,
        "uses_forbidden_metadata_as_feature": False,
    }
    return row, metadata


def _family_options(feature_family: str) -> dict[str, bool]:
    if feature_family == "composition_fraction":
        return {"include_tags": False, "include_geometry": False, "include_pair_histogram": False}
    if feature_family == "tag_composition":
        return {"include_tags": True, "include_geometry": False, "include_pair_histogram": False}
    if feature_family == "geometry_z_profile":
        return {"include_tags": False, "include_geometry": True, "include_pair_histogram": False}
    if feature_family == "composition_tag_geometry":
        return {"include_tags": True, "include_geometry": True, "include_pair_histogram": False}
    if feature_family == "pair_histogram":
        return {"include_tags": False, "include_geometry": False, "include_pair_histogram": True}
    raise ValueError(f"unknown Stage174 feature family {feature_family!r}")


def lowfreq_feature_matrix(
    atoms_list: Sequence[Any],
    *,
    feature_family: str,
    pair_bins: tuple[float, ...] = DEFAULT_PAIR_BINS,
) -> tuple[torch.Tensor, list[str], dict[str, object]]:
    options = _family_options(str(feature_family))
    rows: list[dict[str, float]] = []
    rejected: set[str] = set()
    tags_available = False
    for atoms in atoms_list:
        row, metadata = lowfreq_feature_row_from_atoms(atoms, pair_bins=pair_bins, **options)
        rows.append(row)
        rejected.update(str(key) for key in metadata.get("rejected_metadata_keys", []))
        tags_available = tags_available or bool(metadata.get("tags_available"))
    feature_names = sorted({key for row in rows for key in row})
    forbidden_hits = [name for name in feature_names if any(key in name for key in FORBIDDEN_METADATA_KEYS)]
    if forbidden_hits:
        raise ValueError(f"non-deployable feature names generated: {forbidden_hits[:8]}")
    values = [[float(row.get(name, 0.0)) for name in feature_names] for row in rows]
    metadata = {
        "feature_family": str(feature_family),
        "feature_count": int(len(feature_names)),
        "requires_tags": bool(options["include_tags"]),
        "tags_available": bool(tags_available),
        "include_geometry": bool(options["include_geometry"]),
        "include_pair_histogram": bool(options["include_pair_histogram"]),
        "rejected_metadata_keys": sorted(rejected),
        "uses_forbidden_metadata_as_feature": False,
    }
    return torch.tensor(values, dtype=torch.float64), feature_names, metadata


def evaluate_lowfreq_feature_family(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    atom_counts: torch.Tensor,
    group_labels: Sequence[str],
    group_key: str,
    feature_family: str,
    feature_names: Sequence[str],
    feature_metadata: Mapping[str, object],
    ridge_grid: Sequence[float] = DEFAULT_RIDGE_GRID,
) -> dict[str, Any]:
    metrics = energy_group_loocv_projection_metrics(
        features,
        targets,
        group_labels=[str(label) for label in group_labels],
        group_key=str(group_key),
        atom_counts=atom_counts,
        include_intercept=True,
        standardize_features=True,
        ridge_values=tuple(float(value) for value in ridge_grid),
    )
    metrics.update(
        {
            "feature_family": str(feature_family),
            "feature_names": [str(name) for name in feature_names],
            "feature_count": int(len(feature_names)),
            "feature_metadata": dict(feature_metadata),
            "requires_tags": bool(feature_metadata.get("requires_tags")),
            "uses_forbidden_metadata_as_feature": False,
        }
    )
    return metrics


def _load_atoms(configs: Path, *, limit_configs: int) -> list[Any]:
    import ase.io

    atoms = ase.io.read(str(configs), index=f":{int(limit_configs)}")
    return atoms if isinstance(atoms, list) else [atoms]


def _metric_value(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    return float("inf") if value is None else float(value)


def run_stage174_lowfreq_probe(
    configs: str | Path,
    stage165_json: str | Path,
    *,
    output_json: str | Path | None = None,
    output_md: str | Path | None = None,
    limit_configs: int = 512,
    group_key: str = "case_id",
    stage165_variant: str = "stage157_direct_b32_rel0p25_mixed2048",
    feature_families: Sequence[str] = DEFAULT_FEATURE_FAMILIES,
    ridge_grid: Sequence[float] = DEFAULT_RIDGE_GRID,
) -> dict[str, Any]:
    configs_path = Path(configs)
    targets, atom_counts, target_metadata = load_stage165_case_offset_residual_targets(
        configs_path,
        stage165_json=Path(stage165_json),
        group_key=str(group_key),
        variant=str(stage165_variant) if stage165_variant else None,
        limit_configs=int(limit_configs),
    )
    group_labels = load_config_group_labels(configs_path, group_key=str(group_key), limit_configs=int(limit_configs))
    atoms_list = _load_atoms(configs_path, limit_configs=int(limit_configs))
    rows: list[dict[str, Any]] = []
    for family in feature_families:
        features, feature_names, feature_metadata = lowfreq_feature_matrix(atoms_list, feature_family=str(family))
        rows.append(
            evaluate_lowfreq_feature_family(
                features,
                targets,
                atom_counts=atom_counts,
                group_labels=group_labels,
                group_key=str(group_key),
                feature_family=str(family),
                feature_names=feature_names,
                feature_metadata=feature_metadata,
                ridge_grid=ridge_grid,
            )
        )
    rows.sort(key=lambda row: (_metric_value(row, "energy_per_atom_rmse"), str(row["feature_family"])))
    intercept = rows[0].get("energy_intercept_baseline_per_atom_rmse") if rows else None
    supported = [row for row in rows if bool(row.get("energy_beats_intercept_baseline"))]
    summary = {
        "schema_version": "rtece_stage174_lowfreq_feature_probe_results.v1",
        "stage": "stage174_lowfreq_feature_probe",
        "diagnostic_semantics": "deployable_low_frequency_chemistry_site_front_probe_for_stage170_energy_offsets",
        "target_semantics": "stage165_case_offset_residual_mev_atom",
        "energy_split_mode": "group-loocv",
        "group_key": str(group_key),
        "uses_case_id_as_feature": False,
        "non_deployable_feature_policy": "case_id_source_key_source_frame_group_id_image_id_are_never_features",
        "configs": str(configs_path),
        "limit_configs": int(limit_configs),
        "stage165_json": str(stage165_json),
        "stage165_variant": str(stage165_variant),
        "target_metadata": target_metadata,
        "intercept_baseline_per_atom_rmse": intercept,
        "supported_feature_families": [str(row["feature_family"]) for row in supported],
        "stage174_recommendation": (
            "promote_best_supported_lowfreq_front_module"
            if supported
            else "do_not_promote_current_lowfreq_proxy_prioritize_richer_local_tece_front_or_distillation_coverage"
        ),
        "rows": rows,
    }
    if output_json is not None:
        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if output_md is not None:
        out = Path(output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(render_stage174_markdown(summary), encoding="utf-8")
    return summary


def render_stage174_markdown(summary: Mapping[str, Any]) -> str:
    supported = list(summary.get("supported_feature_families") or [])
    if supported:
        interpretation = (
            "At least one deployable low-frequency family beats the intercept-only group-heldout baseline. "
            "The best supported family is therefore a candidate for the next trainable rTECE front module, "
            "subject to E/F RMSE, max-error, dimer, rattle-relax, and throughput validation."
        )
    else:
        interpretation = (
            "No tested low-frequency proxy beats the intercept-only group-heldout baseline. "
            "This rejects a simple global composition/tag/geometry correction as the next architecture move. "
            "The next branch should use richer local TECE front coordinates or teacher-projected/distilled coverage "
            "rather than adding these proxy summaries directly to the student."
        )
    lines = [
        "# Stage174 Low-Frequency Feature Probe Results",
        "",
        f"- split: `{summary['energy_split_mode']}` by `{summary['group_key']}`",
        f"- target: `{summary['target_semantics']}`",
        f"- intercept baseline: `{summary.get('intercept_baseline_per_atom_rmse')}` meV/atom RMSE",
        f"- recommendation: `{summary['stage174_recommendation']}`",
        "",
        "| feature family | features | requires tags | E RMSE | E MAE | E max | selected ridge | beats intercept |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in summary.get("rows", []):
        lines.append(
            "| {family} | {features} | {tags} | {rmse:.3f} | {mae:.3f} | {maxerr:.3f} | {ridge:g} | {beats} |".format(
                family=row["feature_family"],
                features=int(row["feature_count"]),
                tags=bool(row["requires_tags"]),
                rmse=float(row["energy_per_atom_rmse"]),
                mae=float(row["energy_per_atom_mae"]),
                maxerr=float(row["energy_per_atom_max_abs"]),
                ridge=float(row["energy_selected_ridge"]),
                beats=bool(row["energy_beats_intercept_baseline"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A feature family is architecture-promotable only if it beats the intercept-only group-heldout baseline without using non-deployable metadata as features.",
            "",
            interpretation,
            "",
            "## Next Priority",
            "",
            "- Do not add `case_id`, source metadata, or per-case offsets as model inputs.",
            "- Do not promote the current composition/tag/geometry proxy families into the production student.",
            "- Treat coarse pair histograms as a weak hint only: they are closest to the baseline but still do not pass the group-heldout gate.",
            "- Move the next architecture experiment toward local, learnable TECE front coordinates and force-protected E/F active-set scoring.",
            "",
        ]
    )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--stage165-json", required=True)
    parser.add_argument("--stage165-variant", default="stage157_direct_b32_rel0p25_mixed2048")
    parser.add_argument("--group-key", default="case_id")
    parser.add_argument("--limit-configs", type=int, default=512)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--energy-ridge-grid", default="1e-08,1e-06,0.0001,0.01,1,100")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ridge_grid = tuple(float(part) for part in str(args.energy_ridge_grid).split(",") if part)
    summary = run_stage174_lowfreq_probe(
        args.configs,
        args.stage165_json,
        output_json=args.output_json,
        output_md=args.output_md,
        limit_configs=args.limit_configs,
        group_key=args.group_key,
        stage165_variant=args.stage165_variant,
        ridge_grid=ridge_grid,
    )
    print(json.dumps({"stage": summary["stage"], "recommendation": summary["stage174_recommendation"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
