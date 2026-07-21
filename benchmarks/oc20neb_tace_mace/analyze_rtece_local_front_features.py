#!/usr/bin/env python3
"""Stage175 local TECE-front feature probes for rTECE energy offsets."""

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

from benchmarks.oc20neb_tace_mace.analyze_rtece_lowfreq_features import (
    DEFAULT_RIDGE_GRID,
    FORBIDDEN_METADATA_KEYS,
)
from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import (
    energy_group_loocv_projection_metrics,
    load_config_group_labels,
    load_stage165_case_offset_residual_targets,
)

DEFAULT_SHELL_EDGES = (0.0, 1.6, 2.6, 3.6, 5.0)
DEFAULT_FEATURE_FAMILIES = (
    "local_l0_rank3",
    "local_l0_l1_rank3",
    "local_l0_l1_l2_rank3",
    "local_l0_l1_l2_rank4",
)


def _chemistry_basis(z: int, rank: int) -> np.ndarray:
    value = max(float(z), 1.0) / 92.0
    basis = np.asarray([1.0, value, value * value, math.sqrt(value)], dtype=np.float64)
    return basis[: int(rank)].copy()


def _quadrupole_unit(rhat: np.ndarray) -> np.ndarray:
    return np.outer(rhat, rhat) - np.eye(3, dtype=np.float64) / 3.0


def _shell_index(distance: float, shell_edges: Sequence[float]) -> int | None:
    for idx, (left, right) in enumerate(zip(shell_edges[:-1], shell_edges[1:])):
        if float(left) < distance <= float(right):
            return idx
    return None


def _family_options(feature_family: str) -> dict[str, Any]:
    if feature_family == "local_l0_rank3":
        return {"chemistry_rank": 3, "l_max": 0, "center_condition": True}
    if feature_family == "local_l0_l1_rank3":
        return {"chemistry_rank": 3, "l_max": 1, "center_condition": True}
    if feature_family == "local_l0_l1_l2_rank3":
        return {"chemistry_rank": 3, "l_max": 2, "center_condition": True}
    if feature_family == "local_l0_l1_l2_rank4":
        return {"chemistry_rank": 4, "l_max": 2, "center_condition": True}
    raise ValueError(f"unknown Stage175 local front family {feature_family!r}")


def _vectors_from_center(atoms: Any, center: int, indices: list[int]) -> np.ndarray:
    try:
        return np.asarray(atoms.get_distances(center, indices, mic=True, vector=True), dtype=np.float64)
    except Exception:
        positions = np.asarray(atoms.positions, dtype=np.float64)
        return positions[np.asarray(indices, dtype=np.int64)] - positions[int(center)]


def local_tece_front_feature_row_from_atoms(
    atoms: Any,
    *,
    shell_edges: tuple[float, ...] = DEFAULT_SHELL_EDGES,
    chemistry_rank: int = 3,
    l_max: int = 2,
    center_condition: bool = True,
) -> tuple[dict[str, float], dict[str, object]]:
    """Compute local species-conditioned Cartesian moment scalar contractions.

    The row contains structure-level means over center atoms. The contractions are
    rotationally invariant by construction: ell=1 uses vector norm squared and
    ell=2 uses quadrupole Frobenius norm squared.
    """
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    natoms = int(len(numbers))
    rank = int(chemistry_rank)
    shell_values = tuple(float(value) for value in shell_edges)
    if natoms < 1:
        raise ValueError("atoms must contain at least one atom")
    if rank < 1 or rank > 4:
        raise ValueError("chemistry_rank must be in [1, 4]")
    if int(l_max) < 0 or int(l_max) > 2:
        raise ValueError("l_max must be 0, 1, or 2")
    if len(shell_values) < 2 or any(right <= left for left, right in zip(shell_values[:-1], shell_values[1:])):
        raise ValueError("shell_edges must be strictly increasing")

    center_rank = rank if center_condition else 1
    shell_count = len(shell_values) - 1
    row: dict[str, float] = {}
    for center in range(natoms):
        center_basis = _chemistry_basis(int(numbers[center]), center_rank) if center_condition else np.ones(1, dtype=np.float64)
        neighbor_indices = [idx for idx in range(natoms) if idx != center]
        if not neighbor_indices:
            continue
        vectors = _vectors_from_center(atoms, center, neighbor_indices)
        l0 = np.zeros((shell_count + 1, center_rank, rank), dtype=np.float64)
        l0_r = np.zeros_like(l0)
        l1 = np.zeros((shell_count + 1, center_rank, rank, 3), dtype=np.float64)
        l2 = np.zeros((shell_count + 1, center_rank, rank, 3, 3), dtype=np.float64)
        for local_idx, neigh in enumerate(neighbor_indices):
            vec = np.asarray(vectors[local_idx], dtype=np.float64)
            distance = float(np.linalg.norm(vec))
            if distance <= 1.0e-12 or distance > shell_values[-1]:
                continue
            shell = _shell_index(distance, shell_values)
            if shell is None:
                continue
            rhat = vec / distance
            neigh_basis = _chemistry_basis(int(numbers[neigh]), rank)
            q = _quadrupole_unit(rhat)
            for shell_slot in (shell, shell_count):
                for cb, cb_value in enumerate(center_basis):
                    for nb, nb_value in enumerate(neigh_basis):
                        weight = float(cb_value * nb_value)
                        l0[shell_slot, cb, nb] += weight
                        l0_r[shell_slot, cb, nb] += weight * distance
                        if int(l_max) >= 1:
                            l1[shell_slot, cb, nb] += weight * rhat
                        if int(l_max) >= 2:
                            l2[shell_slot, cb, nb] += weight * q
        for shell_slot in range(shell_count + 1):
            shell_name = f"shell{shell_slot}" if shell_slot < shell_count else "all"
            for cb in range(center_rank):
                for nb in range(rank):
                    suffix = f"{shell_name}_centerb{cb}_neighb{nb}"
                    row[f"local_l0_{suffix}_mean"] = row.get(f"local_l0_{suffix}_mean", 0.0) + float(l0[shell_slot, cb, nb]) / float(natoms)
                    row[f"local_l0_r_{suffix}_mean"] = row.get(f"local_l0_r_{suffix}_mean", 0.0) + float(l0_r[shell_slot, cb, nb]) / float(natoms)
                    if int(l_max) >= 1:
                        value = float(np.dot(l1[shell_slot, cb, nb], l1[shell_slot, cb, nb]))
                        row[f"local_l1_norm2_{suffix}_mean"] = row.get(f"local_l1_norm2_{suffix}_mean", 0.0) + value / float(natoms)
                    if int(l_max) >= 2:
                        value = float(np.sum(l2[shell_slot, cb, nb] * l2[shell_slot, cb, nb]))
                        row[f"local_l2_frob2_{suffix}_mean"] = row.get(f"local_l2_frob2_{suffix}_mean", 0.0) + value / float(natoms)
    rejected = sorted(key for key in FORBIDDEN_METADATA_KEYS if key in getattr(atoms, "info", {}))
    metadata = {
        "tece_front_semantics": "local_species_conditioned_cartesian_moment_scalar_contractions",
        "shell_edges": [float(value) for value in shell_values],
        "chemistry_rank": int(rank),
        "l_max": int(l_max),
        "center_condition": bool(center_condition),
        "rejected_metadata_keys": rejected,
        "uses_forbidden_metadata_as_feature": False,
    }
    return row, metadata


def local_tece_front_feature_matrix(
    atoms_list: Sequence[Any],
    *,
    feature_family: str,
    shell_edges: tuple[float, ...] = DEFAULT_SHELL_EDGES,
) -> tuple[torch.Tensor, list[str], dict[str, object]]:
    options = _family_options(str(feature_family))
    rows: list[dict[str, float]] = []
    rejected: set[str] = set()
    for atoms in atoms_list:
        row, metadata = local_tece_front_feature_row_from_atoms(atoms, shell_edges=shell_edges, **options)
        rows.append(row)
        rejected.update(str(key) for key in metadata.get("rejected_metadata_keys", []))
    feature_names = sorted({key for row in rows for key in row})
    forbidden_hits = [name for name in feature_names if any(key in name for key in FORBIDDEN_METADATA_KEYS)]
    if forbidden_hits:
        raise ValueError(f"non-deployable feature names generated: {forbidden_hits[:8]}")
    values = [[float(row.get(name, 0.0)) for name in feature_names] for row in rows]
    metadata = {
        "feature_family": str(feature_family),
        "feature_count": int(len(feature_names)),
        "tece_front_semantics": "local_species_conditioned_cartesian_moment_scalar_contractions",
        "shell_edges": [float(value) for value in shell_edges],
        "chemistry_rank": int(options["chemistry_rank"]),
        "l_max": int(options["l_max"]),
        "center_condition": bool(options["center_condition"]),
        "rejected_metadata_keys": sorted(rejected),
        "uses_forbidden_metadata_as_feature": False,
    }
    return torch.tensor(values, dtype=torch.float64), feature_names, metadata


def evaluate_local_front_feature_family(
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
            "tece_front_semantics": feature_metadata.get("tece_front_semantics"),
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


def run_stage175_local_front_probe(
    configs: str | Path,
    stage165_json: str | Path,
    *,
    output_json: str | Path | None = None,
    output_md: str | Path | None = None,
    limit_configs: int = 512,
    group_key: str = "case_id",
    stage165_variant: str = "stage157_direct_b32_rel0p25_mixed2048",
    feature_families: Sequence[str] = DEFAULT_FEATURE_FAMILIES,
    shell_edges: Sequence[float] = DEFAULT_SHELL_EDGES,
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
    shell_tuple = tuple(float(value) for value in shell_edges)
    for family in feature_families:
        features, feature_names, feature_metadata = local_tece_front_feature_matrix(
            atoms_list,
            feature_family=str(family),
            shell_edges=shell_tuple,
        )
        rows.append(
            evaluate_local_front_feature_family(
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
    supported = [row for row in rows if bool(row.get("energy_beats_intercept_baseline"))]
    summary = {
        "schema_version": "rtece_stage175_local_tece_front_probe_results.v1",
        "stage": "stage175_local_tece_front_probe",
        "diagnostic_semantics": "local_tece_front_probe_for_stage170_energy_offsets_after_stage174_proxy_rejection",
        "target_semantics": "stage165_case_offset_residual_mev_atom",
        "energy_split_mode": "group-loocv",
        "group_key": str(group_key),
        "uses_case_id_as_feature": False,
        "configs": str(configs_path),
        "limit_configs": int(limit_configs),
        "stage165_json": str(stage165_json),
        "stage165_variant": str(stage165_variant),
        "shell_edges": [float(value) for value in shell_tuple],
        "target_metadata": target_metadata,
        "intercept_baseline_per_atom_rmse": rows[0].get("energy_intercept_baseline_per_atom_rmse") if rows else None,
        "supported_feature_families": [str(row["feature_family"]) for row in supported],
        "stage175_recommendation": (
            "promote_best_supported_local_tece_front_module"
            if supported
            else "do_not_promote_fixed_local_front_prioritize_teacher_projected_or_learnable_tece_front"
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
        out.write_text(render_stage175_markdown(summary), encoding="utf-8")
    return summary


def render_stage175_markdown(summary: Mapping[str, Any]) -> str:
    supported = list(summary.get("supported_feature_families") or [])
    interpretation = (
        "At least one local TECE front feature family beats the intercept baseline; the best supported row is a candidate "
        "for the next trainable rTECE front module."
        if supported
        else "No fixed local TECE front family beats the intercept baseline. This points toward teacher-projected or learnable "
        "local front selection rather than promoting this fixed diagnostic basis directly."
    )
    lines = [
        "# Stage175 Local TECE Front Probe Results",
        "",
        f"- split: `{summary['energy_split_mode']}` by `{summary['group_key']}`",
        f"- target: `{summary['target_semantics']}`",
        f"- intercept baseline: `{summary.get('intercept_baseline_per_atom_rmse')}` meV/atom RMSE",
        f"- recommendation: `{summary['stage175_recommendation']}`",
        "",
        "| feature family | features | Lmax | rank | E RMSE | E MAE | E max | selected ridge | beats intercept |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.get("rows", []):
        metadata = dict(row.get("feature_metadata") or {})
        lines.append(
            "| {family} | {features} | {lmax} | {rank} | {rmse:.3f} | {mae:.3f} | {maxerr:.3f} | {ridge:g} | {beats} |".format(
                family=row["feature_family"],
                features=int(row["feature_count"]),
                lmax=int(metadata.get("l_max", -1)),
                rank=int(metadata.get("chemistry_rank", -1)),
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
            interpretation,
            "",
            "## Next Priority",
            "",
            "- Promote the best supported local front only as an architecture candidate, not as a post-hoc correction.",
            "- Start with the low-rank L0 local density front; the fixed L1/L2 additions did not improve the group-heldout energy-offset RMSE enough to justify their extra feature/cost burden.",
            "- Convert the diagnostic all-distance implementation into a normal neighbor-pass trainable rTECE front before any throughput claim.",
            "- Re-test the promoted front with full DFT/teacher E/F RMSE, E/F max, relative NEB, dimer scan, rattle-relax, and atom-count throughput scaling.",
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
    parser.add_argument("--shell-edges", default="0,1.6,2.6,3.6,5.0")
    parser.add_argument("--energy-ridge-grid", default="1e-08,1e-06,0.0001,0.01,1,100")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    ridge_grid = tuple(float(part) for part in str(args.energy_ridge_grid).split(",") if part)
    shell_edges = tuple(float(part) for part in str(args.shell_edges).split(",") if part)
    summary = run_stage175_local_front_probe(
        args.configs,
        args.stage165_json,
        output_json=args.output_json,
        output_md=args.output_md,
        limit_configs=args.limit_configs,
        group_key=args.group_key,
        stage165_variant=args.stage165_variant,
        shell_edges=shell_edges,
        ridge_grid=ridge_grid,
    )
    print(json.dumps({"stage": summary["stage"], "recommendation": summary["stage175_recommendation"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
