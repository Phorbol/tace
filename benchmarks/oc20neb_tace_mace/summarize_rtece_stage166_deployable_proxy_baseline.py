#!/usr/bin/env python3
"""Stage166: test deployable low-frequency proxy baselines for raw E offsets.

Stage165 showed that oracle per-case offsets explain almost all raw E RMSE.
This script asks a stricter question: can cheap deployable scalar summaries
(composition, cell scale, coarse geometry) explain those offsets without using
case_id as a model feature? case_id is used only to align benchmark group offsets
with structures in the extxyz file.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

DEFAULT_BENCHMARK = Path(
    "runs/oc20neb_tace_mace/rtece-stage157-relative-neb-loss/relative_loss_runs/"
    "stage157_direct_b32_rel0p25_mixed2048/stage157_direct_b32_rel0p25_mixed2048_dft_benchmark.json"
)
DEFAULT_CONFIGS = Path(
    "/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/"
    "Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz"
)
DEFAULT_OUTPUT_JSON = Path(
    "runs/oc20neb_tace_mace/rtece-stage166-deployable-proxy-baseline/"
    "stage166_deployable_proxy_baseline.json"
)
DEFAULT_OUTPUT_MD = Path(
    "runs/oc20neb_tace_mace/rtece-stage166-deployable-proxy-baseline/"
    "stage166_deployable_proxy_baseline.md"
)

DEFAULT_FEATURE_SETS: dict[str, list[str]] = {
    "intercept_only": [],
    "composition": ["natoms"],
    "composition_cell": ["natoms", "volume_per_atom", "xy_area_per_atom", "cell_a", "cell_b", "cell_c"],
    "composition_cell_geometry": [
        "natoms",
        "volume_per_atom",
        "xy_area_per_atom",
        "cell_a",
        "cell_b",
        "cell_c",
        "pos_z_span",
        "pos_z_mean_frac",
        "pos_z_std_frac",
    ],
}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _rmse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(math.sqrt(float(np.mean(values * values)))) if values.size else 0.0


def _mae(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return float(np.mean(np.abs(values))) if values.size else 0.0


def _standardize_train_apply(x_train: np.ndarray, x_apply: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if x_train.shape[1] == 0:
        return x_train.copy(), x_apply.copy()
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1.0e-12] = 1.0
    return (x_train - mean) / std, (x_apply - mean) / std


def _fit_ridge(x: np.ndarray, y: np.ndarray, *, ridge: float) -> np.ndarray:
    design = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    penalty = float(ridge) * np.eye(design.shape[1], dtype=np.float64)
    penalty[0, 0] = 0.0
    lhs = design.T @ design + penalty
    rhs = design.T @ y
    try:
        return np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(lhs, rhs, rcond=None)[0]


def _predict_with_coef(x: np.ndarray, coef: np.ndarray) -> np.ndarray:
    design = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)
    return design @ coef


def _case_order(case_offsets: Mapping[str, float], case_features: Mapping[str, Mapping[str, float]]) -> list[str]:
    return sorted(set(case_offsets) & set(case_features))


def _feature_matrix(
    case_ids: Sequence[str],
    case_features: Mapping[str, Mapping[str, float]],
    feature_names: Sequence[str],
) -> np.ndarray:
    return np.asarray(
        [[_float(case_features[case_id].get(name)) for name in feature_names] for case_id in case_ids],
        dtype=np.float64,
    )


def _loocv_predictions(x: np.ndarray, y: np.ndarray, *, ridge: float) -> np.ndarray | None:
    n = y.shape[0]
    if n < 3:
        return None
    preds = np.zeros_like(y, dtype=np.float64)
    for holdout in range(n):
        keep = np.arange(n) != holdout
        x_train_raw = x[keep]
        x_holdout_raw = x[~keep]
        x_train, x_holdout = _standardize_train_apply(x_train_raw, x_holdout_raw)
        coef = _fit_ridge(x_train, y[keep], ridge=ridge)
        preds[holdout] = _predict_with_coef(x_holdout, coef)[0]
    return preds


def _r2(y: np.ndarray, pred: np.ndarray) -> float | None:
    denom = float(np.sum((y - np.mean(y)) ** 2))
    if denom <= 1.0e-24:
        return None
    return 1.0 - float(np.sum((y - pred) ** 2)) / denom


def fit_proxy_feature_sets(
    *,
    case_offsets_mev_atom: Mapping[str, float],
    case_features: Mapping[str, Mapping[str, float]],
    feature_sets: Mapping[str, Sequence[str]] = DEFAULT_FEATURE_SETS,
    ridge: float = 1.0e-4,
) -> dict[str, Any]:
    case_ids = _case_order(case_offsets_mev_atom, case_features)
    y = np.asarray([_float(case_offsets_mev_atom[case_id]) for case_id in case_ids], dtype=np.float64)
    rows = []
    for name, requested_features in feature_sets.items():
        features = [str(feature) for feature in requested_features]
        x_raw = _feature_matrix(case_ids, case_features, features)
        x, _ = _standardize_train_apply(x_raw, x_raw)
        coef = _fit_ridge(x, y, ridge=ridge)
        train_pred = _predict_with_coef(x, coef)
        loocv_pred = _loocv_predictions(x_raw, y, ridge=ridge)
        row = {
            "feature_set": str(name),
            "feature_names": features,
            "feature_count": len(features),
            "case_count": len(case_ids),
            "ridge": float(ridge),
            "uses_case_id_as_feature": False,
            "train_rmse_mev_atom": _rmse(train_pred - y),
            "train_mae_mev_atom": _mae(train_pred - y),
            "train_r2": _r2(y, train_pred),
            "loocv_rmse_mev_atom": None if loocv_pred is None else _rmse(loocv_pred - y),
            "loocv_mae_mev_atom": None if loocv_pred is None else _mae(loocv_pred - y),
            "loocv_r2": None if loocv_pred is None else _r2(y, loocv_pred),
        }
        rows.append(row)
    rows.sort(key=lambda row: (float("inf") if row["loocv_rmse_mev_atom"] is None else float(row["loocv_rmse_mev_atom"]), float(row["train_rmse_mev_atom"])))
    best_train = min(rows, key=lambda row: float(row["train_rmse_mev_atom"])) if rows else None
    best_loocv = rows[0] if rows else None
    intercept = next((row for row in rows if row["feature_set"] == "intercept_only"), None)
    best_loocv_rmse = None if best_loocv is None else best_loocv.get("loocv_rmse_mev_atom")
    intercept_loocv_rmse = None if intercept is None else intercept.get("loocv_rmse_mev_atom")
    best_loocv_r2 = None if best_loocv is None else best_loocv.get("loocv_r2")
    proxy_supported = (
        best_loocv_rmse is not None
        and intercept_loocv_rmse is not None
        and best_loocv_rmse < 0.9 * float(intercept_loocv_rmse)
        and best_loocv_r2 is not None
        and float(best_loocv_r2) > 0.0
    )
    return {
        "schema_version": "rtece_stage166_deployable_proxy_baseline.v1",
        "diagnostic_semantics": "case_id is used only for grouping offsets to structures; it is never used as a deployable feature",
        "target_semantics": "oracle group_mean_offsets_eV_per_atom converted to meV/atom",
        "case_count": len(case_ids),
        "target_offset_rmse_mev_atom": _rmse(y),
        "target_offset_mae_mev_atom": _mae(y),
        "target_offset_max_abs_mev_atom": float(np.max(np.abs(y))) if y.size else 0.0,
        "best_train_rmse_feature_set": None if best_train is None else best_train["feature_set"],
        "best_loocv_rmse_feature_set": None if best_loocv is None else best_loocv["feature_set"],
        "best_loocv_rmse_mev_atom": best_loocv_rmse,
        "intercept_loocv_rmse_mev_atom": intercept_loocv_rmse,
        "deployable_proxy_supported": bool(proxy_supported),
        "proxy_generalization_status": "proxy_supported" if proxy_supported else "proxy_not_supported_by_loocv",
        "stage166_recommendation": "add_explicit_proxy_baseline_path" if proxy_supported else "do_not_add_global_proxy_baseline_yet_prioritize_richer_local_tece_paths_or_distillation_coverage",
        "case_ids": case_ids,
        "rows": rows,
    }


def _feature_row_from_atoms_list(atoms_list: Sequence[object]) -> dict[str, float]:
    first = atoms_list[0]
    numbers = np.asarray(first.get_atomic_numbers(), dtype=np.int64)
    natoms = float(len(numbers))
    cell = np.asarray(first.cell.array, dtype=np.float64)
    lengths = np.asarray(first.cell.lengths(), dtype=np.float64)
    volume = float(abs(first.cell.volume))
    xy_area = float(np.linalg.norm(np.cross(cell[0], cell[1]))) if cell.shape == (3, 3) else 0.0
    positions = np.asarray(first.positions, dtype=np.float64)
    c_len = max(float(lengths[2]) if lengths.size >= 3 else 0.0, 1.0e-12)
    z_values = positions[:, 2] if positions.size else np.zeros((0,), dtype=np.float64)
    row: dict[str, float] = {
        "natoms": natoms,
        "volume": volume,
        "volume_per_atom": volume / max(natoms, 1.0),
        "xy_area": xy_area,
        "xy_area_per_atom": xy_area / max(natoms, 1.0),
        "cell_a": float(lengths[0]) if lengths.size >= 1 else 0.0,
        "cell_b": float(lengths[1]) if lengths.size >= 2 else 0.0,
        "cell_c": float(lengths[2]) if lengths.size >= 3 else 0.0,
        "pos_z_span": float(np.max(z_values) - np.min(z_values)) if z_values.size else 0.0,
        "pos_z_mean_frac": float(np.mean(z_values) / c_len) if z_values.size else 0.0,
        "pos_z_std_frac": float(np.std(z_values) / c_len) if z_values.size else 0.0,
    }
    for z in sorted(set(numbers.tolist())):
        count = float(np.count_nonzero(numbers == int(z)))
        row[f"count_z{int(z)}"] = count
        row[f"frac_z{int(z)}"] = count / max(natoms, 1.0)
    return row


def load_case_features_from_extxyz(configs: str | Path, *, group_key: str = "case_id", limit_configs: int | None = None) -> dict[str, dict[str, float]]:
    import ase.io

    index = ":" if limit_configs is None else f":{int(limit_configs)}"
    atoms = ase.io.read(str(configs), index=index)
    atoms_list = atoms if isinstance(atoms, list) else [atoms]
    grouped: dict[str, list[object]] = defaultdict(list)
    for item in atoms_list:
        case_id = item.info.get(group_key)
        if case_id is None:
            continue
        grouped[str(case_id)].append(item)
    return {case_id: _feature_row_from_atoms_list(rows) for case_id, rows in grouped.items() if rows}


def _augment_default_feature_sets(case_features: Mapping[str, Mapping[str, float]]) -> dict[str, list[str]]:
    keys = sorted({key for row in case_features.values() for key in row})
    composition_keys = [key for key in keys if key.startswith("frac_z") or key == "natoms"]
    feature_sets = dict(DEFAULT_FEATURE_SETS)
    feature_sets["composition"] = composition_keys
    feature_sets["composition_cell"] = sorted(set(composition_keys + DEFAULT_FEATURE_SETS["composition_cell"]))
    feature_sets["composition_cell_geometry"] = sorted(set(composition_keys + DEFAULT_FEATURE_SETS["composition_cell_geometry"]))
    return feature_sets


def summarize_stage166_from_files(
    *,
    benchmark: str | Path = DEFAULT_BENCHMARK,
    configs: str | Path = DEFAULT_CONFIGS,
    group_key: str = "case_id",
    limit_configs: int | None = None,
    ridge: float = 1.0e-4,
) -> dict[str, Any]:
    payload = json.loads(Path(benchmark).read_text())
    offsets_ev = payload.get("group_mean_offsets_eV_per_atom") or {}
    if not isinstance(offsets_ev, Mapping):
        offsets_ev = {}
    offsets_mev = {str(case_id): _float(value) * 1000.0 for case_id, value in offsets_ev.items()}
    case_features = load_case_features_from_extxyz(configs, group_key=group_key, limit_configs=limit_configs)
    feature_sets = _augment_default_feature_sets(case_features)
    summary = fit_proxy_feature_sets(
        case_offsets_mev_atom=offsets_mev,
        case_features=case_features,
        feature_sets=feature_sets,
        ridge=ridge,
    )
    summary.update({
        "benchmark": str(benchmark),
        "configs": str(configs),
        "group_key": str(group_key),
        "limit_configs": limit_configs,
        "source_variant": payload.get("variant") or Path(benchmark).stem.replace("_dft_benchmark", ""),
        "raw_e_rmse_mev_atom": _float(payload.get("rmse_e_mev_atom")),
        "group_offset_rmse_mev_atom": _float(payload.get("group_mean_offset_rmse_mev_atom")),
        "relative_image_rmse_mev_atom": _float(payload.get("relative_image_rmse_mev_atom")),
        "barrier_rmse_mev_atom": _float(payload.get("barrier_rmse_mev_atom")),
        "force_rmse_mev_a": _float(payload.get("rmse_f_mev_a")),
    })
    return summary


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_stage166_proxy_baseline_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Stage166 Deployable Proxy Baseline",
        "",
        "This tests whether cheap deployable scalar proxies can explain the oracle case offsets from Stage165. case_id is used only for grouping offsets to structures; it is not used as a model feature.",
        "",
        f"Source variant: `{summary.get('source_variant', 'manual')}`. Target offset RMSE: `{_fmt(summary.get('target_offset_rmse_mev_atom'))}` meV/atom.",
        f"Generalization status: `{summary.get('proxy_generalization_status', 'unknown')}`. Recommendation: `{summary.get('stage166_recommendation', 'unknown')}`.",
        "",
        "| feature set | features | train RMSE | train R2 | LOOCV RMSE | LOOCV R2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.get("rows", []):
        lines.append(
            "| {name} | {count} | {train} | {train_r2} | {loocv} | {loocv_r2} |".format(
                name=row.get("feature_set"),
                count=row.get("feature_count"),
                train=_fmt(row.get("train_rmse_mev_atom")),
                train_r2=_fmt(row.get("train_r2")),
                loocv=_fmt(row.get("loocv_rmse_mev_atom")),
                loocv_r2=_fmt(row.get("loocv_r2")),
            )
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "A low train RMSE with poor LOOCV means these cheap global proxies can memorize the 22-case offset pattern but are not a deployable fix. A strong LOOCV result would justify adding an explicit low-frequency scalar baseline path to the student; a weak LOOCV result points back to richer local TECE scalar paths or teacher-generated coverage.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument("--group-key", default="case_id")
    parser.add_argument("--limit-configs", type=int, default=None)
    parser.add_argument("--ridge", type=float, default=1.0e-4)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_stage166_from_files(
        benchmark=args.benchmark,
        configs=args.configs,
        group_key=args.group_key,
        limit_configs=args.limit_configs,
        ridge=args.ridge,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_stage166_proxy_baseline_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
