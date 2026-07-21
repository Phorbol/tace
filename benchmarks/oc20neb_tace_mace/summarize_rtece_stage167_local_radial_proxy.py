#!/usr/bin/env python3
"""Stage167: local radial scalar proxy for Stage165 raw-energy offsets.

Stage166 showed cheap global proxies do not generalize. This stage moves one
step closer to the TECE/NEP/MTP scalar endpoint: one neighbor pass, radial
pair histograms, immediate scalarization, and ridge/LOOCV diagnostics. It is a
proxy diagnostic, not a new model architecture.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from benchmarks.oc20neb_tace_mace.summarize_rtece_stage166_deployable_proxy_baseline import (
    _fit_ridge,
    _loocv_predictions,
    _mae,
    _predict_with_coef,
    _r2,
    _rmse,
    _standardize_train_apply,
    load_case_features_from_extxyz,
)

DEFAULT_BENCHMARK = Path(
    "runs/oc20neb_tace_mace/rtece-stage157-relative-neb-loss/relative_loss_runs/"
    "stage157_direct_b32_rel0p25_mixed2048/stage157_direct_b32_rel0p25_mixed2048_dft_benchmark.json"
)
DEFAULT_CONFIGS = Path(
    "/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/"
    "Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz"
)
DEFAULT_OUTPUT_JSON = Path(
    "runs/oc20neb_tace_mace/rtece-stage167-local-radial-proxy/"
    "stage167_local_radial_proxy.json"
)
DEFAULT_OUTPUT_MD = Path(
    "runs/oc20neb_tace_mace/rtece-stage167-local-radial-proxy/"
    "stage167_local_radial_proxy.md"
)


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _case_order(case_offsets: Mapping[str, float], case_features: Mapping[str, Mapping[str, float]]) -> list[str]:
    return sorted(set(case_offsets) & set(case_features))


def _feature_matrix(case_ids: Sequence[str], case_features: Mapping[str, Mapping[str, float]], feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [[_float(case_features[case_id].get(name)) for name in feature_names] for case_id in case_ids],
        dtype=np.float64,
    )


def local_radial_proxy_features(atoms, *, cutoff: float = 5.0, radial_bins: int = 8) -> dict[str, float]:
    from ase.neighborlist import neighbor_list

    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    natoms = float(len(numbers))
    features: dict[str, float] = {"natoms": natoms}
    if natoms <= 0:
        return features
    for z in sorted(set(numbers.tolist())):
        count = float(np.count_nonzero(numbers == int(z)))
        features[f"frac_z{int(z)}"] = count / natoms
    try:
        src, dst, distances = neighbor_list("ijd", atoms, float(cutoff))
    except Exception:
        src = np.zeros((0,), dtype=np.int64)
        dst = np.zeros((0,), dtype=np.int64)
        distances = np.zeros((0,), dtype=np.float64)
    bins = max(1, int(radial_bins))
    edges = np.linspace(0.0, float(cutoff), bins + 1, dtype=np.float64)
    hist: dict[str, float] = defaultdict(float)
    for i, j, distance in zip(src, dst, distances, strict=False):
        if int(i) >= int(j):
            continue
        d = float(distance)
        if not (0.0 < d < float(cutoff)):
            continue
        bin_id = int(np.searchsorted(edges, d, side="right") - 1)
        bin_id = min(max(bin_id, 0), bins - 1)
        za, zb = sorted((int(numbers[int(i)]), int(numbers[int(j)])))
        hist[f"pair_z{za}_z{zb}_bin{bin_id}"] += 1.0 / natoms
    features.update(hist)
    return features


def _case_radial_features_from_atoms_list(atoms_list: Sequence[object], *, cutoff: float, radial_bins: int) -> dict[str, float]:
    accum: dict[str, float] = defaultdict(float)
    count = 0
    for atoms in atoms_list:
        row = local_radial_proxy_features(atoms, cutoff=cutoff, radial_bins=radial_bins)
        for key, value in row.items():
            accum[key] += float(value)
        count += 1
    if count <= 0:
        return {}
    return {key: value / count for key, value in accum.items()}


def load_case_local_radial_features(
    configs: str | Path,
    *,
    group_key: str = "case_id",
    cutoff: float = 5.0,
    radial_bins: int = 8,
    limit_configs: int | None = None,
) -> dict[str, dict[str, float]]:
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
    return {
        case_id: _case_radial_features_from_atoms_list(rows, cutoff=cutoff, radial_bins=radial_bins)
        for case_id, rows in grouped.items()
        if rows
    }


def _score_feature_set(
    *,
    name: str,
    feature_names: Sequence[str],
    case_ids: Sequence[str],
    case_features: Mapping[str, Mapping[str, float]],
    y: np.ndarray,
    ridge: float,
) -> dict[str, Any]:
    x_raw = _feature_matrix(case_ids, case_features, feature_names)
    x, _ = _standardize_train_apply(x_raw, x_raw)
    coef = _fit_ridge(x, y, ridge=ridge)
    pred = _predict_with_coef(x, coef)
    loocv = _loocv_predictions(x_raw, y, ridge=ridge)
    return {
        "feature_set": name,
        "feature_count": len(feature_names),
        "feature_names": list(feature_names),
        "case_count": len(case_ids),
        "ridge": float(ridge),
        "uses_case_id_as_feature": False,
        "train_rmse_mev_atom": _rmse(pred - y),
        "train_mae_mev_atom": _mae(pred - y),
        "train_r2": _r2(y, pred),
        "loocv_rmse_mev_atom": None if loocv is None else _rmse(loocv - y),
        "loocv_mae_mev_atom": None if loocv is None else _mae(loocv - y),
        "loocv_r2": None if loocv is None else _r2(y, loocv),
    }


def summarize_local_radial_proxy_feature_sets(
    *,
    case_offsets_mev_atom: Mapping[str, float],
    case_features: Mapping[str, Mapping[str, float]],
    global_case_features: Mapping[str, Mapping[str, float]] | None = None,
    ridge: float = 1.0e-3,
) -> dict[str, Any]:
    case_ids = _case_order(case_offsets_mev_atom, case_features)
    y = np.asarray([_float(case_offsets_mev_atom[case_id]) for case_id in case_ids], dtype=np.float64)
    local_keys = sorted({key for case_id in case_ids for key in case_features[case_id] if key.startswith("pair_")})
    frac_keys = sorted({key for case_id in case_ids for key in case_features[case_id] if key.startswith("frac_z") or key == "natoms"})
    rows = [
        _score_feature_set(name="intercept_only", feature_names=[], case_ids=case_ids, case_features=case_features, y=y, ridge=ridge),
        _score_feature_set(name="composition", feature_names=frac_keys, case_ids=case_ids, case_features=case_features, y=y, ridge=ridge),
        _score_feature_set(name="local_radial", feature_names=local_keys, case_ids=case_ids, case_features=case_features, y=y, ridge=ridge),
        _score_feature_set(name="local_radial_plus_composition", feature_names=sorted(set(local_keys + frac_keys)), case_ids=case_ids, case_features=case_features, y=y, ridge=ridge),
    ]
    if global_case_features:
        merged = {case_id: {**global_case_features.get(case_id, {}), **case_features.get(case_id, {})} for case_id in case_ids}
        global_keys = sorted({key for case_id in case_ids for key in global_case_features.get(case_id, {})})
        rows.append(
            _score_feature_set(
                name="local_radial_plus_global",
                feature_names=sorted(set(local_keys + frac_keys + global_keys)),
                case_ids=case_ids,
                case_features=merged,
                y=y,
                ridge=ridge,
            )
        )
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
        "schema_version": "rtece_stage167_local_radial_proxy.v1",
        "diagnostic_semantics": "case_id is used only for grouping offsets to structures; local radial descriptors are deployable scalar proxy features",
        "case_count": len(case_ids),
        "target_offset_rmse_mev_atom": _rmse(y),
        "target_offset_mae_mev_atom": _mae(y),
        "target_offset_max_abs_mev_atom": float(np.max(np.abs(y))) if y.size else 0.0,
        "best_train_rmse_feature_set": None if best_train is None else best_train["feature_set"],
        "best_loocv_rmse_feature_set": None if best_loocv is None else best_loocv["feature_set"],
        "best_loocv_rmse_mev_atom": best_loocv_rmse,
        "intercept_loocv_rmse_mev_atom": intercept_loocv_rmse,
        "local_radial_proxy_supported": bool(proxy_supported),
        "proxy_generalization_status": "local_radial_proxy_supported" if proxy_supported else "local_radial_proxy_not_supported_by_loocv",
        "stage167_recommendation": "promote_local_radial_scalar_baseline_candidate" if proxy_supported else "do_not_promote_simple_local_radial_proxy_prioritize_teacher_coverage_or_higher_body_scalar_paths",
        "case_ids": list(case_ids),
        "rows": rows,
    }


def summarize_stage167_from_files(
    *,
    benchmark: str | Path = DEFAULT_BENCHMARK,
    configs: str | Path = DEFAULT_CONFIGS,
    group_key: str = "case_id",
    cutoff: float = 5.0,
    radial_bins: int = 8,
    limit_configs: int | None = None,
    ridge: float = 1.0e-3,
) -> dict[str, Any]:
    payload = json.loads(Path(benchmark).read_text())
    offsets = payload.get("group_mean_offsets_eV_per_atom") or {}
    case_offsets = {str(case_id): _float(value) * 1000.0 for case_id, value in offsets.items()}
    local_features = load_case_local_radial_features(configs, group_key=group_key, cutoff=cutoff, radial_bins=radial_bins, limit_configs=limit_configs)
    global_features = load_case_features_from_extxyz(configs, group_key=group_key, limit_configs=limit_configs)
    summary = summarize_local_radial_proxy_feature_sets(
        case_offsets_mev_atom=case_offsets,
        case_features=local_features,
        global_case_features=global_features,
        ridge=ridge,
    )
    summary.update({
        "benchmark": str(benchmark),
        "configs": str(configs),
        "group_key": str(group_key),
        "cutoff": float(cutoff),
        "radial_bins": int(radial_bins),
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


def render_stage167_local_radial_proxy_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Stage167 Local Radial Proxy",
        "",
        "This tests one neighbor-pass local radial scalar proxies for Stage165 oracle case offsets. case_id is used only for grouping offsets to structures; it is not used as a model feature.",
        "",
        f"Source variant: `{summary.get('source_variant', 'manual')}`. Target offset RMSE: `{_fmt(summary.get('target_offset_rmse_mev_atom'))}` meV/atom.",
        f"Generalization status: `{summary.get('proxy_generalization_status', 'unknown')}`. Recommendation: `{summary.get('stage167_recommendation', 'unknown')}`.",
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
        "If local radial proxies improve LOOCV over Stage166 global proxies, a deployable scalar baseline or richer local radial path is worth promoting. If they only reduce train error, the case-offset mode requires better deployment coverage, higher-body scalar contractions, or teacher response distillation rather than another simple baseline.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument("--group-key", default="case_id")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--radial-bins", type=int, default=8)
    parser.add_argument("--limit-configs", type=int, default=None)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_stage167_from_files(
        benchmark=args.benchmark,
        configs=args.configs,
        group_key=args.group_key,
        cutoff=args.cutoff,
        radial_bins=args.radial_bins,
        limit_configs=args.limit_configs,
        ridge=args.ridge,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_stage167_local_radial_proxy_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
