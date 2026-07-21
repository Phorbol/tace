#!/usr/bin/env python3
"""Stage168: rTECE semantic descriptor proxies for raw-energy offsets.

Stage166/167 showed cheap global and local radial proxies do not generalize for
Stage165 oracle case offsets. This stage tests the actual implemented rTECE
semantic descriptor families: if these descriptors explain case offsets under
LOOCV, the energy problem is likely training/downfolding/distillation. If they
only fit train cases, the current path family still lacks deployable baseline
coverage.
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
import torch

from benchmarks.oc20neb_tace_mace.summarize_rtece_stage166_deployable_proxy_baseline import (
    _case_order,
    _feature_matrix,
    _fit_ridge,
    _float,
    _loocv_predictions,
    _mae,
    _predict_with_coef,
    _r2,
    _rmse,
    _standardize_train_apply,
)
from tace.interface.ase.rtece_calculator import atoms_to_rtece_graph
from tace.models.rtece_scalar import (
    RTECEScalarConfig,
    build_rtece_config_from_path_ids,
    descriptor_dim,
    rtece_descriptors,
    rtece_route_contract,
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
    "runs/oc20neb_tace_mace/rtece-stage168-rtece-descriptor-proxy/"
    "stage168_rtece_descriptor_proxy.json"
)
DEFAULT_OUTPUT_MD = Path(
    "runs/oc20neb_tace_mace/rtece-stage168-rtece-descriptor-proxy/"
    "stage168_rtece_descriptor_proxy.md"
)


def rtece_descriptor_proxy_configs(
    *,
    num_radial: int = 8,
    hidden_channels: tuple[int, ...] = (64, 64),
) -> dict[str, RTECEScalarConfig]:
    common: dict[str, Any] = {
        "cutoff": 5.0,
        "num_radial": int(num_radial),
        "hidden_channels": tuple(int(value) for value in hidden_channels),
    }
    cross_rank = min(max(2, int(num_radial) // 2), int(num_radial))
    return {
        "t4_pair_density": build_rtece_config_from_path_ids(
            "stage168_t4_pair_density",
            ("atomic.radial_density",),
            **common,
        ),
        "t3_element_density": build_rtece_config_from_path_ids(
            "stage168_t3_element_density",
            ("atomic.radial_density", "atomic.element_density"),
            **common,
        ),
        "t3_atomic_l2_cross": build_rtece_config_from_path_ids(
            "stage168_t3_atomic_l2_cross",
            (
                "atomic.radial_density",
                "atomic.element_density",
                "atomic.vector_norm",
                "atomic.quadrupole_norm",
                "atomic.vector_cross_radial_dot",
                "atomic.quadrupole_cross_radial_frobenius",
            ),
            moment_l_max=2,
            atomic_cross_radial_sketch_channels=cross_rank,
            **common,
        ),
        "t3_cavity_edge": build_rtece_config_from_path_ids(
            "stage168_t3_cavity_edge",
            (
                "atomic.radial_density",
                "atomic.element_density",
                "edge.cavity.vector_dot",
                "edge.cavity.quadrupole_frobenius",
                "edge.cavity.target_vector_projection",
                "edge.cavity.source_vector_projection",
                "edge.cavity.target_quadrupole_projection",
                "edge.cavity.source_quadrupole_projection",
                "edge.direct.radial",
            ),
            moment_l_max=2,
            **common,
        ),
        "t3_full_moment_shell": build_rtece_config_from_path_ids(
            "stage168_t3_full_moment_shell",
            (
                "atomic.radial_density",
                "atomic.element_density",
                "edge.full_moment.vector_shell_dot",
                "edge.full_moment.vector_cross_shell_dot",
                "edge.full_moment.quadrupole_shell_frobenius",
                "edge.full_moment.quadrupole_cross_shell_frobenius",
                "edge.full_moment.vector_shell_contrast_projection",
            ),
            moment_l_max=2,
            **common,
        ),
    }


def _descriptor_stats(values: np.ndarray, *, prefix: str) -> dict[str, float]:
    if values.ndim != 2:
        raise ValueError("rTECE descriptor values must be rank-2 [atoms, descriptors]")
    if values.shape[0] == 0:
        mean = np.zeros((values.shape[1],), dtype=np.float64)
        std = np.zeros_like(mean)
    else:
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
    row: dict[str, float] = {}
    for idx, value in enumerate(mean.tolist()):
        row[f"{prefix}.mean.d{idx}"] = float(value)
    for idx, value in enumerate(std.tolist()):
        row[f"{prefix}.std.d{idx}"] = float(value)
    row[f"{prefix}.descriptor_dim"] = float(values.shape[1])
    return row


def rtece_descriptor_features_for_atoms(
    atoms,
    *,
    configs: Mapping[str, RTECEScalarConfig],
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    neighborlist_backend: str = "matscipy",
) -> dict[str, float]:
    row: dict[str, float] = {}
    graphs_by_cutoff: dict[float, Any] = {}
    for name, config in configs.items():
        cutoff = float(config.cutoff)
        graph = graphs_by_cutoff.get(cutoff)
        if graph is None:
            graph = atoms_to_rtece_graph(
                atoms,
                cutoff=cutoff,
                device=device,
                dtype=dtype,
                neighborlist_backend=neighborlist_backend,
            )
            graphs_by_cutoff[cutoff] = graph
        with torch.no_grad():
            descriptors = rtece_descriptors(graph, config).detach().cpu().numpy().astype(np.float64, copy=False)
        row.update(_descriptor_stats(descriptors, prefix=str(name)))
    return row


def _average_feature_rows(rows: Sequence[Mapping[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    averaged: dict[str, float] = {}
    for key in keys:
        values = [_float(row.get(key)) for row in rows]
        averaged[key] = float(np.mean(values)) if values else 0.0
    return averaged


def write_case_feature_cache(
    path: str | Path,
    *,
    case_features: Mapping[str, Mapping[str, float]],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_features = {
        str(case_id): {str(key): float(value) for key, value in sorted(row.items())}
        for case_id, row in sorted(case_features.items())
    }
    payload = {
        "schema_version": "rtece_stage168_case_feature_cache.v1",
        "cache_semantics": "case-level averaged rTECE semantic descriptor statistics; case_id is a grouping key, not a model feature",
        "case_count": len(normalized_features),
        "feature_count": len({key for row in normalized_features.values() for key in row}),
        "metadata": dict(metadata or {}),
        "case_features": normalized_features,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def load_case_feature_cache(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text())
    if payload.get("schema_version") != "rtece_stage168_case_feature_cache.v1":
        raise ValueError("expected rtece_stage168_case_feature_cache.v1 cache")
    raw_features = payload.get("case_features")
    if not isinstance(raw_features, Mapping):
        raise ValueError("case feature cache is missing case_features")
    payload["case_features"] = {
        str(case_id): {str(key): _float(value) for key, value in dict(row).items()}
        for case_id, row in raw_features.items()
    }
    payload["case_count"] = len(payload["case_features"])
    return payload


def case_rtece_descriptor_features_from_atoms(
    atoms_list: Sequence[object],
    *,
    configs: Mapping[str, RTECEScalarConfig],
    group_key: str = "case_id",
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    neighborlist_backend: str = "matscipy",
) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for index, atoms in enumerate(atoms_list):
        case_id = atoms.info.get(group_key, f"config_{index}")
        grouped[str(case_id)].append(
            rtece_descriptor_features_for_atoms(
                atoms,
                configs=configs,
                device=device,
                dtype=dtype,
                neighborlist_backend=neighborlist_backend,
            )
        )
    return {case_id: _average_feature_rows(rows) for case_id, rows in grouped.items() if rows}


def select_case_balanced_atoms(
    atoms_list: Sequence[object],
    *,
    group_key: str = "case_id",
    max_configs_per_case: int | None = None,
) -> list[object]:
    if max_configs_per_case is None:
        return list(atoms_list)
    max_per_case = int(max_configs_per_case)
    if max_per_case <= 0:
        raise ValueError("max_configs_per_case must be positive when provided")
    counts: dict[str, int] = defaultdict(int)
    selected: list[object] = []
    for index, atoms in enumerate(atoms_list):
        case_id = str(atoms.info.get(group_key, f"config_{index}"))
        if counts[case_id] >= max_per_case:
            continue
        selected.append(atoms)
        counts[case_id] += 1
    return selected


def load_case_rtece_descriptor_features(
    configs_file: str | Path,
    *,
    rtece_configs: Mapping[str, RTECEScalarConfig],
    group_key: str = "case_id",
    limit_configs: int | None = None,
    max_configs_per_case: int | None = None,
    neighborlist_backend: str = "matscipy",
) -> dict[str, dict[str, float]]:
    import ase.io

    index = ":" if limit_configs is None else f":{int(limit_configs)}"
    atoms = ase.io.read(str(configs_file), index=index)
    atoms_list = atoms if isinstance(atoms, list) else [atoms]
    atoms_list = select_case_balanced_atoms(
        atoms_list,
        group_key=group_key,
        max_configs_per_case=max_configs_per_case,
    )
    return case_rtece_descriptor_features_from_atoms(
        atoms_list,
        configs=rtece_configs,
        group_key=group_key,
        neighborlist_backend=neighborlist_backend,
    )


def _features_for_prefix(case_features: Mapping[str, Mapping[str, float]], prefix: str) -> list[str]:
    text = str(prefix)
    return sorted({key for row in case_features.values() for key in row if key.startswith(text)})


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
        "feature_set": str(name),
        "feature_count": len(feature_names),
        "feature_names_sample": list(feature_names)[:24],
        "feature_names_omitted": max(0, len(feature_names) - 24),
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


def summarize_rtece_descriptor_proxy_feature_sets(
    *,
    case_offsets_mev_atom: Mapping[str, float],
    case_features: Mapping[str, Mapping[str, float]],
    feature_groups: Mapping[str, str] | None = None,
    ridge: float = 1.0e-3,
) -> dict[str, Any]:
    case_ids = _case_order(case_offsets_mev_atom, case_features)
    y = np.asarray([_float(case_offsets_mev_atom[case_id]) for case_id in case_ids], dtype=np.float64)
    groups = dict(feature_groups or {})
    if not groups:
        prefixes = sorted({key.split(".", 1)[0] for row in case_features.values() for key in row if "." in key})
        groups = {prefix: f"{prefix}." for prefix in prefixes}
    rows = [
        _score_feature_set(
            name="intercept_only",
            feature_names=[],
            case_ids=case_ids,
            case_features=case_features,
            y=y,
            ridge=ridge,
        )
    ]
    cumulative: list[str] = []
    for name, prefix in groups.items():
        features = _features_for_prefix(case_features, prefix)
        rows.append(
            _score_feature_set(
                name=str(name),
                feature_names=features,
                case_ids=case_ids,
                case_features=case_features,
                y=y,
                ridge=ridge,
            )
        )
        cumulative.extend(features)
    if len(groups) > 1:
        rows.append(
            _score_feature_set(
                name="all_rtece_descriptor_groups",
                feature_names=sorted(set(cumulative)),
                case_ids=case_ids,
                case_features=case_features,
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
        "schema_version": "rtece_stage168_rtece_descriptor_proxy.v1",
        "diagnostic_semantics": "case_id is used only for grouping offsets to structures; rTECE descriptors are deployable semantic path features",
        "target_semantics": "oracle group_mean_offsets_eV_per_atom converted to meV/atom",
        "case_count": len(case_ids),
        "target_offset_rmse_mev_atom": _rmse(y),
        "target_offset_mae_mev_atom": _mae(y),
        "target_offset_max_abs_mev_atom": float(np.max(np.abs(y))) if y.size else 0.0,
        "best_train_rmse_feature_set": None if best_train is None else best_train["feature_set"],
        "best_loocv_rmse_feature_set": None if best_loocv is None else best_loocv["feature_set"],
        "best_loocv_rmse_mev_atom": best_loocv_rmse,
        "intercept_loocv_rmse_mev_atom": intercept_loocv_rmse,
        "rtece_descriptor_proxy_supported": bool(proxy_supported),
        "proxy_generalization_status": "rtece_descriptor_proxy_supported" if proxy_supported else "rtece_descriptor_proxy_not_supported_by_loocv",
        "stage168_recommendation": "promote_rtece_semantic_baseline_or_distillation_target" if proxy_supported else "do_not_treat_current_rtece_descriptors_as_sufficient_energy_baseline_prioritize_teacher_coverage_and_new_scalar_paths",
        "case_ids": list(case_ids),
        "rows": rows,
    }


def summarize_stage168_from_files(
    *,
    benchmark: str | Path = DEFAULT_BENCHMARK,
    configs_file: str | Path = DEFAULT_CONFIGS,
    group_key: str = "case_id",
    num_radial: int = 8,
    limit_configs: int | None = None,
    max_configs_per_case: int | None = None,
    ridge: float = 1.0e-3,
    neighborlist_backend: str = "matscipy",
    feature_cache_json: str | Path | None = None,
    write_feature_cache_json: str | Path | None = None,
) -> dict[str, Any]:
    payload = json.loads(Path(benchmark).read_text())
    offsets = payload.get("group_mean_offsets_eV_per_atom") or {}
    case_offsets = {str(case_id): _float(value) * 1000.0 for case_id, value in offsets.items()}
    configs = rtece_descriptor_proxy_configs(num_radial=num_radial)
    feature_cache_used = False
    if feature_cache_json is not None and Path(feature_cache_json).exists():
        cache_payload = load_case_feature_cache(feature_cache_json)
        case_features = cache_payload["case_features"]
        feature_cache_used = True
    else:
        case_features = load_case_rtece_descriptor_features(
            configs_file,
            rtece_configs=configs,
            group_key=group_key,
            limit_configs=limit_configs,
            max_configs_per_case=max_configs_per_case,
            neighborlist_backend=neighborlist_backend,
        )
        if write_feature_cache_json is not None:
            write_case_feature_cache(
                write_feature_cache_json,
                case_features=case_features,
                metadata={
                    "benchmark": str(benchmark),
                    "configs": str(configs_file),
                    "group_key": str(group_key),
                    "num_radial": int(num_radial),
                    "limit_configs": limit_configs,
                    "max_configs_per_case": max_configs_per_case,
                    "neighborlist_backend": str(neighborlist_backend),
                    "descriptor_config_names": list(configs),
                },
            )
    summary = summarize_rtece_descriptor_proxy_feature_sets(
        case_offsets_mev_atom=case_offsets,
        case_features=case_features,
        feature_groups={name: f"{name}." for name in configs},
        ridge=ridge,
    )
    summary.update(
        {
            "benchmark": str(benchmark),
            "configs": str(configs_file),
            "group_key": str(group_key),
            "num_radial": int(num_radial),
            "limit_configs": limit_configs,
            "max_configs_per_case": max_configs_per_case,
            "neighborlist_backend": str(neighborlist_backend),
            "feature_cache_json": None if feature_cache_json is None else str(feature_cache_json),
            "write_feature_cache_json": None if write_feature_cache_json is None else str(write_feature_cache_json),
            "feature_cache_used": bool(feature_cache_used),
            "source_variant": payload.get("variant") or Path(benchmark).stem.replace("_dft_benchmark", ""),
            "raw_e_rmse_mev_atom": _float(payload.get("rmse_e_mev_atom")),
            "group_offset_rmse_mev_atom": _float(payload.get("group_mean_offset_rmse_mev_atom")),
            "relative_image_rmse_mev_atom": _float(payload.get("relative_image_rmse_mev_atom")),
            "barrier_rmse_mev_atom": _float(payload.get("barrier_rmse_mev_atom")),
            "force_rmse_mev_a": _float(payload.get("rmse_f_mev_a")),
            "descriptor_configs": {
                name: {
                    "variant": config.variant,
                    "descriptor_dim": descriptor_dim(config),
                    "route": rtece_route_contract(config),
                    "scalar_path_ids": list(config.scalar_path_ids or ()),
                }
                for name, config in configs.items()
            },
        }
    )
    return summary


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_stage168_rtece_descriptor_proxy_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Stage168 rTECE Descriptor Proxy",
        "",
        "This tests implemented rTECE semantic descriptor families against Stage165 oracle case offsets. case_id is used only for grouping offsets to structures; it is not used as a model feature.",
        "",
        f"Source variant: `{summary.get('source_variant', 'manual')}`. Target offset RMSE: `{_fmt(summary.get('target_offset_rmse_mev_atom'))}` meV/atom.",
        f"Generalization status: `{summary.get('proxy_generalization_status', 'unknown')}`. Recommendation: `{summary.get('stage168_recommendation', 'unknown')}`.",
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
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "If an implemented rTECE descriptor family improves LOOCV, the absolute-energy failure is likely a training, downfolding, or distillation issue. If it only reduces train error, current semantic paths are not sufficient deployable coordinates for the low-frequency energy baseline, and the next priority should be teacher coverage plus new high-value scalar or edge-relational paths.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--configs", type=Path, default=DEFAULT_CONFIGS)
    parser.add_argument("--group-key", default="case_id")
    parser.add_argument("--num-radial", type=int, default=8)
    parser.add_argument("--limit-configs", type=int, default=None)
    parser.add_argument("--max-configs-per-case", type=int, default=None)
    parser.add_argument("--ridge", type=float, default=1.0e-3)
    parser.add_argument("--neighborlist-backend", default="matscipy")
    parser.add_argument("--feature-cache-json", type=Path, default=None)
    parser.add_argument("--write-feature-cache-json", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_stage168_from_files(
        benchmark=args.benchmark,
        configs_file=args.configs,
        group_key=args.group_key,
        num_radial=args.num_radial,
        limit_configs=args.limit_configs,
        max_configs_per_case=args.max_configs_per_case,
        ridge=args.ridge,
        neighborlist_backend=args.neighborlist_backend,
        feature_cache_json=args.feature_cache_json,
        write_feature_cache_json=args.write_feature_cache_json,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(render_stage168_rtece_descriptor_proxy_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
