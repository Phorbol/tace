#!/usr/bin/env python3
"""Stratify rTECE force errors by element and lightweight focus groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from benchmarks.oc20neb_tace_mace.analyze_rtece_projection_error import _load_graphs
from benchmarks.oc20neb_tace_mace.make_rtece_projection_weights import _load_reference_forces
from tace.models.rtece_workflow import load_checkpoint


def _as_force_matrix(values: Any, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float64, device="cpu")
    if tensor.ndim != 2 or tensor.shape[1] != 3:
        raise ValueError(f"{name} must have shape [atoms, 3], got {tuple(tensor.shape)}")
    if tensor.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one atom")
    if not bool(torch.isfinite(tensor).all().item()):
        raise ValueError(f"{name} must be finite")
    return tensor


def _focus_masks(symbols: list[str]) -> dict[str, list[int]]:
    c_or_n = {"C", "N"}
    chno = {"C", "H", "N", "O"}
    return {
        "C_or_N": [index for index, symbol in enumerate(symbols) if symbol in c_or_n],
        "CHNO": [index for index, symbol in enumerate(symbols) if symbol in chno],
        "not_CHNO": [index for index, symbol in enumerate(symbols) if symbol not in chno],
    }


def _force_error_row(label: str, indices: list[int], abs_error: torch.Tensor, signed_error: torch.Tensor) -> dict[str, Any]:
    if indices:
        index_tensor = torch.tensor(indices, dtype=torch.long)
        group_abs = abs_error[index_tensor]
        group_signed = signed_error[index_tensor]
        count = int(group_abs.shape[0])
        component_count = int(group_abs.numel())
        mae = float(group_abs.mean().item())
        rmse = float(torch.sqrt(group_signed.square().mean()).item())
        max_abs = float(group_abs.max().item())
        component_l1 = float(group_abs.sum().item())
        component_l2 = float(torch.linalg.vector_norm(group_signed).item())
    else:
        count = 0
        component_count = 0
        mae = 0.0
        rmse = 0.0
        max_abs = 0.0
        component_l1 = 0.0
        component_l2 = 0.0
    return {
        "label": str(label),
        "count": count,
        "component_count": component_count,
        "atom_fraction": float(count / abs_error.shape[0]) if abs_error.shape[0] else 0.0,
        "mae_f_mev_a": mae,
        "rmse_f_mev_a": rmse,
        "max_abs_f_mev_a": max_abs,
        "component_l1_mev_a": component_l1,
        "component_l2_mev_a": component_l2,
    }


def _element_rows(symbols: list[str], abs_error: torch.Tensor, signed_error: torch.Tensor) -> list[dict[str, Any]]:
    by_label: dict[str, list[int]] = {}
    for index, symbol in enumerate(symbols):
        by_label.setdefault(str(symbol), []).append(index)
    rows = [_force_error_row(label, indices, abs_error, signed_error) for label, indices in by_label.items()]
    return sorted(rows, key=lambda row: (-float(row["mae_f_mev_a"]), str(row["label"])))


def stratify_symbol_force_errors(
    *,
    symbols: list[str],
    predicted_forces: Any,
    reference_forces: Any,
    target_force_source: str,
    focus_selection_label: str = "C_or_N",
    focus_excess_weight: float = 2.0,
) -> dict[str, Any]:
    pred = _as_force_matrix(predicted_forces, "predicted_forces")
    ref = _as_force_matrix(reference_forces, "reference_forces")
    if pred.shape != ref.shape:
        raise ValueError(f"predicted/reference force shapes differ: {tuple(pred.shape)} vs {tuple(ref.shape)}")
    if len(symbols) != int(pred.shape[0]):
        raise ValueError(f"symbols/forces length mismatch: {len(symbols)} vs {int(pred.shape[0])}")

    signed_error = (pred - ref) * 1000.0
    abs_error = signed_error.abs()
    focus_rows = [
        _force_error_row(label, indices, abs_error, signed_error)
        for label, indices in _focus_masks(symbols).items()
    ]
    focus_rows.sort(key=lambda row: (-float(row["mae_f_mev_a"]), str(row["label"])))
    global_mae = float(abs_error.mean().item())
    focus_by_label = {str(row["label"]): row for row in focus_rows}
    selection_focus = focus_by_label.get(str(focus_selection_label))
    focus_mae = float(selection_focus["mae_f_mev_a"]) if selection_focus is not None else 0.0
    focus_excess = max(0.0, focus_mae - global_mae)
    selection_score = global_mae + float(focus_excess_weight) * focus_excess
    return {
        "schema_version": "rtece_force_error_stratification.v1",
        "target_force_source": str(target_force_source),
        "num_atoms": int(pred.shape[0]),
        "num_force_components": int(pred.numel()),
        "mae_f_mev_a": global_mae,
        "rmse_f_mev_a": float(torch.sqrt(signed_error.square().mean()).item()),
        "max_abs_f_mev_a": float(abs_error.max().item()),
        "selection_focus_label": str(focus_selection_label),
        "selection_focus_count": int(selection_focus["count"]) if selection_focus is not None else 0,
        "selection_focus_mae_f_mev_a": float(focus_mae),
        "selection_focus_excess_mae_f_mev_a": float(focus_excess),
        "selection_focus_excess_weight": float(focus_excess_weight),
        "selection_score_mev_a": float(selection_score),
        "selection_score_definition": "global_mae + focus_excess_weight * max(0, focus_mae - global_mae)",
        "elements": _element_rows([str(symbol) for symbol in symbols], abs_error, signed_error),
        "focus_groups": focus_rows,
    }


def load_symbols_from_extxyz(configs: Path, *, limit_configs: int) -> list[str]:
    import ase.io

    atoms_list = ase.io.read(str(configs), index=f":{int(limit_configs)}")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    symbols: list[str] = []
    for atoms in atoms_list:
        symbols.extend(atoms.get_chemical_symbols())
    return symbols


def predict_forces_from_checkpoint(
    checkpoint: Path,
    configs: Path,
    *,
    limit_configs: int,
    cutoff: float | None,
    default_dtype: str,
    neighborlist_backend: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    dtype = torch.float64 if default_dtype == "float64" else torch.float32
    device = torch.device("cpu")
    model, config, metadata = load_checkpoint(checkpoint, dtype=dtype, device=device)
    effective_cutoff = float(cutoff) if cutoff is not None else float(config.cutoff)
    graphs = _load_graphs(
        configs,
        cutoff=effective_cutoff,
        device=device,
        dtype=dtype,
        limit_configs=int(limit_configs),
        neighborlist_backend=neighborlist_backend,
    )
    was_training = model.training
    model.eval()
    predicted = []
    for graph in graphs:
        predicted.append(model(graph)["forces"].detach().cpu())
    if was_training:
        model.train()
    return torch.cat(predicted, dim=0).to(dtype=torch.float64), {
        "cutoff": effective_cutoff,
        "dtype": str(dtype).replace("torch.", ""),
        "checkpoint_scalar_path_ids": list(config.scalar_path_ids or []),
        "checkpoint_tece_path_manifest_hash": (metadata.get("tece_path_manifest") or {}).get("manifest_hash"),
    }


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# rTECE Force Error Stratification",
        "",
        f"- route: `{payload.get('route', 'NA')}`",
        f"- target force source: `{payload.get('target_force_source', 'NA')}`",
        f"- atoms: {payload['num_atoms']}",
        f"- force MAE/RMSE: {payload['mae_f_mev_a']:.3f} / {payload['rmse_f_mev_a']:.3f} meV/A",
        f"- selection score: {payload.get('selection_score_mev_a', payload['mae_f_mev_a']):.3f} meV/A",
        f"- selection focus: `{payload.get('selection_focus_label', 'NA')}` excess weight {payload.get('selection_focus_excess_weight', 'NA')}",
        "",
        "## Focus Groups",
        "",
        "| group | atoms | atom frac | F MAE | F RMSE | max abs F |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in payload["focus_groups"]:
        lines.append(
            "| {label} | {count} | {atom_fraction:.3f} | {mae_f_mev_a:.3f} | {rmse_f_mev_a:.3f} | {max_abs_f_mev_a:.3f} |".format(**row)
        )
    lines.extend([
        "",
        "## Elements",
        "",
        "| element | atoms | atom frac | F MAE | F RMSE | max abs F |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in payload["elements"]:
        lines.append(
            "| {label} | {count} | {atom_fraction:.3f} | {mae_f_mev_a:.3f} | {rmse_f_mev_a:.3f} | {max_abs_f_mev_a:.3f} |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--route", default=None)
    parser.add_argument("--limit-configs", type=int, required=True)
    parser.add_argument("--target-force-array", default="teacher_forces")
    parser.add_argument("--focus-selection-label", default="C_or_N")
    parser.add_argument("--focus-excess-weight", type=float, default=2.0)
    parser.add_argument("--cutoff", type=float, default=None)
    parser.add_argument("--default-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--neighborlist-backend", choices=("ase", "vesin", "matscipy"), default="matscipy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predicted, model_meta = predict_forces_from_checkpoint(
        args.checkpoint,
        args.configs,
        limit_configs=int(args.limit_configs),
        cutoff=args.cutoff,
        default_dtype=str(args.default_dtype),
        neighborlist_backend=str(args.neighborlist_backend),
    )
    reference_parts = _load_reference_forces(
        args.configs,
        force_array=str(args.target_force_array),
        dtype=torch.float64,
        limit_configs=int(args.limit_configs),
    )
    reference = torch.cat(reference_parts, dim=0).to(dtype=torch.float64)
    symbols = load_symbols_from_extxyz(args.configs, limit_configs=int(args.limit_configs))
    payload = stratify_symbol_force_errors(
        symbols=symbols,
        predicted_forces=predicted,
        reference_forces=reference,
        target_force_source=str(args.target_force_array),
        focus_selection_label=str(args.focus_selection_label),
        focus_excess_weight=float(args.focus_excess_weight),
    )
    payload.update(
        {
            "route": args.route,
            "checkpoint": str(args.checkpoint),
            "configs": str(args.configs),
            "limit_configs": int(args.limit_configs),
            "neighborlist_backend": str(args.neighborlist_backend),
            **model_meta,
        }
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(format_markdown(payload), encoding="utf-8")
    print(args.output_json)
    if args.output_md is not None:
        print(args.output_md)


if __name__ == "__main__":
    main()
