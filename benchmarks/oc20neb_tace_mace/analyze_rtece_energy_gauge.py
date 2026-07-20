#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from benchmarks.oc20neb_tace_mace.benchmark_models import reference_arrays, summarize_errors
from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import (
    build_atom_graph,
    choose_rtece_force_mode,
    extxyz_index,
    load_atoms_window,
)
from benchmarks.oc20neb_tace_mace.train_rtece_scalar import load_checkpoint


def composition_counts_from_atoms(atoms) -> dict[int, int]:
    numbers = np.asarray(atoms.get_atomic_numbers(), dtype=np.int64)
    return {int(z): int((numbers == z).sum()) for z in sorted(set(numbers.tolist()))}


def _normalise_counts(counts: Sequence[Mapping[int | str, int | float]]) -> list[dict[int, float]]:
    normalised: list[dict[int, float]] = []
    for row in counts:
        normalised.append({int(z): float(v) for z, v in row.items() if float(v) != 0.0})
    return normalised


def _design_matrix(
    counts: Sequence[Mapping[int | str, int | float]],
    *,
    elements: Sequence[int],
) -> np.ndarray:
    rows = _normalise_counts(counts)
    return np.asarray([[float(row.get(int(z), 0.0)) for z in elements] for row in rows], dtype=np.float64)


def _energy_summary(pred_e: np.ndarray, ref_e: np.ndarray, natoms: np.ndarray) -> dict[str, float]:
    signed = (np.asarray(pred_e, dtype=np.float64).reshape(-1) - np.asarray(ref_e, dtype=np.float64).reshape(-1)) / np.asarray(natoms, dtype=np.float64).reshape(-1) * 1000.0
    abs_err = np.abs(signed)
    bias = float(np.mean(signed)) if signed.size else 0.0
    return {
        "mae_e_mev_atom": float(np.mean(abs_err)) if abs_err.size else 0.0,
        "rmse_e_mev_atom": float(math.sqrt(float(np.mean(signed**2)))) if signed.size else 0.0,
        "bias_e_mev_atom": bias,
        "mean_signed_e_mev_atom": bias,
        "max_abs_e_mev_atom": float(np.max(abs_err)) if abs_err.size else 0.0,
    }


def fit_residual_energy_calibration(
    counts: Sequence[Mapping[int | str, int | float]],
    pred_e: Sequence[float] | np.ndarray,
    ref_e: Sequence[float] | np.ndarray,
    *,
    elements: Sequence[int],
    mode: str,
    ridge: float = 1.0e-12,
) -> dict[str, object]:
    pred = np.asarray(pred_e, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref_e, dtype=np.float64).reshape(-1)
    if pred.shape != ref.shape:
        raise ValueError(f"pred_e and ref_e shape mismatch: {pred.shape} vs {ref.shape}")
    if len(counts) != pred.shape[0]:
        raise ValueError(f"counts length {len(counts)} does not match energies length {pred.shape[0]}")
    residual = ref - pred
    elements_i = [int(z) for z in elements]
    if mode == "none":
        return {"kind": "none", "elements_z": elements_i}
    if mode == "global":
        return {
            "kind": "global_total_energy_shift",
            "elements_z": elements_i,
            "shift_eV": float(np.mean(residual)),
        }
    if mode != "per_element":
        raise ValueError(f"unknown calibration mode: {mode}")
    design = _design_matrix(counts, elements=elements_i)
    lhs = design.T @ design
    if ridge > 0.0:
        lhs = lhs + float(ridge) * np.eye(lhs.shape[0], dtype=np.float64)
    rhs = design.T @ residual
    try:
        values = np.linalg.solve(lhs, rhs)
    except np.linalg.LinAlgError:
        values = np.linalg.lstsq(design, residual, rcond=None)[0]
    fitted = design @ values
    calib_natoms = design.sum(axis=1)
    calib_natoms[calib_natoms <= 0.0] = 1.0
    return {
        "kind": "per_element_residual_e0",
        "elements_z": elements_i,
        "ridge": float(ridge),
        "residual_e0_by_z_eV": {str(z): float(v) for z, v in zip(elements_i, values, strict=True)},
        "calibration_residual_metrics": _energy_summary(pred + fitted, ref, calib_natoms),
    }


def apply_energy_calibration(
    counts: Sequence[Mapping[int | str, int | float]],
    pred_e: Sequence[float] | np.ndarray,
    calibration: Mapping[str, object],
) -> np.ndarray:
    pred = np.asarray(pred_e, dtype=np.float64).reshape(-1)
    kind = str(calibration.get("kind", "none"))
    if kind == "none":
        return pred.copy()
    if len(counts) != pred.shape[0]:
        raise ValueError(f"counts length {len(counts)} does not match energies length {pred.shape[0]}")
    if kind == "global_total_energy_shift":
        return pred + float(calibration.get("shift_eV", 0.0))
    if kind == "per_element_residual_e0":
        residual_e0 = {int(z): float(v) for z, v in dict(calibration.get("residual_e0_by_z_eV", {})).items()}
        correction = []
        for row in _normalise_counts(counts):
            correction.append(sum(float(n) * residual_e0.get(int(z), 0.0) for z, n in row.items()))
        return pred + np.asarray(correction, dtype=np.float64)
    raise ValueError(f"unknown calibration kind: {kind}")


def _predict_rtece(
    *,
    model_path: Path,
    atoms_list: list,
    device_name: str,
    dtype_name: str,
    graph_construction_backend: str,
    force_mode_request: str,
) -> tuple[np.ndarray, np.ndarray, object]:
    dtype = torch.float64 if dtype_name == "float64" else torch.float32
    requested = torch.device(device_name)
    device = requested if requested.type == "cpu" or torch.cuda.is_available() else torch.device("cpu")
    model, config = load_checkpoint(model_path, dtype=dtype, device=device)
    force_mode = choose_rtece_force_mode(force_mode_request, config, device_type=device.type, dtype=dtype)
    if force_mode != "autograd":
        raise ValueError("stage146 energy gauge diagnostic currently requires --force-mode autograd for comparable E/F outputs")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    pred_e: list[np.ndarray] = []
    pred_f: list[np.ndarray] = []
    for atoms in atoms_list:
        graph = build_atom_graph(
            atoms,
            graph_construction_backend=graph_construction_backend,
            cutoff=float(config.cutoff),
            device=device,
            dtype=dtype,
        )
        out = model(graph)
        pred_e.append(out["energy"].detach().cpu().numpy().reshape(-1))
        pred_f.append(out["forces"].detach().cpu().numpy().reshape(-1, 3))
    return np.concatenate(pred_e, axis=0), np.concatenate(pred_f, axis=0), config


def run_energy_gauge_diagnostic(args: argparse.Namespace) -> dict[str, object]:
    calib_atoms = load_atoms_window(args.configs, start_config=args.calib_start, limit_configs=args.calib_limit)
    eval_atoms = load_atoms_window(args.configs, start_config=args.eval_start, limit_configs=args.eval_limit)
    calib_ref_e, calib_ref_f, calib_natoms = reference_arrays(calib_atoms, args.energy_key, args.forces_key)
    eval_ref_e, eval_ref_f, eval_natoms = reference_arrays(eval_atoms, args.energy_key, args.forces_key)
    calib_pred_e, calib_pred_f, config = _predict_rtece(
        model_path=args.model,
        atoms_list=calib_atoms,
        device_name=args.device,
        dtype_name=args.default_dtype,
        graph_construction_backend=args.graph_construction_backend,
        force_mode_request=args.force_mode,
    )
    eval_pred_e, eval_pred_f, _ = _predict_rtece(
        model_path=args.model,
        atoms_list=eval_atoms,
        device_name=args.device,
        dtype_name=args.default_dtype,
        graph_construction_backend=args.graph_construction_backend,
        force_mode_request=args.force_mode,
    )
    calib_counts = [composition_counts_from_atoms(atoms) for atoms in calib_atoms]
    eval_counts = [composition_counts_from_atoms(atoms) for atoms in eval_atoms]
    elements = sorted({int(z) for row in calib_counts for z in row} | {int(z) for row in eval_counts for z in row})
    calibration_payload = {}
    for mode in ("none", "global", "per_element"):
        calibration = fit_residual_energy_calibration(
            calib_counts,
            calib_pred_e,
            calib_ref_e,
            elements=elements,
            mode=mode,
            ridge=float(args.ridge),
        )
        corrected_eval_e = apply_energy_calibration(eval_counts, eval_pred_e, calibration)
        metrics = summarize_errors(corrected_eval_e, eval_pred_f, eval_ref_e, eval_ref_f, eval_natoms)
        calibration_payload[mode] = {
            "calibration": calibration,
            "eval_metrics": metrics,
        }
    return {
        "schema_version": "rtece_stage146_energy_gauge.v1",
        "model": str(args.model),
        "configs": str(args.configs),
        "variant": str(args.variant),
        "energy_key": str(args.energy_key),
        "forces_key": str(args.forces_key),
        "device": str(args.device),
        "default_dtype": str(args.default_dtype),
        "graph_construction_backend": str(args.graph_construction_backend),
        "force_mode": str(args.force_mode),
        "model_atomic_energies": {str(k): float(v) for k, v in (config.atomic_energies or {}).items()},
        "model_energy_per_atom_shift": float(config.energy_per_atom_shift),
        "calibration_window": {
            "start": int(args.calib_start),
            "limit": int(args.calib_limit),
            "extxyz_index": extxyz_index(start_config=args.calib_start, limit_configs=args.calib_limit),
            "configs": len(calib_atoms),
        },
        "eval_window": {
            "start": int(args.eval_start),
            "limit": int(args.eval_limit),
            "extxyz_index": extxyz_index(start_config=args.eval_start, limit_configs=args.eval_limit),
            "configs": len(eval_atoms),
        },
        "elements_z": elements,
        "calibration_metrics_before_fit": summarize_errors(calib_pred_e, calib_pred_f, calib_ref_e, calib_ref_f, calib_natoms),
        "eval_calibrations": calibration_payload,
    }


def write_markdown(payload: Mapping[str, object], path: Path) -> None:
    rows = []
    for mode, item in dict(payload["eval_calibrations"]).items():
        metrics = dict(item["eval_metrics"])
        rows.append(
            f"| {mode} | {metrics['rmse_e_mev_atom']:.3f} | {metrics['mae_e_mev_atom']:.3f} | {metrics['max_abs_e_mev_atom']:.3f} | {metrics['bias_e_mev_atom']:.3f} | {metrics['rmse_f_mev_a']:.3f} | {metrics['mae_f_mev_a']:.3f} |"
        )
    text = "\n".join(
        [
            "# Stage146 Energy Gauge Diagnostic",
            "",
            "Calibration and evaluation windows are disjoint; per-element residual E0 here is a diagnostic for gauge transfer, not a final benchmark correction.",
            "",
            f"- model: `{payload['model']}`",
            f"- calibration: `{dict(payload['calibration_window'])['extxyz_index']}`",
            f"- evaluation: `{dict(payload['eval_window'])['extxyz_index']}`",
            f"- model energy reference: per-element={bool(dict(payload['model_atomic_energies']))}, global_shift={payload['model_energy_per_atom_shift']}",
            "",
            "| calibration | E RMSE meV/atom | E MAE meV/atom | E max meV/atom | E bias meV/atom | F RMSE meV/A | F MAE meV/A |",
            "|---|---:|---:|---:|---:|---:|---:|",
            *rows,
            "",
        ]
    )
    path.write_text(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose rTECE energy gauge errors with disjoint calibration/evaluation windows.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", default="rtece")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--default-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--energy-key", default="energy")
    parser.add_argument("--forces-key", default="forces")
    parser.add_argument("--calib-start", type=int, default=0)
    parser.add_argument("--calib-limit", type=int, default=128)
    parser.add_argument("--eval-start", type=int, default=128)
    parser.add_argument("--eval-limit", type=int, default=256)
    parser.add_argument("--ridge", type=float, default=1.0e-8)
    parser.add_argument("--force-mode", choices=("autograd",), default="autograd")
    parser.add_argument("--graph-construction-backend", choices=("ase_neighborlist", "matscipy_neighborlist", "torch_radius_nopbc"), default="matscipy_neighborlist")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_energy_gauge_diagnostic(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_markdown(payload, args.output.with_suffix(".md"))


if __name__ == "__main__":
    main()
