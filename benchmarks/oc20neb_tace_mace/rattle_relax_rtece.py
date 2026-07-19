#!/usr/bin/env python3
"""Run bounded rTECE rattle+relax physical-generalization diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def classify_focus_groups(symbols: list[str]) -> list[str]:
    symbol_set = {str(symbol) for symbol in symbols}
    groups: list[str] = []
    if symbol_set & {"C", "N"}:
        groups.append("C_or_N")
    has_cn = bool(symbol_set & {"C", "N"})
    has_chno = bool(symbol_set & {"C", "H", "N", "O"})
    if has_chno:
        groups.append("CHNO")
    if has_chno and not has_cn:
        groups.append("CHNO_no_CN")
    if not groups:
        groups.append("not_CHNO")
    return groups


def positions_rmsd(positions: Any, reference_positions: Any) -> float:
    import numpy as np

    pos = np.asarray(positions, dtype=np.float64)
    ref = np.asarray(reference_positions, dtype=np.float64)
    if pos.shape != ref.shape:
        raise ValueError(f"position shapes differ: {pos.shape} vs {ref.shape}")
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"positions must have shape [atoms, 3], got {pos.shape}")
    if pos.shape[0] < 1:
        raise ValueError("positions must contain at least one atom")
    diff = pos - ref
    return float(math.sqrt(float(np.mean(np.sum(diff * diff, axis=1)))))


def deterministic_rattle_atoms(atoms, *, rattle_std_a: float, seed: int):
    import numpy as np

    if float(rattle_std_a) < 0.0:
        raise ValueError("rattle_std_a must be non-negative")
    rattled = atoms.copy()
    rng = np.random.default_rng(int(seed))
    displacement = rng.normal(loc=0.0, scale=float(rattle_std_a), size=(len(rattled), 3))
    displacement -= displacement.mean(axis=0, keepdims=True)
    rattled.positions = rattled.positions + displacement
    return rattled


def _mean(values: list[float]) -> float | None:
    return float(sum(values) / len(values)) if values else None


def _record_group_row(label: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    converged = [bool(record.get("converged")) for record in records]
    final_rmsd = [float(record["final_rmsd_a"]) for record in records]
    initial_rmsd = [float(record["initial_rmsd_a"]) for record in records]
    max_fmax = [float(record["max_fmax_ev_a"]) for record in records]
    return {
        "label": str(label),
        "count": int(count),
        "converged_count": int(sum(1 for value in converged if value)),
        "converged_fraction": float(sum(1 for value in converged if value) / count) if count else 0.0,
        "mean_initial_rmsd_a": _mean(initial_rmsd),
        "mean_final_rmsd_a": _mean(final_rmsd),
        "max_final_rmsd_a": float(max(final_rmsd)) if final_rmsd else None,
        "mean_max_fmax_ev_a": _mean(max_fmax),
        "max_fmax_ev_a": float(max(max_fmax)) if max_fmax else None,
    }


def summarize_relax_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("relax summary requires at least one record")
    base = _record_group_row("all", records)
    group_names = ["C_or_N", "CHNO", "CHNO_no_CN", "not_CHNO"]
    group_rows = []
    for group in group_names:
        group_records = [record for record in records if group in record.get("focus_groups", [])]
        if group_records:
            group_rows.append(_record_group_row(group, group_records))
    return {
        "schema_version": "rtece_rattle_relax_summary.v1",
        "num_configs": int(len(records)),
        "converged_count": base["converged_count"],
        "converged_fraction": base["converged_fraction"],
        "mean_initial_rmsd_a": base["mean_initial_rmsd_a"],
        "mean_final_rmsd_a": base["mean_final_rmsd_a"],
        "max_final_rmsd_a": base["max_final_rmsd_a"],
        "mean_max_fmax_ev_a": base["mean_max_fmax_ev_a"],
        "max_fmax_ev_a": base["max_fmax_ev_a"],
        "focus_groups": group_rows,
    }


def make_rtece_calculator(model, config, *, device: Any, dtype: Any, force_mode: str, neighborlist_backend: str):
    from tace.interface.ase import RTECEAseCalc

    return RTECEAseCalc(
        model,
        config=config,
        dtype=dtype,
        device=device,
        force_mode=force_mode,
        neighborlist_backend=neighborlist_backend,
    )


def relax_rattled_structure(
    atoms,
    *,
    calculator,
    config_index: int,
    rattle_std_a: float,
    rattle_seed: int,
    fmax: float,
    max_steps: int,
) -> dict[str, Any]:
    from ase.optimize import LBFGS
    import numpy as np

    reference = atoms.copy()
    initial = deterministic_rattle_atoms(atoms, rattle_std_a=float(rattle_std_a), seed=int(rattle_seed) + int(config_index))
    initial_rmsd = positions_rmsd(initial.positions, reference.positions)
    initial.calc = calculator
    fmax_history: list[float] = []
    energy_history: list[float] = []
    start = time.perf_counter()

    def record_step() -> None:
        forces = np.asarray(initial.get_forces(), dtype=np.float64)
        fmax_value = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0
        fmax_history.append(fmax_value)
        energy_history.append(float(initial.get_potential_energy()))

    record_step()
    optimizer = LBFGS(initial, logfile=None)
    optimizer.attach(record_step, interval=1)
    converged = bool(optimizer.run(fmax=float(fmax), steps=int(max_steps)))
    wall_s = time.perf_counter() - start
    final_forces = np.asarray(initial.get_forces(), dtype=np.float64)
    final_fmax = float(np.linalg.norm(final_forces, axis=1).max()) if len(final_forces) else 0.0
    max_fmax = float(max([final_fmax, *fmax_history])) if fmax_history else final_fmax
    symbols = reference.get_chemical_symbols()
    return {
        "config_index": int(config_index),
        "formula": reference.get_chemical_formula(),
        "num_atoms": int(len(reference)),
        "focus_groups": classify_focus_groups(symbols),
        "rattle_std_a": float(rattle_std_a),
        "rattle_seed": int(rattle_seed) + int(config_index),
        "initial_rmsd_a": float(initial_rmsd),
        "final_rmsd_a": positions_rmsd(initial.positions, reference.positions),
        "converged": converged,
        "steps_recorded": int(max(0, len(fmax_history) - 1)),
        "max_steps": int(max_steps),
        "initial_fmax_ev_a": float(fmax_history[0]) if fmax_history else None,
        "final_fmax_ev_a": final_fmax,
        "max_fmax_ev_a": max_fmax,
        "initial_energy_eV": float(energy_history[0]) if energy_history else None,
        "final_energy_eV": float(initial.get_potential_energy()),
        "wall_s": float(wall_s),
    }


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# rTECE Rattle Relax",
        "",
        f"- route: `{payload.get('route', 'NA')}`",
        f"- checkpoint: `{payload.get('checkpoint', 'NA')}`",
        f"- configs: `{payload.get('configs', 'NA')}`",
        "",
        "## Summary",
        "",
        "| group | count | converged frac | mean initial RMSD | mean final RMSD | max final RMSD | max fmax |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    summary = payload["summary"]
    rows = [{"label": "all", **summary}]
    rows.extend(summary.get("focus_groups", []))
    for row in rows:
        lines.append(
            "| {label} | {count} | {conv:.3f} | {init:.4g} | {final:.4g} | {max_rmsd:.4g} | {max_f:.4g} |".format(
                label=row["label"],
                count=row.get("count", summary["num_configs"]),
                conv=float(row["converged_fraction"]),
                init=float(row["mean_initial_rmsd_a"] or 0.0),
                final=float(row["mean_final_rmsd_a"] or 0.0),
                max_rmsd=float(row["max_final_rmsd_a"] or 0.0),
                max_f=float(row["max_fmax_ev_a"] or 0.0),
            )
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
    parser.add_argument("--start-config", type=int, default=0)
    parser.add_argument("--limit-configs", type=int, default=2)
    parser.add_argument("--rattle-std", type=float, default=0.05)
    parser.add_argument("--rattle-seed", type=int, default=20260718)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--default-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--force-mode", default="autograd")
    parser.add_argument("--neighborlist-backend", choices=("ase", "vesin", "matscipy"), default="matscipy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import ase.io
    import torch
    from tace.models.rtece_workflow import load_checkpoint

    if int(args.limit_configs) < 1:
        raise ValueError("--limit-configs must be positive")
    if int(args.start_config) < 0:
        raise ValueError("--start-config must be non-negative")
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if args.default_dtype == "float64" else torch.float32
    model, config, metadata = load_checkpoint(args.checkpoint, dtype=dtype, device=device)
    calculator = make_rtece_calculator(
        model,
        config,
        device=device,
        dtype=dtype,
        force_mode=str(args.force_mode),
        neighborlist_backend=str(args.neighborlist_backend),
    )
    stop = int(args.start_config) + int(args.limit_configs)
    atoms_list = ase.io.read(str(args.configs), index=f"{int(args.start_config)}:{stop}")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    records = []
    for offset, atoms in enumerate(atoms_list):
        config_index = int(args.start_config) + offset
        records.append(
            relax_rattled_structure(
                atoms,
                calculator=calculator,
                config_index=config_index,
                rattle_std_a=float(args.rattle_std),
                rattle_seed=int(args.rattle_seed),
                fmax=float(args.fmax),
                max_steps=int(args.max_steps),
            )
        )
    payload = {
        "schema_version": "rtece_rattle_relax.v1",
        "route": args.route,
        "checkpoint": str(args.checkpoint),
        "configs": str(args.configs),
        "start_config": int(args.start_config),
        "limit_configs": int(args.limit_configs),
        "rattle_std_a": float(args.rattle_std),
        "rattle_seed": int(args.rattle_seed),
        "fmax": float(args.fmax),
        "max_steps": int(args.max_steps),
        "force_mode": str(args.force_mode),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "neighborlist_backend": str(args.neighborlist_backend),
        "cutoff": float(config.cutoff),
        "checkpoint_scalar_path_ids": list(config.scalar_path_ids or []),
        "checkpoint_tece_path_manifest_hash": (metadata.get("tece_path_manifest") or {}).get("manifest_hash"),
        "summary": summarize_relax_records(records),
        "records": records,
    }
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
