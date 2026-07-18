#!/usr/bin/env python3
"""Run bounded rTECE dimer scans for physical smoothness diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def dimer_distances_from_covalent_radii(
    symbol_a: str,
    symbol_b: str,
    *,
    num_points: int,
    min_scale: float = 0.5,
    max_scale: float = 5.0,
) -> list[float]:
    """Return linearly spaced dimer distances from scaled covalent-radius sums."""
    if int(num_points) < 2:
        raise ValueError("num_points must be at least 2")
    if float(min_scale) <= 0.0 or float(max_scale) <= 0.0:
        raise ValueError("distance scales must be positive")
    if float(min_scale) >= float(max_scale):
        raise ValueError("min_scale must be smaller than max_scale")
    from ase.data import atomic_numbers, covalent_radii

    try:
        radius_a = float(covalent_radii[atomic_numbers[str(symbol_a)]])
        radius_b = float(covalent_radii[atomic_numbers[str(symbol_b)]])
    except KeyError as exc:
        raise ValueError(f"unknown chemical symbol in pair {symbol_a!r}-{symbol_b!r}") from exc
    radius_sum = radius_a + radius_b
    if radius_sum <= 0.0:
        raise ValueError(f"covalent radius sum must be positive for {symbol_a}-{symbol_b}")
    import numpy as np

    return [float(value) for value in np.linspace(float(min_scale) * radius_sum, float(max_scale) * radius_sum, int(num_points))]


def build_dimer_atoms(symbol_a: str, symbol_b: str, distance_a: float):
    from ase import Atoms

    if float(distance_a) <= 0.0:
        raise ValueError("dimer distance must be positive")
    return Atoms(
        symbols=[str(symbol_a), str(symbol_b)],
        positions=[[0.0, 0.0, 0.0], [float(distance_a), 0.0, 0.0]],
        pbc=False,
    )


def _is_finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def summarize_dimer_scan_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("dimer scan summary requires at least one row")
    ordered = sorted(rows, key=lambda row: float(row["distance_a"]))
    energies = [float(row["energy_eV"]) for row in ordered if _is_finite(row.get("energy_eV"))]
    force_parallel = [float(row["force_parallel_ev_a"]) for row in ordered if _is_finite(row.get("force_parallel_ev_a"))]
    max_force_norms = [float(row["max_force_norm_ev_a"]) for row in ordered if _is_finite(row.get("max_force_norm_ev_a"))]
    num_nonfinite_energy = sum(1 for row in ordered if not _is_finite(row.get("energy_eV")))
    num_nonfinite_force = sum(
        1
        for row in ordered
        if not _is_finite(row.get("force_parallel_ev_a")) or not _is_finite(row.get("max_force_norm_ev_a"))
    )
    energy_steps = [abs(energies[index + 1] - energies[index]) for index in range(len(energies) - 1)]
    force_steps = [abs(force_parallel[index + 1] - force_parallel[index]) for index in range(len(force_parallel) - 1)]
    short_row = ordered[0]
    long_row = ordered[-1]
    short_energy = float(short_row["energy_eV"]) if _is_finite(short_row.get("energy_eV")) else None
    long_energy = float(long_row["energy_eV"]) if _is_finite(long_row.get("energy_eV")) else None
    short_force = float(short_row["force_parallel_ev_a"]) if _is_finite(short_row.get("force_parallel_ev_a")) else None
    return {
        "num_points": int(len(ordered)),
        "num_nonfinite_energy": int(num_nonfinite_energy),
        "num_nonfinite_force": int(num_nonfinite_force),
        "has_nonfinite": bool(num_nonfinite_energy or num_nonfinite_force),
        "min_distance_a": float(min(float(row["distance_a"]) for row in ordered)),
        "max_distance_a": float(max(float(row["distance_a"]) for row in ordered)),
        "energy_min_eV": float(min(energies)) if energies else None,
        "energy_max_eV": float(max(energies)) if energies else None,
        "energy_range_eV": float(max(energies) - min(energies)) if energies else None,
        "max_abs_force_ev_a": float(max(abs(value) for value in max_force_norms)) if max_force_norms else None,
        "max_abs_energy_step_eV": float(max(energy_steps)) if energy_steps else 0.0,
        "max_abs_force_step_ev_a": float(max(force_steps)) if force_steps else 0.0,
        "short_distance_a": float(short_row["distance_a"]),
        "long_distance_a": float(long_row["distance_a"]),
        "short_minus_long_energy_eV": float(short_energy - long_energy) if short_energy is not None and long_energy is not None else None,
        "short_force_parallel_ev_a": short_force,
        "short_force_repulsive": bool(short_force < 0.0) if short_force is not None else False,
    }


def scan_dimer_pair(
    model,
    config,
    *,
    symbol_a: str,
    symbol_b: str,
    distances: list[float],
    device: Any,
    dtype: Any,
    force_mode: str,
    neighborlist_backend: str,
) -> list[dict[str, Any]]:
    import torch
    from benchmarks.oc20neb_tace_mace.train_rtece_scalar import atoms_to_rtece_graph
    from tace.models.rtece_workflow import predict

    rows: list[dict[str, Any]] = []
    for distance in distances:
        atoms = build_dimer_atoms(symbol_a, symbol_b, float(distance))
        graph = atoms_to_rtece_graph(
            atoms,
            cutoff=float(config.cutoff),
            device=device,
            dtype=dtype,
            neighborlist_backend=neighborlist_backend,
        )
        out = predict(model, graph, force_mode=force_mode)
        energy = float(out["energy"].detach().cpu().reshape(-1)[0])
        forces = out["forces"].detach().cpu().to(dtype=torch.float64)
        force_parallel = float(forces[0, 0].item())
        rows.append(
            {
                "pair": f"{symbol_a}-{symbol_b}",
                "symbol_a": str(symbol_a),
                "symbol_b": str(symbol_b),
                "distance_a": float(distance),
                "energy_eV": energy,
                "force_parallel_ev_a": force_parallel,
                "max_force_norm_ev_a": float(torch.linalg.vector_norm(forces, dim=1).max().item()),
                "force_balance_error_ev_a": float(torch.linalg.vector_norm(forces.sum(dim=0)).item()),
            }
        )
    return rows


def parse_pair(value: str) -> tuple[str, str]:
    parts = [part.strip() for part in value.replace(",", "-").split("-") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("dimer pair must look like C-N or C,N")
    return parts[0], parts[1]


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# rTECE Dimer Scan",
        "",
        f"- route: `{payload.get('route', 'NA')}`",
        f"- checkpoint: `{payload.get('checkpoint', 'NA')}`",
        f"- force mode: `{payload.get('force_mode', 'NA')}`",
        "",
        "## Summary",
        "",
        "| pair | points | nonfinite | short repulsive | short dE | short F | max |F| | max dE step | max dF step |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in payload["pair_summaries"]:
        summary = item["summary"]
        lines.append(
            "| {pair} | {num_points} | {nonfinite} | {repulsive} | {short_de:.6g} | {short_force:.6g} | {fmax:.6g} | {de:.6g} | {df:.6g} |".format(
                pair=item["pair"],
                num_points=summary["num_points"],
                nonfinite=int(summary["has_nonfinite"]),
                repulsive=int(summary["short_force_repulsive"]),
                short_de=float(summary["short_minus_long_energy_eV"] or 0.0),
                short_force=float(summary["short_force_parallel_ev_a"] or 0.0),
                fmax=float(summary["max_abs_force_ev_a"] or 0.0),
                de=float(summary["max_abs_energy_step_eV"] or 0.0),
                df=float(summary["max_abs_force_step_ev_a"] or 0.0),
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument("--route", default=None)
    parser.add_argument("--pairs", nargs="+", type=parse_pair, default=[("C", "N"), ("C", "O"), ("N", "H"), ("O", "H")])
    parser.add_argument("--num-points", type=int, default=24)
    parser.add_argument("--min-scale", type=float, default=0.5)
    parser.add_argument("--max-scale", type=float, default=5.0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--default-dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--force-mode", default="autograd")
    parser.add_argument("--neighborlist-backend", choices=("ase", "vesin", "matscipy"), default="ase")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import torch
    from tace.models.rtece_workflow import load_checkpoint

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    dtype = torch.float64 if args.default_dtype == "float64" else torch.float32
    model, config, metadata = load_checkpoint(args.checkpoint, dtype=dtype, device=device)
    pair_payloads = []
    for symbol_a, symbol_b in args.pairs:
        distances = dimer_distances_from_covalent_radii(
            symbol_a,
            symbol_b,
            num_points=int(args.num_points),
            min_scale=float(args.min_scale),
            max_scale=float(args.max_scale),
        )
        rows = scan_dimer_pair(
            model,
            config,
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            distances=distances,
            device=device,
            dtype=dtype,
            force_mode=str(args.force_mode),
            neighborlist_backend=str(args.neighborlist_backend),
        )
        pair_payloads.append({"pair": f"{symbol_a}-{symbol_b}", "summary": summarize_dimer_scan_rows(rows), "rows": rows})
    payload = {
        "schema_version": "rtece_dimer_scan.v1",
        "route": args.route,
        "checkpoint": str(args.checkpoint),
        "force_mode": str(args.force_mode),
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "num_points": int(args.num_points),
        "min_scale": float(args.min_scale),
        "max_scale": float(args.max_scale),
        "cutoff": float(config.cutoff),
        "checkpoint_scalar_path_ids": list(config.scalar_path_ids or []),
        "checkpoint_tece_path_manifest_hash": (metadata.get("tece_path_manifest") or {}).get("manifest_hash"),
        "pair_summaries": pair_payloads,
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
