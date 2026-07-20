#!/usr/bin/env python3
"""Run bounded dimer and rattle/relax probes for Stage145 community baselines."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from benchmarks.oc20neb_tace_mace.benchmark_stage145_community import run_nep_prediction
from benchmarks.oc20neb_tace_mace.dimer_scan_rtece import (
    build_dimer_atoms,
    dimer_distances_from_covalent_radii,
    summarize_dimer_scan_rows,
)
from benchmarks.oc20neb_tace_mace.rattle_relax_rtece import relax_rattled_structure, summarize_relax_records


class NEPPredictionCalculator:
    implemented_properties = ["energy", "forces"]

    def __init__(self, model_artifact: Path, *, run_dir: Path, energy_key: str = "energy", forces_key: str = "forces") -> None:
        from ase.calculators.calculator import Calculator

        class _Calc(Calculator):
            implemented_properties = ["energy", "forces"]

            def __init__(self, outer: "NEPPredictionCalculator") -> None:
                super().__init__()
                self.outer = outer
                self.call_index = 0

            def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=None):  # type: ignore[override]
                super().calculate(atoms, properties, system_changes)
                if atoms is None:
                    raise ValueError("atoms is required")
                self.call_index += 1
                local = atoms.copy()
                local.info[self.outer.energy_key] = 0.0
                local.arrays[self.outer.forces_key] = np.zeros((len(local), 3), dtype=np.float64)
                work_dir = self.outer.run_dir / f"nep_single_{self.call_index:06d}"
                pred_e, pred_f, _ = run_nep_prediction(
                    model_artifact=self.outer.model_artifact,
                    atoms_list=[local],
                    energy_key=self.outer.energy_key,
                    forces_key=self.outer.forces_key,
                    work_dir=work_dir,
                )
                self.results["energy"] = float(pred_e.reshape(-1)[0])
                self.results["forces"] = np.asarray(pred_f, dtype=np.float64).reshape(len(local), 3)

        self.model_artifact = Path(model_artifact)
        self.run_dir = Path(run_dir)
        self.energy_key = str(energy_key)
        self.forces_key = str(forces_key)
        self.calculator = _Calc(self)


def make_calculator(engine: str, model_artifact: Path, *, run_dir: Path):
    if engine == "deepmd":
        from deepmd.calculator import DP

        return DP(model=str(model_artifact))
    if engine == "nep":
        return NEPPredictionCalculator(model_artifact, run_dir=run_dir).calculator
    raise ValueError(f"unsupported engine {engine!r}")


def scan_pair_with_calculator(calculator, *, symbol_a: str, symbol_b: str, num_points: int, min_scale: float, max_scale: float) -> dict[str, Any]:
    rows = []
    for distance in dimer_distances_from_covalent_radii(
        symbol_a,
        symbol_b,
        num_points=int(num_points),
        min_scale=float(min_scale),
        max_scale=float(max_scale),
    ):
        atoms = build_dimer_atoms(symbol_a, symbol_b, float(distance))
        atoms.calc = calculator
        try:
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=np.float64)
            force_parallel = float(forces[0, 0])
            max_force = float(np.linalg.norm(forces, axis=1).max()) if len(forces) else 0.0
            balance = float(np.linalg.norm(forces.sum(axis=0))) if len(forces) else 0.0
        except Exception as exc:
            energy = float("nan")
            force_parallel = float("nan")
            max_force = float("nan")
            balance = float("nan")
            rows.append(
                {
                    "pair": f"{symbol_a}-{symbol_b}",
                    "symbol_a": str(symbol_a),
                    "symbol_b": str(symbol_b),
                    "distance_a": float(distance),
                    "energy_eV": energy,
                    "force_parallel_ev_a": force_parallel,
                    "max_force_norm_ev_a": max_force,
                    "force_balance_error_ev_a": balance,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        rows.append(
            {
                "pair": f"{symbol_a}-{symbol_b}",
                "symbol_a": str(symbol_a),
                "symbol_b": str(symbol_b),
                "distance_a": float(distance),
                "energy_eV": energy,
                "force_parallel_ev_a": force_parallel,
                "max_force_norm_ev_a": max_force,
                "force_balance_error_ev_a": balance,
            }
        )
    return {"pair": f"{symbol_a}-{symbol_b}", "rows": rows, "summary": summarize_dimer_scan_rows(rows)}


def load_relax_configs(configs: Path, *, start_config: int, limit_configs: int):
    import ase.io

    stop = int(start_config) + int(limit_configs)
    atoms_list = ase.io.read(str(configs), index=f"{int(start_config)}:{stop}")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    if not atoms_list:
        raise ValueError(f"no configurations read from {configs}")
    return atoms_list


def run_physical(args: argparse.Namespace) -> dict[str, Any]:
    calc = make_calculator(args.engine, args.model_artifact, run_dir=args.run_dir)
    pair_payloads = [
        scan_pair_with_calculator(
            calc,
            symbol_a=a,
            symbol_b=b,
            num_points=args.dimer_points,
            min_scale=args.min_scale,
            max_scale=args.max_scale,
        )
        for a, b in args.pairs
    ]
    atoms_list = load_relax_configs(args.configs, start_config=args.start_config, limit_configs=args.limit_configs)
    records = []
    for offset, atoms in enumerate(atoms_list):
        records.append(
            relax_rattled_structure(
                atoms,
                calculator=calc,
                config_index=int(args.start_config) + offset,
                rattle_std_a=float(args.rattle_std),
                rattle_seed=int(args.rattle_seed),
                fmax=float(args.fmax),
                max_steps=int(args.max_steps),
            )
        )
    rattle_summary = summarize_relax_records(records)
    dimer_gate = all(
        (not item["summary"].get("has_nonfinite")) and bool(item["summary"].get("short_force_repulsive"))
        for item in pair_payloads
    )
    rattle_gate = bool(
        (rattle_summary.get("max_fmax_ev_a") is not None and float(rattle_summary["max_fmax_ev_a"]) <= 1.0)
        and (rattle_summary.get("max_final_rmsd_a") is not None and float(rattle_summary["max_final_rmsd_a"]) <= 0.25)
    )
    payload = {
        "schema_version": "community_baseline_physical_pareto.v1",
        "row_name": str(args.row_name),
        "engine": str(args.engine),
        "status": "completed",
        "model_artifact": str(args.model_artifact),
        "configs": str(args.configs),
        "dimer_scan": {"pair_summaries": pair_payloads},
        "rattle_relax": {"records": records, "summary": rattle_summary},
        "dimer_gate_pass": bool(dimer_gate),
        "rattle_gate_pass": bool(rattle_gate),
        "physical_gate_pass": bool(dimer_gate and rattle_gate),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _write_error(output: Path, *, args: argparse.Namespace, message: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "schema_version": "community_baseline_physical_pareto.v1",
                "row_name": str(args.row_name),
                "engine": str(args.engine),
                "status": "error",
                "error": str(message),
                "model_artifact": str(args.model_artifact),
                "configs": str(args.configs),
                "physical_gate_pass": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_pair(value: str) -> tuple[str, str]:
    parts = [part.strip() for part in value.replace(",", "-").split("-") if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("pair must look like C-N")
    return parts[0], parts[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("deepmd", "nep"), required=True)
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--row-name", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pairs", nargs="+", type=parse_pair, default=[("C", "N"), ("C", "O"), ("N", "H"), ("O", "H")])
    parser.add_argument("--dimer-points", type=int, default=16)
    parser.add_argument("--min-scale", type=float, default=0.5)
    parser.add_argument("--max-scale", type=float, default=5.0)
    parser.add_argument("--start-config", type=int, default=0)
    parser.add_argument("--limit-configs", type=int, default=2)
    parser.add_argument("--rattle-std", type=float, default=0.05)
    parser.add_argument("--rattle-seed", type=int, default=20260721)
    parser.add_argument("--fmax", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        payload = run_physical(args)
    except Exception as exc:
        _write_error(args.output_json, args=args, message=f"{type(exc).__name__}: {exc}")
        raise
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
