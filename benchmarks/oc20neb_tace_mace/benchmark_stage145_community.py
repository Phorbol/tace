#!/usr/bin/env python3
"""Benchmark Stage145 community baselines on the common held-out DFT split."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from benchmarks.oc20neb_tace_mace.benchmark_models import cuda_memory, reference_arrays, summarize_errors
from benchmarks.oc20neb_tace_mace.relative_energy_metrics import (
    atoms_group_values,
    energy_error_decomposition_metrics,
    relative_energy_group_metrics,
)



def _empty_memory_payload() -> dict[str, float | None]:
    return {"peak_allocated_mb": None, "peak_reserved_mb": None, "peak_memory_mb": None}


def _memory_payload_from_torch(torch_module) -> dict[str, float | None]:
    payload = cuda_memory(torch_module)
    reserved = payload.get("peak_reserved_mb")
    allocated = payload.get("peak_allocated_mb")
    payload["peak_memory_mb"] = reserved if reserved is not None else allocated
    return payload


def _maybe_reset_cuda_peak_memory(device: str):
    if str(device) != "cuda":
        return None
    try:
        import torch
    except Exception:
        return None
    if not torch.cuda.is_available():
        return torch
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return torch


def _maybe_collect_cuda_peak_memory(torch_module) -> dict[str, float | None]:
    if torch_module is None:
        return _empty_memory_payload()
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
    return _memory_payload_from_torch(torch_module)


def _community_memory_payload(engine: str, torch_module) -> dict[str, float | str | None]:
    if str(engine) == "nep":
        return {
            **_empty_memory_payload(),
            "memory_measurement_protocol": "external_nep_process_not_captured",
        }
    return {
        **_maybe_collect_cuda_peak_memory(torch_module),
        "memory_measurement_protocol": "torch_cuda_peak_memory_for_inprocess_backends_or_null",
    }

def _flatten_relative_energy_metrics(metrics: dict[str, object]) -> dict[str, object]:
    flattened = dict(metrics)
    schema = flattened.pop("schema_version", None)
    if schema is not None:
        flattened["relative_energy_metric_schema_version"] = schema
    return flattened


def _flatten_energy_decomposition_metrics(metrics: dict[str, object]) -> dict[str, object]:
    flattened = dict(metrics)
    schema = flattened.pop("schema_version", None)
    if schema is not None:
        flattened["energy_decomposition_metric_schema_version"] = schema
    return flattened


def _load_atoms(configs: Path, *, start_config: int, limit_configs: int):
    import ase.io

    if int(start_config) < 0:
        raise ValueError("start_config must be non-negative")
    if int(limit_configs) < 1:
        raise ValueError("limit_configs must be positive")
    stop = int(start_config) + int(limit_configs)
    atoms_list = ase.io.read(str(configs), index=f"{int(start_config)}:{stop}")
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    if not atoms_list:
        raise ValueError(f"no configurations read from {configs}")
    return atoms_list


def _atoms_energy(atoms, energy_key: str, config_idx: int) -> float:
    if energy_key in atoms.info:
        return float(atoms.info[energy_key])
    if atoms.calc is not None and energy_key in getattr(atoms.calc, "results", {}):
        return float(atoms.calc.results[energy_key])
    try:
        return float(atoms.get_potential_energy())
    except Exception as exc:  # noqa: BLE001 - preserve config context for benchmark JSON.
        raise KeyError(f"configuration {config_idx} is missing energy key {energy_key!r}") from exc


def _atoms_forces(atoms, forces_key: str, config_idx: int) -> np.ndarray:
    if forces_key in atoms.arrays:
        return np.asarray(atoms.arrays[forces_key], dtype=np.float64)
    if atoms.calc is not None and forces_key in getattr(atoms.calc, "results", {}):
        return np.asarray(atoms.calc.results[forces_key], dtype=np.float64)
    try:
        return np.asarray(atoms.get_forces(), dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 - preserve config context for benchmark JSON.
        raise KeyError(f"configuration {config_idx} is missing forces key {forces_key!r}") from exc


def _write_nep_prediction_xyz(atoms_list, path: Path, *, energy_key: str, forces_key: str) -> None:
    import ase.io

    serializable = []
    for config_idx, atoms in enumerate(atoms_list):
        item = atoms.copy()
        energy = _atoms_energy(atoms, energy_key, config_idx)
        forces = _atoms_forces(atoms, forces_key, config_idx)
        item.info.clear()
        item.info["energy"] = energy
        for key in list(item.arrays.keys()):
            if key not in {"numbers", "positions"}:
                del item.arrays[key]
        item.arrays["force"] = forces
        serializable.append(item)
    ase.io.write(path, serializable, format="extxyz")


def _nep_type_symbols(train_dir: Path) -> list[str]:
    summary = train_dir / "conversion_summary.json"
    if summary.exists():
        payload = json.loads(summary.read_text())
        values = payload.get("type_map") or []
        if values:
            return [str(value) for value in values]
    nep_in = train_dir / "nep.in"
    if nep_in.exists():
        for line in nep_in.read_text().splitlines():
            parts = line.split()
            if parts and parts[0] == "type" and len(parts) >= 3:
                return [str(value) for value in parts[2:]]
    raise FileNotFoundError(f"cannot infer NEP type symbols from {train_dir}")


def run_nep_prediction(
    *,
    model_artifact: Path,
    atoms_list,
    energy_key: str,
    forces_key: str,
    work_dir: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_dir = model_artifact.parent
    type_symbols = _nep_type_symbols(train_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_artifact, work_dir / "nep.txt")
    _write_nep_prediction_xyz(atoms_list, work_dir / "train.xyz", energy_key=energy_key, forces_key=forces_key)
    (work_dir / "nep.in").write_text(
        "type         {n} {symbols}\nprediction   1\n".format(n=len(type_symbols), symbols=" ".join(type_symbols)),
        encoding="utf-8",
    )
    start = time.perf_counter()
    completed = subprocess.run(["nep"], cwd=work_dir, check=False, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(f"nep prediction failed with code {completed.returncode}: {completed.stderr[-2000:]}")
    energy_out = np.loadtxt(work_dir / "energy_train.out", ndmin=2)
    force_out = np.loadtxt(work_dir / "force_train.out", ndmin=2)
    natoms = np.asarray([len(atoms) for atoms in atoms_list], dtype=np.float64)
    pred_e = energy_out[:, 0].reshape(-1) * natoms
    pred_f = force_out[:, :3].reshape(-1, 3)
    return pred_e, pred_f, {
        "engine_protocol": "gpumd_nep_prediction_1",
        "prediction_work_dir": str(work_dir),
        "prediction_wall_s": float(elapsed),
        "stdout_tail": completed.stdout[-2000:],
        "stderr_tail": completed.stderr[-2000:],
    }


def run_deepmd_ase(
    *,
    model_artifact: Path,
    atoms_list,
    device: str,
    warmup_passes: int,
    measure_passes: int,
) -> tuple[np.ndarray, np.ndarray, list[float], dict[str, Any]]:
    from deepmd.calculator import DP

    calc = DP(model=str(model_artifact))
    pred_e: list[float] = []
    pred_f: list[np.ndarray] = []

    def forward_once(*, collect: bool) -> None:
        if collect:
            pred_e.clear()
            pred_f.clear()
        for atoms in atoms_list:
            local = atoms.copy()
            local.calc = calc
            energy = float(local.get_potential_energy())
            forces = np.asarray(local.get_forces(), dtype=np.float64)
            if collect:
                pred_e.append(energy)
                pred_f.append(forces)

    for _ in range(max(0, int(warmup_passes))):
        forward_once(collect=False)
    pass_times: list[float] = []
    for pass_idx in range(max(1, int(measure_passes))):
        start = time.perf_counter()
        forward_once(collect=pass_idx == 0)
        pass_times.append(time.perf_counter() - start)
    return np.asarray(pred_e, dtype=np.float64), np.concatenate(pred_f, axis=0), pass_times, {
        "engine_protocol": "deepmd_ase_DP",
        "device_request": str(device),
    }


def _write_error(output: Path, *, args: argparse.Namespace, message: str) -> None:
    payload = {
        "schema_version": "community_baseline_dft_benchmark.v1",
        "row_name": str(args.row_name),
        "engine": str(args.engine),
        "status": "error",
        "error": str(message),
        "model_artifact": str(args.model_artifact),
        "configs": str(args.configs),
        **_empty_memory_payload(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    atoms_list = _load_atoms(args.configs, start_config=args.start_config, limit_configs=args.limit_configs)
    ref_e, ref_f, natoms = reference_arrays(atoms_list, args.energy_key, args.forces_key)
    torch_for_memory = _maybe_reset_cuda_peak_memory(args.device)
    total_atoms = int(natoms.sum())
    if args.engine == "deepmd":
        pred_e, pred_f, pass_times, metadata = run_deepmd_ase(
            model_artifact=args.model_artifact,
            atoms_list=atoms_list,
            device=args.device,
            warmup_passes=args.warmup_passes,
            measure_passes=args.measure_passes,
        )
    elif args.engine == "nep":
        pass_times = []
        pred_e = pred_f = None
        metadata = None
        for pass_idx in range(max(1, int(args.measure_passes))):
            work_dir = args.output.parent / f"{args.row_name}_nep_prediction_pass{pass_idx}"
            p_e, p_f, meta = run_nep_prediction(
                model_artifact=args.model_artifact,
                atoms_list=atoms_list,
                energy_key=args.energy_key,
                forces_key=args.forces_key,
                work_dir=work_dir,
            )
            pass_times.append(float(meta["prediction_wall_s"]))
            if pass_idx == 0:
                pred_e, pred_f, metadata = p_e, p_f, meta
        assert pred_e is not None and pred_f is not None and metadata is not None
    else:
        raise ValueError(f"unsupported engine {args.engine!r}")
    metrics = summarize_errors(pred_e, pred_f, ref_e, ref_f, natoms)
    group_ids, image_indices = atoms_group_values(atoms_list, "case_id", "source_frame")
    relative_metrics = relative_energy_group_metrics(
        pred_e,
        ref_e,
        natoms,
        group_ids,
        image_indices=image_indices,
    )
    decomposition_metrics = energy_error_decomposition_metrics(
        pred_e,
        ref_e,
        natoms,
        group_ids,
        image_indices=image_indices,
    )
    mean_time = float(np.mean(pass_times)) if pass_times else None
    memory_payload = _community_memory_payload(args.engine, torch_for_memory)
    payload = {
        "schema_version": "community_baseline_dft_benchmark.v1",
        "row_name": str(args.row_name),
        "engine": str(args.engine),
        "status": "completed",
        "model_artifact": str(args.model_artifact),
        "configs": str(args.configs),
        "energy_key": str(args.energy_key),
        "forces_key": str(args.forces_key),
        "start_config": int(args.start_config),
        "limit_configs": int(args.limit_configs),
        "num_configs": int(len(atoms_list)),
        "total_atoms": total_atoms,
        "pass_times_s": [float(value) for value in pass_times],
        "mean_time_s": mean_time,
        "atoms_per_second": float(total_atoms / mean_time) if mean_time and mean_time > 0 else None,
        **memory_payload,
        "relative_energy_errors_available": True,
        **metrics,
        **_flatten_relative_energy_metrics(relative_metrics),
        **_flatten_energy_decomposition_metrics(decomposition_metrics),
        **metadata,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("deepmd", "nep"), required=True)
    parser.add_argument("--model-artifact", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--row-name", required=True)
    parser.add_argument("--energy-key", default="dft_energy")
    parser.add_argument("--forces-key", default="dft_forces")
    parser.add_argument("--start-config", type=int, default=0)
    parser.add_argument("--limit-configs", type=int, default=128)
    parser.add_argument("--warmup-passes", type=int, default=1)
    parser.add_argument("--measure-passes", type=int, default=3)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        payload = run_benchmark(args)
    except Exception as exc:
        _write_error(args.output, args=args, message=f"{type(exc).__name__}: {exc}")
        raise
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
