#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from benchmarks.oc20neb_tace_mace.benchmark_models import log
from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import load_atoms_window
from benchmarks.oc20neb_tace_mace.rtece_scalar_model import collate_graphs
from benchmarks.oc20neb_tace_mace.train_rtece_scalar import atoms_to_graph, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile scalar-sketched rTECE force paths.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-output", type=Path, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--default-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--start-config", type=int, default=0)
    parser.add_argument("--limit-configs", type=int, default=1024)
    parser.add_argument("--warmup-passes", type=int, default=3)
    parser.add_argument("--profile-passes", type=int, default=5)
    parser.add_argument("--top-ops", type=int, default=20)
    parser.add_argument(
        "--force-mode",
        choices=("autograd", "analytic_pair", "analytic_pair_triton_force", "analytic_element_triton_force", "analytic_density", "analytic_element_packed"),
        default="analytic_pair",
    )
    return parser.parse_args()


def event_time(event: Any, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def profile_rows_from_events(events: list[Any], *, limit: int) -> list[dict[str, Any]]:
    rows = []
    for event in events:
        device_time = event_time(event, "device_time_total", "cuda_time_total", "self_device_time_total", "self_cuda_time_total")
        cpu_time = event_time(event, "cpu_time_total")
        self_cpu_time = event_time(event, "self_cpu_time_total")
        rows.append(
            {
                "name": str(getattr(event, "key", "")),
                "count": int(getattr(event, "count", 0) or 0),
                "device_time_total_us": device_time,
                "cpu_time_total_us": cpu_time,
                "self_cpu_time_total_us": self_cpu_time,
            }
        )
    rows.sort(key=lambda row: (row["device_time_total_us"], row["cpu_time_total_us"]), reverse=True)
    return rows[: max(1, limit)]


def run_model(model, graph, force_mode: str):
    if force_mode == "analytic_pair":
        return model.forward_pair_analytic_forces(graph)
    if force_mode == "analytic_pair_triton_force":
        return model.forward_pair_triton_force_analytic_forces(graph)
    if force_mode == "analytic_element_triton_force":
        return model.forward_element_density_triton_force_analytic_forces(graph)
    if force_mode == "analytic_density":
        return model.forward_density_analytic_forces(graph)
    if force_mode == "analytic_element_packed":
        return model.forward_element_density_packed_analytic_forces(graph)
    return model(graph)


def main() -> None:
    args = parse_args()
    dtype = torch.float64 if args.default_dtype == "float64" else torch.float32
    requested = torch.device(args.device)
    device = requested if requested.type == "cpu" or torch.cuda.is_available() else torch.device("cpu")
    model, config = load_checkpoint(args.model, dtype=dtype, device=device)
    for param in model.parameters():
        param.requires_grad_(False)

    atoms_list = load_atoms_window(
        args.configs,
        start_config=args.start_config,
        limit_configs=args.limit_configs,
    )
    graph = collate_graphs(
        [atoms_to_graph(atoms, cutoff=config.cutoff, device=device, dtype=dtype)[0] for atoms in atoms_list]
    )
    atoms = int(sum(len(atoms) for atoms in atoms_list))

    for _ in range(max(0, args.warmup_passes)):
        run_model(model, graph, args.force_mode)
    if device.type == "cuda":
        torch.cuda.synchronize()

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(activities=activities, record_shapes=False, profile_memory=True) as prof:
        for _ in range(max(1, args.profile_passes)):
            run_model(model, graph, args.force_mode)
            if device.type == "cuda":
                torch.cuda.synchronize()
            prof.step()

    events = prof.key_averages()
    top_ops = profile_rows_from_events(list(events), limit=args.top_ops)
    if args.trace_output is not None:
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(args.trace_output))

    payload = {
        "backend": "rtece_scalar",
        "model": str(args.model),
        "configs_path": str(args.configs),
        "start_config": args.start_config,
        "configs": len(atoms_list),
        "atoms": atoms,
        "device": str(device),
        "default_dtype": args.default_dtype,
        "force_mode": args.force_mode,
        "profile_passes": args.profile_passes,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "trace_output": str(args.trace_output) if args.trace_output is not None else None,
        "top_ops": top_ops,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
