#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch

from benchmarks.oc20neb_tace_mace.benchmark_models import (
    cuda_memory,
    log,
    reference_arrays,
    summarize_errors,
)
from benchmarks.oc20neb_tace_mace.rtece_scalar_model import RTECEGraph, RTECEScalarConfig, collate_graphs
from benchmarks.oc20neb_tace_mace.train_rtece_scalar import atoms_to_graph, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a scalar-sketched rTECE prototype.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--default-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--start-config", type=int, default=0)
    parser.add_argument("--limit-configs", type=int, default=128)
    parser.add_argument("--measure-passes", type=int, default=3)
    parser.add_argument(
        "--force-mode",
        choices=("auto", "autograd", "analytic_pair", "analytic_pair_triton_force", "analytic_element_triton_force", "analytic_element_triton_descriptor_force", "analytic_density", "analytic_element_packed"),
        default="auto",
    )
    parser.add_argument(
        "--include-graph-construction",
        action="store_true",
        help="Include ASE neighbor-list graph construction inside the timed loop.",
    )
    parser.add_argument(
        "--batch-graph-construction",
        action="store_true",
        help="When graph construction is timed, rebuild all graphs and collate before one batched model pass.",
    )
    parser.add_argument(
        "--replay-cached-graph",
        action="store_true",
        help="Prebuild the graph once, then time position refresh plus one batched model pass.",
    )
    parser.add_argument(
        "--trajectory-replay-steps",
        type=int,
        default=0,
        help="Run a synthetic MD-style cached-graph trajectory with this many force steps per timed pass.",
    )
    parser.add_argument(
        "--trajectory-rebuild-interval",
        type=int,
        default=0,
        help="In trajectory replay mode, rebuild the batched graph every K force steps; 0 disables rebuilds.",
    )
    parser.add_argument(
        "--trajectory-displacement-std",
        type=float,
        default=0.0,
        help="Deterministic per-step synthetic displacement scale in Angstrom for trajectory replay.",
    )
    return parser.parse_args()


def extxyz_index(*, start_config: int, limit_configs: int | None) -> str:
    if start_config < 0:
        raise ValueError(f"start_config must be non-negative, got {start_config}")
    if limit_configs is not None and limit_configs < 1:
        raise ValueError(f"limit_configs must be positive when set, got {limit_configs}")
    if limit_configs is None:
        return ":" if start_config == 0 else f"{start_config}:"
    stop_config = start_config + limit_configs
    return f":{stop_config}" if start_config == 0 else f"{start_config}:{stop_config}"


def choose_rtece_force_mode(
    requested: str,
    config: RTECEScalarConfig,
    *,
    device_type: str,
    dtype: torch.dtype,
) -> str:
    if requested != "auto":
        return requested
    eligible_element_density = (
        bool(config.use_element_density)
        and not bool(config.use_density_quadratic)
        and not bool(config.use_vector_moments)
        and not bool(config.use_atomic_moments)
        and int(config.num_edge_sketches) == 0
        and device_type == "cuda"
        and dtype == torch.float32
    )
    if eligible_element_density:
        return "analytic_element_triton_descriptor_force"
    return "autograd"


def validate_graph_construction_args(
    *,
    include_graph_construction: bool,
    batch_graph_construction: bool,
    replay_cached_graph: bool,
    trajectory_replay_steps: int = 0,
    trajectory_rebuild_interval: int = 0,
) -> None:
    if trajectory_replay_steps < 0:
        raise ValueError(f"--trajectory-replay-steps must be non-negative, got {trajectory_replay_steps}")
    if trajectory_rebuild_interval < 0:
        raise ValueError(f"--trajectory-rebuild-interval must be non-negative, got {trajectory_rebuild_interval}")
    if batch_graph_construction and not include_graph_construction:
        raise ValueError("--batch-graph-construction requires --include-graph-construction")
    if replay_cached_graph and include_graph_construction:
        raise ValueError("--replay-cached-graph is separate from --include-graph-construction timing")
    if replay_cached_graph and batch_graph_construction:
        raise ValueError("--replay-cached-graph cannot be combined with --batch-graph-construction")
    if trajectory_replay_steps > 0:
        if include_graph_construction or batch_graph_construction or replay_cached_graph:
            raise ValueError("--trajectory-replay-steps is separate from graph-construction and cached-graph replay modes")
        if trajectory_rebuild_interval == 0:
            return
        if trajectory_rebuild_interval < 1:
            raise ValueError("--trajectory-rebuild-interval must be positive when enabled")
    elif trajectory_rebuild_interval:
        raise ValueError("--trajectory-rebuild-interval requires --trajectory-replay-steps")


def synthetic_trajectory_positions(
    base_positions: torch.Tensor,
    *,
    step: int,
    displacement_std: float,
) -> torch.Tensor:
    if step == 0 or displacement_std == 0.0:
        return base_positions
    atom_index = torch.arange(base_positions.shape[0], device=base_positions.device, dtype=base_positions.dtype)
    phases = atom_index[:, None] * base_positions.new_tensor([12.9898, 78.233, 37.719])
    phases = phases + float(step) * base_positions.new_tensor([0.37, 0.53, 0.71])
    return base_positions + float(displacement_std) * torch.sin(phases)


def atoms_to_geometry_graph(
    atoms,
    *,
    cutoff: float,
    device: torch.device,
    dtype: torch.dtype,
) -> RTECEGraph:
    from ase.neighborlist import neighbor_list

    src, dst = neighbor_list("ij", atoms, cutoff)
    if len(src) == 0:
        edge_index_np = np.zeros((2, 0), dtype=np.int64)
    else:
        edge_index_np = np.stack([src, dst], axis=0)
    return RTECEGraph(
        z=torch.tensor(atoms.numbers, dtype=torch.long, device=device),
        pos=torch.tensor(atoms.positions, dtype=dtype, device=device),
        edge_index=torch.tensor(edge_index_np, dtype=torch.long, device=device),
        batch=torch.zeros(len(atoms), dtype=torch.long, device=device),
    )


def replay_graph_positions(template: RTECEGraph, positions: torch.Tensor) -> RTECEGraph:
    if tuple(positions.shape) != tuple(template.pos.shape):
        raise ValueError(f"cached positions shape {tuple(positions.shape)} does not match graph shape {tuple(template.pos.shape)}")
    return RTECEGraph(
        z=template.z,
        pos=positions.to(device=template.pos.device, dtype=template.pos.dtype),
        edge_index=template.edge_index,
        batch=template.batch,
    )


def load_atoms_window(configs: Path, *, start_config: int, limit_configs: int | None):
    import ase.io

    index = extxyz_index(start_config=start_config, limit_configs=limit_configs)
    log(f"reading configs {configs} index={index}")
    atoms_list = ase.io.read(str(configs), index=index)
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    if not atoms_list:
        raise ValueError(f"no configurations read from {configs} index={index}")
    log(f"read {len(atoms_list)} configs")
    return atoms_list


def main() -> None:
    args = parse_args()
    validate_graph_construction_args(
        include_graph_construction=bool(args.include_graph_construction),
        batch_graph_construction=bool(args.batch_graph_construction),
        replay_cached_graph=bool(args.replay_cached_graph),
        trajectory_replay_steps=int(args.trajectory_replay_steps),
        trajectory_rebuild_interval=int(args.trajectory_rebuild_interval),
    )
    dtype = torch.float64 if args.default_dtype == "float64" else torch.float32
    requested = torch.device(args.device)
    device = requested if requested.type == "cpu" or torch.cuda.is_available() else torch.device("cpu")
    model, config = load_checkpoint(args.model, dtype=dtype, device=device)
    force_mode = choose_rtece_force_mode(args.force_mode, config, device_type=device.type, dtype=dtype)
    for param in model.parameters():
        param.requires_grad_(False)

    atoms_list = load_atoms_window(
        args.configs,
        start_config=args.start_config,
        limit_configs=args.limit_configs,
    )
    ref_e, ref_f, natoms = reference_arrays(atoms_list, "energy", "forces")
    prebuilt_graph = None
    replay_template = None
    replay_positions = None
    trajectory_template = None
    trajectory_base_positions = None
    if not args.include_graph_construction:
        prebuilt_graph = collate_graphs(
            [
                atoms_to_graph(atoms, cutoff=config.cutoff, device=device, dtype=dtype)[0]
                for atoms in atoms_list
            ]
        )
        if args.trajectory_replay_steps > 0:
            trajectory_template = prebuilt_graph
            trajectory_base_positions = prebuilt_graph.pos.detach().clone()
            prebuilt_graph = None
        elif args.replay_cached_graph:
            replay_template = prebuilt_graph
            replay_positions = prebuilt_graph.pos.detach().clone()
            prebuilt_graph = None

    def rebuild_batched_graph_from_positions(positions: torch.Tensor) -> RTECEGraph:
        positions_cpu = positions.detach().cpu().numpy()
        rebuilt = []
        offset = 0
        for atoms, count in zip(atoms_list, natoms, strict=True):
            count_int = int(count)
            atoms_copy = atoms.copy()
            atoms_copy.positions = positions_cpu[offset : offset + count_int]
            rebuilt.append(atoms_to_geometry_graph(atoms_copy, cutoff=config.cutoff, device=device, dtype=dtype))
            offset += count_int
        return collate_graphs(rebuilt)

    def run_model(graph):
        if force_mode == "analytic_pair":
            return model.forward_pair_analytic_forces(graph)
        if force_mode == "analytic_pair_triton_force":
            return model.forward_pair_triton_force_analytic_forces(graph)
        if force_mode == "analytic_element_triton_force":
            return model.forward_element_density_triton_force_analytic_forces(graph)
        if force_mode == "analytic_element_triton_descriptor_force":
            return model.forward_element_density_triton_descriptor_force_analytic_forces(graph)
        if force_mode == "analytic_density":
            return model.forward_density_analytic_forces(graph)
        if force_mode == "analytic_element_packed":
            return model.forward_element_density_packed_analytic_forces(graph)
        return model(graph)

    def forward_once(collect: bool):
        if prebuilt_graph is not None:
            out = run_model(prebuilt_graph)
            if not collect:
                return []
            return [
                {
                    "energy": out["energy"].detach().cpu().numpy(),
                    "forces": out["forces"].detach().cpu().numpy(),
                }
            ]

        if replay_template is not None:
            if replay_positions is None:
                raise RuntimeError("cached replay positions were not initialized")
            graph = replay_graph_positions(replay_template, replay_positions.clone())
            out = run_model(graph)
            if not collect:
                return []
            return [
                {
                    "energy": out["energy"].detach().cpu().numpy(),
                    "forces": out["forces"].detach().cpu().numpy(),
                }
            ]

        if trajectory_template is not None:
            if trajectory_base_positions is None:
                raise RuntimeError("trajectory replay positions were not initialized")
            graph_template = trajectory_template
            collected = []
            for step in range(int(args.trajectory_replay_steps)):
                positions = synthetic_trajectory_positions(
                    trajectory_base_positions,
                    step=step,
                    displacement_std=float(args.trajectory_displacement_std),
                )
                if step > 0 and args.trajectory_rebuild_interval and step % int(args.trajectory_rebuild_interval) == 0:
                    graph_template = rebuild_batched_graph_from_positions(positions)
                    graph = graph_template
                else:
                    graph = replay_graph_positions(graph_template, positions)
                out = run_model(graph)
                if collect and step == 0:
                    collected.append(
                        {
                            "energy": out["energy"].detach().cpu().numpy(),
                            "forces": out["forces"].detach().cpu().numpy(),
                        }
                    )
            return collected

        if args.batch_graph_construction:
            graph = collate_graphs(
                [
                    atoms_to_graph(atoms, cutoff=config.cutoff, device=device, dtype=dtype)[0]
                    for atoms in atoms_list
                ]
            )
            out = run_model(graph)
            if not collect:
                return []
            return [
                {
                    "energy": out["energy"].detach().cpu().numpy(),
                    "forces": out["forces"].detach().cpu().numpy(),
                }
            ]

        outputs = []
        for atoms in atoms_list:
            graph, _, _ = atoms_to_graph(atoms, cutoff=config.cutoff, device=device, dtype=dtype)
            out = run_model(graph)
            if collect:
                outputs.append(
                    {
                        "energy": out["energy"].detach().cpu().numpy(),
                        "forces": out["forces"].detach().cpu().numpy(),
                    }
                )
        return outputs

    forward_once(False)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    pass_times = []
    first_outputs = []
    for pass_idx in range(max(1, args.measure_passes)):
        start = time.perf_counter()
        outputs = forward_once(pass_idx == 0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        pass_times.append(time.perf_counter() - start)
        if pass_idx == 0:
            first_outputs = outputs

    pred_e = np.concatenate([item["energy"].reshape(-1) for item in first_outputs], axis=0)
    pred_f = np.concatenate([item["forces"].reshape(-1, 3) for item in first_outputs], axis=0)
    seconds_per_pass = float(np.mean(pass_times))
    atoms = int(natoms.sum())
    force_steps_per_pass = int(args.trajectory_replay_steps) if int(args.trajectory_replay_steps) > 0 else 1
    atom_steps_per_pass = atoms * force_steps_per_pass
    config_steps_per_pass = len(atoms_list) * force_steps_per_pass
    payload = {
        "backend": "rtece_scalar",
        "variant": args.variant,
        "model": str(args.model),
        "configs_path": str(args.configs),
        "start_config": args.start_config,
        "extxyz_index": extxyz_index(
            start_config=args.start_config,
            limit_configs=args.limit_configs,
        ),
        "configs": len(atoms_list),
        "atoms": atoms,
        "device": str(device),
        "default_dtype": args.default_dtype,
        "model_class": model.__class__.__name__,
        "requested_force_mode": args.force_mode,
        "force_mode": force_mode,
        "hidden_channels": list(model.config.hidden_channels),
        "num_radial": int(model.config.num_radial),
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "includes_graph_construction": bool(args.include_graph_construction),
        "batched_graph_construction": bool(args.batch_graph_construction),
        "replay_cached_graph": bool(args.replay_cached_graph),
        "trajectory_replay": trajectory_template is not None,
        "trajectory_replay_steps": int(args.trajectory_replay_steps),
        "trajectory_rebuild_interval": int(args.trajectory_rebuild_interval),
        "trajectory_displacement_std": float(args.trajectory_displacement_std),
        "prebuilt_batched_graph": prebuilt_graph is not None,
        "measure_passes": args.measure_passes,
        "force_steps_per_pass": force_steps_per_pass,
        "atom_steps_per_pass": atom_steps_per_pass,
        "pass_times_s": pass_times,
        "seconds_per_pass": seconds_per_pass,
        "atoms_per_second": atom_steps_per_pass / seconds_per_pass,
        "configs_per_second": config_steps_per_pass / seconds_per_pass,
        **summarize_errors(pred_e, pred_f, ref_e, ref_f, natoms),
        **cuda_memory(torch),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
