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
    load_atoms,
    reference_arrays,
    summarize_errors,
)
from benchmarks.oc20neb_tace_mace.train_rtece_scalar import atoms_to_graph, load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a scalar-sketched rTECE prototype.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--configs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--default-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--limit-configs", type=int, default=128)
    parser.add_argument("--measure-passes", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float64 if args.default_dtype == "float64" else torch.float32
    requested = torch.device(args.device)
    device = requested if requested.type == "cpu" or torch.cuda.is_available() else torch.device("cpu")
    model, config = load_checkpoint(args.model, dtype=dtype, device=device)
    for param in model.parameters():
        param.requires_grad_(False)

    atoms_list = load_atoms(args.configs, args.limit_configs)
    ref_e, ref_f, natoms = reference_arrays(atoms_list, "energy", "forces")

    def forward_once(collect: bool):
        outputs = []
        for atoms in atoms_list:
            graph, _, _ = atoms_to_graph(atoms, cutoff=config.cutoff, device=device, dtype=dtype)
            out = model(graph)
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
    payload = {
        "backend": "rtece_scalar",
        "variant": args.variant,
        "model": str(args.model),
        "configs_path": str(args.configs),
        "configs": len(atoms_list),
        "atoms": atoms,
        "device": str(device),
        "default_dtype": args.default_dtype,
        "model_class": model.__class__.__name__,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "includes_graph_construction": True,
        "measure_passes": args.measure_passes,
        "pass_times_s": pass_times,
        "seconds_per_pass": seconds_per_pass,
        "atoms_per_second": atoms / seconds_per_pass,
        "configs_per_second": len(atoms_list) / seconds_per_pass,
        **summarize_errors(pred_e, pred_f, ref_e, ref_f, natoms),
        **cuda_memory(torch),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
