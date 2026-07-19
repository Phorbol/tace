#!/usr/bin/env python3
"""Benchmark TACE and MACE models on the OC20NEB fullcase-200 subset.

The script measures graph-level inference after graph construction so model
compute is compared without repeatedly charging ASE parsing and neighbor-list
construction. Accuracy is reported against extxyz `energy` and `forces` keys.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


def log(message: str) -> None:
    print(f'[benchmark_models] {message}', file=sys.stderr, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('mace', 'tace'), required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--configs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--default-dtype', choices=('float32', 'float64'), default='float32')
    parser.add_argument('--energy-key', default='energy')
    parser.add_argument('--forces-key', default='forces')
    parser.add_argument('--head', default='Default')
    parser.add_argument('--limit-configs', type=int, default=None)
    parser.add_argument('--warmup-passes', type=int, default=1)
    parser.add_argument('--measure-passes', type=int, default=3)
    parser.add_argument('--enable-cueq', action='store_true')
    parser.add_argument('--mace-accel-backend', choices=('none', 'cueq', 'oeq', 'hybrid'), default='none')
    parser.add_argument('--tace-accel-backend', choices=('none', 'oeq', 'cue', 'compile', 'eqt'), default='none')
    parser.add_argument('--tace-strict-load', type=int, default=1)
    parser.add_argument('--ema', type=int, default=1, help='TACE EMA flag passed to load_tace')
    parser.add_argument('--nl-backend', choices=('ase', 'vesin', 'matscipy'), default='matscipy')
    return parser.parse_args()


def load_atoms(configs: Path, limit: int | None):
    import ase.io

    index = ':' if limit is None else f':{int(limit)}'
    log(f'reading configs {configs} index={index}')
    atoms_list = ase.io.read(str(configs), index=index)
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    if not atoms_list:
        raise ValueError(f'no configurations read from {configs}')
    log(f'read {len(atoms_list)} configs')
    return atoms_list


def reference_arrays(atoms_list, energy_key: str, forces_key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    energies: list[float] = []
    forces: list[np.ndarray] = []
    natoms: list[int] = []
    for idx, atoms in enumerate(atoms_list):
        if energy_key in atoms.info:
            energy = float(atoms.info[energy_key])
        elif atoms.calc is not None and energy_key in getattr(atoms.calc, 'results', {}):
            energy = float(atoms.calc.results[energy_key])
        else:
            try:
                energy = float(atoms.get_potential_energy())
            except Exception as exc:
                raise KeyError(f'configuration {idx} is missing energy key {energy_key!r}') from exc
        if forces_key in atoms.arrays:
            force = np.asarray(atoms.arrays[forces_key], dtype=np.float64)
        elif atoms.calc is not None and forces_key in getattr(atoms.calc, 'results', {}):
            force = np.asarray(atoms.calc.results[forces_key], dtype=np.float64)
        else:
            try:
                force = np.asarray(atoms.get_forces(), dtype=np.float64)
            except Exception as exc:
                raise KeyError(f'configuration {idx} is missing forces key {forces_key!r}') from exc
        energies.append(energy)
        forces.append(force)
        natoms.append(len(atoms))
    return np.asarray(energies, dtype=np.float64), np.concatenate(forces, axis=0), np.asarray(natoms, dtype=np.float64)


def summarize_errors(pred_e: np.ndarray, pred_f: np.ndarray, ref_e: np.ndarray, ref_f: np.ndarray, natoms: np.ndarray) -> dict[str, float]:
    signed_e_err_mev_atom = (pred_e.reshape(-1) - ref_e.reshape(-1)) / natoms * 1000.0
    abs_e_err_mev_atom = np.abs(signed_e_err_mev_atom)
    abs_f_err_mev_a = np.abs(pred_f.reshape(-1, 3) - ref_f.reshape(-1, 3)) * 1000.0
    energy_bias = float(np.mean(signed_e_err_mev_atom))
    return {
        'mae_e_mev_atom': float(np.mean(abs_e_err_mev_atom)),
        'rmse_e_mev_atom': float(math.sqrt(float(np.mean(signed_e_err_mev_atom**2)))),
        'bias_e_mev_atom': energy_bias,
        'mean_signed_e_mev_atom': energy_bias,
        'max_abs_e_mev_atom': float(np.max(abs_e_err_mev_atom)),
        'mae_f_mev_a': float(np.mean(abs_f_err_mev_a)),
        'rmse_f_mev_a': float(math.sqrt(float(np.mean(abs_f_err_mev_a**2)))),
        'max_abs_f_mev_a': float(np.max(abs_f_err_mev_a)),
    }


def cuda_memory(torch_module) -> dict[str, float | None]:
    if not torch_module.cuda.is_available():
        return {'peak_allocated_mb': None, 'peak_reserved_mb': None}
    return {
        'peak_allocated_mb': float(torch_module.cuda.max_memory_allocated() / 1024**2),
        'peak_reserved_mb': float(torch_module.cuda.max_memory_reserved() / 1024**2),
    }


def benchmark_loop(torch_module, forward_once, *, device: str, warmup_passes: int, measure_passes: int) -> tuple[list[dict[str, Any]], list[float]]:
    for _ in range(max(0, warmup_passes)):
        forward_once(collect=False)
    if device == 'cuda' and torch_module.cuda.is_available():
        torch_module.cuda.synchronize()
        torch_module.cuda.reset_peak_memory_stats()
    pass_times: list[float] = []
    first_outputs: list[dict[str, Any]] = []
    for pass_idx in range(max(1, measure_passes)):
        start = time.perf_counter()
        outputs = forward_once(collect=(pass_idx == 0))
        if device == 'cuda' and torch_module.cuda.is_available():
            torch_module.cuda.synchronize()
        pass_times.append(time.perf_counter() - start)
        if pass_idx == 0:
            first_outputs = outputs
    return first_outputs, pass_times


def run_mace(args: argparse.Namespace, atoms_list) -> dict[str, Any]:
    log('importing MACE backend')
    import torch
    from mace import data
    from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
    from mace.cli.convert_e3nn_oeq import run as run_e3nn_to_oeq
    from mace.cli.convert_e3nn_hybrid import run as run_e3nn_to_hybrid
    from mace.tools import torch_geometric, torch_tools, utils
    log('imported MACE backend')

    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device) if args.device == 'cuda' else torch.device('cpu')
    log(f'loading MACE model {args.model}')
    model = torch.load(f=args.model, map_location=device)
    log(f'loaded MACE model class={model.__class__.__name__}')
    mace_accel_backend = 'cueq' if args.enable_cueq else args.mace_accel_backend
    if mace_accel_backend == 'cueq':
        log('converting MACE model to cueq')
        model = run_e3nn_to_cueq(model, device=device)
        log('converted MACE model to cueq')
    elif mace_accel_backend == 'oeq':
        log('converting MACE model to oeq')
        model = run_e3nn_to_oeq(model, device=device)
        log('converted MACE model to oeq')
    elif mace_accel_backend == 'hybrid':
        log('converting MACE model to cueq+oeq hybrid')
        model = run_e3nn_to_hybrid(model, device=device)
        log('converted MACE model to cueq+oeq hybrid')
    dtype = torch.float32 if args.default_dtype == 'float32' else torch.float64
    model = model.to(device=device, dtype=dtype)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    head_name = args.head if args.head else 'Default'
    configs = [data.config_from_atoms(atoms, head_name=head_name) for atoms in atoms_list]
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    heads = getattr(model, 'heads', None)
    log(f'building MACE graphs for {len(configs)} configs')
    dataset = [
        data.AtomicData.from_config(config, z_table=z_table, cutoff=float(model.r_max), heads=heads)
        for config in configs
    ]
    log('built MACE graphs')
    loader = torch_geometric.dataloader.DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    def forward_once(*, collect: bool) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        for batch in loader:
            batch = batch.to(device)
            with torch.enable_grad():
                output = model(batch.to_dict(), compute_stress=False)
            if collect:
                collected.append({
                    'energy': output['energy'].detach().cpu().numpy(),
                    'forces': output['forces'].detach().cpu().numpy(),
                })
        return collected

    log('running MACE benchmark loop')
    outputs, pass_times = benchmark_loop(torch, forward_once, device=args.device, warmup_passes=args.warmup_passes, measure_passes=args.measure_passes)
    log('finished MACE benchmark loop')
    pred_e = np.concatenate([item['energy'].reshape(-1) for item in outputs], axis=0)
    pred_f = np.concatenate([item['forces'].reshape(-1, 3) for item in outputs], axis=0)
    return {
        'backend': 'mace',
        'mace_accel_backend': mace_accel_backend,
        'model_class': model.__class__.__name__,
        'num_parameters': int(sum(p.numel() for p in model.parameters())),
        'pred_energy_ev': pred_e,
        'pred_forces_ev_a': pred_f,
        'pass_times_s': pass_times,
        **cuda_memory(torch),
    }


def run_tace(args: argparse.Namespace, atoms_list) -> dict[str, Any]:
    log('importing TACE backend')
    accel_env = {
        'none': ('0', '0', '0', '0'),
        'oeq': ('1', '0', '0', '0'),
        'cue': ('0', '1', '0', '0'),
        'compile': ('0', '0', '0', '1'),
        'eqt': ('0', '0', '1', '0'),
    }[args.tace_accel_backend]
    os.environ['TACE_USE_OEQ'] = accel_env[0]
    os.environ['TACE_USE_CUE'] = accel_env[1]
    os.environ['TACE_USE_EQT'] = accel_env[2]
    os.environ['TACE_USE_COMPILE'] = accel_env[3]
    log(
        'TACE accel env '
        f'backend={args.tace_accel_backend} '
        f'TACE_USE_OEQ={os.environ["TACE_USE_OEQ"]} '
        f'TACE_USE_CUE={os.environ["TACE_USE_CUE"]} '
        f'TACE_USE_EQT={os.environ["TACE_USE_EQT"]} '
        f'TACE_USE_COMPILE={os.environ["TACE_USE_COMPILE"]}'
    )
    import torch
    from torch_geometric.loader import DataLoader
    from tace.dataset.graph import from_atoms
    from tace.dataset.quantity import KEYS, PROPERTY, KeySpecification, update_keyspec_from_kwargs
    from tace.dataset.read import check_keys
    from tace.lightning import load_tace
    from tace.utils._global import DTYPE
    from tace.utils.utils import num_params
    log('imported TACE backend')

    device = torch.device(args.device if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    key_spec = KeySpecification()
    key_values = dict(KEYS)
    key_values.update({'energy_key': args.energy_key, 'forces_key': args.forces_key, 'direct_forces_key': args.forces_key})
    update_keyspec_from_kwargs(key_spec, key_values)
    log(f'loading TACE model {args.model}')
    model = load_tace(str(args.model), str(device), strict=bool(args.tace_strict_load), use_ema=args.ema)
    log(f'loaded TACE model class={model.__class__.__name__}')
    cutoff = model.get_cutoff()
    max_neighbors = model.get_max_neighbors()
    element = model.get_torch_element()
    target_property = model.get_target_property()
    embedding_property = model.get_embedding_property()
    dtype = DTYPE[args.default_dtype]
    torch.set_default_dtype(dtype)
    model = model.to(device=device, dtype=dtype)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    log(f'building TACE graphs for {len(atoms_list)} configs')
    dataset = [
        from_atoms(
            element,
            atoms,
            cutoff,
            max_neighbors='inf' if max_neighbors is None else max_neighbors,
            keyspec=key_spec,
            target_property=target_property,
            embedding_property=embedding_property,
            neighborlist_backend=args.nl_backend,
        )
        for atoms in check_keys(atoms_list, target_property, key_spec, embedding_property, training=True)
    ]
    log('built TACE graphs')
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    def forward_once(*, collect: bool) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        for batch in loader:
            batch = batch.to(device)
            for target in target_property:
                for requires_grad_p in PROPERTY[target]['requires_grad_with']:
                    if hasattr(batch, requires_grad_p):
                        getattr(batch, requires_grad_p).requires_grad_(True)
            with torch.enable_grad():
                output = model(batch)
            if collect:
                collected.append({
                    'energy': output['energy'].detach().cpu().numpy(),
                    'forces': output['forces'].detach().cpu().numpy(),
                })
        return collected

    outputs, pass_times = benchmark_loop(torch, forward_once, device=args.device, warmup_passes=args.warmup_passes, measure_passes=args.measure_passes)
    pred_e = np.concatenate([item['energy'].reshape(-1) for item in outputs], axis=0)
    pred_f = np.concatenate([item['forces'].reshape(-1, 3) for item in outputs], axis=0)
    return {
        'backend': 'tace',
        'tace_accel_backend': args.tace_accel_backend,
        'tace_strict_load': bool(args.tace_strict_load),
        'model_class': model.__class__.__name__,
        'num_parameters': int(num_params(model)),
        'pred_energy_ev': pred_e,
        'pred_forces_ev_a': pred_f,
        'pass_times_s': pass_times,
        **cuda_memory(torch),
    }


def main() -> None:
    args = parse_args()
    atoms_list = load_atoms(args.configs, args.limit_configs)
    ref_e, ref_f, natoms = reference_arrays(atoms_list, args.energy_key, args.forces_key)
    total_atoms = int(np.sum(natoms))
    started = time.time()
    if args.backend == 'mace':
        result = run_mace(args, atoms_list)
    else:
        result = run_tace(args, atoms_list)
    pass_times = [float(x) for x in result.pop('pass_times_s')]
    pred_e = result.pop('pred_energy_ev')
    pred_f = result.pop('pred_forces_ev_a')
    metrics = summarize_errors(pred_e, pred_f, ref_e, ref_f, natoms)
    mean_pass = float(np.mean(pass_times))
    summary = {
        **result,
        'configs': len(atoms_list),
        'atoms': total_atoms,
        'batch_size': int(args.batch_size),
        'device': args.device,
        'default_dtype': args.default_dtype,
        'warmup_passes': int(args.warmup_passes),
        'measure_passes': int(args.measure_passes),
        'pass_times_s': pass_times,
        'seconds_per_pass': mean_pass,
        'configs_per_second': float(len(atoms_list) / mean_pass),
        'atoms_per_second': float(total_atoms / mean_pass),
        'wall_seconds_including_setup': float(time.time() - started),
        'model': str(args.model),
        'configs_path': str(args.configs),
        'energy_key': args.energy_key,
        'forces_key': args.forces_key,
        **metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + '\n')
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
