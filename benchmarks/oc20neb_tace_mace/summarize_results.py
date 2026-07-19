#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
from pathlib import Path


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def dmon_peak_fb_mb(path):
    peak = None
    p = Path(path)
    if not p.exists():
        return None
    for line in p.read_text(errors='ignore').splitlines():
        parts = line.split()
        if len(parts) < 16 or not re.match(r'^\d{2}:\d{2}:\d{2}$', parts[0]):
            continue
        try:
            fb = int(float(parts[15]))
        except ValueError:
            continue
        peak = fb if peak is None else max(peak, fb)
    return peak


def parse_tace_validation(log_path):
    text = Path(log_path).read_text(errors='ignore').replace('\r', '\n')
    current_epoch = None
    records = []
    current = None
    epoch_re = re.compile(r'\[Epoch\s+(\d+)/(\d+)\]')
    metric_re = re.compile(r'\[INFO\]\s+(val/[A-Za-z0-9_]+):\s+([-+0-9.eE]+)')
    for line in text.splitlines():
        m = epoch_re.search(line)
        if m:
            current_epoch = int(m.group(1))
        m = metric_re.search(line)
        if not m:
            continue
        name, value = m.group(1), float(m.group(2))
        if current is None or current.get('epoch') != current_epoch:
            current = {'epoch': current_epoch}
            records.append(current)
        current[name] = value
    complete = [r for r in records if 'val/energy_per_atom_mae' in r and 'val/forces_mae' in r]
    return complete


def sacct_elapsed(job_id):
    if not job_id:
        return None
    try:
        out = subprocess.check_output(
            ['sacct', '-j', str(job_id), '--parsable2', '--noheader', '--format=JobID,State,ExitCode,ElapsedRaw'],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
    except Exception:
        return None
    for line in out.splitlines():
        parts = line.split('|')
        if len(parts) >= 4 and parts[0] == str(job_id):
            try:
                return {'state': parts[1], 'exit_code': parts[2], 'elapsed_s': int(parts[3])}
            except ValueError:
                return {'state': parts[1], 'exit_code': parts[2], 'elapsed_s': None}
    return None


def fmt(x, digits=3):
    if x is None:
        return 'NA'
    if isinstance(x, str):
        return x
    return f'{x:.{digits}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mace-matrix-summary', required=True)
    ap.add_argument('--mace-benchmark', required=True)
    ap.add_argument('--tace-benchmark', required=True)
    ap.add_argument('--tace-log', required=True)
    ap.add_argument('--tace-dmon', required=True)
    ap.add_argument('--tace-job-id')
    ap.add_argument('--benchmark-dmon')
    ap.add_argument('--output-json', required=True)
    ap.add_argument('--output-md', required=True)
    args = ap.parse_args()

    matrix = load_json(args.mace_matrix_summary)[0]
    mace_bench = load_json(args.mace_benchmark)
    tace_bench = load_json(args.tace_benchmark)
    tace_vals = parse_tace_validation(args.tace_log)
    last_tace_val = tace_vals[-1] if tace_vals else {}
    best_tace_force = min((r.get('val/forces_mae') for r in tace_vals if r.get('val/forces_mae') is not None), default=None)
    best_tace_energy = min((r.get('val/energy_per_atom_mae') for r in tace_vals if r.get('val/energy_per_atom_mae') is not None), default=None)
    tace_sacct = sacct_elapsed(args.tace_job_id)

    comparison = {
        'mace_training': {
            'final_update': matrix.get('final_update'),
            'final_epoch': matrix.get('final_epoch'),
            'final_mae_e_mev_atom': matrix.get('final_mae_e_mev_atom'),
            'final_mae_f_mev_a': matrix.get('final_mae_f_mev_a'),
            'best_mae_e_mev_atom': matrix.get('best_mae_e_mev_atom'),
            'best_mae_f_mev_a': matrix.get('best_mae_f_mev_a'),
            'mean_seconds_per_epoch': matrix.get('mean_seconds_per_epoch'),
            'seconds_per_update': matrix.get('seconds_per_update'),
            'updates_per_second': matrix.get('updates_per_second'),
            'max_fb_memory_mb': matrix.get('max_fb_memory_mb'),
            'num_parameters': mace_bench.get('num_parameters'),
        },
        'tace_training': {
            'last_validation': last_tace_val,
            'best_val_energy_per_atom_mae': best_tace_energy,
            'best_val_forces_mae': best_tace_force,
            'max_fb_memory_mb': dmon_peak_fb_mb(args.tace_dmon),
            'sacct': tace_sacct,
            'num_parameters': tace_bench.get('num_parameters'),
        },
        'mace_inference': mace_bench,
        'tace_inference': tace_bench,
        'benchmark_peak_fb_mb': dmon_peak_fb_mb(args.benchmark_dmon) if args.benchmark_dmon else None,
        'sources': {
            'mace_matrix_summary': args.mace_matrix_summary,
            'mace_benchmark': args.mace_benchmark,
            'tace_benchmark': args.tace_benchmark,
            'tace_log': args.tace_log,
            'tace_dmon': args.tace_dmon,
            'benchmark_dmon': args.benchmark_dmon,
        },
    }

    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(comparison, indent=2, sort_keys=True), encoding='utf-8')

    mace_train = comparison['mace_training']
    tace_train = comparison['tace_training']
    rows = [
        ('params', mace_train.get('num_parameters'), tace_train.get('num_parameters')),
        ('train final E MAE (meV/atom)', mace_train.get('final_mae_e_mev_atom'), last_tace_val.get('val/energy_per_atom_mae')),
        ('train final F MAE (meV/A)', mace_train.get('final_mae_f_mev_a'), last_tace_val.get('val/forces_mae')),
        ('train best E MAE (meV/atom)', mace_train.get('best_mae_e_mev_atom'), best_tace_energy),
        ('train best F MAE (meV/A)', mace_train.get('best_mae_f_mev_a'), best_tace_force),
        ('train peak FB memory (MB)', mace_train.get('max_fb_memory_mb'), tace_train.get('max_fb_memory_mb')),
        ('train seconds/update', mace_train.get('seconds_per_update'), (tace_train.get('sacct') or {}).get('elapsed_s') / 200000 if (tace_train.get('sacct') or {}).get('elapsed_s') else None),
        ('infer configs/s', mace_bench.get('configs_per_second'), tace_bench.get('configs_per_second')),
        ('infer atoms/s', mace_bench.get('atoms_per_second'), tace_bench.get('atoms_per_second')),
        ('infer PyTorch peak allocated MB', mace_bench.get('peak_allocated_mb'), tace_bench.get('peak_allocated_mb')),
        ('infer PyTorch peak reserved MB', mace_bench.get('peak_reserved_mb'), tace_bench.get('peak_reserved_mb')),
        ('infer E MAE (meV/atom)', mace_bench.get('mae_e_mev_atom'), tace_bench.get('mae_e_mev_atom')),
        ('infer F MAE (meV/A)', mace_bench.get('mae_f_mev_a'), tace_bench.get('mae_f_mev_a')),
    ]
    md = ['# OC20NEB TACE vs MACE comparison', '', '| metric | MACE | TACE |', '|---|---:|---:|']
    for name, mace, tace in rows:
        md.append(f'| {name} | {fmt(mace)} | {fmt(tace)} |')
    md.extend(['', '## Notes', '', '- Training metrics use the validation metrics reported by each training pipeline.', '- Inference metrics use `benchmark_models.py` on the same validation subset and batch size.', '- MACE training baseline is from the existing `matrix_summary.json`; TACE training metrics are parsed from the TACE log.', ''])
    out_md.write_text('\n'.join(md), encoding='utf-8')
    print(out_md)
    print(out_json)


if __name__ == '__main__':
    main()
