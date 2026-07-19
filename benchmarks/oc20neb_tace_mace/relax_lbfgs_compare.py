#!/usr/bin/env python3
"""
LBFGS relaxation head-to-head: TACE teacher vs rTECE scalar.

Reads one structure from OC20NEB, relaxes with each model via ASE LBFGS
(fmax=5e-2 eV/A, max 300 steps), and writes per-step logs for comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ase.io
import numpy as np
import torch
from ase.calculators.calculator import Calculator
from ase.optimize import LBFGS

# ── rTECE imports ──────────────────────────────────────────────────
from tace.interface.ase.rtece_calculator import atoms_to_rtece_graph
from tace.models.rtece_scalar import RTECEGraph, RTECEScalarConfig, RTECEScalarModel
from tace.models.rtece_workflow import load_checkpoint as load_rtece


# ══════════════════════════════════════════════════════════════════════
# rTECE ASE Calculator (minimal wrapper)
# ══════════════════════════════════════════════════════════════════════

class RTECECalculator(Calculator):
    implemented_properties = ["energy", "forces", "free_energy"]

    def __init__(
        self,
        model: RTECEScalarModel,
        config: RTECEScalarConfig,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        neighborlist_backend: str = "matscipy",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model = model
        self._config = config
        self._device = device
        self._dtype = dtype
        self._neighborlist_backend = str(neighborlist_backend)
        self._model.eval()

    def atoms_to_graph(self, atoms, cutoff: float, device: torch.device, dtype: torch.dtype) -> RTECEGraph:
        return atoms_to_rtece_graph(
            atoms,
            cutoff=float(cutoff),
            device=device,
            dtype=dtype,
            neighborlist_backend=self._neighborlist_backend,
        )

    def calculate(self, atoms=None, properties=None, system_changes=None):
        super().calculate(atoms, properties, system_changes)
        graph = self.atoms_to_graph(
            atoms,
            cutoff=self._config.cutoff,
            device=torch.device(self._device),
            dtype=self._dtype,
        )
        out = self._model(graph)
        energy = out["energy"].detach().cpu().item()
        forces = out["forces"].detach().cpu().numpy()
        self.results["energy"] = energy
        self.results["free_energy"] = energy
        self.results["forces"] = forces


# ══════════════════════════════════════════════════════════════════════
# Relaxation runner
# ══════════════════════════════════════════════════════════════════════

def relax_with_log(
    atoms,
    calculator: Calculator,
    label: str,
    log_path: Path,
    *,
    fmax: float = 5e-2,
    max_steps: int = 300,
    device: str = "cuda",
) -> dict:
    """Run LBFGS relaxation, writing per-step energy/force/timing to *log_path*."""

    atoms = atoms.copy()
    atoms.calc = calculator

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []
    steps_data: list[dict] = []

    t_start = time.perf_counter()

    def write_header():
        h = f"{'step':>5s}  {'E (eV)':>14s}  {'Fmax (eV/A)':>13s}  {'wall (s)':>10s}  {'atoms/s':>10s}"
        log_lines.append(h)
        log_lines.append("-" * len(h))

    def log_step():
        e = float(atoms.get_potential_energy())
        fmax = float((atoms.get_forces() ** 2).sum(axis=1).max() ** 0.5)
        wall = time.perf_counter() - t_start
        natoms = len(atoms)
        atom_s = natoms * (len(steps_data) + 1) / max(wall, 1e-6)
        line = f"{len(steps_data):5d}  {e:14.6f}  {fmax:13.6f}  {wall:10.2f}  {atom_s:10.0f}"
        log_lines.append(line)
        steps_data.append({"step": len(steps_data), "energy": e, "fmax": fmax, "wall_s": wall})

    write_header()

    optimizer = LBFGS(atoms, logfile=None)  # we handle logging ourselves
    optimizer.attach(log_step, interval=1)

    converged = optimizer.run(fmax=fmax, steps=max_steps)
    wall_total = time.perf_counter() - t_start

    e_final = float(atoms.get_potential_energy())
    f_final = atoms.get_forces()
    fmax_final = float((f_final ** 2).sum(axis=1).max() ** 0.5)
    natoms = len(atoms)

    log_lines.append("")
    log_lines.append(f"converged: {converged}")
    log_lines.append(f"steps: {len(steps_data)}")
    log_lines.append(f"wall_total_s: {wall_total:.2f}")
    log_lines.append(f"atoms: {natoms}")
    log_lines.append(f"final_energy_eV: {e_final:.8f}")
    log_lines.append(f"final_fmax_eV_per_A: {fmax_final:.8f}")
    log_lines.append(f"throughput_atom_step_per_s: {natoms * len(steps_data) / max(wall_total, 1e-6):.1f}")

    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    return {
        "label": label,
        "converged": converged,
        "steps": len(steps_data),
        "wall_total_s": wall_total,
        "atoms": natoms,
        "initial_energy_eV": steps_data[0]["energy"] if steps_data else None,
        "final_energy_eV": e_final,
        "final_fmax_eV_per_A": fmax_final,
        "throughput_atom_step_per_s": natoms * len(steps_data) / max(wall_total, 1e-6),
        "steps_data": steps_data,
    }, atoms  # return relaxed atoms for geometry comparison


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LBFGS relaxation: TACE teacher vs rTECE")
    p.add_argument("--teacher", type=Path, required=True, help="TACE teacher .ckpt")
    p.add_argument("--rtece", type=Path, required=True, help="rTECE scalar .pt checkpoint")
    p.add_argument("--rtece-fast", type=Path, default=None, help="optional second rTECE for speed comparison")
    p.add_argument("--configs", type=Path, required=True, help="OC20NEB extxyz file")
    p.add_argument("--config-index", type=int, default=0, help="which structure to relax")
    p.add_argument("--out-dir", type=Path, required=True, help="output directory for logs")
    p.add_argument("--fmax", type=float, default=5e-2, help="force convergence (eV/A)")
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--default-dtype", choices=("float32", "float64"), default="float32")
    p.add_argument("--skip-teacher", action="store_true", help="skip teacher relaxation")
    p.add_argument("--skip-rtece", action="store_true", help="skip rTECE relaxation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = torch.float64 if args.default_dtype == "float64" else torch.float32
    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU", file=sys.stderr)
        device_str = "cpu"

    # ── read structure ──────────────────────────────────────────────
    atoms_list = ase.io.read(str(args.configs), index=f"{args.config_index}")
    if isinstance(atoms_list, list):
        atoms = atoms_list[0]
    else:
        atoms = atoms_list

    natoms = len(atoms)
    formula = atoms.get_chemical_formula()
    print(f"structure: index={args.config_index}  formula={formula}  natoms={natoms}")
    print(f"pbc: {atoms.pbc}  cell: {atoms.cell.array.diagonal() if atoms.cell is not None else 'N/A'}")

    results: dict[str, dict] = {}
    relaxed_atoms: dict[str, ase.Atoms] = {}
    initial_atoms = atoms.copy()

    # ── TACE teacher ─────────────────────────────────────────────────
    if not args.skip_teacher:
        print("\n── Loading TACE teacher ──")
        import os
        os.environ.setdefault("TACE_USE_OEQ", "1")

        from tace.interface.ase import TACEAseCalc

        tace_calc = TACEAseCalc(
            str(args.teacher),
            dtype=args.default_dtype,
            device=device_str,
            neighborlist_backend="ase",
        )

        teacher_log = args.out_dir / "relax_teacher.log"
        print(f"  relaxing teacher → {teacher_log}")
        t0 = time.perf_counter()
        results["teacher"], relaxed_atoms["teacher"] = relax_with_log(
            atoms, tace_calc, "TACE_teacher", teacher_log,
            fmax=args.fmax, max_steps=args.max_steps, device=device_str,
        )
        print(f"  done in {time.perf_counter() - t0:.0f}s  "
              f"steps={results['teacher']['steps']}  "
              f"E_final={results['teacher']['final_energy_eV']:.4f}  "
              f"Fmax_final={results['teacher']['final_fmax_eV_per_A']:.4f}")

    # ── rTECE ────────────────────────────────────────────────────────
    if not args.skip_rtece:
        print("\n── Loading rTECE ──")
        rtece_model, rtece_config, rtece_meta = load_rtece(
            args.rtece,
            dtype=dtype,
            device=device_str,
        )
        variant = rtece_config.variant
        hidden = list(rtece_config.hidden_channels)
        nradial = rtece_config.num_radial
        nparams = sum(p.numel() for p in rtece_model.parameters())
        print(f"  variant={variant}  hidden={hidden}  radial={nradial}  params={nparams}")

        rtece_calc = RTECECalculator(
            rtece_model, rtece_config,
            device=device_str, dtype=dtype,
            neighborlist_backend="matscipy",
        )
        # warm-up pass
        _ = rtece_model(rtece_calc.atoms_to_graph(
            atoms,
            cutoff=rtece_config.cutoff,
            device=torch.device(device_str),
            dtype=dtype,
        ))
        if device_str == "cuda":
            torch.cuda.synchronize()

        rtece_log = args.out_dir / "relax_rtece.log"
        print(f"  relaxing rTECE → {rtece_log}")
        t0 = time.perf_counter()
        results["rtece"], relaxed_atoms["rtece"] = relax_with_log(
            atoms, rtece_calc, "rTECE", rtece_log,
            fmax=args.fmax, max_steps=args.max_steps, device=device_str,
        )
        print(f"  done in {time.perf_counter() - t0:.0f}s  "
              f"steps={results['rtece']['steps']}  "
              f"E_final={results['rtece']['final_energy_eV']:.4f}  "
              f"Fmax_final={results['rtece']['final_fmax_eV_per_A']:.4f}")

    # ── summary ──────────────────────────────────────────────────────
    print("\n═══ Comparison ═══")
    header = f"{'model':<20s}  {'steps':>5s}  {'wall (s)':>10s}  {'at·step/s':>12s}  {'E_final (eV)':>14s}  {'Fmax_final':>12s}"
    print(header)
    print("-" * len(header))
    for key in ("teacher", "rtece", "rtece_fast"):
        if key in results:
            r = results[key]
            print(f"{r['label']:<20s}  {r['steps']:5d}  {r['wall_total_s']:10.1f}  "
                  f"{r['throughput_atom_step_per_s']:12.0f}  {r['final_energy_eV']:14.6f}  "
                  f"{r['final_fmax_eV_per_A']:12.6f}")

    if "teacher" in results and "rtece" in results:
        speedup = results["teacher"]["wall_total_s"] / max(results["rtece"]["wall_total_s"], 1e-6)
        print(f"\nspeedup (rTECE vs teacher): {speedup:.1f}×")
        de = results["rtece"]["final_energy_eV"] - results["teacher"]["final_energy_eV"]
        print(f"ΔE (rTECE - teacher): {de:.4f} eV  ({de / natoms * 1000:.2f} meV/atom)")

    # ── rTECE fast ───────────────────────────────────────────────────
    if args.rtece_fast:
        print("\n── Loading rTECE fast ──")
        rtece_fast_model, rtece_fast_config, rtece_fast_meta = load_rtece(
            args.rtece_fast,
            dtype=dtype,
            device=device_str,
        )
        variant = rtece_fast_config.variant
        hidden = list(rtece_fast_config.hidden_channels)
        nradial = rtece_fast_config.num_radial
        nparams = sum(p.numel() for p in rtece_fast_model.parameters())
        print(f"  variant={variant}  hidden={hidden}  radial={nradial}  params={nparams}")

        rtece_fast_calc = RTECECalculator(
            rtece_fast_model, rtece_fast_config,
            device=device_str, dtype=dtype,
            neighborlist_backend="matscipy",
        )
        _ = rtece_fast_model(rtece_fast_calc.atoms_to_graph(
            atoms,
            cutoff=rtece_fast_config.cutoff,
            device=torch.device(device_str),
            dtype=dtype,
        ))
        if device_str == "cuda":
            torch.cuda.synchronize()

        rtece_fast_log = args.out_dir / "relax_rtece_fast.log"
        print(f"  relaxing rTECE-fast → {rtece_fast_log}")
        t0 = time.perf_counter()
        results["rtece_fast"], relaxed_atoms["rtece_fast"] = relax_with_log(
            atoms, rtece_fast_calc, "rTECE_fast", rtece_fast_log,
            fmax=args.fmax, max_steps=args.max_steps, device=device_str,
        )
        print(f"  done in {time.perf_counter() - t0:.0f}s  "
              f"steps={results['rtece_fast']['steps']}  "
              f"E_final={results['rtece_fast']['final_energy_eV']:.4f}  "
              f"Fmax_final={results['rtece_fast']['final_fmax_eV_per_A']:.4f}")

    # ── save relaxed structures ──────────────────────────────────────
    for key, label in (
        ("teacher", "TACE_teacher"),
        ("rtece", "rTECE_SOTA"),
        ("rtece_fast", "rTECE_fast"),
    ):
        if key in relaxed_atoms:
            xyz_path = args.out_dir / f"relaxed_{key}.extxyz"
            ase.io.write(str(xyz_path), relaxed_atoms[key], format="extxyz")
            print(f"relaxed structure → {xyz_path}")

    # ── save initial structure ───────────────────────────────────────
    init_path = args.out_dir / "initial.extxyz"
    ase.io.write(str(init_path), initial_atoms, format="extxyz")
    print(f"initial structure → {init_path}")

    # ── geometry comparison ──────────────────────────────────────────
    if "teacher" in relaxed_atoms and "rtece" in relaxed_atoms:
        t_pos = relaxed_atoms["teacher"].get_positions()
        r_pos = relaxed_atoms["rtece"].get_positions()
        # align via optimal rotation (undo any rigid translation/rotation)
        try:
            from ase.build import minimize_rotation_and_translation
            # compute displacement for each atom
            disp = np.linalg.norm(r_pos - t_pos, axis=1)
            rmsd = float(np.sqrt(np.mean(disp ** 2)))
            max_disp = float(np.max(disp))
            print(f"geometry RMSD (rTECE vs teacher): {rmsd:.4f} Å")
            print(f"max atom displacement: {max_disp:.4f} Å")
            # per-element breakdown
            symbols = relaxed_atoms["teacher"].get_chemical_symbols()
            for el in sorted(set(symbols)):
                mask = np.array([s == el for s in symbols])
                el_disp = disp[mask]
                print(f"  {el}: mean={el_disp.mean():.4f} Å  max={el_disp.max():.4f} Å")
        except Exception as exc:
            print(f"(geometry comparison skipped: {exc})")

    summary_path = args.out_dir / "relax_summary.json"
    # strip per-step data for the summary
    summary_results = {k: {sk: sv for sk, sv in v.items() if sk != "steps_data"} for k, v in results.items()}
    summary_results["args"] = {
        "teacher": str(args.teacher),
        "rtece": str(args.rtece),
        "configs": str(args.configs),
        "config_index": args.config_index,
        "formula": formula,
        "natoms": natoms,
        "fmax": args.fmax,
        "max_steps": args.max_steps,
    }
    def json_safe(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

    summary_path.write_text(json.dumps(summary_results, indent=2, sort_keys=True, default=json_safe) + "\n", encoding="utf-8")
    print(f"\nsummary → {summary_path}")


if __name__ == "__main__":
    main()
