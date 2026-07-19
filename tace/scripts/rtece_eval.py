from __future__ import annotations

import argparse
import copy
import time
import ase.io

from tace.interface.ase import RTECEAseCalc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict energies and forces with a scalar rTECE checkpoint.")
    parser.add_argument("-i", "--input", required=True, help="Path to an ASE-readable structure file")
    parser.add_argument("-o", "--output", default="rtece_predict.xyz", help="Path to an ASE-writeable output file")
    parser.add_argument("-m", "--model", required=True, help="Path to an rTECE scalar checkpoint")
    parser.add_argument("--index", default=":", help="ASE read index, default ':'")
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"), help="Tensor precision")
    parser.add_argument("--device", default=None, help="Inference device, default auto")
    parser.add_argument("--force-mode", default="autograd", help="rTECE force backend, e.g. autograd or analytic_pair")
    parser.add_argument(
        "--neighborlist-backend",
        default="matscipy",
        choices=("ase", "vesin", "matscipy"),
        help="Neighbor-list backend for ASE graph construction",
    )
    parser.add_argument("--energy-key", default="rTECE_energy", help="Output atoms.info key for predicted energy")
    parser.add_argument("--forces-key", default="rTECE_forces", help="Output atoms.arrays key for predicted forces")
    return parser.parse_args()


def _as_list(atoms_or_list):
    return atoms_or_list if isinstance(atoms_or_list, list) else [atoms_or_list]


def main() -> None:
    args = parse_args()
    atoms_list = _as_list(ase.io.read(args.input, index=args.index))
    output_atoms = [copy.deepcopy(atoms) for atoms in atoms_list]
    calc = RTECEAseCalc(
        args.model,
        dtype=args.dtype,
        device=args.device,
        force_mode=args.force_mode,
        neighborlist_backend=args.neighborlist_backend,
    )

    start = time.time()
    total_atoms = 0
    for atoms in output_atoms:
        atoms.calc = calc
        atoms.info[args.energy_key] = float(atoms.get_potential_energy())
        atoms.arrays[args.forces_key] = atoms.get_forces()
        total_atoms += len(atoms)
        atoms.calc = None

    ase.io.write(args.output, output_atoms, format="extxyz")
    elapsed = max(time.time() - start, 1.0e-12)
    print(f"Predicted {len(output_atoms)} structures / {total_atoms} atoms in {elapsed:.3f} s")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
