#!/usr/bin/env python3
"""Create a projection-constrained rTECE init-state checkpoint.

This is the first production training hook for Stage180-style renormalized
initialization.  The current artifact produced by analyze_rtece_projection_error
records path-level projection losses but not closed-form Schur coefficients, so
this initializer uses that artifact as an auditable path/config contract and then
runs a short deterministic E/F prefit inside the retained student model space.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tace.lightning.rtece import build_training_config, parse_scalar_path_ids
from tace.models.rtece_scalar import RTECEScalarModel
from tace.models.rtece_workflow import save_checkpoint, train_steps
from tace.scripts.rtece_train_scalar import fit_atomic_energies, load_samples, parse_force_focus_elements


def load_projection_payload(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != "rtece_projection_diagnostic.v1":
        raise ValueError(f"projection JSON {path} is not an rtece_projection_diagnostic.v1 artifact")
    if not isinstance(payload.get("rows"), list) or not payload["rows"]:
        raise ValueError(f"projection JSON {path} does not contain diagnostic rows")
    return payload


def select_projection_row(payload: dict[str, Any], candidate: str) -> dict[str, Any]:
    matches = [row for row in payload.get("rows", []) if row.get("candidate") == candidate]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one projection row for candidate {candidate!r}, found {len(matches)}")
    return dict(matches[0])


def validate_projection_contract(
    payload: dict[str, Any],
    row: dict[str, Any],
    *,
    scalar_path_ids: tuple[str, ...],
    num_radial: int,
    species_basis_channels: int,
    species_basis_mode: str,
    local_l0_chemistry_rank: int,
    atomic_cross_radial_sketch_channels: int,
) -> None:
    if tuple(row.get("candidate_scalar_path_ids") or ()) != tuple(scalar_path_ids):
        raise ValueError("projection candidate path ids do not match requested student scalar path ids")
    checks = {
        "num_radial": int(num_radial),
        "species_basis_channels": int(species_basis_channels),
        "species_basis_mode": str(species_basis_mode),
        "local_l0_chemistry_rank": int(local_l0_chemistry_rank),
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise ValueError(f"projection artifact {key}={payload.get(key)!r} does not match requested {expected!r}")
    candidate_channels = int(payload.get("candidate_atomic_cross_radial_sketch_channels", payload.get("atomic_cross_radial_sketch_channels")))
    if candidate_channels != int(atomic_cross_radial_sketch_channels):
        raise ValueError(
            "projection candidate atomic_cross_radial_sketch_channels="
            f"{candidate_channels} does not match requested {atomic_cross_radial_sketch_channels}"
        )


def projection_metadata(path: str | Path, payload: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    compact_keys = [
        "candidate",
        "candidate_dim",
        "reference_dim",
        "relative_residual",
        "relative_energy_residual",
        "relative_force_residual",
        "energy_per_atom_rmse",
        "energy_per_atom_max_abs",
        "force_rmse",
        "force_max_abs",
        "energy_beats_intercept_baseline",
        "energy_underdetermined",
        "force_underdetermined",
    ]
    return {
        "projection_json": str(path),
        "projection_schema_version": payload.get("schema_version"),
        "projection_configs": payload.get("configs"),
        "projection_limit_configs": payload.get("limit_configs"),
        "reference_path_ids": list(payload.get("reference_path_ids") or []),
        "candidate_row": {key: row.get(key) for key in compact_keys if key in row},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection-json", type=Path, required=True)
    parser.add_argument("--candidate", default="scratch_same_student")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--scalar-path-ids", required=True)
    parser.add_argument("--limit-configs", type=int, default=256)
    parser.add_argument("--init-steps", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--force-weight", type=float, default=10.0)
    parser.add_argument("--force-focus-elements", default=None)
    parser.add_argument("--force-focus-weight", type=float, default=1.0)
    parser.add_argument("--hidden-channels", default="64,64")
    parser.add_argument("--num-radial", type=int, default=8)
    parser.add_argument("--moment-l-max", type=int, choices=(0, 1, 2), default=None)
    parser.add_argument("--species-basis-channels", type=int, default=0)
    parser.add_argument("--species-basis-mode", choices=("fixed_z_power", "learnable_embedding"), default="fixed_z_power")
    parser.add_argument("--local-l0-chemistry-rank", type=int, default=0)
    parser.add_argument("--atomic-cross-radial-sketch-channels", type=int, default=2)
    parser.add_argument("--atomic-cross-radial-projection", choices=("fixed_shell_mean", "learnable", "pod_fixed"), default="fixed_shell_mean")
    parser.add_argument("--learnable-radial-mixing", action="store_true")
    parser.add_argument("--no-fit-energy-shift", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--default-dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--neighborlist-backend", choices=("ase", "vesin", "matscipy"), default="matscipy")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))
    dtype = torch.float64 if args.default_dtype == "float64" else torch.float32
    requested = torch.device(args.device)
    device = requested if requested.type == "cpu" or torch.cuda.is_available() else torch.device("cpu")
    scalar_path_ids = parse_scalar_path_ids(args.scalar_path_ids)
    if scalar_path_ids is None:
        raise ValueError("--scalar-path-ids must contain at least one path id")

    projection_payload = load_projection_payload(args.projection_json)
    projection_row = select_projection_row(projection_payload, str(args.candidate))
    validate_projection_contract(
        projection_payload,
        projection_row,
        scalar_path_ids=scalar_path_ids,
        num_radial=args.num_radial,
        species_basis_channels=args.species_basis_channels,
        species_basis_mode=args.species_basis_mode,
        local_l0_chemistry_rank=args.local_l0_chemistry_rank,
        atomic_cross_radial_sketch_channels=args.atomic_cross_radial_sketch_channels,
    )

    config = build_training_config(
        variant=args.variant,
        scalar_path_ids=scalar_path_ids,
        hidden_channels=args.hidden_channels,
        num_radial=args.num_radial,
        moment_l_max=args.moment_l_max,
        species_basis_channels=args.species_basis_channels,
        species_basis_mode=args.species_basis_mode,
        local_l0_chemistry_rank=args.local_l0_chemistry_rank,
        atomic_cross_radial_sketch_channels=args.atomic_cross_radial_sketch_channels,
        atomic_cross_radial_projection=args.atomic_cross_radial_projection,
        learnable_radial_mixing=args.learnable_radial_mixing,
    )
    samples = load_samples(
        args.train_file,
        cutoff=config.cutoff,
        device=device,
        dtype=dtype,
        limit_configs=args.limit_configs,
        neighborlist_backend=args.neighborlist_backend,
    )
    if not args.no_fit_energy_shift:
        config = replace(config, atomic_energies=fit_atomic_energies(samples))
        samples = load_samples(
            args.train_file,
            cutoff=config.cutoff,
            device=device,
            dtype=dtype,
            limit_configs=args.limit_configs,
            neighborlist_backend=args.neighborlist_backend,
        )
    model = RTECEScalarModel(config).to(device=device, dtype=dtype)
    force_focus_atomic_numbers = parse_force_focus_elements(args.force_focus_elements)
    init_summary = train_steps(
        model,
        samples,
        max_steps=int(args.init_steps),
        lr=float(args.lr),
        energy_weight=float(args.energy_weight),
        force_weight=float(args.force_weight),
        force_focus_atomic_numbers=force_focus_atomic_numbers,
        force_focus_weight=float(args.force_focus_weight),
    )
    metadata = {
        "init_method": "projection_constrained_energy_force_prefit.v1",
        "init_role": "stage180_renorm_initialized_same_student",
        "init_steps": int(args.init_steps),
        "init_lr": float(args.lr),
        "init_energy_weight": float(args.energy_weight),
        "init_force_weight": float(args.force_weight),
        "init_force_focus_atomic_numbers": list(force_focus_atomic_numbers),
        "init_force_focus_weight": float(args.force_focus_weight),
        "init_train_file": str(args.train_file),
        "init_limit_configs": int(args.limit_configs),
        "init_summary": init_summary,
        "projection": projection_metadata(args.projection_json, projection_payload, projection_row),
    }
    save_checkpoint(args.output, model.to("cpu"), config, metadata=metadata)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": str(args.output), "summary": str(summary_path), **metadata}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
