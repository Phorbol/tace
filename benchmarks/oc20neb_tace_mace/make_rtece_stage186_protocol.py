from __future__ import annotations

import argparse
from pathlib import Path

from tace.models.rtece_protocol import (
    build_stage186_operator_manifest,
    write_operator_manifest,
)
from tace.models.rtece_scalar import build_rtece_config_from_path_ids


def _parse_path_ids(value: str) -> tuple[str, ...]:
    path_ids = tuple(part.strip() for part in value.split(",") if part.strip())
    if not path_ids:
        raise argparse.ArgumentTypeError("path id list must not be empty")
    return path_ids


def _parse_retained_group(value: str) -> tuple[str, tuple[str, ...]]:
    name, separator, raw_path_ids = value.partition(":")
    if not separator or not name.strip():
        raise argparse.ArgumentTypeError("retained group must be NAME:path_id,path_id")
    return name.strip(), _parse_path_ids(raw_path_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--reference-path-ids", type=_parse_path_ids, required=True)
    parser.add_argument("--retained-group", action="append", type=_parse_retained_group, required=True)
    parser.add_argument("--implementation-revision", required=True)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--num-radial", type=int, default=8)
    parser.add_argument("--species-basis-channels", type=int, default=0)
    parser.add_argument(
        "--species-basis-mode",
        choices=("fixed_z_power", "learnable_embedding"),
        default="fixed_z_power",
    )
    parser.add_argument("--local-l0-chemistry-rank", type=int, default=0)
    parser.add_argument("--atomic-cross-radial-sketch-channels", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retained_groups = dict(args.retained_group)
    if len(retained_groups) != len(args.retained_group):
        raise ValueError("retained group names must be unique")
    config = build_rtece_config_from_path_ids(
        "rtece_stage186_operator_reference",
        args.reference_path_ids,
        cutoff=float(args.cutoff),
        num_radial=int(args.num_radial),
        species_basis_channels=int(args.species_basis_channels),
        species_basis_mode=str(args.species_basis_mode),
        local_l0_chemistry_rank=int(args.local_l0_chemistry_rank),
        atomic_cross_radial_sketch_channels=int(
            args.atomic_cross_radial_sketch_channels
        ),
    )
    manifest = build_stage186_operator_manifest(
        config,
        retained_groups=retained_groups,
        implementation_revision=str(args.implementation_revision),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    write_operator_manifest(args.output_json, manifest)
    print(args.output_json)


if __name__ == "__main__":
    main()
