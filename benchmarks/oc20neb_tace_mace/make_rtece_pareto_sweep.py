#!/usr/bin/env python3
"""Generate rTECE Pareto-sweep Slurm wrappers from TECE design-space axes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.oc20neb_tace_mace.submit_rtece_scalar_matrix import write_rtece_matrix_wrapper


def _row(
    name: str,
    *,
    scalar_path_ids: str | None,
    moment_l_max: int,
    learnable_radial_mixing: bool,
    short_range_repulsion_potential: str = "softplus_overlap",
    tece_axes: tuple[str, ...] = (),
    hidden_channels: str = "16,16",
    num_radial: int = 8,
    species_basis_channels: int = 0,
    species_basis_mode: str = "fixed_z_power",
    atomic_cross_radial_sketch_channels: int = 2,
    atomic_cross_radial_projection: str = "fixed_shell_mean",
    radial_edge_sketch_channels: int = 0,
    descriptor_conditioner: str = "none",
    descriptor_conditioner_hidden_channels: int = 0,
    descriptor_bottleneck_dim: int = 0,
    train_variant: str | None = None,
) -> dict[str, Any]:
    use_short_range = short_range_repulsion_potential != "softplus_overlap"
    axes = [
        "angular_bandwidth_l_max",
        "scalar_path_density",
        "radial_rank",
        "scalar_head_capacity",
    ]
    if learnable_radial_mixing:
        axes.append("trainable_feature_extractor")
    if species_basis_channels and species_basis_mode == "learnable_embedding":
        axes.append("trainable_species_basis")
    if use_short_range:
        axes.append("short_range_physical_prior")
    if descriptor_bottleneck_dim:
        axes.append("descriptor_bottleneck")
    axes.extend(axis for axis in tece_axes if axis not in axes)
    return {
        "name": name,
        "variant": name,
        "train_variant": train_variant or name,
        "scalar_path_ids": scalar_path_ids,
        "moment_l_max": int(moment_l_max),
        "learnable_radial_mixing": bool(learnable_radial_mixing),
        "short_range_repulsion_potential": short_range_repulsion_potential,
        "use_short_range_repulsion": bool(use_short_range),
        "hidden_channels": hidden_channels,
        "num_radial": int(num_radial),
        "species_basis_channels": int(species_basis_channels),
        "species_basis_mode": str(species_basis_mode),
        "atomic_cross_radial_sketch_channels": int(atomic_cross_radial_sketch_channels),
        "atomic_cross_radial_projection": str(atomic_cross_radial_projection),
        "radial_edge_sketch_channels": int(radial_edge_sketch_channels),
        "descriptor_conditioner": str(descriptor_conditioner),
        "descriptor_conditioner_hidden_channels": int(descriptor_conditioner_hidden_channels),
        "descriptor_bottleneck_dim": int(descriptor_bottleneck_dim),
        "tece_axes": axes,
    }


def _parameter_estimates(row: dict[str, Any]) -> dict[str, int]:
    from tace.models.rtece_scalar import (
        RTECEScalarModel,
        build_rtece_config,
        build_rtece_config_from_path_ids,
        config_with_moment_l_max,
    )

    hidden_channels = tuple(int(part.strip()) for part in str(row.get("hidden_channels", "64,64")).split(",") if part.strip())
    scalar_path_ids = row.get("scalar_path_ids")
    if scalar_path_ids is not None:
        config = build_rtece_config_from_path_ids(
            str(row.get("variant") or row["name"]),
            tuple(part.strip() for part in str(scalar_path_ids).split(",") if part.strip()),
            hidden_channels=hidden_channels,
            num_radial=int(row.get("num_radial", 8)),
            moment_l_max=int(row.get("moment_l_max", 2)),
            learnable_radial_mixing=bool(row.get("learnable_radial_mixing", False)),
            species_basis_channels=int(row.get("species_basis_channels", 0)),
            species_basis_mode=str(row.get("species_basis_mode", "fixed_z_power")),
            atomic_cross_radial_sketch_channels=int(row.get("atomic_cross_radial_sketch_channels", 2)),
            atomic_cross_radial_projection=str(row.get("atomic_cross_radial_projection", "fixed_shell_mean")),
            descriptor_conditioner=str(row.get("descriptor_conditioner", "none")),
            descriptor_conditioner_hidden_channels=int(row.get("descriptor_conditioner_hidden_channels", 0)),
            descriptor_bottleneck_dim=int(row.get("descriptor_bottleneck_dim", 0)),
            use_short_range_repulsion=bool(row.get("use_short_range_repulsion", False)),
            short_range_repulsion_potential=str(row.get("short_range_repulsion_potential", "softplus_overlap")),
        )
    else:
        config = replace(
            build_rtece_config(str(row.get("train_variant") or row.get("variant") or row["name"])),
            hidden_channels=hidden_channels,
            num_radial=int(row.get("num_radial", 8)),
            learnable_radial_mixing=bool(row.get("learnable_radial_mixing", False)),
            radial_edge_sketch_channels=int(row.get("radial_edge_sketch_channels", 0)),
            descriptor_bottleneck_dim=int(row.get("descriptor_bottleneck_dim", 0)),
            use_short_range_repulsion=bool(row.get("use_short_range_repulsion", False)),
            short_range_repulsion_potential=str(row.get("short_range_repulsion_potential", "softplus_overlap")),
        )
        config = config_with_moment_l_max(config, int(row.get("moment_l_max", 2)))
    model = RTECEScalarModel(config)
    representation_prefixes = (
        "radial_mixing",
        "atomic_cross_radial_projection",
        "species_basis_embedding",
        "descriptor_conditioner",
        "descriptor_bottleneck",
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    representation = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith(representation_prefixes)
    )
    readout = sum(parameter.numel() for name, parameter in model.named_parameters() if name.startswith("energy_head"))
    return {
        "num_parameters_estimate": int(total),
        "representation_parameters_estimate": int(representation),
        "readout_parameters_estimate": int(readout),
    }


def _with_parameter_estimates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    estimated = []
    for row in rows:
        item = dict(row)
        item.update(_parameter_estimates(item))
        estimated.append(item)
    return estimated


def default_pareto_rows() -> list[dict[str, Any]]:
    return [
        _row(
            "l0_scalar_fixed",
            scalar_path_ids="atomic.radial_density",
            moment_l_max=0,
            learnable_radial_mixing=False,
        ),
        _row(
            "l0_scalar_learnable_zbl",
            scalar_path_ids="atomic.radial_density",
            moment_l_max=0,
            learnable_radial_mixing=True,
            short_range_repulsion_potential="zbl",
        ),
        _row(
            "l1_vector_learnable_zbl",
            scalar_path_ids="atomic.radial_density,atomic.vector_norm",
            moment_l_max=1,
            learnable_radial_mixing=True,
            short_range_repulsion_potential="zbl",
        ),
        _row(
            "l2_atomic_learnable_zbl",
            scalar_path_ids="atomic.radial_density,atomic.vector_norm,atomic.quadrupole_norm",
            moment_l_max=2,
            learnable_radial_mixing=True,
            short_range_repulsion_potential="zbl",
        ),
        _row(
            "l1_vector_cross_radial_learnable_zbl",
            scalar_path_ids="atomic.radial_density,atomic.vector_norm,atomic.vector_cross_radial_dot",
            moment_l_max=1,
            learnable_radial_mixing=True,
            short_range_repulsion_potential="zbl",
            tece_axes=("cross_radial_invariants",),
        ),
        _row(
            "l2_atomic_cross_radial_learnable_zbl",
            scalar_path_ids=(
                "atomic.radial_density,atomic.vector_norm,atomic.vector_cross_radial_dot,"
                "atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius"
            ),
            moment_l_max=2,
            learnable_radial_mixing=True,
            short_range_repulsion_potential="zbl",
            tece_axes=("cross_radial_invariants",),
        ),
        _row(
            "l1_vector_cross_radial_k3_learnable_zbl",
            scalar_path_ids="atomic.radial_density,atomic.vector_norm,atomic.vector_cross_radial_dot",
            moment_l_max=1,
            learnable_radial_mixing=True,
            short_range_repulsion_potential="zbl",
            tece_axes=("cross_radial_invariants",),
            atomic_cross_radial_sketch_channels=3,
        ),
        _row(
            "l2_atomic_cross_radial_k3_learnable_zbl",
            scalar_path_ids=(
                "atomic.radial_density,atomic.vector_norm,atomic.vector_cross_radial_dot,"
                "atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius"
            ),
            moment_l_max=2,
            learnable_radial_mixing=True,
            short_range_repulsion_potential="zbl",
            tece_axes=("cross_radial_invariants",),
            atomic_cross_radial_sketch_channels=3,
        ),
        _row(
            "l1_cavity_vector_learnable_zbl",
            scalar_path_ids="atomic.radial_density,edge.cavity.vector_dot,edge.direct.radial",
            moment_l_max=1,
            learnable_radial_mixing=True,
            short_range_repulsion_potential="zbl",
            tece_axes=("edge_relational_scalar_sketches",),
        ),
        _row(
            "l2_cavity_vector_quad_learnable_zbl",
            scalar_path_ids="atomic.radial_density,edge.cavity.vector_dot,edge.cavity.quadrupole_frobenius,edge.direct.radial",
            moment_l_max=2,
            learnable_radial_mixing=True,
            short_range_repulsion_potential="zbl",
            tece_axes=("edge_relational_scalar_sketches",),
        ),
    ]


def stage115_ef_active_rows() -> list[dict[str, Any]]:
    rows = [
        _row(
            "l0_radial",
            scalar_path_ids="atomic.radial_density",
            moment_l_max=0,
            learnable_radial_mixing=True,
            short_range_repulsion_potential="zbl",
            tece_axes=("stage114_ef_active_selection", "throughput_endpoint"),
        ),
        _row(
            "l1_cross_k2",
            scalar_path_ids="atomic.radial_density,atomic.vector_norm,atomic.vector_cross_radial_dot",
            moment_l_max=1,
            learnable_radial_mixing=True,
            short_range_repulsion_potential="zbl",
            tece_axes=("cross_radial_invariants", "stage114_ef_active_selection"),
        ),
        _row(
            "l2_atomic_no_edge_k2",
            scalar_path_ids=(
                "atomic.radial_density,atomic.vector_norm,atomic.vector_cross_radial_dot,"
                "atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius"
            ),
            moment_l_max=2,
            learnable_radial_mixing=True,
            short_range_repulsion_potential="zbl",
            tece_axes=(
                "cross_radial_invariants",
                "stage114_ef_active_selection",
                "atomic_l2_scalar_paths",
            ),
        ),
    ]
    for row in rows:
        row["stage_basis"] = "stage114_ef_active_rank"
        row["stage114_source"] = (
            "runs/oc20neb_tace_mace/rtece-stage114-ef-active-rank/stage114_notes.md"
        )
    return rows


def stage116_capacity_ladder_rows() -> list[dict[str, Any]]:
    l1_paths = "atomic.radial_density,atomic.vector_norm,atomic.vector_cross_radial_dot"
    l2_paths = (
        "atomic.radial_density,atomic.vector_norm,atomic.vector_cross_radial_dot,"
        "atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius"
    )
    specs = [
        ("l1_cross_k2_h64", l1_paths, 1, "64,64"),
        ("l1_cross_k2_h128", l1_paths, 1, "128,128"),
        ("l1_cross_k2_h128x3", l1_paths, 1, "128,128,128"),
        ("l2_atomic_no_edge_k2_h64", l2_paths, 2, "64,64"),
        ("l2_atomic_no_edge_k2_h128", l2_paths, 2, "128,128"),
        ("l2_atomic_no_edge_k2_h128x3", l2_paths, 2, "128,128,128"),
    ]
    rows = [
        _row(
            name,
            scalar_path_ids=paths,
            moment_l_max=moment_l_max,
            learnable_radial_mixing=True,
            short_range_repulsion_potential="zbl",
            tece_axes=(
                "cross_radial_invariants",
                "stage114_ef_active_selection",
                "scalar_head_capacity_ladder",
            ),
            hidden_channels=hidden_channels,
        )
        for name, paths, moment_l_max, hidden_channels in specs
    ]
    for row in rows:
        row["stage_basis"] = "stage116_capacity_ladder"
        row["stage115_source"] = (
            "runs/oc20neb_tace_mace/rtece-stage115-ef-active-wrappers/stage115_notes.md"
        )
        row["stage116_design_source"] = (
            "runs/oc20neb_tace_mace/rtece-stage116-capacity-ladder-design/stage116_notes.md"
        )
    return rows




def stage118_representation_ladder_rows() -> list[dict[str, Any]]:
    atomic_l2_paths = (
        "atomic.radial_density,atomic.vector_norm,atomic.vector_cross_radial_dot,"
        "atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius"
    )
    cavity_l2_paths = (
        "atomic.radial_density,edge.cavity.vector_dot,edge.cavity.quadrupole_frobenius,"
        "edge.direct.radial"
    )
    species_cavity_l2_paths = (
        "atomic.radial_density,atomic.species_basis_density,edge.cavity.vector_dot,"
        "edge.cavity.quadrupole_frobenius,edge.direct.radial"
    )
    rows = [
        _row(
            "l2_atomic_cross_h128",
            scalar_path_ids=atomic_l2_paths,
            moment_l_max=2,
            learnable_radial_mixing=True,
            short_range_repulsion_potential="zbl",
            tece_axes=(
                "representation_ladder",
                "cross_radial_invariants",
                "stage116_fixed_head_capacity",
            ),
            hidden_channels="128,128",
            atomic_cross_radial_sketch_channels=3,
            atomic_cross_radial_projection="learnable",
        ),
        _row(
            "l2_cavity_edge_h128",
            scalar_path_ids=cavity_l2_paths,
            moment_l_max=2,
            learnable_radial_mixing=True,
            short_range_repulsion_potential="zbl",
            tece_axes=(
                "representation_ladder",
                "cavity_edge_relational_scalar_sketches",
                "direct_edge_radial_path",
                "stage116_fixed_head_capacity",
            ),
            hidden_channels="128,128",
        ),
        _row(
            "l2_cavity_radial_edge_h128",
            scalar_path_ids=None,
            moment_l_max=2,
            learnable_radial_mixing=True,
            short_range_repulsion_potential="zbl",
            tece_axes=(
                "representation_ladder",
                "cavity_edge_relational_scalar_sketches",
                "low_rank_radial_edge_moment_sketches",
                "cross_radial_edge_invariants",
                "stage116_fixed_head_capacity",
            ),
            hidden_channels="128,128",
            radial_edge_sketch_channels=2,
            train_variant="rtece_cavity_radial_edge_sketch14",
        ),
        _row(
            "l2_species_cavity_edge_h128",
            scalar_path_ids=species_cavity_l2_paths,
            moment_l_max=2,
            learnable_radial_mixing=True,
            short_range_repulsion_potential="zbl",
            tece_axes=(
                "representation_ladder",
                "low_rank_neighbor_species_basis",
                "cavity_edge_relational_scalar_sketches",
                "direct_edge_radial_path",
                "stage116_fixed_head_capacity",
            ),
            hidden_channels="128,128",
            species_basis_channels=4,
        ),
        _row(
            "l2_conditioned_cavity_edge_h128",
            scalar_path_ids=cavity_l2_paths,
            moment_l_max=2,
            learnable_radial_mixing=True,
            short_range_repulsion_potential="zbl",
            tece_axes=(
                "representation_ladder",
                "cavity_edge_relational_scalar_sketches",
                "trainable_scalar_descriptor_conditioner",
                "stage116_fixed_head_capacity",
            ),
            hidden_channels="128,128",
            descriptor_conditioner="residual_mlp",
            descriptor_conditioner_hidden_channels=64,
        ),
        _row(
            "l2_cavity_edge_h128x3",
            scalar_path_ids=cavity_l2_paths,
            moment_l_max=2,
            learnable_radial_mixing=True,
            short_range_repulsion_potential="zbl",
            tece_axes=(
                "representation_ladder",
                "cavity_edge_relational_scalar_sketches",
                "scalar_head_capacity_ladder",
            ),
            hidden_channels="128,128,128",
        ),
    ]
    for row in rows:
        row["stage_basis"] = "stage118_representation_ladder"
        row["stage116_design_source"] = (
            "runs/oc20neb_tace_mace/rtece-stage116-capacity-ladder-design/stage116_notes.md"
        )
        row["stage118_design_source"] = (
            "runs/oc20neb_tace_mace/rtece-stage118-representation-ladder-design/stage118_notes.md"
        )
    return rows


def stage119_frontloaded_representation_rows() -> list[dict[str, Any]]:
    species_paths = "atomic.radial_density,atomic.species_basis_density"
    l1_species_cross_paths = (
        "atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,"
        "atomic.vector_cross_radial_dot"
    )
    l2_species_cross_paths = (
        "atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,"
        "atomic.vector_cross_radial_dot,atomic.quadrupole_norm,"
        "atomic.quadrupole_cross_radial_frobenius"
    )
    species_cavity_paths = (
        "atomic.radial_density,atomic.species_basis_density,edge.cavity.vector_dot,"
        "edge.cavity.quadrupole_frobenius,edge.direct.radial"
    )
    species_cavity_atomic_paths = (
        "atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,"
        "atomic.vector_cross_radial_dot,atomic.quadrupole_norm,"
        "atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot,"
        "edge.cavity.quadrupole_frobenius,edge.direct.radial"
    )
    specs = [
        ("l0_species8_learnembed_h64", species_paths, 0, 8, ("low_rank_neighbor_species_basis",)),
        ("l1_species16_cross_learnembed_h64", l1_species_cross_paths, 1, 16, ("cross_radial_invariants", "trainable_cross_radial_projection")),
        ("l2_species16_atomic_cross_learnembed_h64", l2_species_cross_paths, 2, 16, ("cross_radial_invariants", "trainable_cross_radial_projection", "atomic_l2_scalar_paths")),
        ("l2_species16_cavity_edge_learnembed_h64", species_cavity_paths, 2, 16, ("cavity_edge_relational_scalar_sketches", "direct_edge_radial_path")),
        ("l2_species32_cavity_edge_learnembed_h64", species_cavity_paths, 2, 32, ("cavity_edge_relational_scalar_sketches", "direct_edge_radial_path")),
        ("l2_species32_cavity_atomic_cross_learnembed_h64", species_cavity_atomic_paths, 2, 32, ("cross_radial_invariants", "trainable_cross_radial_projection", "cavity_edge_relational_scalar_sketches", "direct_edge_radial_path", "atomic_l2_scalar_paths")),
    ]
    rows = []
    for name, paths, moment_l_max, species_channels, axes in specs:
        row = _row(
            name,
            scalar_path_ids=paths,
            moment_l_max=moment_l_max,
            learnable_radial_mixing=True,
            short_range_repulsion_potential="zbl",
            tece_axes=("frontloaded_representation_capacity", *axes),
            hidden_channels="64,64",
            species_basis_channels=species_channels,
            species_basis_mode="learnable_embedding",
            atomic_cross_radial_sketch_channels=3,
            atomic_cross_radial_projection="learnable" if "cross_radial" in paths else "fixed_shell_mean",
        )
        row["stage_basis"] = "stage119_frontloaded_representation_ladder"
        row["capacity_allocation"] = "frontloaded_representation_not_readout"
        row["review_basis"] = "rTECE_review fixed-feature/species-collision critique plus TECE_design_space T3 scalar-sketch axes"
        row["stage119_design_source"] = "runs/oc20neb_tace_mace/rtece-stage119-frontloaded-representation-design/stage119_notes.md"
        rows.append(row)
    return _with_parameter_estimates(rows)



def stage120_descriptor_bottleneck_rows() -> list[dict[str, Any]]:
    stage119 = {row["name"]: row for row in stage119_frontloaded_representation_rows()}
    specs = [
        ("l0_species8_bneck16_h64", "l0_species8_learnembed_h64", 16),
        ("l0_species8_bneck32_h64", "l0_species8_learnembed_h64", 32),
        ("l2_species32_cavity_atomic_bneck16_h64", "l2_species32_cavity_atomic_cross_learnembed_h64", 16),
        ("l2_species32_cavity_atomic_bneck32_h64", "l2_species32_cavity_atomic_cross_learnembed_h64", 32),
    ]
    rows = []
    for name, source_name, bottleneck_dim in specs:
        base = dict(stage119[source_name])
        base["name"] = name
        base["variant"] = name
        base["train_variant"] = name
        base["descriptor_bottleneck_dim"] = int(bottleneck_dim)
        base["stage_basis"] = "stage120_descriptor_bottleneck_ladder"
        base["capacity_allocation"] = "front_bottleneck_path_mixer_not_wider_head"
        base["stage120_design_source"] = "runs/oc20neb_tace_mace/rtece-stage119-frontloaded-representation-design/stage119_results.md"
        base["tece_axes"] = list(dict.fromkeys([*base.get("tece_axes", []), "descriptor_bottleneck", "front_low_rank_path_mixer"]))
        rows.append(base)
    return _with_parameter_estimates(rows)



def stage121_active_frontloaded_rows() -> list[dict[str, Any]]:
    l1_paths = (
        "atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,"
        "atomic.vector_cross_radial_dot"
    )
    l2_paths = (
        "atomic.radial_density,atomic.species_basis_density,atomic.vector_norm,"
        "atomic.vector_cross_radial_dot,atomic.quadrupole_norm,"
        "atomic.quadrupole_cross_radial_frobenius"
    )
    specs = [
        ("l1_active_species16_bneck16_h64", l1_paths, 1, 16),
        ("l1_active_species16_bneck32_h64", l1_paths, 1, 32),
        ("l2_active_species16_bneck16_h64", l2_paths, 2, 16),
        ("l2_active_species16_bneck32_h64", l2_paths, 2, 32),
    ]
    rows = []
    for name, paths, moment_l_max, bottleneck_dim in specs:
        row = _row(
            name,
            scalar_path_ids=paths,
            moment_l_max=moment_l_max,
            learnable_radial_mixing=True,
            short_range_repulsion_potential="zbl",
            tece_axes=(
                "stage114_ef_active_selection",
                "frontloaded_representation_capacity",
                "low_rank_neighbor_species_basis",
                "cross_radial_invariants",
                "trainable_cross_radial_projection",
                "front_low_rank_path_mixer",
            ) + (("atomic_l2_scalar_paths",) if moment_l_max >= 2 else ()),
            hidden_channels="64,64",
            species_basis_channels=16,
            species_basis_mode="learnable_embedding",
            atomic_cross_radial_sketch_channels=3,
            atomic_cross_radial_projection="learnable",
            descriptor_bottleneck_dim=bottleneck_dim,
        )
        row["stage_basis"] = "stage121_active_frontloaded_representation"
        row["capacity_allocation"] = "active_set_frontloaded_representation_not_wider_head"
        row["stage114_source"] = "runs/oc20neb_tace_mace/rtece-stage114-ef-active-rank/stage114_notes.md"
        row["stage119_source"] = "runs/oc20neb_tace_mace/rtece-stage119-frontloaded-representation-design/stage119_results.md"
        row["stage120_source"] = "runs/oc20neb_tace_mace/rtece-stage120-descriptor-bottleneck-wrappers/rtece_pareto_sweep_index.json"
        row["review_basis"] = (
            "Stage114 E/F active atomic paths plus rTECE_review species-collision and fixed-feature critiques; "
            "capacity is added in trainable radial/species/cross/bottleneck representation layers, not by widening the final head."
        )
        rows.append(row)
    return _with_parameter_estimates(rows)

def preflight_extxyz_file(path: str | Path, *, limit_configs: int | None = None) -> dict[str, int | str]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"extxyz preflight file not found: {source}")
    import ase.io

    frames = 0
    atoms_total = 0
    index = ":" if limit_configs is None else f":{int(limit_configs)}"
    try:
        for atoms in ase.io.iread(str(source), index=index, format="extxyz"):
            frames += 1
            atoms_total += len(atoms)
    except Exception as exc:  # noqa: BLE001 - convert ASE parser detail into workflow context.
        raise ValueError(
            f"extxyz preflight failed for {source}: readable frames={frames}, "
            f"readable atoms={atoms_total}; parser error: {exc}"
        ) from exc
    if limit_configs is not None and frames < int(limit_configs):
        raise ValueError(
            f"extxyz preflight failed for {source}: requested frames={int(limit_configs)}, "
            f"readable frames={frames}, readable atoms={atoms_total}"
        )
    return {"path": str(source), "frames": frames, "atoms": atoms_total}


def _preflight_sweep_inputs(
    *,
    train_file: str,
    train_valid_file: str,
    dft_valid_file: str,
    teacher_valid_file: str,
    limit_configs: int,
    valid_limit_configs: int,
    bench_limit_configs: int,
) -> dict[str, dict[str, int | str]]:
    return {
        "train_file": preflight_extxyz_file(train_file, limit_configs=limit_configs),
        "train_valid_file": preflight_extxyz_file(train_valid_file, limit_configs=valid_limit_configs),
        "dft_valid_file": preflight_extxyz_file(dft_valid_file, limit_configs=bench_limit_configs),
        "teacher_valid_file": preflight_extxyz_file(teacher_valid_file, limit_configs=bench_limit_configs),
    }


def write_pareto_sweep(
    output_dir: str | Path,
    *,
    run_root: str,
    train_file: str,
    train_valid_file: str,
    dft_valid_file: str,
    teacher_valid_file: str,
    rows: list[dict[str, Any]] | None = None,
    row_set: str | None = None,
    limit_configs: int = 2048,
    valid_limit_configs: int = 256,
    bench_limit_configs: int = 1024,
    max_steps: int = 20000,
    lr: str | float = "1e-3",
    force_weight: float = 10.0,
    force_mode: str = "autograd",
    measure_passes: int = 5,
    default_dtype: str = "float32",
    trainer_backend: str = "lightning",
    batch_size: int = 8,
    valid_batch_size: int = 16,
    lr_scheduler: str = "plateau",
    lr_patience: int = 100,
    lr_factor: float = 0.5,
    lr_warmup_steps: int = 500,
    early_stopping_patience: int | None = 400,
    gradient_clip_val: float = 0.0,
    eval_interval: int = 500,
    min_eval_step: int = 2000,
    preflight_extxyz: bool = False,
) -> dict[str, Any]:
    preflight = None
    if preflight_extxyz:
        preflight = _preflight_sweep_inputs(
            train_file=train_file,
            train_valid_file=train_valid_file,
            dft_valid_file=dft_valid_file,
            teacher_valid_file=teacher_valid_file,
            limit_configs=limit_configs,
            valid_limit_configs=valid_limit_configs,
            bench_limit_configs=bench_limit_configs,
        )
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    selected_row_set = row_set or ("custom" if rows is not None else "design-space-default")
    sweep_rows = [dict(row) for row in (rows or default_pareto_rows())]
    indexed_rows = []
    for row in sweep_rows:
        name = str(row["name"])
        train_variant = str(row.get("train_variant") or row.get("variant") or name)
        scalar_path_ids = row.get("scalar_path_ids")
        row_dir = target / name
        wrapper = write_rtece_matrix_wrapper(
            row_dir,
            variants=train_variant,
            run_root=f"{run_root.rstrip('/')}/{name}",
            train_file=train_file,
            train_valid_file=train_valid_file,
            dft_valid_file=dft_valid_file,
            teacher_valid_file=teacher_valid_file,
            limit_configs=limit_configs,
            valid_limit_configs=valid_limit_configs,
            bench_limit_configs=bench_limit_configs,
            max_steps=max_steps,
            lr=lr,
            hidden_channels=str(row.get("hidden_channels", "16,16")),
            num_radial=int(row.get("num_radial", 8)),
            moment_l_max=int(row["moment_l_max"]),
            scalar_path_ids=None if scalar_path_ids is None else str(scalar_path_ids),
            species_basis_channels=int(row.get("species_basis_channels", 0)),
            species_basis_mode=str(row.get("species_basis_mode", "fixed_z_power")),
            atomic_cross_radial_sketch_channels=int(row.get("atomic_cross_radial_sketch_channels", 2)),
            atomic_cross_radial_projection=str(row.get("atomic_cross_radial_projection", "fixed_shell_mean")),
            force_weight=force_weight,
            use_short_range_repulsion=bool(row.get("use_short_range_repulsion", False)),
            short_range_repulsion_potential=str(row.get("short_range_repulsion_potential", "softplus_overlap")),
            learnable_radial_mixing=bool(row.get("learnable_radial_mixing", False)),
            descriptor_conditioner=str(row.get("descriptor_conditioner", "none")),
            descriptor_conditioner_hidden_channels=int(row.get("descriptor_conditioner_hidden_channels", 0)),
            descriptor_bottleneck_dim=int(row.get("descriptor_bottleneck_dim", 0)),
            force_mode=force_mode,
            measure_passes=measure_passes,
            default_dtype=default_dtype,
            eval_interval=eval_interval,
            min_eval_step=min_eval_step,
            checkpoint_name="rtece_scalar_best.pt",
            trainer_backend=trainer_backend,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            lr_scheduler=lr_scheduler,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
            lr_warmup_steps=lr_warmup_steps,
            early_stopping_patience=early_stopping_patience,
            gradient_clip_val=gradient_clip_val,
        )
        indexed = dict(row)
        indexed["wrapper"] = str(wrapper)
        indexed["sbatch_command"] = ["sbatch", str(wrapper)]
        indexed_rows.append(indexed)
    payload = {
        "schema_version": "rtece_pareto_sweep.v1",
        "row_set": selected_row_set,
        "run_root": run_root,
        "rows": indexed_rows,
        "review_basis": [
            "TECE_design_space: angular bandwidth / radial rank / sparse scalar path density",
            "rTECE_review: cavity relational paths, E0/ZBL physical prior, production training/inference",
        ],
    }
    if preflight is not None:
        payload["preflight_extxyz"] = preflight
    index_path = target / "rtece_pareto_sweep_index.json"
    index_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate rTECE Pareto-sweep wrappers without submitting jobs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--train-valid-file", required=True)
    parser.add_argument("--dft-valid-file", required=True)
    parser.add_argument("--teacher-valid-file", required=True)
    parser.add_argument(
        "--row-set",
        choices=(
            "design-space-default",
            "ef-active-stage115",
            "capacity-ladder-stage116",
            "representation-ladder-stage118",
            "frontloaded-representation-stage119",
            "descriptor-bottleneck-stage120",
            "active-frontloaded-stage121",
        ),
        default="design-space-default",
    )
    parser.add_argument("--limit-configs", type=int, default=2048)
    parser.add_argument("--valid-limit-configs", type=int, default=256)
    parser.add_argument("--bench-limit-configs", type=int, default=1024)
    parser.add_argument("--max-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--valid-batch-size", type=int, default=16)
    parser.add_argument("--lr-warmup-steps", type=int, default=500)
    parser.add_argument("--early-stopping-patience", type=int, default=400)
    parser.add_argument(
        "--preflight-extxyz",
        action="store_true",
        help="Read requested extxyz windows before writing wrappers, so malformed data fails before queue submission.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.row_set == "ef-active-stage115":
        rows = stage115_ef_active_rows()
    elif args.row_set == "capacity-ladder-stage116":
        rows = stage116_capacity_ladder_rows()
    elif args.row_set == "representation-ladder-stage118":
        rows = stage118_representation_ladder_rows()
    elif args.row_set == "frontloaded-representation-stage119":
        rows = stage119_frontloaded_representation_rows()
    elif args.row_set == "descriptor-bottleneck-stage120":
        rows = stage120_descriptor_bottleneck_rows()
    elif args.row_set == "active-frontloaded-stage121":
        rows = stage121_active_frontloaded_rows()
    else:
        rows = None
    payload = write_pareto_sweep(
        args.output_dir,
        run_root=args.run_root,
        train_file=args.train_file,
        train_valid_file=args.train_valid_file,
        dft_valid_file=args.dft_valid_file,
        teacher_valid_file=args.teacher_valid_file,
        rows=rows,
        row_set=args.row_set,
        limit_configs=args.limit_configs,
        valid_limit_configs=args.valid_limit_configs,
        bench_limit_configs=args.bench_limit_configs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        valid_batch_size=args.valid_batch_size,
        lr_warmup_steps=args.lr_warmup_steps,
        early_stopping_patience=args.early_stopping_patience,
        preflight_extxyz=args.preflight_extxyz,
    )
    print(
        json.dumps(
            {
                "index": str(Path(args.output_dir) / "rtece_pareto_sweep_index.json"),
                "row_set": payload["row_set"],
                "rows": len(payload["rows"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
