from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

import ase.io
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.oc20neb_tace_mace.benchmark_rtece_scalar import (
    cell_list_oracle_work_metadata,
    synthetic_trajectory_positions,
    torch_radius_nopbc_graph,
)
from benchmarks.oc20neb_tace_mace.rtece_scalar_model import (
    RTECEGraph,
    RTECEScalarConfig,
    cell_list_packed_element_density_descriptors,
    packed_element_density_descriptors,
)

CONFIGS = Path(
    "/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz"
)
OUT = Path("runs/oc20neb_tace_mace/rtece-stage52-cell-list-fused-descriptor")
LIMITS = [128, 512]
CUTOFF = 5.0
TRAJECTORY_STEP = 19
DISPLACEMENT_STD = 0.001
REPEATS = 3


def build_template(atoms_list):
    z_parts = []
    pos_parts = []
    batch_parts = []
    for graph_id, atoms in enumerate(atoms_list):
        count = len(atoms)
        z_parts.append(torch.tensor(atoms.numbers, dtype=torch.long))
        pos_parts.append(torch.tensor(atoms.positions, dtype=torch.float64))
        batch_parts.append(torch.full((count,), graph_id, dtype=torch.long))
    z = torch.cat(z_parts, dim=0)
    pos = torch.cat(pos_parts, dim=0)
    batch = torch.cat(batch_parts, dim=0)
    return RTECEGraph(
        z=z,
        pos=pos,
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        batch=batch,
    )


def median_seconds(fn):
    times = []
    value = None
    for _ in range(REPEATS):
        start = time.perf_counter()
        value = fn()
        times.append(time.perf_counter() - start)
    return value, statistics.median(times)


def main():
    atoms_all = ase.io.read(str(CONFIGS), index=f":{max(LIMITS)}")
    config = RTECEScalarConfig(
        variant="rtece_element_density",
        cutoff=CUTOFF,
        num_radial=8,
        hidden_channels=(24, 24),
        use_element_density=True,
    )
    rows = []
    for limit in LIMITS:
        template = build_template(atoms_all[:limit])
        positions = synthetic_trajectory_positions(
            template.pos,
            step=TRAJECTORY_STEP,
            displacement_std=DISPLACEMENT_STD,
        )
        moved_template = RTECEGraph(
            z=template.z,
            pos=positions,
            edge_index=template.edge_index,
            batch=template.batch,
        )
        metadata = cell_list_oracle_work_metadata(template, positions, cutoff=CUTOFF)

        direct_graph, direct_graph_s = median_seconds(
            lambda: torch_radius_nopbc_graph(template, positions, cutoff=CUTOFF)
        )
        direct_desc, direct_desc_s = median_seconds(
            lambda: packed_element_density_descriptors(direct_graph, config)
        )
        fused_desc, fused_desc_s = median_seconds(
            lambda: cell_list_packed_element_density_descriptors(moved_template, config)
        )
        max_abs_diff = (direct_desc - fused_desc).abs().max().item() if direct_desc.numel() else 0.0
        if max_abs_diff > 1e-10:
            raise RuntimeError(f"descriptor mismatch at limit={limit}: {max_abs_diff}")
        if metadata["active_directed_edges"] != int(direct_graph.edge_index.shape[1]):
            raise RuntimeError(
                f"active edge mismatch at limit={limit}: "
                f"oracle={metadata['active_directed_edges']} loop={direct_graph.edge_index.shape[1]}"
            )

        row = {
            "limit_configs": limit,
            "cutoff": CUTOFF,
            "trajectory_step": TRAJECTORY_STEP,
            "trajectory_displacement_std": DISPLACEMENT_STD,
            "num_radial": config.num_radial,
            "descriptor_dim": 2 * config.num_radial,
            "max_abs_descriptor_diff": max_abs_diff,
            "direct_radius_graph_median_s_cpu": direct_graph_s,
            "direct_edge_descriptor_median_s_cpu": direct_desc_s,
            "cell_list_fused_descriptor_median_s_cpu": fused_desc_s,
            **metadata,
        }
        rows.append(row)
        (OUT / f"descriptor_oracle_limit{limit}.json").write_text(
            json.dumps(row, indent=2, sort_keys=True) + "\n"
        )
    (OUT / "descriptor_oracle_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print("limit\tatoms\tactive_edges\tcell_candidates\tmax_abs_diff\tdirect_graph_s\tdirect_desc_s\tfused_desc_s")
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["limit_configs"]),
                    str(row["num_atoms"]),
                    str(row["active_directed_edges"]),
                    str(row["cell_candidate_directed_pairs"]),
                    f"{row['max_abs_descriptor_diff']:.3e}",
                    f"{row['direct_radius_graph_median_s_cpu']:.6f}",
                    f"{row['direct_edge_descriptor_median_s_cpu']:.6f}",
                    f"{row['cell_list_fused_descriptor_median_s_cpu']:.6f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
