from __future__ import annotations

import json
import sys
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
from benchmarks.oc20neb_tace_mace.rtece_scalar_model import RTECEGraph

CONFIGS = Path(
    "/home/gengjianrui/workdir_sjtu-caoxiaoming/gengjianrui/Phorbol-mace-dpa4-training-accel/runs/oc20neb_fullcase200_fps_extxyz/valid.extxyz"
)
OUT = Path("runs/oc20neb_tace_mace/rtece-stage51-cell-list-oracle-work")
LIMITS = [1024, 2048, 4096, 8192, 10000]
CUTOFF = 5.0
TRAJECTORY_STEP = 19
DISPLACEMENT_STD = 0.001


def build_template(atoms_list):
    z_parts = []
    pos_parts = []
    batch_parts = []
    for graph_id, atoms in enumerate(atoms_list):
        count = len(atoms)
        z_parts.append(torch.tensor(atoms.numbers, dtype=torch.long))
        pos_parts.append(torch.tensor(atoms.positions, dtype=torch.float32))
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


def main():
    rows = []
    atoms_all = ase.io.read(str(CONFIGS), index=":10000")
    for limit in LIMITS:
        atoms_list = atoms_all[:limit]
        template = build_template(atoms_list)
        positions = synthetic_trajectory_positions(
            template.pos,
            step=TRAJECTORY_STEP,
            displacement_std=DISPLACEMENT_STD,
        )
        metadata = cell_list_oracle_work_metadata(template, positions, cutoff=CUTOFF)
        loop_graph = torch_radius_nopbc_graph(template, positions, cutoff=CUTOFF)
        if metadata["active_directed_edges"] != int(loop_graph.edge_index.shape[1]):
            raise RuntimeError(
                f"active edge mismatch at limit={limit}: "
                f"oracle={metadata['active_directed_edges']} loop={loop_graph.edge_index.shape[1]}"
            )
        payload = {
            "limit_configs": limit,
            "cutoff": CUTOFF,
            "trajectory_step": TRAJECTORY_STEP,
            "trajectory_displacement_std": DISPLACEMENT_STD,
            **metadata,
        }
        rows.append(payload)
        (OUT / f"cell_list_oracle_limit{limit}.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (OUT / "cell_list_oracle_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    print("limit\tatoms\tpadded\tall_nonself\tcell_candidates\tactive\tcand/padded\tcand/active\tmax_cell_occ")
    for row in rows:
        print(
            "\t".join(
                [
                    str(row["limit_configs"]),
                    str(row["num_atoms"]),
                    str(row["padded_pair_slots"]),
                    str(row["all_pair_nonself_slots"]),
                    str(row["cell_candidate_directed_pairs"]),
                    str(row["active_directed_edges"]),
                    f"{row['candidate_to_padded_ratio']:.6f}",
                    f"{row['candidate_to_active_ratio']:.6f}",
                    str(row["max_cell_occupancy"]),
                ]
            )
        )


if __name__ == "__main__":
    main()
