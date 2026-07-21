from __future__ import annotations

import json
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_stage179_validation_matrix_separates_required_hypotheses_from_optimizations(tmp_path: Path):
    from benchmarks.oc20neb_tace_mace.make_rtece_stage179_scientific_validation import (
        build_stage179_validation_matrix,
    )

    stage178 = tmp_path / "stage178" / "stage178_benchmark_analysis.json"
    stage177 = tmp_path / "stage177" / "stage177_pareto_audit.json"
    _write_json(
        stage178,
        {
            "rattle_relax_policy": "continuous_rmsd_after_relax_not_binary_gate",
            "best_by_error": {"variant": "stage178_l0_local_species"},
            "rows": [
                {
                    "variant": "stage178_l0_local_species",
                    "dft_f_rmse_mev_a": 116.0,
                    "dft_e_rmse_mev_atom": 116.0,
                    "dft_f_max_mev_a": 1704.0,
                    "atoms_per_second_limit1024": 1_790_000.0,
                    "focus_rattle_final_rmsd_a": 0.234,
                    "excluded_by_rattle_gate": False,
                },
                {
                    "variant": "stage178_t3_minimal_cavity_direct",
                    "dft_f_rmse_mev_a": 121.0,
                    "dft_e_rmse_mev_atom": 165.0,
                    "dft_f_max_mev_a": 4502.0,
                    "atoms_per_second_limit1024": 782_000.0,
                    "focus_rattle_final_rmsd_a": 0.300,
                    "excluded_by_rattle_gate": False,
                },
            ],
            "interpretation": [
                "Stage178 tests front-loaded representation capacity, not final-head widening.",
                "Stage177 NEP/DPA rows remain smoke baselines under non-matching throughput protocols.",
            ],
        },
    )
    _write_json(
        stage177,
        {
            "rows": [
                {
                    "candidate": "stage176_local_l0_rank3",
                    "family": "rtece",
                    "dft_f_rmse_mev_a": 142.0,
                    "dft_e_rmse_mev_atom": 295.0,
                    "atoms_per_second": 3_988_000.0,
                    "throughput_status": "atom_count_scaling_found_memory_missing",
                },
                {
                    "candidate": "nep4_mixed_smoke",
                    "family": "nep",
                    "dft_f_rmse_mev_a": 143.0,
                    "dft_e_rmse_mev_atom": 23.8,
                    "atoms_per_second": 103_000.0,
                    "throughput_status": "community_engine_wall_time",
                },
            ],
            "comparison_caveats": ["non_matching_throughput_protocols"],
        },
    )

    matrix = build_stage179_validation_matrix(stage178_analysis=stage178, stage177_audit=stage177)

    required_ids = {row["id"] for row in matrix["required_scientific_hypotheses"]}
    optional_ids = {row["id"] for row in matrix["nice_to_have_optimizations"]}
    assert "operator_space_renormalization_beats_from_scratch" in required_ids
    assert "matched_nep_dpa_protocol" in required_ids
    assert "triton_or_nvalchemi_kernel_fusion" in optional_ids

    statuses = {row["id"]: row["status"] for row in matrix["required_scientific_hypotheses"]}
    assert statuses["early_scalarization_hardware_hypothesis"] == "supported"
    assert statuses["edge_relational_rtece_advantage"] == "contradicted_or_unrealized_current_recipe"
    assert statuses["operator_space_renormalization_beats_from_scratch"] == "missing"
    assert statuses["matched_nep_dpa_protocol"] == "protocol_mismatched"

    assert matrix["rattle_relax_policy"] == "continuous_rmsd_after_relax_not_binary_gate"
    assert matrix["stage178_best_current_endpoint"] == "stage178_l0_local_species"
    assert matrix["current_user_readiness"]["status"] == "research_prototype_not_user_ready"
