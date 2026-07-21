from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_stage178_analysis_treats_rattle_as_continuous_relax_metric(tmp_path: Path):
    from benchmarks.oc20neb_tace_mace.make_rtece_stage178_benchmark_analysis import (
        build_stage178_analysis,
    )

    root = tmp_path / "stage178"
    diagnostics = root / "diagnostics"
    results = root / "results"
    candidates = ("l0_best_error_legacy_rattle_false", "l1_worse_error_legacy_rattle_true")
    metrics = {
        candidates[0]: {
            "e_rmse": 10.0,
            "f_rmse": 20.0,
            "f_max": 200.0,
            "rattle_rmsd": 0.25,
            "legacy_rattle": False,
            "atoms_per_second": 1000.0,
        },
        candidates[1]: {
            "e_rmse": 30.0,
            "f_rmse": 40.0,
            "f_max": 300.0,
            "rattle_rmsd": 0.20,
            "legacy_rattle": True,
            "atoms_per_second": 800.0,
        },
    }
    for name, row in metrics.items():
        _write_json(
            diagnostics / name / f"{name}_dft_benchmark.json",
            {
                "variant": name,
                "num_parameters": 123,
                "rmse_e_mev_atom": row["e_rmse"],
                "mae_e_mev_atom": row["e_rmse"] / 2,
                "max_abs_e_mev_atom": row["e_rmse"] * 2,
                "rmse_f_mev_a": row["f_rmse"],
                "mae_f_mev_a": row["f_rmse"] / 2,
                "max_abs_f_mev_a": row["f_max"],
                "relative_image_rmse_mev_atom": 5.0,
                "barrier_rmse_mev_atom": 6.0,
                "atoms_per_second": row["atoms_per_second"],
                "seconds_per_pass": 0.1,
                "peak_allocated_mb": 100.0,
                "peak_reserved_mb": 200.0,
            },
        )
        _write_json(
            diagnostics / name / f"{name}_teacher_benchmark.json",
            {
                "variant": name,
                "rmse_e_mev_atom": row["e_rmse"] + 1,
                "rmse_f_mev_a": row["f_rmse"] + 1,
            },
        )
        _write_json(
            diagnostics / name / f"{name}_scaling_limit1024.json",
            {
                "variant": name,
                "configs": 1024,
                "atoms": 50000,
                "atoms_per_second": row["atoms_per_second"] * 2,
                "seconds_per_pass": 0.2,
                "peak_allocated_mb": 300.0,
                "peak_reserved_mb": 400.0,
            },
        )
        _write_json(
            diagnostics / name / f"{name}_physical_pareto.json",
            {
                "rows": [
                    {
                        "variant": name,
                        "physical_gate_pass": False,
                        "rattle_gate_pass": row["legacy_rattle"],
                        "dimer_gate_pass": True,
                        "focus_rattle_final_rmsd_a": row["rattle_rmsd"],
                        "focus_rattle_max_fmax_ev_a": 1.0,
                        "physical_score": 1.0,
                    }
                ]
            },
        )
        _write_json(
            results / name / "train_summary.json",
            {"steps": 100, "best_step": 50, "best_valid_loss": 0.1},
        )

    analysis = build_stage178_analysis(stage178_root=root)

    assert analysis["rattle_relax_policy"] == "continuous_rmsd_after_relax_not_binary_gate"
    assert analysis["best_by_error"]["variant"] == candidates[0]
    l0 = next(row for row in analysis["rows"] if row["variant"] == candidates[0])
    assert l0["legacy_rattle_gate_pass"] is False
    assert l0["rattle_relax_decision"] == "continuous_metric_only"
    assert l0["excluded_by_rattle_gate"] is False
    assert l0["focus_rattle_final_rmsd_a"] == pytest.approx(0.25)
