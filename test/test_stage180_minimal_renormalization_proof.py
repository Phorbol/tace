from __future__ import annotations

import subprocess
import sys


def test_stage180_scaffold_declares_fixed_path_set_and_honest_tooling_gaps(tmp_path):
    from benchmarks.oc20neb_tace_mace.make_rtece_stage180_minimal_renormalization_proof import (
        make_stage180_manifest,
        audit_stage180_manifest,
    )

    payload = make_stage180_manifest(output_root=tmp_path / "stage180")
    audit = audit_stage180_manifest(payload)

    assert audit["contract_pass"], audit["failed_checks"]
    assert payload["stage"] == "stage180_minimal_renormalization_proof"
    assert payload["scientific_question"].startswith("Does TECE-style renormalized initialization")
    assert payload["fixed_student_path_ids"] == [
        "atomic.radial_density",
        "atomic.species_basis_density",
        "atomic.local_l0_lowrank_density",
        "atomic.vector_norm",
        "atomic.vector_cross_radial_dot",
    ]

    arms = {row["id"]: row for row in payload["comparison_arms"]}
    assert set(arms) == {
        "scratch_same_student",
        "linear_projection_diagnostic",
        "renorm_initialized_same_student",
        "renorm_initialized_teacher_residual_distill",
    }
    assert arms["scratch_same_student"]["runnable_now"] is True
    assert arms["linear_projection_diagnostic"]["runnable_now"] is True
    assert arms["renorm_initialized_same_student"]["runnable_now"] is False
    assert arms["renorm_initialized_same_student"]["blocking_tooling"] == [
        "checkpoint_initialization_from_projection_coefficients",
        "train_entrypoint_init_checkpoint_or_init_state",
    ]
    assert arms["renorm_initialized_teacher_residual_distill"]["runnable_now"] is False
    assert "teacher_residual_cache_or_extxyz_labels" in arms["renorm_initialized_teacher_residual_distill"]["blocking_tooling"]

    wrappers = payload["artifacts"]["wrappers"]
    assert set(wrappers) == {
        "projection_diagnostic",
        "scratch_train",
        "benchmark_physical_after_training",
    }
    assert audit["no_forbidden_sbatch_flags"] is True
    assert payload["success_criteria"]["primary_claim_requires"] == [
        "renorm_initialized_same_student beats scratch_same_student under matched E/F/relative/physical metrics",
        "or the result is recorded as falsifying the current renormalization recipe",
    ]


def test_stage180_generator_runs_as_direct_script(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/oc20neb_tace_mace/make_rtece_stage180_minimal_renormalization_proof.py",
            "--output-root",
            str(tmp_path / "stage180"),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "stage180" / "stage180_manifest.json").exists()
    assert (tmp_path / "stage180" / "wrappers" / "stage180_projection_diagnostic_no_export.sbatch").exists()


def test_stage180_nondefault_output_root_keeps_implementation_plan_local(tmp_path):
    from benchmarks.oc20neb_tace_mace.make_rtece_stage180_minimal_renormalization_proof import make_stage180_manifest

    output_root = tmp_path / "stage180"
    payload = make_stage180_manifest(output_root=output_root)

    assert payload["artifacts"]["implementation_plan"] == str(output_root / "stage180_implementation_plan.md")


def test_stage180_projection_wrapper_passes_local_l0_projection_args(tmp_path):
    from benchmarks.oc20neb_tace_mace.make_rtece_stage180_minimal_renormalization_proof import (
        make_stage180_manifest,
        write_projection_wrapper,
    )

    payload = make_stage180_manifest(output_root=tmp_path / "stage180")
    wrapper = write_projection_wrapper(
        tmp_path / "stage180" / "wrappers" / "stage180_projection_diagnostic_no_export.sbatch",
        payload,
    )
    text = wrapper.read_text(encoding="utf-8")

    assert "--species-basis-mode learnable_embedding" in text
    assert "--local-l0-chemistry-rank 4" in text
    assert "--force-component-sample-count 64" in text
    assert "--force-component-sample-count 6000" not in text
