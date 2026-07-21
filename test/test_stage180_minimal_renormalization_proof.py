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
    assert arms["renorm_initialized_same_student"]["runnable_now"] is True
    assert arms["renorm_initialized_same_student"]["blocking_tooling"] == []
    assert arms["renorm_initialized_same_student"]["init_state"].endswith("rtece_scalar_init.pt")
    assert arms["renorm_initialized_teacher_residual_distill"]["runnable_now"] is False
    assert "teacher_residual_cache_or_extxyz_labels" in arms["renorm_initialized_teacher_residual_distill"]["blocking_tooling"]

    wrappers = payload["artifacts"]["wrappers"]
    assert set(wrappers) == {
        "projection_diagnostic",
        "projection_initializer",
        "scratch_train",
        "renorm_initialized_train",
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



def test_rtece_train_init_state_loader_rejects_config_mismatch(tmp_path):
    import pytest
    import torch
    from dataclasses import replace

    from tace.lightning.rtece import load_rtece_init_state
    from tace.models.rtece_scalar import RTECEScalarModel, build_rtece_config_from_path_ids
    from tace.models.rtece_workflow import save_checkpoint

    config = build_rtece_config_from_path_ids(
        "init_state_unit",
        ["atomic.radial_density"],
        num_radial=3,
        hidden_channels=(4,),
    )
    source = RTECEScalarModel(config).to(dtype=torch.float32)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.fill_(0.125)
    init_path = tmp_path / "rtece_scalar_init.pt"
    save_checkpoint(init_path, source, config, metadata={"init_method": "unit_test"})

    target = RTECEScalarModel(config).to(dtype=torch.float32)
    metadata = load_rtece_init_state(init_path, target, config, dtype=torch.float32)

    assert metadata["init_state"] == str(init_path)
    assert metadata["init_metadata"]["init_method"] == "unit_test"
    for name, value in target.state_dict().items():
        assert torch.allclose(value, source.state_dict()[name].to(value.dtype))

    mismatch = replace(config, num_radial=4)
    with pytest.raises(ValueError, match="init-state config mismatch"):
        load_rtece_init_state(init_path, RTECEScalarModel(mismatch), mismatch, dtype=torch.float32)


def test_stage180_manifest_makes_renorm_init_arm_runnable_with_safe_wrappers(tmp_path):
    from benchmarks.oc20neb_tace_mace.make_rtece_stage180_minimal_renormalization_proof import (
        audit_stage180_manifest,
        make_stage180_manifest,
        materialize_stage180,
    )

    payload = make_stage180_manifest(output_root=tmp_path / "stage180")
    result = materialize_stage180(payload)
    payload = result["payload"]
    audit = audit_stage180_manifest(payload)
    arms = {row["id"]: row for row in payload["comparison_arms"]}
    wrappers = payload["artifacts"]["wrappers"]

    assert arms["renorm_initialized_same_student"]["runnable_now"] is True
    assert arms["renorm_initialized_same_student"]["blocking_tooling"] == []
    assert set(wrappers) == {
        "projection_diagnostic",
        "projection_initializer",
        "scratch_train",
        "renorm_initialized_train",
        "benchmark_physical_after_training",
    }
    assert "--init-state" in (tmp_path / "stage180" / "wrappers" / "stage180_renorm_init_train_no_export.sbatch").read_text(encoding="utf-8")
    assert "initialize_rtece_from_projection.py" in (tmp_path / "stage180" / "wrappers" / "stage180_projection_initializer_no_export.sbatch").read_text(encoding="utf-8")
    assert audit["contract_pass"], audit["failed_checks"]
    assert audit["no_forbidden_sbatch_flags"] is True


def test_stage180_initializer_uses_training_limit_for_matching_e0s(tmp_path):
    from benchmarks.oc20neb_tace_mace.make_rtece_stage180_minimal_renormalization_proof import (
        make_stage180_manifest,
        write_projection_initializer_wrapper,
    )

    payload = make_stage180_manifest(output_root=tmp_path / "stage180")
    wrapper = write_projection_initializer_wrapper(
        tmp_path / "stage180" / "wrappers" / "stage180_projection_initializer_no_export.sbatch",
        payload,
    )
    text = wrapper.read_text(encoding="utf-8")

    assert '--limit-configs "${LIMIT_CONFIGS}"' in text
    assert '--limit-configs "${PROJECTION_LIMIT_CONFIGS}"' not in text
