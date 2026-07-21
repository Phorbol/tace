from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


def test_rmd17_npz_to_extxyz_converts_kcal_to_ev(tmp_path):
    from ase.io import read
    from benchmarks.oc20neb_tace_mace.prepare_stage181_conventional_datasets import (
        KCAL_MOL_TO_EV,
        convert_rmd17_npz_to_extxyz,
    )

    source = tmp_path / "rmd17_ethanol.npz"
    out = tmp_path / "ethanol.extxyz"
    np.savez(
        source,
        nuclear_charges=np.array([6, 1], dtype=np.int64),
        coords=np.array([[[0.0, 0.0, 0.0], [1.1, 0.0, 0.0]]], dtype=np.float64),
        energies=np.array([10.0], dtype=np.float64),
        forces=np.array([[[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]]], dtype=np.float64),
    )

    summary = convert_rmd17_npz_to_extxyz(source, out, molecule="ethanol", limit_configs=None)
    atoms = read(out, index=0)

    assert summary["source_units"] == {"energy": "kcal/mol", "forces": "kcal/mol/A", "distance": "A"}
    assert summary["target_units"] == {"energy": "eV", "forces": "eV/A", "distance": "A"}
    assert summary["num_configs"] == 1
    assert atoms.get_potential_energy() == pytest.approx(10.0 * KCAL_MOL_TO_EV)
    assert atoms.get_forces()[0, 1] == pytest.approx(2.0 * KCAL_MOL_TO_EV)
    assert atoms.info["dataset"] == "rMD17"
    assert atoms.info["molecule"] == "ethanol"


def test_3bpa_manifest_declares_splits_units_and_validation_roles(tmp_path):
    from benchmarks.oc20neb_tace_mace.prepare_stage181_conventional_datasets import make_3bpa_dataset_manifest

    root = tmp_path / "dataset_3BPA"
    manifest = make_3bpa_dataset_manifest(root)

    assert manifest["dataset"] == "3BPA"
    assert manifest["units"] == {"energy": "eV", "forces": "eV/A", "distance": "A"}
    assert manifest["labels"] == {"energy_key": "energy", "forces_key": "forces"}
    assert manifest["splits"]["train_300K"]["role"] == "train_id_300K"
    assert manifest["splits"]["test_1200K"]["role"] == "strong_temperature_ood"
    assert manifest["splits"]["test_dih"]["role"] == "dihedral_pes_ood"
    assert manifest["splits"]["iso_atoms"]["role"] == "isolated_atom_reference"


def test_stage181_manifest_and_wrappers_are_sai_safe(tmp_path):
    from benchmarks.oc20neb_tace_mace.make_rtece_stage181_conventional_md import (
        audit_stage181_manifest,
        make_stage181_manifest,
        materialize_stage181,
    )

    payload = make_stage181_manifest(output_root=tmp_path / "stage181", dataset_root=tmp_path / "dataset_3BPA")
    result = materialize_stage181(payload)
    audit = audit_stage181_manifest(payload)

    assert result["audit"]["contract_pass"]
    assert audit["contract_pass"], audit["failed_checks"]
    assert payload["stage"] == "stage181_conventional_md"
    assert payload["dataset_matrix"]["primary"] == "3BPA"
    assert payload["dataset_matrix"]["secondary"] == "rMD17"
    assert payload["student_config"]["local_l0_chemistry_rank"] == 4

    wrappers = payload["artifacts"]["wrappers"]
    assert set(wrappers) == {
        "train_3bpa_300k",
        "benchmark_3bpa",
        "train_3bpa_mixedT",
        "benchmark_3bpa_mixedT",
        "train_rmd17_ethanol",
        "benchmark_rmd17_ethanol",
    }
    wrapper_text = "\n".join(Path(path).read_text(encoding="utf-8") for path in wrappers.values())
    assert "--export" not in wrapper_text
    assert "#SBATCH --mem" not in wrapper_text
    assert "--cpus-per-task" not in wrapper_text
    assert "set -u" not in wrapper_text
    assert "train_300K.xyz" in wrapper_text
    assert "test_300K.xyz" in wrapper_text
    assert "test_600K.xyz" in wrapper_text
    assert "test_1200K.xyz" in wrapper_text
    assert "test_dih.xyz" in wrapper_text
    assert "--species-basis-mode learnable_embedding" in wrapper_text
    assert "--local-l0-chemistry-rank 4" in wrapper_text
    assert "--no-fit-energy-shift" not in wrapper_text


def test_3bpa_validation_reads_extxyz_splits_and_label_keys(tmp_path):
    from ase import Atoms
    from ase.io import write
    from benchmarks.oc20neb_tace_mace.prepare_stage181_conventional_datasets import validate_3bpa_dataset

    root = tmp_path / "dataset_3BPA"
    root.mkdir()
    configs = []
    for idx in range(2):
        atoms = Atoms(numbers=[6, 1], positions=[[0.0, 0.0, 0.0], [1.0 + 0.1 * idx, 0.0, 0.0]])
        atoms.info["energy"] = -1.0 + 0.1 * idx
        atoms.arrays["forces"] = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=np.float64)
        configs.append(atoms)
    split_filenames = [
        "train_300K.xyz",
        "train_mixedT.xyz",
        "test_300K.xyz",
        "test_600K.xyz",
        "test_1200K.xyz",
        "test_dih.xyz",
    ]
    for filename in split_filenames:
        write(root / filename, configs, format="extxyz")
    iso = Atoms(numbers=[1], positions=[[0.0, 0.0, 0.0]])
    iso.info["energy"] = 0.0
    iso.arrays["forces"] = np.zeros((1, 3), dtype=np.float64)
    write(root / "iso_atoms.xyz", [iso], format="extxyz")

    report = validate_3bpa_dataset(root, max_configs_per_split=1)

    assert report["schema_version"] == "rtece_stage181_3bpa_validation.v1"
    assert report["contract_pass"], report["failed_checks"]
    assert report["units"] == {"energy": "eV", "forces": "eV/A", "distance": "A"}
    assert report["splits"]["train_300K"]["num_configs_checked"] == 1
    assert report["splits"]["train_300K"]["num_atoms_first"] == 2
    assert report["splits"]["test_dih"]["has_energy"]
    assert report["splits"]["test_dih"]["has_forces"]


def test_stage181_manifest_includes_rmd17_smoke_wrappers(tmp_path):
    from benchmarks.oc20neb_tace_mace.make_rtece_stage181_conventional_md import (
        audit_stage181_manifest,
        make_stage181_manifest,
        materialize_stage181,
    )

    payload = make_stage181_manifest(
        output_root=tmp_path / "stage181",
        dataset_root=tmp_path / "dataset_3BPA",
        rmd17_root=tmp_path / "rMD17",
    )
    materialize_stage181(payload)
    audit = audit_stage181_manifest(payload)

    assert audit["contract_pass"], audit["failed_checks"]
    assert payload["rmd17_smoke"]["molecule"] == "ethanol"
    assert payload["rmd17_smoke"]["train_configs_recommended_max"] == 1000
    wrappers = payload["artifacts"]["wrappers"]
    assert set(wrappers) == {
        "train_3bpa_300k",
        "benchmark_3bpa",
        "train_3bpa_mixedT",
        "benchmark_3bpa_mixedT",
        "train_rmd17_ethanol",
        "benchmark_rmd17_ethanol",
    }
    wrapper_text = "\n".join(Path(path).read_text(encoding="utf-8") for path in wrappers.values())
    assert "converted_extxyz/rmd17_ethanol_train.extxyz" in wrapper_text
    assert "converted_extxyz/rmd17_ethanol_valid.extxyz" in wrapper_text
    assert "converted_extxyz/rmd17_ethanol_test.extxyz" in wrapper_text
    assert "--export" not in wrapper_text
    assert "#SBATCH --mem" not in wrapper_text
    assert "--cpus-per-task" not in wrapper_text
    assert "set -u" not in wrapper_text



def test_rmd17_npz_to_extxyz_splits_uses_ev_units_and_contiguous_disjoint_splits(tmp_path):
    from ase.io import read
    from benchmarks.oc20neb_tace_mace.prepare_stage181_conventional_datasets import (
        KCAL_MOL_TO_EV,
        convert_rmd17_npz_to_extxyz_splits,
    )

    source = tmp_path / "rmd17_ethanol.npz"
    coords = np.zeros((8, 2, 3), dtype=np.float64)
    coords[:, 1, 0] = np.arange(8, dtype=np.float64) + 1.0
    forces = np.ones((8, 2, 3), dtype=np.float64)
    forces[:, 0, 0] = np.arange(8, dtype=np.float64)
    np.savez(
        source,
        nuclear_charges=np.array([6, 1], dtype=np.int64),
        coords=coords,
        energies=np.arange(8, dtype=np.float64),
        forces=forces,
    )

    summary = convert_rmd17_npz_to_extxyz_splits(
        source,
        tmp_path / "converted_extxyz",
        molecule="ethanol",
        train_count=3,
        valid_count=2,
        test_count=2,
    )

    assert summary["schema_version"] == "rtece_stage181_rmd17_split_conversion.v1"
    assert summary["splits"]["train"]["indices"] == [0, 1, 2]
    assert summary["splits"]["valid"]["indices"] == [3, 4]
    assert summary["splits"]["test"]["indices"] == [5, 6]
    train = read(summary["splits"]["train"]["output_extxyz"], index=":")
    valid = read(summary["splits"]["valid"]["output_extxyz"], index=":")
    test = read(summary["splits"]["test"]["output_extxyz"], index=":")
    assert len(train) == 3
    assert len(valid) == 2
    assert len(test) == 2
    assert train[2].info["source_frame"] == 2
    assert valid[0].info["source_frame"] == 3
    assert test[1].get_potential_energy() == pytest.approx(6.0 * KCAL_MOL_TO_EV)
    assert test[1].get_forces()[0, 0] == pytest.approx(6.0 * KCAL_MOL_TO_EV)



def test_stage181_manifest_includes_3bpa_mixed_temperature_comparison(tmp_path):
    from benchmarks.oc20neb_tace_mace.make_rtece_stage181_conventional_md import (
        audit_stage181_manifest,
        make_stage181_manifest,
        materialize_stage181,
    )

    payload = make_stage181_manifest(output_root=tmp_path / "stage181", dataset_root=tmp_path / "dataset_3BPA")
    materialize_stage181(payload)
    audit = audit_stage181_manifest(payload)

    assert audit["contract_pass"], audit["failed_checks"]
    assert payload["distribution_coverage_comparison"]["baseline_train_split"] == "train_300K"
    assert payload["distribution_coverage_comparison"]["coverage_train_split"] == "train_mixedT"
    assert "test_1200K" in payload["distribution_coverage_comparison"]["ood_splits"]
    wrappers = payload["artifacts"]["wrappers"]
    assert "train_3bpa_mixedT" in wrappers
    assert "benchmark_3bpa_mixedT" in wrappers
    wrapper_text = "\n".join(Path(path).read_text(encoding="utf-8") for path in wrappers.values())
    assert '"${DATASET_ROOT}/train_mixedT.xyz"' in wrapper_text
    assert "stage181_3bpa_trainMixedT" in wrapper_text
    assert "stage181_3bpa_trainMixedT/test_1200K_benchmark.json" in wrapper_text
    assert "--export" not in wrapper_text
    assert "#SBATCH --mem" not in wrapper_text
    assert "--cpus-per-task" not in wrapper_text
    assert "set -u" not in wrapper_text


def test_stage182_3bpa_closure_manifest_compares_rtece_nep_and_dpalike(tmp_path):
    from benchmarks.oc20neb_tace_mace.make_rtece_stage182_3bpa_closure import (
        BENCHMARK_SPLITS,
        audit_stage182_manifest,
        make_stage182_manifest,
        materialize_stage182,
    )

    payload = make_stage182_manifest(output_root=tmp_path / "stage182", dataset_root=tmp_path / "dataset_3BPA")
    result = materialize_stage182(payload)
    audit = audit_stage182_manifest(payload)

    assert result["audit"]["contract_pass"]
    assert audit["contract_pass"], audit["failed_checks"]
    assert payload["stage"] == "stage182_3bpa_conventional_closure"
    assert payload["dataset"]["name"] == "3BPA"
    assert payload["dataset"]["units"] == {"energy": "eV", "forces": "eV/A", "distance": "A"}
    assert payload["dataset"]["benchmark_splits"] == BENCHMARK_SPLITS
    assert {row["name"] for row in payload["rows"]} == {
        "rtece_l1_local_l0_train300k",
        "nep4_train300k",
        "deepmd_dpa1_zero_train300k",
    }
    assert payload["comparison_contract"]["primary_ranking_metric"] == "rmse_f_mev_a"
    for metric in [
        "rmse_e_mev_atom",
        "max_abs_e_mev_atom",
        "rmse_f_mev_a",
        "max_abs_f_mev_a",
        "atoms_per_second",
        "peak_memory_mb",
    ]:
        assert metric in payload["comparison_contract"]["required_metrics"]
    wrappers = result["wrappers"]
    assert set(wrappers) == {row["name"] for row in payload["rows"]}
    benchmark_wrappers = result["benchmark_wrappers"]
    assert set(benchmark_wrappers) == {row["name"] for row in payload["rows"]}


def test_stage182_3bpa_closure_wrappers_are_sai_safe_and_split_complete(tmp_path):
    from pathlib import Path
    from benchmarks.oc20neb_tace_mace.make_rtece_stage182_3bpa_closure import (
        BENCHMARK_SPLITS,
        make_stage182_manifest,
        materialize_stage182,
    )

    payload = make_stage182_manifest(output_root=tmp_path / "stage182", dataset_root=tmp_path / "dataset_3BPA")
    result = materialize_stage182(payload)
    wrapper_text = "\n".join(Path(path).read_text(encoding="utf-8") for group in ["wrappers", "benchmark_wrappers"] for path in result[group].values())

    for forbidden in ["--export", "#SBATCH --mem", "--mem=", "#SBATCH --cpus-per-task", "--cpus-per-task", "set -u", "export "]:
        assert forbidden not in wrapper_text
    assert "train_300K.xyz" in wrapper_text
    assert "train_mixedT.xyz" not in wrapper_text
    for split in BENCHMARK_SPLITS:
        assert f"{split}.xyz" in wrapper_text
    assert "tace.scripts.rtece_train_scalar" in wrapper_text
    assert "convert_stage145_nep.py" in wrapper_text
    assert "convert_stage145_deepmd.py" in wrapper_text
    assert "benchmark_rtece_scalar.py" in wrapper_text
    assert "benchmark_stage145_community.py" in wrapper_text
    assert "module load gpumd" in wrapper_text
    assert "module load deepmd-kit" in wrapper_text


def test_stage182_summary_marks_missing_results_and_orders_by_force_rmse(tmp_path):
    import json
    from benchmarks.oc20neb_tace_mace.make_rtece_stage182_3bpa_closure import (
        make_stage182_manifest,
        summarize_stage182_results,
    )

    payload = make_stage182_manifest(output_root=tmp_path / "stage182", dataset_root=tmp_path / "dataset_3BPA")
    rows = {row["name"]: row for row in payload["rows"]}
    rtece_out = Path(rows["rtece_l1_local_l0_train300k"]["benchmark_outputs"]["test_300K"])
    nep_out = Path(rows["nep4_train300k"]["benchmark_outputs"]["test_300K"])
    for path, rmse_f, atoms_s in [(rtece_out, 120.0, 1.5e6), (nep_out, 80.0, 7.5e6)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "status": "completed",
            "rmse_e_mev_atom": 3.0,
            "max_abs_e_mev_atom": 9.0,
            "rmse_f_mev_a": rmse_f,
            "max_abs_f_mev_a": 400.0,
            "atoms_per_second": atoms_s,
        }), encoding="utf-8")

    summary = summarize_stage182_results(payload)

    assert summary["schema_version"] == "rtece_stage182_3bpa_closure_summary.v1"
    assert summary["rows"][0]["row_name"] == "nep4_train300k"
    assert summary["rows"][0]["split"] == "test_300K"
    assert any(row["status"] == "missing" and row["row_name"] == "deepmd_dpa1_zero_train300k" for row in summary["rows"])
    assert "rmse_f_mev_a" in summary["markdown"]
    assert "missing" in summary["markdown"]


def test_stage183_3bpa_representation_ladder_manifest_is_doc_grounded_and_nested(tmp_path):
    from benchmarks.oc20neb_tace_mace.make_rtece_stage183_3bpa_representation_ladder import (
        BENCHMARK_SPLITS,
        audit_stage183_manifest,
        make_stage183_manifest,
        materialize_stage183,
    )

    payload = make_stage183_manifest(output_root=tmp_path / "stage183", dataset_root=tmp_path / "dataset_3BPA")
    result = materialize_stage183(payload)
    audit = audit_stage183_manifest(payload)

    assert result["audit"]["contract_pass"]
    assert audit["contract_pass"], audit["failed_checks"]
    assert payload["stage"] == "stage183_3bpa_representation_ladder"
    assert payload["dataset"]["name"] == "3BPA"
    assert payload["dataset"]["benchmark_splits"] == BENCHMARK_SPLITS
    assert payload["stage182_analysis"]["primary_failure_mode"] == "dihedral_pes_gap"
    assert payload["stage182_analysis"]["priority_axes"] == ["l2_atomic_quadrupole", "t3_cavity_edge_relational"]
    assert "TECE_design_space.md" in "\n".join(payload["theory_alignment"])
    assert "rTECE_review.md" in "\n".join(payload["theory_alignment"])

    rows = payload["rows"]
    assert [row["name"] for row in rows] == [
        "stage183_l0_local_species",
        "stage183_l1_cross",
        "stage183_l2_atomic_quadrupole",
        "stage183_t3_cavity_vecq",
    ]
    previous_paths: set[str] = set()
    for row in rows:
        current_paths = set(row["student_config"]["scalar_path_ids"])
        assert previous_paths <= current_paths
        previous_paths = current_paths
        assert row["student_config"]["hidden_channels"] == "64,64"
        assert row["capacity_allocation"] != "widen_final_head_only"
    assert "atomic.quadrupole_norm" in rows[2]["student_config"]["scalar_path_ids"]
    assert "atomic.quadrupole_cross_radial_frobenius" in rows[2]["student_config"]["scalar_path_ids"]
    assert "edge.cavity.vector_dot" in rows[3]["student_config"]["scalar_path_ids"]
    assert "edge.cavity.quadrupole_frobenius" in rows[3]["student_config"]["scalar_path_ids"]


def test_stage183_3bpa_wrappers_are_sai_safe_and_use_production_label_keys(tmp_path):
    from benchmarks.oc20neb_tace_mace.make_rtece_stage183_3bpa_representation_ladder import (
        BENCHMARK_SPLITS,
        FORBIDDEN_SBATCH_TOKENS,
        make_stage183_manifest,
        materialize_stage183,
    )

    payload = make_stage183_manifest(output_root=tmp_path / "stage183", dataset_root=tmp_path / "dataset_3BPA")
    result = materialize_stage183(payload)
    wrapper_text = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for group in ["wrappers", "benchmark_wrappers"]
        for path in result[group].values()
    )

    for forbidden in FORBIDDEN_SBATCH_TOKENS:
        assert forbidden not in wrapper_text
    assert "tace.scripts.rtece_train_scalar" in wrapper_text
    assert "benchmark_rtece_scalar.py" in wrapper_text
    assert "--energy-key energy" in wrapper_text
    assert "--forces-key forces" in wrapper_text
    assert "--moment-l-max 2" in wrapper_text
    assert "--atomic-cross-radial-projection learnable" in wrapper_text
    assert "--scalar-path-ids atomic.radial_density,atomic.species_basis_density,atomic.local_l0_lowrank_density,atomic.vector_norm,atomic.vector_cross_radial_dot,atomic.quadrupole_norm,atomic.quadrupole_cross_radial_frobenius,edge.cavity.vector_dot,edge.cavity.quadrupole_frobenius" in wrapper_text
    for split in BENCHMARK_SPLITS:
        assert f"{split}.xyz" in wrapper_text


def test_stage183_analysis_prioritizes_l2_and_edge_from_stage182_dih_gap(tmp_path):
    import json
    from benchmarks.oc20neb_tace_mace.make_rtece_stage183_3bpa_representation_ladder import analyze_stage182_for_stage183

    summary_path = tmp_path / "stage182_summary.json"
    summary_path.write_text(json.dumps({
        "rows": [
            {"row_name": "nep4_train300k", "engine": "nep", "split": "test_300K", "status": "completed", "rmse_f_mev_a": 100.0, "rmse_e_mev_atom": 2.0},
            {"row_name": "rtece_l1_local_l0_train300k", "engine": "rtece", "split": "test_300K", "status": "completed", "rmse_f_mev_a": 120.0, "rmse_e_mev_atom": 3.0},
            {"row_name": "nep4_train300k", "engine": "nep", "split": "test_dih", "status": "completed", "rmse_f_mev_a": 100.0, "rmse_e_mev_atom": 2.0},
            {"row_name": "rtece_l1_local_l0_train300k", "engine": "rtece", "split": "test_dih", "status": "completed", "rmse_f_mev_a": 220.0, "rmse_e_mev_atom": 14.0},
        ],
    }), encoding="utf-8")

    analysis = analyze_stage182_for_stage183(summary_path)

    assert analysis["primary_failure_mode"] == "dihedral_pes_gap"
    assert analysis["force_rmse_ratios"]["test_dih"] > analysis["force_rmse_ratios"]["test_300K"]
    assert analysis["energy_rmse_ratios"]["test_dih"] > analysis["energy_rmse_ratios"]["test_300K"]
    assert analysis["priority_axes"] == ["l2_atomic_quadrupole", "t3_cavity_edge_relational"]
    assert analysis["next_decision"] == "representation_ladder_before_more_kernel_work"


def test_stage183_summary_ranks_rows_by_force_rmse_and_includes_train_loss(tmp_path):
    import json
    from benchmarks.oc20neb_tace_mace.make_rtece_stage183_3bpa_representation_ladder import (
        make_stage183_manifest,
        summarize_stage183_results,
    )

    payload = make_stage183_manifest(output_root=tmp_path / "stage183", dataset_root=tmp_path / "dataset_3BPA")
    rows = {row["name"]: row for row in payload["rows"]}
    for row_name, valid_loss in [("stage183_l0_local_species", 0.3), ("stage183_l2_atomic_quadrupole", 0.1)]:
        summary_path = Path(rows[row_name]["train_dir"]) / "train_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps({"best_valid_loss": valid_loss, "best_step": 100}), encoding="utf-8")
    for row_name, rmse_f in [("stage183_l0_local_species", 200.0), ("stage183_l2_atomic_quadrupole", 120.0)]:
        out = Path(rows[row_name]["benchmark_outputs"]["test_300K"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "status": "completed",
            "rmse_e_mev_atom": 5.0,
            "max_abs_e_mev_atom": 10.0,
            "rmse_f_mev_a": rmse_f,
            "max_abs_f_mev_a": 500.0,
            "atoms_per_second": 1.0e6,
            "peak_reserved_mb": 900.0,
        }), encoding="utf-8")

    summary = summarize_stage183_results(payload)

    assert summary["schema_version"] == "rtece_stage183_representation_ladder_summary.v1"
    assert summary["rows"][0]["row_name"] == "stage183_l2_atomic_quadrupole"
    assert summary["rows"][0]["train_best_valid_loss"] == 0.1
    assert summary["rows"][0]["peak_memory_mb"] == 900.0
    assert any(row["status"] == "missing" and row["row_name"] == "stage183_t3_cavity_vecq" for row in summary["rows"])
    assert "train_best_valid_loss" in summary["markdown"]
    assert "stage183_l2_atomic_quadrupole" in summary["markdown"]


def test_stage184_3bpa_physical_manifest_tracks_continuous_rattle_and_l2_priority(tmp_path):
    from benchmarks.oc20neb_tace_mace.make_rtece_stage184_3bpa_physical_validation import (
        audit_stage184_manifest,
        make_stage184_manifest,
        materialize_stage184,
    )

    payload = make_stage184_manifest(output_root=tmp_path / "stage184", dataset_root=tmp_path / "dataset_3BPA")
    result = materialize_stage184(payload)
    audit = audit_stage184_manifest(payload)

    assert result["audit"]["contract_pass"]
    assert audit["contract_pass"], audit["failed_checks"]
    assert payload["stage"] == "stage184_3bpa_physical_validation"
    assert payload["stage183_decision"]["current_pareto_candidate"] == "stage183_l2_atomic_quadrupole"
    assert payload["physical_contract"]["rattle_policy"] == "continuous_rmsd_not_binary_gate"
    assert payload["physical_contract"]["dimer_distance_scales"] == {"min": 0.5, "max": 5.0}
    assert payload["physical_contract"]["rattle_configs"] == "test_300K.xyz"
    assert {row["name"] for row in payload["rows"]} == {
        "stage183_l0_local_species",
        "stage183_l1_cross",
        "stage183_l2_atomic_quadrupole",
        "stage183_t3_cavity_vecq",
        "nep4_train300k",
    }
    assert any(row["engine"] == "nep" for row in payload["rows"])
    assert "TECE_design_space.md" in "\n".join(payload["theory_alignment"])
    assert "rTECE_review.md" in "\n".join(payload["theory_alignment"])


def test_stage184_3bpa_physical_wrappers_are_sai_safe_and_reuse_existing_tools(tmp_path):
    from benchmarks.oc20neb_tace_mace.make_rtece_stage184_3bpa_physical_validation import (
        FORBIDDEN_SBATCH_TOKENS,
        make_stage184_manifest,
        materialize_stage184,
    )

    payload = make_stage184_manifest(output_root=tmp_path / "stage184", dataset_root=tmp_path / "dataset_3BPA")
    result = materialize_stage184(payload)
    wrapper_text = "\n".join(Path(path).read_text(encoding="utf-8") for path in result["wrappers"].values())

    for forbidden in FORBIDDEN_SBATCH_TOKENS:
        assert forbidden not in wrapper_text
    assert "dimer_scan_rtece.py" in wrapper_text
    assert "rattle_relax_rtece.py" in wrapper_text
    assert "physical_stage145_community.py" in wrapper_text
    assert "summarize_rtece_stage184" not in wrapper_text
    assert "--min-scale 0.5" in wrapper_text
    assert "--max-scale 5.0" in wrapper_text
    assert "--configs" in wrapper_text
    assert "test_300K.xyz" in wrapper_text
    assert "--rattle-std 0.05" in wrapper_text
    assert "module load gpumd" in wrapper_text


def test_stage184_summary_reads_rtece_and_nep_physical_outputs_as_continuous_metrics(tmp_path):
    import json
    from benchmarks.oc20neb_tace_mace.make_rtece_stage184_3bpa_physical_validation import (
        make_stage184_manifest,
        summarize_stage184_results,
    )

    payload = make_stage184_manifest(output_root=tmp_path / "stage184", dataset_root=tmp_path / "dataset_3BPA")
    rows = {row["name"]: row for row in payload["rows"]}
    rtece = rows["stage183_l2_atomic_quadrupole"]
    Path(rtece["dimer_json"]).parent.mkdir(parents=True, exist_ok=True)
    Path(rtece["dimer_json"]).write_text(json.dumps({
        "pair_summaries": [
            {"pair": "C-N", "summary": {"has_nonfinite": False, "short_force_repulsive": True, "short_minus_long_energy_eV": 1.0}},
            {"pair": "C-O", "summary": {"has_nonfinite": False, "short_force_repulsive": True, "short_minus_long_energy_eV": 0.8}},
        ]
    }), encoding="utf-8")
    Path(rtece["rattle_json"]).write_text(json.dumps({
        "summary": {"num_configs": 2, "converged_fraction": 0.5, "mean_final_rmsd_a": 0.12, "max_final_rmsd_a": 0.2, "max_fmax_ev_a": 0.6, "focus_groups": []}
    }), encoding="utf-8")
    nep = rows["nep4_train300k"]
    Path(nep["community_physical_json"]).parent.mkdir(parents=True, exist_ok=True)
    Path(nep["community_physical_json"]).write_text(json.dumps({
        "status": "completed",
        "dimer_scan": {"pair_summaries": [{"pair": "C-N", "summary": {"has_nonfinite": False, "short_force_repulsive": False}}]},
        "rattle_relax": {"summary": {"num_configs": 1, "converged_fraction": 1.0, "mean_final_rmsd_a": 0.05, "max_final_rmsd_a": 0.05, "max_fmax_ev_a": 0.1, "focus_groups": []}},
    }), encoding="utf-8")

    summary = summarize_stage184_results(payload)

    assert summary["schema_version"] == "rtece_stage184_physical_validation_summary.v1"
    l2_row = next(row for row in summary["rows"] if row["row_name"] == "stage183_l2_atomic_quadrupole")
    nep_row = next(row for row in summary["rows"] if row["row_name"] == "nep4_train300k")
    assert l2_row["status"] == "completed"
    assert l2_row["dimer_short_repulsive_fraction"] == 1.0
    assert l2_row["rattle_mean_final_rmsd_a"] == 0.12
    assert nep_row["dimer_short_repulsive_fraction"] == 0.0
    assert summary["rattle_policy"] == "continuous_rmsd_not_binary_gate"
    assert "rattle_mean_final_rmsd_a" in summary["markdown"]


def test_stage185_rmd17_manifest_declares_unit_conversion_split_policy_and_doc_alignment(tmp_path):
    from benchmarks.oc20neb_tace_mace.make_rtece_stage185_rmd17_representation_ladder import (
        KCAL_MOL_TO_EV,
        audit_stage185_manifest,
        make_stage185_manifest,
        materialize_stage185,
    )

    payload = make_stage185_manifest(output_root=tmp_path / "stage185", rmd17_root=tmp_path / "rMD17")
    result = materialize_stage185(payload)
    audit = audit_stage185_manifest(payload)

    assert result["audit"]["contract_pass"]
    assert audit["contract_pass"], audit["failed_checks"]
    assert payload["stage"] == "stage185_rmd17_representation_ladder"
    assert payload["dataset"]["name"] == "rMD17"
    assert payload["dataset"]["molecule"] == "ethanol"
    assert payload["dataset"]["source"]["figshare_article"] == "12672038"
    assert payload["dataset"]["source"]["version"] == 4
    assert payload["dataset"]["source_units"] == {"energy": "kcal/mol", "forces": "kcal/mol/A", "distance": "A"}
    assert payload["dataset"]["target_units"] == {"energy": "eV", "forces": "eV/A", "distance": "A"}
    assert payload["dataset"]["conversion_factor_energy"] == KCAL_MOL_TO_EV
    assert payload["split_policy"]["train_count"] <= 1000
    assert payload["split_policy"]["indexing"] == "contiguous_time_ordered_blocks"
    assert payload["comparison_contract"]["primary_ranking_metric"] == "rmse_f_mev_a"
    assert "TECE_design_space.md" in "\n".join(payload["theory_alignment"])
    assert "rTECE_review.md" in "\n".join(payload["theory_alignment"])


def test_stage185_rmd17_rows_reuse_nested_stage183_representation_ladder_without_head_widening(tmp_path):
    from benchmarks.oc20neb_tace_mace.make_rtece_stage185_rmd17_representation_ladder import (
        make_stage185_manifest,
    )

    payload = make_stage185_manifest(output_root=tmp_path / "stage185", rmd17_root=tmp_path / "rMD17")
    rows = payload["rows"]

    assert [row["name"] for row in rows] == [
        "stage185_rmd17_l0_local_species",
        "stage185_rmd17_l1_cross",
        "stage185_rmd17_l2_atomic_quadrupole",
        "stage185_rmd17_t3_cavity_vecq",
    ]
    previous_paths: set[str] = set()
    for row in rows:
        current_paths = set(row["student_config"]["scalar_path_ids"])
        assert previous_paths <= current_paths
        previous_paths = current_paths
        assert row["student_config"]["hidden_channels"] == "64,64"
        assert row["capacity_allocation"] != "widen_final_head_only"
        assert row["dataset_role"] == "single_molecule_rmd17_temporal_generalization"
    assert "atomic.quadrupole_norm" in rows[2]["student_config"]["scalar_path_ids"]
    assert "edge.cavity.vector_dot" in rows[3]["student_config"]["scalar_path_ids"]


def test_stage185_rmd17_wrappers_are_sai_safe_and_include_download_convert_train_benchmark(tmp_path):
    from pathlib import Path
    from benchmarks.oc20neb_tace_mace.make_rtece_stage185_rmd17_representation_ladder import (
        FORBIDDEN_SBATCH_TOKENS,
        make_stage185_manifest,
        materialize_stage185,
    )

    payload = make_stage185_manifest(output_root=tmp_path / "stage185", rmd17_root=tmp_path / "rMD17")
    result = materialize_stage185(payload)
    wrapper_text = "\n".join(
        Path(path).read_text(encoding="utf-8")
        for group in ["prep_wrapper", "wrappers", "benchmark_wrappers"]
        for path in ([result[group]] if group == "prep_wrapper" else result[group].values())
    )

    for forbidden in FORBIDDEN_SBATCH_TOKENS:
        assert forbidden not in wrapper_text
    assert "Missing local rMD17 source npz" in wrapper_text
    assert "https://ndownloader.figshare.com/files/62265733" in json.dumps(payload)
    assert "rmd17-npz-to-splits" in wrapper_text
    assert "converted_extxyz/rmd17_ethanol_train.extxyz" in wrapper_text
    assert "converted_extxyz/rmd17_ethanol_valid.extxyz" in wrapper_text
    assert "converted_extxyz/rmd17_ethanol_test.extxyz" in wrapper_text
    assert "tace.scripts.rtece_train_scalar" in wrapper_text
    assert "benchmark_rtece_scalar.py" in wrapper_text
    assert "--energy-key energy" in wrapper_text
    assert "--forces-key forces" in wrapper_text
    assert "--valid-limit-configs 1000" in wrapper_text
    assert "--limit-configs 1000" in wrapper_text
    assert "--measure-passes 5" in wrapper_text


def test_stage185_script_runs_as_direct_cli_entrypoint(tmp_path):
    import subprocess
    import sys

    script = Path("benchmarks/oc20neb_tace_mace/make_rtece_stage185_rmd17_representation_ladder.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--output-root",
            str(tmp_path / "stage185"),
            "--rmd17-root",
            str(tmp_path / "rMD17"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert (tmp_path / "stage185" / "stage185_manifest.json").exists()
    assert (tmp_path / "stage185" / "stage185_manifest_audit.json").exists()


def test_stage185_prep_wrapper_uses_sai_accepted_short_walltime(tmp_path):
    from pathlib import Path
    from benchmarks.oc20neb_tace_mace.make_rtece_stage185_rmd17_representation_ladder import (
        make_stage185_manifest,
        materialize_stage185,
    )

    payload = make_stage185_manifest(output_root=tmp_path / "stage185", rmd17_root=tmp_path / "rMD17")
    result = materialize_stage185(payload)
    prep_text = Path(result["prep_wrapper"]).read_text(encoding="utf-8")

    assert "#SBATCH --qos=flood-1o2gpu" in prep_text
    assert "#SBATCH --time=03:55:00" in prep_text
    assert "#SBATCH --time=05:55:00" not in prep_text


def test_stage185_prep_wrapper_is_compute_node_offline_and_points_to_login_download(tmp_path):
    from pathlib import Path
    from benchmarks.oc20neb_tace_mace.make_rtece_stage185_rmd17_representation_ladder import (
        make_stage185_manifest,
        materialize_stage185,
    )

    payload = make_stage185_manifest(output_root=tmp_path / "stage185", rmd17_root=tmp_path / "rMD17")
    result = materialize_stage185(payload)
    prep_text = Path(result["prep_wrapper"]).read_text(encoding="utf-8")

    assert "curl " not in prep_text
    assert "Missing local rMD17 source npz" in prep_text
    assert "https://ndownloader.figshare.com/files/62265733" in prep_text
    assert "Run this on a login node" in prep_text

