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
