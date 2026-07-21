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
    assert set(wrappers) == {"train_3bpa_300k", "benchmark_3bpa"}
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
