from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms


ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "benchmarks" / "oc20neb_tace_mace"


def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, BENCH / relative_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_copy_atoms_with_teacher_labels_preserves_reference_labels():
    distill = load_module("distill_tace_labels", "distill_tace_labels.py")
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    atoms.info["energy"] = -1.25
    atoms.arrays["forces"] = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])

    labeled = distill.copy_atoms_with_teacher_labels(
        [atoms],
        energies=np.array([-1.5]),
        forces=np.array([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]]),
        target_energy_key="energy",
        target_forces_key="forces",
        reference_prefix="dft_",
    )

    assert len(labeled) == 1
    assert labeled[0] is not atoms
    assert labeled[0].info["energy"] == pytest.approx(-1.5)
    assert labeled[0].arrays["forces"] == pytest.approx(
        np.array([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]])
    )
    assert labeled[0].info["dft_energy"] == pytest.approx(-1.25)
    assert labeled[0].arrays["dft_forces"] == pytest.approx(
        np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    )


def test_copy_atoms_with_teacher_labels_rejects_force_length_mismatch():
    distill = load_module("distill_tace_labels", "distill_tace_labels.py")
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    atoms.info["energy"] = -1.25
    atoms.arrays["forces"] = np.zeros((2, 3))

    with pytest.raises(ValueError, match="forces rows"):
        distill.copy_atoms_with_teacher_labels(
            [atoms],
            energies=np.array([-1.5]),
            forces=np.zeros((1, 3)),
            target_energy_key="energy",
            target_forces_key="forces",
            reference_prefix="dft_",
        )



def test_copy_atoms_with_teacher_labels_preserves_calculator_results():
    from ase.calculators.singlepoint import SinglePointCalculator

    distill = load_module("distill_tace_labels", "distill_tace_labels.py")
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    dft_forces = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(atoms, energy=-1.25, forces=dft_forces)

    labeled = distill.copy_atoms_with_teacher_labels(
        [atoms],
        energies=np.array([-1.5]),
        forces=np.array([[0.2, 0.0, 0.0], [-0.2, 0.0, 0.0]]),
        target_energy_key="energy",
        target_forces_key="forces",
        reference_prefix="dft_",
    )

    assert labeled[0].info["dft_energy"] == pytest.approx(-1.25)
    assert labeled[0].arrays["dft_forces"] == pytest.approx(dft_forces)

def test_mix_teacher_and_dft_labels_preserves_sources_and_writes_targets():
    mix = load_module("mix_tece_distill_labels", "mix_tece_distill_labels.py")
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    atoms.info["energy"] = -1.5
    atoms.info["dft_energy"] = -1.0
    atoms.arrays["forces"] = np.array([[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0]])
    atoms.arrays["dft_forces"] = np.array([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]])

    mixed = mix.mix_atoms_labels(
        [atoms],
        teacher_weight=0.75,
        target_energy_key="energy",
        target_forces_key="forces",
        dft_energy_key="dft_energy",
        dft_forces_key="dft_forces",
        teacher_prefix="teacher_",
    )

    assert mixed[0].info["energy"] == pytest.approx(0.75 * -1.5 + 0.25 * -1.0)
    assert mixed[0].arrays["forces"] == pytest.approx(
        0.75 * atoms.arrays["forces"] + 0.25 * atoms.arrays["dft_forces"]
    )
    assert mixed[0].info["teacher_energy"] == pytest.approx(-1.5)
    assert mixed[0].arrays["teacher_forces"] == pytest.approx(atoms.arrays["forces"])
    assert mixed[0].info["dft_energy"] == pytest.approx(-1.0)
    assert mixed[0].arrays["dft_forces"] == pytest.approx(atoms.arrays["dft_forces"])


def test_mix_teacher_and_dft_labels_rejects_bad_weight():
    mix = load_module("mix_tece_distill_labels", "mix_tece_distill_labels.py")
    with pytest.raises(ValueError, match="teacher_weight"):
        mix.mix_atoms_labels([], teacher_weight=1.5)

def test_mix_teacher_and_dft_labels_reads_teacher_from_calculator_results():
    from ase.calculators.singlepoint import SinglePointCalculator

    mix = load_module("mix_tece_distill_labels", "mix_tece_distill_labels.py")
    atoms = Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]])
    teacher_forces = np.array([[0.4, 0.0, 0.0], [-0.4, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(atoms, energy=-1.5, forces=teacher_forces)
    atoms.info["dft_energy"] = -1.0
    atoms.arrays["dft_forces"] = np.array([[0.0, 0.2, 0.0], [0.0, -0.2, 0.0]])

    mixed = mix.mix_atoms_labels([atoms], teacher_weight=0.5)

    assert mixed[0].info["energy"] == pytest.approx(-1.25)
    assert mixed[0].arrays["forces"] == pytest.approx(
        0.5 * teacher_forces + 0.5 * atoms.arrays["dft_forces"]
    )
    assert mixed[0].info["teacher_energy"] == pytest.approx(-1.5)
    assert mixed[0].arrays["teacher_forces"] == pytest.approx(teacher_forces)


def test_concat_extxyz_datasets_preserves_order_and_records_source(tmp_path):
    import ase.io

    concat = load_module("concat_extxyz_datasets", "concat_extxyz_datasets.py")
    base = Atoms("H", positions=[[0.0, 0.0, 0.0]])
    base.info["energy"] = -1.0
    base.arrays["forces"] = np.zeros((1, 3))
    rattled = Atoms("He", positions=[[0.1, 0.0, 0.0]])
    rattled.info["energy"] = -2.0
    rattled.arrays["forces"] = np.ones((1, 3))
    base_path = tmp_path / "base.extxyz"
    rattle_path = tmp_path / "teacher_rattle.extxyz"
    output = tmp_path / "augmented.extxyz"
    summary_path = tmp_path / "summary.json"
    ase.io.write(base_path, [base], format="extxyz")
    ase.io.write(rattle_path, [rattled], format="extxyz")

    summary = concat.concat_extxyz_datasets(
        inputs=[base_path, rattle_path],
        output=output,
        summary_path=summary_path,
        source_labels=["base_mixed", "teacher_rattle"],
    )

    merged = ase.io.read(output, index=":")
    assert [atoms.get_chemical_formula() for atoms in merged] == ["H", "He"]
    assert merged[0].info["rtece_concat_source"] == "base_mixed"
    assert merged[1].info["rtece_concat_source"] == "teacher_rattle"
    assert summary["schema_version"] == "rtece_concat_extxyz.v1"
    assert summary["configs"] == 2
    assert summary["sources"][1]["label"] == "teacher_rattle"
    assert summary_path.exists()


def test_stage117_rattle_distill_manifest_defines_teacher_fake_label_pipeline(tmp_path):
    stage117 = load_module("make_rtece_stage117_rattle_distill_plan", "make_rtece_stage117_rattle_distill_plan.py")

    manifest = stage117.make_stage117_manifest(
        output_root=tmp_path / "stage117",
        base_train="mixed_train_tw0.75.extxyz",
        teacher_model="teacher.pt",
        valid_file="teacher_valid.extxyz",
        dft_valid_file="mixed_valid_tw0.75_regen.extxyz",
        source_limit_configs=128,
        copies_per_config=2,
        rattle_std_a=0.05,
        seed=20260719,
        row_set="capacity-ladder-stage116",
        lr_warmup_steps=500,
    )

    assert manifest["schema_version"] == "rtece_stage117_rattle_distill_plan.v1"
    assert manifest["distillation_semantics"] == "teacher_fake_labels_on_rattled_geometries"
    assert manifest["rattle_label_policy"] == "pure_teacher_targets_preserve_source_labels_as_metadata"
    assert manifest["source_limit_configs"] == 128
    assert manifest["copies_per_config"] == 2
    assert manifest["row_set"] == "capacity-ladder-stage116"
    assert manifest["lr_warmup_steps"] == 500
    assert manifest["artifacts"]["rattled_configs"].endswith("rattled_source128_copies2_std0p05.extxyz")
    assert manifest["artifacts"]["teacher_labeled_rattles"].endswith("teacher_labeled_rattles.extxyz")
    assert manifest["artifacts"]["augmented_train"].endswith("augmented_train_base_plus_teacher_rattles.extxyz")
    commands = "\n".join(step["command"] for step in manifest["steps"])
    assert "make_rattle_distill_configs.py" in commands
    assert "distill_tace_labels.py" in commands
    assert "concat_extxyz_datasets.py" in commands
    assert "make_rtece_pareto_sweep.py" in commands
    assert "--lr-warmup-steps 500" in commands
    assert "mix_tece_distill_labels.py" not in commands
    assert "--export" not in commands


def test_stage117_manifest_audit_rejects_missing_warmup_contract(tmp_path):
    stage117 = load_module("make_rtece_stage117_rattle_distill_plan", "make_rtece_stage117_rattle_distill_plan.py")
    manifest = stage117.make_stage117_manifest(
        output_root=tmp_path / "stage117",
        base_train="mixed_train_tw0.75.extxyz",
        teacher_model="teacher.pt",
        valid_file="teacher_valid.extxyz",
        dft_valid_file="mixed_valid_tw0.75_regen.extxyz",
        source_limit_configs=128,
        copies_per_config=1,
        rattle_std_a=0.05,
        seed=20260719,
        row_set="capacity-ladder-stage116",
        lr_warmup_steps=500,
    )
    for step in manifest["steps"]:
        if step["id"] == "make_capacity_ladder_wrappers":
            step["command"] = step["command"].replace(" --lr-warmup-steps 500", "")

    audit = stage117.audit_stage117_manifest(manifest)

    assert audit["contract_pass"] is False
    assert "lr_warmup_contract" in audit["failed_checks"]


def test_stage117_manifest_audit_rejects_mixed_rattle_labels(tmp_path):
    stage117 = load_module("make_rtece_stage117_rattle_distill_plan", "make_rtece_stage117_rattle_distill_plan.py")
    manifest = stage117.make_stage117_manifest(
        output_root=tmp_path / "stage117",
        base_train="mixed_train_tw0.75.extxyz",
        teacher_model="teacher.pt",
        valid_file="teacher_valid.extxyz",
        dft_valid_file="mixed_valid_tw0.75_regen.extxyz",
        source_limit_configs=128,
        copies_per_config=1,
        rattle_std_a=0.05,
        seed=20260719,
        row_set="capacity-ladder-stage116",
        lr_warmup_steps=500,
    )
    manifest["steps"][1]["command"] += " && python benchmarks/oc20neb_tace_mace/mix_tece_distill_labels.py"
    manifest["rattle_label_policy"] = "mixed_teacher_and_source_labels"

    audit = stage117.audit_stage117_manifest(manifest)

    assert audit["contract_pass"] is False
    assert "rattle_label_policy" in audit["failed_checks"]
    assert "no_mix_tece_distill_labels" in audit["failed_checks"]


def test_stage117_manifest_audit_accepts_clean_teacher_fake_label_contract(tmp_path):
    stage117 = load_module("make_rtece_stage117_rattle_distill_plan", "make_rtece_stage117_rattle_distill_plan.py")
    manifest = stage117.make_stage117_manifest(
        output_root=tmp_path / "stage117",
        base_train="mixed_train_tw0.75.extxyz",
        teacher_model="teacher.pt",
        valid_file="teacher_valid.extxyz",
        dft_valid_file="mixed_valid_tw0.75_regen.extxyz",
        source_limit_configs=128,
        copies_per_config=2,
        rattle_std_a=0.05,
        seed=20260719,
        row_set="capacity-ladder-stage116",
        lr_warmup_steps=500,
    )

    audit = stage117.audit_stage117_manifest(manifest)

    assert audit["schema_version"] == "rtece_stage117_manifest_audit.v1"
    assert audit["contract_pass"] is True
    assert audit["failed_checks"] == []
    assert audit["checked"]["augmented_limit_configs"] == 384
    assert audit["checked"]["lr_warmup_steps"] == 500
    assert audit["checks"]["lr_warmup_contract"] is True
    assert audit["checked"]["queue_policy"] == "manifest_only_no_sbatch_submission"


def minimal_base_config():
    return {
        "misc": {"project_name": "oc20neb_fullcase200_tace"},
        "dataset": {
            "train_file": "old_train.extxyz",
            "valid_file": "old_valid.extxyz",
            "train_dataloader": {"batch_size": 8},
            "valid_dataloader": {"batch_size": 16},
        },
        "model": {
            "config": {
                "mmax": 3,
                "Lmax": 2,
                "lmax": 3,
                "num_layers": 3,
                "num_channel": 48,
                "edge_embedding": {"type": "nonlinear"},
                "edge_update": {"type": "element2"},
                "radial_basis": {"num_radial_basis": 8, "hidden": [64, 64]},
                "atomic_basis": {
                    "type": "cgtp",
                    "nonlinear": "sigmoid_gate",
                    "edge_nonlinear": "so2_sigmoid_gate",
                    "edge_info_type": "mlp",
                },
                "product_basis": {"type": "cgtp", "correlation": 2},
                "readout_emlp": {"hidden": [16]},
            }
        },
    }


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("scalar_fast", {"Lmax": 0, "lmax": 1, "mmax": 0, "edge_update": "identity"}),
        ("edge_min", {"Lmax": 1, "lmax": 2, "mmax": 1, "edge_update": "element2"}),
        ("edge_min_mixed", {"Lmax": 1, "lmax": 2, "mmax": 1, "edge_update": "element2"}),
        ("edge_radial_mixed", {"Lmax": 1, "lmax": 2, "mmax": 1, "edge_update": "element2"}),
        ("edge_radial_active6_mixed", {"Lmax": 1, "lmax": 2, "mmax": 1, "edge_update": "element2"}),
        ("edge_radial_active4_mixed", {"Lmax": 1, "lmax": 2, "mmax": 1, "edge_update": "element2"}),
        ("edge_readout_mixed", {"Lmax": 1, "lmax": 2, "mmax": 1, "edge_update": "element2"}),
        ("edge_radial_slim_mixed", {"Lmax": 1, "lmax": 2, "mmax": 1, "edge_update": "element2"}),
        ("path_scalar", {"Lmax": 1, "lmax": 2, "mmax": 1, "edge_update": "identity"}),
    ],
)
def test_build_variant_config_applies_tece_projection_axes(variant, expected):
    variants = load_module("make_tece_student_configs", "make_tece_student_configs.py")

    cfg = variants.build_variant_config(
        minimal_base_config(),
        variant,
        train_file="teacher_train.extxyz",
        valid_file="teacher_valid.extxyz",
    )

    model_cfg = cfg["model"]["config"]
    assert cfg["dataset"]["train_file"] == "teacher_train.extxyz"
    assert cfg["dataset"]["valid_file"] == "teacher_valid.extxyz"
    assert cfg["misc"]["project_name"] == f"oc20neb_fullcase200_{variant}"
    assert model_cfg["Lmax"] == expected["Lmax"]
    assert model_cfg["lmax"] == expected["lmax"]
    assert model_cfg["mmax"] == expected["mmax"]
    assert model_cfg["edge_update"]["type"] == expected["edge_update"]
    if variant == "edge_radial_mixed":
        assert model_cfg["num_channel"] == 32
        assert model_cfg["radial_basis"]["hidden"] == [64, 64]
        assert model_cfg["readout_emlp"]["hidden"] == [16]
        assert "active_indices" not in model_cfg["radial_basis"]
    elif variant == "edge_radial_active6_mixed":
        assert model_cfg["num_channel"] == 32
        assert model_cfg["radial_basis"]["hidden"] == [64, 64]
        assert model_cfg["radial_basis"]["active_indices"] == [0, 1, 2, 3, 4, 5]
        assert model_cfg["readout_emlp"]["hidden"] == [16]
    elif variant == "edge_radial_active4_mixed":
        assert model_cfg["num_channel"] == 32
        assert model_cfg["radial_basis"]["hidden"] == [64, 64]
        assert model_cfg["radial_basis"]["active_indices"] == [0, 1, 2, 3]
        assert model_cfg["readout_emlp"]["hidden"] == [16]
    elif variant == "edge_readout_mixed":
        assert model_cfg["num_channel"] == 32
        assert model_cfg["radial_basis"]["hidden"] == [48, 48]
        assert model_cfg["readout_emlp"]["hidden"] == [16]
    elif variant == "edge_radial_slim_mixed":
        assert model_cfg["num_channel"] == 32
        assert model_cfg["radial_basis"]["hidden"] == [64, 32]
        assert model_cfg["readout_emlp"]["hidden"] == [16]


def test_build_variant_config_rejects_unknown_variant():
    variants = load_module("make_tece_student_configs", "make_tece_student_configs.py")

    with pytest.raises(ValueError, match="unknown variant"):
        variants.build_variant_config(
            minimal_base_config(),
            "random_tiny",
            train_file="teacher_train.extxyz",
            valid_file="teacher_valid.extxyz",
        )



def test_normalize_radial_active_indices_defaults_to_full_basis():
    from tace.models._e3nn.representation import normalize_radial_active_indices

    assert normalize_radial_active_indices(4, None) == (0, 1, 2, 3)


def test_normalize_radial_active_indices_validates_compact_subset():
    from tace.models._e3nn.representation import normalize_radial_active_indices

    assert normalize_radial_active_indices(8, [0, 2, 5]) == (0, 2, 5)
    with pytest.raises(ValueError, match="duplicate"):
        normalize_radial_active_indices(8, [0, 2, 2])
    with pytest.raises(ValueError, match="out of range"):
        normalize_radial_active_indices(8, [0, 8])
    with pytest.raises(ValueError, match="non-empty"):
        normalize_radial_active_indices(8, [])

def test_distill_summary_orders_students_by_throughput_and_keeps_error_axes():
    summary = load_module("summarize_tece_distill", "summarize_tece_distill.py")
    rows = [
        summary.make_student_row(
            "edge_min",
            dft_benchmark={"atoms_per_second": 120.0, "mae_f_mev_a": 32.0, "mae_e_mev_atom": 5.0},
            teacher_benchmark={"mae_f_mev_a": 12.0, "mae_e_mev_atom": 2.0},
        ),
        summary.make_student_row(
            "scalar_fast",
            dft_benchmark={"atoms_per_second": 240.0, "mae_f_mev_a": 45.0, "mae_e_mev_atom": 8.0},
            teacher_benchmark={"mae_f_mev_a": 18.0, "mae_e_mev_atom": 3.0},
        ),
    ]

    ranked = summary.rank_student_rows(rows)
    markdown = summary.format_markdown(ranked, baselines=[])

    assert [row["variant"] for row in ranked] == ["scalar_fast", "edge_min"]
    assert "teacher F MAE" in markdown
    assert "DFT F MAE" in markdown
    assert markdown.index("scalar_fast") < markdown.index("edge_min")
