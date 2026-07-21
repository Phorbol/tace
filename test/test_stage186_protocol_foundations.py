from __future__ import annotations

import hashlib
import json
import math

import pytest

from tace.models.rtece_protocol import (
    build_compatibility_tuple,
    build_stage186_operator_manifest,
    canonical_json_bytes,
    load_operator_manifest,
    sha256_file,
    sha256_json,
    validate_compatibility_tuple,
    validate_projection_operator_binding,
    write_canonical_json,
    write_operator_manifest,
)
from tace.models.rtece_scalar import build_rtece_config, build_rtece_config_from_path_ids


def _compatibility_kwargs() -> dict[str, str]:
    return {
        "operator_manifest_hash": "sha256:" + "1" * 64,
        "feature_schema_hash": "sha256:" + "2" * 64,
        "model_config_hash": "sha256:" + "3" * 64,
        "implementation_revision": "git:abc123",
        "teacher_checkpoint_hash": "sha256:" + "4" * 64,
        "teacher_config_hash": "sha256:" + "5" * 64,
        "data_manifest_hash": "sha256:" + "6" * 64,
    }


def _compatibility_payload() -> dict[str, str]:
    return build_compatibility_tuple(**_compatibility_kwargs())


def test_stage186_canonical_json_is_ascii_compact_and_rejects_nan():
    left = {"b": [2, 1], "a": {"text": "\u00e9"}}
    right = {"a": {"text": "\u00e9"}, "b": [2, 1]}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_json_bytes(left) == b'{"a":{"text":"\\u00e9"},"b":[2,1]}'
    assert sha256_json(left) == sha256_json(right)
    assert sha256_json(left).startswith("sha256:")
    assert len(sha256_json(left)) == 71
    with pytest.raises(ValueError):
        canonical_json_bytes({"value": math.nan})


def test_stage186_compatibility_tuple_rejects_blank_missing_and_extra_fields():
    payload = _compatibility_payload()

    assert validate_compatibility_tuple(payload) == payload
    with pytest.raises(ValueError, match="data_manifest_hash"):
        validate_compatibility_tuple(
            {key: value for key, value in payload.items() if key != "data_manifest_hash"}
        )
    with pytest.raises(ValueError, match="implementation_revision"):
        build_compatibility_tuple(
            **{**_compatibility_kwargs(), "implementation_revision": " "}
        )
    with pytest.raises(ValueError, match="implementation_revision"):
        validate_compatibility_tuple({**payload, "implementation_revision": " "})
    with pytest.raises(ValueError, match="unexpected_field"):
        validate_compatibility_tuple({**payload, "unexpected_field": "value"})


@pytest.mark.parametrize(
    "field",
    (
        "operator_manifest_hash",
        "feature_schema_hash",
        "model_config_hash",
        "teacher_checkpoint_hash",
        "teacher_config_hash",
        "data_manifest_hash",
    ),
)
def test_stage186_build_rejects_malformed_semantic_hashes(field):
    with pytest.raises(ValueError, match=field):
        build_compatibility_tuple(**{**_compatibility_kwargs(), field: "sha256:" + "a" * 63})


@pytest.mark.parametrize(
    "field",
    (
        "operator_manifest_hash",
        "feature_schema_hash",
        "model_config_hash",
        "teacher_checkpoint_hash",
        "teacher_config_hash",
        "data_manifest_hash",
        "compatibility_hash",
    ),
)
@pytest.mark.parametrize(
    "malformed_hash",
    ("sha256:" + "a" * 63, "sha256:" + "g" * 64, "SHA256:" + "a" * 64),
)
def test_stage186_validate_rejects_malformed_semantic_hashes(field, malformed_hash):
    with pytest.raises(ValueError, match=field):
        validate_compatibility_tuple({**_compatibility_payload(), field: malformed_hash})


def test_stage186_uppercase_semantic_hashes_are_preserved():
    uppercase_hash = "sha256:" + "A" * 64
    payload = build_compatibility_tuple(
        **{**_compatibility_kwargs(), "operator_manifest_hash": uppercase_hash}
    )

    assert payload["operator_manifest_hash"] == uppercase_hash
    assert validate_compatibility_tuple(payload)["operator_manifest_hash"] == uppercase_hash


def test_stage186_compatibility_tuple_rejects_direct_hash_tampering():
    payload = _compatibility_payload()

    with pytest.raises(ValueError, match="compatibility hash mismatch"):
        validate_compatibility_tuple(
            {**payload, "compatibility_hash": "sha256:" + "0" * 64}
        )


def test_stage186_compatibility_tuple_rejects_tampered_expected_hash():
    payload = _compatibility_payload()

    with pytest.raises(ValueError, match="compatibility hash mismatch"):
        validate_compatibility_tuple(
            payload,
            {**payload, "compatibility_hash": "sha256:" + "0" * 64},
        )


def test_stage186_compatibility_tuple_compares_every_expected_field():
    payload = _compatibility_payload()
    expected = build_compatibility_tuple(
        **{**_compatibility_kwargs(), "teacher_config_hash": "sha256:" + "9" * 64}
    )

    with pytest.raises(ValueError, match="compatibility mismatch"):
        validate_compatibility_tuple(payload, expected)


def test_stage186_file_hash_and_canonical_json_writer(tmp_path):
    path = tmp_path / "protocol.json"
    payload = {"b": [2], "a": 1}

    assert write_canonical_json(path, payload) == sha256_json(payload)
    assert path.read_text(encoding="ascii") == '{\n  "a": 1,\n  "b": [\n    2\n  ]\n}\n'
    assert sha256_file(path) == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads(path.read_text(encoding="ascii")) == payload


def test_stage186_projection_binds_exact_operator_order_and_hash(tmp_path):
    manifest = build_stage186_operator_manifest(
        build_rtece_config("rtece_atomic_moments"),
        retained_groups={"L0": ["atomic.radial_density"]},
        implementation_revision="git:test",
    )

    assert manifest.schema_version == "rtece_stage186_operator_manifest.v1"
    assert manifest.operator_manifest_hash.startswith("sha256:")
    path = tmp_path / "operator_manifest.json"
    assert write_operator_manifest(path, manifest) == manifest.operator_manifest_hash
    assert load_operator_manifest(path) == manifest

    payload = {
        "operator_manifest_schema": manifest.schema_version,
        "operator_manifest_hash": manifest.operator_manifest_hash,
        "operator_path_ids": list(manifest.operator_path_ids),
    }
    validate_projection_operator_binding(payload, manifest)
    payload["operator_path_ids"] = list(reversed(payload["operator_path_ids"]))
    with pytest.raises(ValueError, match="operator path order"):
        validate_projection_operator_binding(payload, manifest)


def test_stage186_operator_manifest_rejects_invalid_groups_and_hash_tampering(tmp_path):
    config = build_rtece_config("rtece_atomic_moments")
    manifest = build_stage186_operator_manifest(
        config,
        retained_groups={
            "L0": ["atomic.radial_density"],
            "L2": [
                "atomic.radial_density",
                "atomic.vector_norm",
                "atomic.quadrupole_norm",
            ],
        },
        implementation_revision="git:test",
    )
    with pytest.raises(ValueError, match="unknown retained operator"):
        build_stage186_operator_manifest(
            config,
            retained_groups={"L0": ["atomic.not_real"]},
            implementation_revision="git:test",
        )
    with pytest.raises(ValueError, match="nested"):
        build_stage186_operator_manifest(
            config,
            retained_groups={
                "L0": ["atomic.radial_density", "atomic.vector_norm"],
                "L2": ["atomic.radial_density"],
            },
            implementation_revision="git:test",
        )
    duplicate_config = build_rtece_config_from_path_ids(
        "duplicate_paths",
        ["atomic.radial_density", "atomic.radial_density"],
    )
    with pytest.raises(ValueError, match="duplicate operator path"):
        build_stage186_operator_manifest(
            duplicate_config,
            retained_groups={"L0": ["atomic.radial_density"]},
            implementation_revision="git:test",
        )

    path = tmp_path / "tampered_operator_manifest.json"
    write_operator_manifest(path, manifest)
    payload = json.loads(path.read_text(encoding="ascii"))
    payload["implementation_revision"] = "git:tampered"
    path.write_text(json.dumps(payload), encoding="ascii")
    with pytest.raises(ValueError, match="operator manifest hash mismatch"):
        load_operator_manifest(path)


def test_stage186_projection_cli_emits_operator_binding(tmp_path):
    import subprocess
    import sys

    import ase.io
    from ase import Atoms
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    configs = tmp_path / "h2_stage186.xyz"
    output = tmp_path / "projection_stage186.json"
    manifest_path = tmp_path / "operator_manifest.json"
    atoms_list = [
        Atoms("H2", positions=[[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
        for distance in (0.70, 0.80, 0.90, 1.00)
    ]
    ase.io.write(configs, atoms_list, format="extxyz")

    path_ids = ("atomic.radial_density", "edge.direct.radial")
    config = build_rtece_config_from_path_ids(
        "stage186_cli_reference",
        path_ids,
        num_radial=3,
    )
    manifest = build_stage186_operator_manifest(
        config,
        retained_groups={
            "L0": ["atomic.radial_density"],
            "L1": list(path_ids),
        },
        implementation_revision="git:test",
    )
    write_operator_manifest(manifest_path, manifest)

    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/oc20neb_tace_mace/analyze_rtece_projection_error.py",
            "--configs",
            str(configs),
            "--output-json",
            str(output),
            "--operator-manifest",
            str(manifest_path),
            "--reference-path-ids",
            ",".join(path_ids),
            "--candidate",
            "baseline:atomic.radial_density",
            "--candidate",
            "full_reference:atomic.radial_density,edge.direct.radial",
            "--num-radial",
            "3",
            "--limit-configs",
            "4",
            "--neighborlist-backend",
            "ase",
            "--active-set-baseline-candidate",
            "baseline",
            "--active-set-projection-weight",
            "1.0",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text(encoding="ascii"))
    assert payload["schema_version"] == "rtece_projection_diagnostic.v1"
    assert payload["operator_manifest_schema"] == manifest.schema_version
    assert payload["operator_manifest_hash"] == manifest.operator_manifest_hash
    assert payload["operator_path_ids"] == list(manifest.operator_path_ids)
    assert payload["active_set_rows"]
    assert all(
        row["operator_manifest_hash"] == manifest.operator_manifest_hash
        and row["operator_path_ids"] == list(manifest.operator_path_ids)
        for row in payload["active_set_rows"]
    )


    wrong_num_radial_command = list(result.args)
    wrong_num_radial_command[
        wrong_num_radial_command.index("--num-radial") + 1
    ] = "4"
    wrong_num_radial = subprocess.run(
        wrong_num_radial_command,
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert wrong_num_radial.returncode != 0
    assert "legacy path manifest" in wrong_num_radial.stderr

    wrong_dtype = subprocess.run(
        [*result.args, "--default-dtype", "float32"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert wrong_dtype.returncode != 0
    assert "feature dtype" in wrong_dtype.stderr


def test_stage186_protocol_cli_writes_loadable_operator_manifest(tmp_path):
    import subprocess
    import sys

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    output = tmp_path / "operator_manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            "benchmarks/oc20neb_tace_mace/make_rtece_stage186_protocol.py",
            "--output-json",
            str(output),
            "--reference-path-ids",
            "atomic.radial_density,edge.direct.radial",
            "--retained-group",
            "L0:atomic.radial_density",
            "--retained-group",
            "L1:atomic.radial_density,edge.direct.radial",
            "--implementation-revision",
            "git:test",
            "--num-radial",
            "3",
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stderr
    manifest = load_operator_manifest(output)
    assert manifest.operator_path_ids == (
        "atomic.radial_density",
        "edge.direct.radial",
    )
    assert manifest.retained_groups == (
        ("L0", ("atomic.radial_density",)),
        ("L1", ("atomic.radial_density", "edge.direct.radial")),
    )
