from __future__ import annotations

import hashlib
import json

import pytest

from tace.models.rtece_protocol import (
    build_compatibility_tuple,
    canonical_json_bytes,
    sha256_file,
    sha256_json,
    validate_compatibility_tuple,
    write_canonical_json,
)


def test_stage186_canonical_json_and_hash_ignore_mapping_order():
    left = {"b": [2, 1], "a": {"x": 1.0}}
    right = {"a": {"x": 1.0}, "b": [2, 1]}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert sha256_json(left) == sha256_json(right)
    assert sha256_json(left).startswith("sha256:")
    assert len(sha256_json(left)) == 71


def test_stage186_compatibility_tuple_rejects_missing_or_tampered_fields():
    payload = build_compatibility_tuple(
        operator_manifest_hash="sha256:" + "1" * 64,
        feature_schema_hash="sha256:" + "2" * 64,
        model_config_hash="sha256:" + "3" * 64,
        implementation_revision="git:abc123",
        teacher_checkpoint_hash="sha256:" + "4" * 64,
        teacher_config_hash="sha256:" + "5" * 64,
        data_manifest_hash="sha256:" + "6" * 64,
    )

    assert validate_compatibility_tuple(payload) == payload
    with pytest.raises(ValueError, match="data_manifest_hash"):
        validate_compatibility_tuple(
            {key: value for key, value in payload.items() if key != "data_manifest_hash"}
        )
    with pytest.raises(ValueError, match="compatibility mismatch"):
        validate_compatibility_tuple(
            payload,
            {**payload, "teacher_config_hash": "sha256:" + "9" * 64},
        )


def test_stage186_file_hash_and_canonical_json_writer(tmp_path):
    path = tmp_path / "protocol.json"
    payload = {"b": [2], "a": 1}

    assert write_canonical_json(path, payload) == sha256_json(payload)
    assert path.read_text(encoding="ascii") == '{\n  "a": 1,\n  "b": [\n    2\n  ]\n}\n'
    assert sha256_file(path) == "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    assert json.loads(path.read_text(encoding="ascii")) == payload
