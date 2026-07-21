from __future__ import annotations

import hashlib
import json
import math

import pytest

from tace.models.rtece_protocol import (
    build_compatibility_tuple,
    canonical_json_bytes,
    sha256_file,
    sha256_json,
    validate_compatibility_tuple,
    write_canonical_json,
)


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
