from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Mapping


STAGE186_COMPATIBILITY_SCHEMA = "rtece_stage186_compatibility.v1"
COMPATIBILITY_FIELDS = (
    "operator_manifest_hash",
    "feature_schema_hash",
    "model_config_hash",
    "implementation_revision",
    "teacher_checkpoint_hash",
    "teacher_config_hash",
    "data_manifest_hash",
)

_COMPATIBILITY_HASH_FIELD = "compatibility_hash"
_COMPATIBILITY_SCHEMA_FIELD = "schema_version"
_COMPATIBILITY_KEYS = frozenset(
    (*COMPATIBILITY_FIELDS, _COMPATIBILITY_HASH_FIELD, _COMPATIBILITY_SCHEMA_FIELD)
)
_SEMANTIC_HASH_FIELDS = frozenset(
    field for field in COMPATIBILITY_FIELDS if field.endswith("_hash")
)
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-fA-F]{64}")


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def sha256_json(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def build_compatibility_tuple(
    *,
    operator_manifest_hash: str,
    feature_schema_hash: str,
    model_config_hash: str,
    implementation_revision: str,
    teacher_checkpoint_hash: str,
    teacher_config_hash: str,
    data_manifest_hash: str,
) -> dict[str, str]:
    values = {
        "operator_manifest_hash": operator_manifest_hash,
        "feature_schema_hash": feature_schema_hash,
        "model_config_hash": model_config_hash,
        "implementation_revision": implementation_revision,
        "teacher_checkpoint_hash": teacher_checkpoint_hash,
        "teacher_config_hash": teacher_config_hash,
        "data_manifest_hash": data_manifest_hash,
    }
    _validate_compatibility_values(values)
    payload = {
        _COMPATIBILITY_SCHEMA_FIELD: STAGE186_COMPATIBILITY_SCHEMA,
        **values,
    }
    payload[_COMPATIBILITY_HASH_FIELD] = sha256_json(payload)
    return payload


def validate_compatibility_tuple(
    payload: Mapping[str, object],
    expected: Mapping[str, object] | None = None,
) -> dict[str, str]:
    validated = _validate_compatibility_payload(payload)
    if expected is not None:
        expected_values = _validate_compatibility_payload(expected)
        for field in (
            _COMPATIBILITY_SCHEMA_FIELD,
            *COMPATIBILITY_FIELDS,
            _COMPATIBILITY_HASH_FIELD,
        ):
            if validated[field] != expected_values[field]:
                raise ValueError(f"compatibility mismatch for {field}")
    return validated


def write_canonical_json(path: str | Path, payload: object) -> str:
    Path(path).write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="ascii",
    )
    return sha256_json(payload)


def _validate_compatibility_payload(
    payload: Mapping[str, object],
    *,
    verify_hash: bool = True,
) -> dict[str, str]:
    if not isinstance(payload, Mapping):
        raise ValueError("compatibility payload must be a mapping")

    payload_keys = set(payload)
    missing = _COMPATIBILITY_KEYS - payload_keys
    if missing:
        raise ValueError(f"missing compatibility field: {sorted(missing)[0]}")
    extra = payload_keys - _COMPATIBILITY_KEYS
    if extra:
        raise ValueError(f"unexpected compatibility field: {sorted(extra)[0]}")

    schema_version = _require_nonblank_string(
        payload[_COMPATIBILITY_SCHEMA_FIELD], _COMPATIBILITY_SCHEMA_FIELD
    )
    if schema_version != STAGE186_COMPATIBILITY_SCHEMA:
        raise ValueError(f"unsupported compatibility schema: {schema_version}")

    values = {field: payload[field] for field in COMPATIBILITY_FIELDS}
    _validate_compatibility_values(values)
    compatibility_hash = _require_sha256_string(
        payload[_COMPATIBILITY_HASH_FIELD], _COMPATIBILITY_HASH_FIELD
    )
    validated = {
        _COMPATIBILITY_SCHEMA_FIELD: schema_version,
        **{field: values[field] for field in COMPATIBILITY_FIELDS},
        _COMPATIBILITY_HASH_FIELD: compatibility_hash,
    }
    if verify_hash:
        expected_hash = sha256_json(
            {
                _COMPATIBILITY_SCHEMA_FIELD: schema_version,
                **{field: values[field] for field in COMPATIBILITY_FIELDS},
            }
        )
        if compatibility_hash != expected_hash:
            raise ValueError("compatibility hash mismatch")
    return validated


def _validate_compatibility_values(values: Mapping[str, object]) -> None:
    for field in COMPATIBILITY_FIELDS:
        if field in _SEMANTIC_HASH_FIELDS:
            _require_sha256_string(values[field], field)
        else:
            _require_nonblank_string(values[field], field)


def _require_sha256_string(value: object, field: str) -> str:
    value = _require_nonblank_string(value, field)
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field} must be a sha256:<64 hex digits> string")
    return value


def _require_nonblank_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonblank string")
    return value


__all__ = [
    "COMPATIBILITY_FIELDS",
    "STAGE186_COMPATIBILITY_SCHEMA",
    "build_compatibility_tuple",
    "canonical_json_bytes",
    "sha256_file",
    "sha256_json",
    "validate_compatibility_tuple",
    "write_canonical_json",
]
