from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
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

STAGE186_OPERATOR_MANIFEST_SCHEMA = "rtece_stage186_operator_manifest.v1"
_OPERATOR_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "legacy_path_manifest_schema",
        "legacy_path_manifest_hash",
        "operator_path_ids",
        "retained_groups",
        "feature_schema",
        "implementation_revision",
        "operator_manifest_hash",
    }
)
_LEGACY_MANIFEST_HASH_PATTERN = re.compile(r"[0-9a-fA-F]{16}")

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-fA-F]{64}")

@dataclass(frozen=True)
class OperatorEntry:
    index: int
    path_id: str


@dataclass(frozen=True)
class OperatorManifest:
    schema_version: str
    legacy_path_manifest_schema: str
    legacy_path_manifest_hash: str
    operators: tuple[OperatorEntry, ...]
    retained_groups: tuple[tuple[str, tuple[str, ...]], ...]
    feature_dtype: str
    feature_distance_unit: str
    implementation_revision: str
    operator_manifest_hash: str

    @property
    def operator_path_ids(self) -> tuple[str, ...]:
        return tuple(entry.path_id for entry in self.operators)

    @property
    def feature_schema(self) -> dict[str, object]:
        return {
            "units": {"distance": self.feature_distance_unit},
            "dtype": self.feature_dtype,
        }


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



def build_stage186_operator_manifest(
    config: object,
    *,
    retained_groups: Mapping[str, list[str] | tuple[str, ...]],
    implementation_revision: str,
) -> OperatorManifest:
    from tace.models.rtece_scalar import rtece_path_manifest

    legacy = rtece_path_manifest(config)
    operator_path_ids = [str(row["id"]) for row in legacy["scalar_paths"]]
    payload: dict[str, object] = {
        "schema_version": STAGE186_OPERATOR_MANIFEST_SCHEMA,
        "legacy_path_manifest_schema": str(legacy["schema_version"]),
        "legacy_path_manifest_hash": str(legacy["manifest_hash"]),
        "operator_path_ids": operator_path_ids,
        "retained_groups": {
            str(name): list(path_ids) for name, path_ids in retained_groups.items()
        },
        "feature_schema": {"units": {"distance": "A"}, "dtype": "float64"},
        "implementation_revision": implementation_revision,
    }
    payload["operator_manifest_hash"] = sha256_json(payload)
    return _operator_manifest_from_payload(payload)


def write_operator_manifest(path: str | Path, manifest: OperatorManifest) -> str:
    payload = _operator_manifest_to_payload(manifest)
    validated = _operator_manifest_from_payload(payload)
    if validated != manifest:
        raise ValueError("operator manifest dataclass is not canonical")
    write_canonical_json(path, payload)
    return manifest.operator_manifest_hash


def load_operator_manifest(path: str | Path) -> OperatorManifest:
    try:
        payload = json.loads(Path(path).read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid operator manifest JSON: {path}") from exc
    return _operator_manifest_from_payload(payload)


def validate_projection_operator_binding(
    payload: Mapping[str, object],
    operator_manifest: OperatorManifest,
) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("projection payload must be a mapping")
    schema = _require_nonblank_string(
        payload.get("operator_manifest_schema"), "operator_manifest_schema"
    )
    if schema != operator_manifest.schema_version:
        raise ValueError("operator manifest schema mismatch")
    manifest_hash = _require_sha256_string(
        payload.get("operator_manifest_hash"), "operator_manifest_hash"
    )
    if manifest_hash != operator_manifest.operator_manifest_hash:
        raise ValueError("operator manifest hash mismatch")
    operator_path_ids = _normalize_operator_path_ids(
        payload.get("operator_path_ids"), field="operator_path_ids"
    )
    if operator_path_ids != operator_manifest.operator_path_ids:
        raise ValueError("operator path order mismatch")
    reference_path_ids = payload.get("reference_path_ids")
    if reference_path_ids is not None:
        normalized_reference = _normalize_operator_path_ids(
            reference_path_ids, field="reference_path_ids"
        )
        if normalized_reference != operator_manifest.operator_path_ids:
            raise ValueError("reference operator path order mismatch")

    for collection_name in ("rows", "active_set_rows"):
        rows = payload.get(collection_name)
        if rows is None:
            continue
        if not isinstance(rows, list):
            raise ValueError(f"{collection_name} must be a list")
        for row_index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ValueError(f"{collection_name}[{row_index}] must be a mapping")
            candidate_paths = _projection_row_path_ids(row)
            if candidate_paths is not None:
                _validate_ordered_operator_subset(
                    candidate_paths,
                    operator_manifest.operator_path_ids,
                    field=f"{collection_name}[{row_index}] operator paths",
                )
            if collection_name == "active_set_rows":
                _validate_row_operator_binding(row, operator_manifest, row_index)


def _operator_manifest_to_payload(manifest: OperatorManifest) -> dict[str, object]:
    return {
        "schema_version": manifest.schema_version,
        "legacy_path_manifest_schema": manifest.legacy_path_manifest_schema,
        "legacy_path_manifest_hash": manifest.legacy_path_manifest_hash,
        "operator_path_ids": list(manifest.operator_path_ids),
        "retained_groups": {
            name: list(path_ids) for name, path_ids in manifest.retained_groups
        },
        "feature_schema": manifest.feature_schema,
        "implementation_revision": manifest.implementation_revision,
        "operator_manifest_hash": manifest.operator_manifest_hash,
    }


def _operator_manifest_from_payload(payload: object) -> OperatorManifest:
    if not isinstance(payload, Mapping):
        raise ValueError("operator manifest payload must be a mapping")
    keys = set(payload)
    missing = _OPERATOR_MANIFEST_KEYS - keys
    if missing:
        raise ValueError(f"missing operator manifest field: {sorted(missing)[0]}")
    extra = keys - _OPERATOR_MANIFEST_KEYS
    if extra:
        raise ValueError(f"unexpected operator manifest field: {sorted(extra)[0]}")

    schema = _require_nonblank_string(payload["schema_version"], "schema_version")
    if schema != STAGE186_OPERATOR_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported operator manifest schema: {schema}")
    legacy_schema = _require_nonblank_string(
        payload["legacy_path_manifest_schema"], "legacy_path_manifest_schema"
    )
    if legacy_schema != "rtece_path_manifest.v1":
        raise ValueError(f"unsupported legacy path manifest schema: {legacy_schema}")
    legacy_hash = _require_nonblank_string(
        payload["legacy_path_manifest_hash"], "legacy_path_manifest_hash"
    )
    if _LEGACY_MANIFEST_HASH_PATTERN.fullmatch(legacy_hash) is None:
        raise ValueError("legacy_path_manifest_hash must contain 16 hex digits")
    operator_path_ids = _normalize_operator_path_ids(
        payload["operator_path_ids"], field="operator_path_ids"
    )
    retained_groups = _normalize_retained_groups(
        payload["retained_groups"], operator_path_ids
    )
    feature_dtype, distance_unit = _validate_feature_schema(payload["feature_schema"])
    implementation_revision = _require_nonblank_string(
        payload["implementation_revision"], "implementation_revision"
    )
    manifest_hash = _require_sha256_string(
        payload["operator_manifest_hash"], "operator_manifest_hash"
    )

    normalized_without_hash: dict[str, object] = {
        "schema_version": schema,
        "legacy_path_manifest_schema": legacy_schema,
        "legacy_path_manifest_hash": legacy_hash,
        "operator_path_ids": list(operator_path_ids),
        "retained_groups": {
            name: list(path_ids) for name, path_ids in retained_groups
        },
        "feature_schema": {
            "units": {"distance": distance_unit},
            "dtype": feature_dtype,
        },
        "implementation_revision": implementation_revision,
    }
    if sha256_json(normalized_without_hash) != manifest_hash:
        raise ValueError("operator manifest hash mismatch")
    return OperatorManifest(
        schema_version=schema,
        legacy_path_manifest_schema=legacy_schema,
        legacy_path_manifest_hash=legacy_hash,
        operators=tuple(
            OperatorEntry(index=index, path_id=path_id)
            for index, path_id in enumerate(operator_path_ids)
        ),
        retained_groups=retained_groups,
        feature_dtype=feature_dtype,
        feature_distance_unit=distance_unit,
        implementation_revision=implementation_revision,
        operator_manifest_hash=manifest_hash,
    )


def _normalize_operator_path_ids(
    value: object,
    *,
    field: str,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{field} must be a nonempty path sequence")
    path_ids = tuple(_require_nonblank_string(item, field) for item in value)
    if len(set(path_ids)) != len(path_ids):
        raise ValueError(f"duplicate operator path in {field}")
    return path_ids


def _normalize_retained_groups(
    value: object,
    operator_path_ids: tuple[str, ...],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if not isinstance(value, Mapping):
        raise ValueError("retained_groups must be a mapping")
    operator_set = set(operator_path_ids)
    groups: list[tuple[str, tuple[str, ...]]] = []
    previous: set[str] = set()
    for raw_name in sorted(value):
        name = _require_nonblank_string(raw_name, "retained group name")
        path_ids = _normalize_operator_path_ids(
            value[raw_name], field=f"retained group {name}"
        )
        unknown = [path_id for path_id in path_ids if path_id not in operator_set]
        if unknown:
            raise ValueError(f"unknown retained operator {unknown[0]!r} in group {name!r}")
        expected_order = tuple(
            path_id for path_id in operator_path_ids if path_id in set(path_ids)
        )
        if path_ids != expected_order:
            raise ValueError(f"retained group {name!r} does not follow operator path order")
        current = set(path_ids)
        if not previous.issubset(current):
            raise ValueError("retained groups must be nested in sorted group-name order")
        groups.append((name, path_ids))
        previous = current
    return tuple(groups)


def _validate_feature_schema(value: object) -> tuple[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"units", "dtype"}:
        raise ValueError("feature_schema must contain exactly units and dtype")
    units = value["units"]
    if not isinstance(units, Mapping) or set(units) != {"distance"}:
        raise ValueError("feature_schema units must contain exactly distance")
    dtype = _require_nonblank_string(value["dtype"], "feature_schema.dtype")
    distance_unit = _require_nonblank_string(
        units["distance"], "feature_schema.units.distance"
    )
    if dtype != "float64" or distance_unit != "A":
        raise ValueError("Stage186 operator feature_schema must use float64 and A")
    return dtype, distance_unit


def _validate_ordered_operator_subset(
    path_ids: tuple[str, ...],
    operator_path_ids: tuple[str, ...],
    *,
    field: str,
) -> None:
    operator_set = set(operator_path_ids)
    unknown = [path_id for path_id in path_ids if path_id not in operator_set]
    if unknown:
        raise ValueError(f"unknown operator path {unknown[0]!r} in {field}")
    expected = tuple(path_id for path_id in operator_path_ids if path_id in set(path_ids))
    if path_ids != expected:
        raise ValueError(f"{field} must follow operator path order")


def _projection_row_path_ids(row: Mapping[str, object]) -> tuple[str, ...] | None:
    for key in ("path_ids", "candidate_scalar_path_ids", "scalar_path_ids"):
        if row.get(key) is not None:
            return _normalize_operator_path_ids(row[key], field=key)
    return None


def _validate_row_operator_binding(
    row: Mapping[str, object],
    operator_manifest: OperatorManifest,
    row_index: int,
) -> None:
    expected = {
        "operator_manifest_schema": operator_manifest.schema_version,
        "operator_manifest_hash": operator_manifest.operator_manifest_hash,
        "operator_path_ids": list(operator_manifest.operator_path_ids),
    }
    for field, expected_value in expected.items():
        if row.get(field) != expected_value:
            raise ValueError(f"active_set_rows[{row_index}] {field} mismatch")
__all__ = [
    "OperatorEntry",
    "OperatorManifest",
    "STAGE186_OPERATOR_MANIFEST_SCHEMA",
    "build_stage186_operator_manifest",
    "load_operator_manifest",
    "validate_projection_operator_binding",
    "write_operator_manifest",
    "COMPATIBILITY_FIELDS",
    "STAGE186_COMPATIBILITY_SCHEMA",
    "build_compatibility_tuple",
    "canonical_json_bytes",
    "sha256_file",
    "sha256_json",
    "validate_compatibility_tuple",
    "write_canonical_json",
]
