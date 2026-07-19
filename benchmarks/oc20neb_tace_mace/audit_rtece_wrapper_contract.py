#!/usr/bin/env python3
"""Audit rTECE Pareto-sweep wrapper contracts before spending queue time."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
from typing import Any

FORBIDDEN_SBATCH_OPTIONS = ("--export", "--mem", "--cpus-per-task")

DEFAULT_STAGE116_EXPECTED_EXPORTS = {
    "LIMIT_CONFIGS": "2048",
    "VALID_LIMIT_CONFIGS": "256",
    "BENCH_LIMIT_CONFIGS": "1024",
    "MAX_STEPS": "20000",
    "TRAINER_BACKEND": "lightning",
    "BATCH_SIZE": "8",
    "VALID_BATCH_SIZE": "16",
    "LR_SCHEDULER": "plateau",
    "LR_PATIENCE": "100",
    "LR_FACTOR": "0.5",
    "LR_WARMUP_STEPS": "500",
    "EARLY_STOPPING_PATIENCE": "400",
    "FORCE_MODE": "autograd",
    "DEFAULT_DTYPE": "float32",
    "USE_SHORT_RANGE_REPULSION": "1",
    "SHORT_RANGE_REPULSION_POTENTIAL": "zbl",
    "LEARNABLE_RADIAL_MIXING": "1",
    "DFT_VALID_FILE": "runs/oc20neb_tace_mace/tece-distill-20260717/mixed_valid_tw0.75_regen.extxyz",
}


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def parse_export_lines(text: str) -> dict[str, str]:
    exports: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("export "):
            continue
        body = line[len("export ") :].strip()
        if "=" not in body:
            continue
        key, value = body.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            parts = shlex.split(value)
            if len(parts) == 1:
                value = parts[0]
        except ValueError:
            pass
        exports[key] = value
    return exports


def forbidden_sbatch_options(text: str) -> list[str]:
    found: list[str] = []
    for option in FORBIDDEN_SBATCH_OPTIONS:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped.startswith("#SBATCH"):
                continue
            if option in stripped:
                found.append(option)
                break
    return found


def audit_wrapper_file(
    wrapper_path: str | Path,
    *,
    expected_exports: dict[str, str],
) -> dict[str, Any]:
    path = Path(wrapper_path)
    text = path.read_text(encoding="utf-8")
    exports = parse_export_lines(text)
    missing = sorted(key for key in expected_exports if key not in exports)
    mismatched = {
        key: {"expected": str(expected), "actual": str(exports.get(key))}
        for key, expected in expected_exports.items()
        if key in exports and str(exports.get(key)) != str(expected)
    }
    forbidden = forbidden_sbatch_options(text)
    contract_pass = not missing and not mismatched and not forbidden
    return {
        "wrapper": str(path),
        "contract_pass": bool(contract_pass),
        "forbidden_sbatch_options": forbidden,
        "missing_exports": missing,
        "mismatched_exports": mismatched,
        "checked_exports": {key: exports.get(key) for key in sorted(expected_exports)},
    }


def audit_wrapper_index(
    index_path: str | Path,
    *,
    expected_exports: dict[str, str] | None = None,
) -> dict[str, Any]:
    expected_exports = dict(expected_exports or DEFAULT_STAGE116_EXPECTED_EXPORTS)
    index = load_json(index_path)
    rows: list[dict[str, Any]] = []
    for index_row in index.get("rows", []):
        name = str(index_row.get("name") or index_row.get("variant"))
        wrapper = index_row.get("wrapper")
        if not wrapper:
            row = {
                "name": name,
                "contract_pass": False,
                "wrapper": None,
                "forbidden_sbatch_options": [],
                "missing_exports": sorted(expected_exports),
                "mismatched_exports": {},
                "checked_exports": {},
            }
        else:
            row = audit_wrapper_file(wrapper, expected_exports=expected_exports)
            row["name"] = name
            row["variant"] = index_row.get("variant") or name
            row["tece_axes"] = index_row.get("tece_axes", [])
            row["hidden_channels"] = index_row.get("hidden_channels")
            row["moment_l_max"] = index_row.get("moment_l_max")
            row["scalar_path_ids"] = index_row.get("scalar_path_ids")
        rows.append(row)
    failed = [str(row["name"]) for row in rows if not bool(row.get("contract_pass"))]
    return {
        "schema_version": "rtece_wrapper_contract_audit.v1",
        "source_index": str(index_path),
        "row_set": index.get("row_set"),
        "contract_pass": not failed,
        "failed_rows": failed,
        "expected_exports": expected_exports,
        "forbidden_sbatch_options": list(FORBIDDEN_SBATCH_OPTIONS),
        "review_basis": [
            "SAI: no sbatch --export, --mem, or --cpus-per-task in wrappers",
            "TECE_design_space Stage E: comparable hardware Pareto rows require identical data/training/evaluation contracts",
            "rTECE_review: report E/F RMSE, high-force tails, physical diagnostics only after contract-equivalent runs",
        ],
        "rows": rows,
    }


def _parse_expect_export(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        result[key] = value
    return result


def format_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# rTECE Wrapper Contract Audit: {payload.get('row_set')}",
        "",
        f"- contract pass: `{payload.get('contract_pass')}`",
        f"- failed rows: {', '.join(payload.get('failed_rows') or []) if payload.get('failed_rows') else 'None'}",
        "",
        "| row | pass | forbidden sbatch | missing exports | mismatched exports | hidden | Lmax |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for row in payload.get("rows", []):
        mismatched = ", ".join(sorted((row.get("mismatched_exports") or {}).keys())) or "None"
        missing = ", ".join(row.get("missing_exports") or []) or "None"
        forbidden = ", ".join(row.get("forbidden_sbatch_options") or []) or "None"
        lines.append(
            "| {name} | {passed} | {forbidden} | {missing} | {mismatched} | {hidden} | {lmax} |".format(
                name=row.get("name"),
                passed=row.get("contract_pass"),
                forbidden=forbidden,
                missing=missing,
                mismatched=mismatched,
                hidden=row.get("hidden_channels"),
                lmax=row.get("moment_l_max"),
            )
        )
    lines.extend(["", "## Review Basis", ""])
    lines.extend(f"- {item}" for item in payload.get("review_basis", []))
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--stage116-defaults", action="store_true", help="Use the current Stage116 capacity-ladder contract defaults.")
    parser.add_argument("--expect-export", action="append", default=[], help="Expected wrapper export in KEY=VALUE form; can be repeated.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expect_export:
        expected = _parse_expect_export(args.expect_export)
    elif args.stage116_defaults:
        expected = DEFAULT_STAGE116_EXPECTED_EXPORTS
    else:
        expected = {}
    payload = audit_wrapper_index(args.index, expected_exports=expected)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(format_markdown(payload), encoding="utf-8")
    print(args.output_md)
    print(args.output_json)
    if not payload["contract_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
