#!/usr/bin/env python3
"""Collect lightweight Stage145 community-baseline training status."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


DEFAULT_MANIFEST = Path("runs/oc20neb_tace_mace/community-baselines-stage145/stage145_manifest.json")
DEFAULT_OUTPUT_ROOT = Path("runs/oc20neb_tace_mace/community-baselines-stage145")


def _last_numeric_row(path: Path) -> list[float] | None:
    if not path.exists():
        return None
    last: list[float] | None = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            last = [float(part) for part in line.split()]
        except ValueError:
            continue
    return last


def _deepmd_status(row: Mapping[str, Any], train_dir: Path, job_state: Mapping[str, Any] | None) -> dict[str, Any]:
    values = _last_numeric_row(train_dir / "lcurve.out")
    target = int(row.get("stop_batch", 0) or 0)
    latest_step = int(values[0]) if values else None
    frozen = (train_dir / "frozen_model.pth").exists()
    completed = bool(latest_step is not None and target and latest_step >= target) or frozen
    status = "completed" if completed else ("running" if values else "missing")
    return {
        "training_status": status,
        "latest_step": latest_step,
        "target_step": target or None,
        "latest_rmse_val": values[1] if values and len(values) > 1 else None,
        "latest_rmse_trn": values[2] if values and len(values) > 2 else None,
        "latest_rmse_e_val": values[3] if values and len(values) > 3 else None,
        "latest_rmse_e_train": values[4] if values and len(values) > 4 else None,
        "latest_rmse_f_val": values[5] if values and len(values) > 5 else None,
        "latest_rmse_f_train": values[6] if values and len(values) > 6 else None,
        "latest_lr": values[7] if values and len(values) > 7 else None,
        "model_artifact_present": frozen,
        "training_curve": str(train_dir / "lcurve.out"),
        "job_state": dict(job_state or {}),
    }


def _nep_status(row: Mapping[str, Any], train_dir: Path, job_state: Mapping[str, Any] | None) -> dict[str, Any]:
    values = _last_numeric_row(train_dir / "loss.out")
    target = int(row.get("generation", 0) or 0)
    latest_step = int(values[0]) if values else None
    completed = bool(latest_step is not None and target and latest_step >= target)
    state = str((job_state or {}).get("state", "")).upper()
    if completed:
        status = "completed"
    elif state == "RUNNING" or values:
        status = "running"
    else:
        status = "missing"
    return {
        "training_status": status,
        "latest_step": latest_step,
        "target_step": target or None,
        "latest_total_loss": values[1] if values and len(values) > 1 else None,
        "latest_l1reg_loss": values[2] if values and len(values) > 2 else None,
        "latest_l2reg_loss": values[3] if values and len(values) > 3 else None,
        "latest_rmse_e_train": values[4] if values and len(values) > 4 else None,
        "latest_rmse_f_train": values[5] if values and len(values) > 5 else None,
        "latest_rmse_v_train": values[6] if values and len(values) > 6 else None,
        "latest_rmse_e_val": values[7] if values and len(values) > 7 else None,
        "latest_rmse_f_val": values[8] if values and len(values) > 8 else None,
        "latest_rmse_v_val": values[9] if values and len(values) > 9 else None,
        "model_artifact_present": (train_dir / "nep.txt").exists(),
        "training_curve": str(train_dir / "loss.out"),
        "job_state": dict(job_state or {}),
    }


def collect_stage145_training_status(
    manifest: Mapping[str, Any],
    output_root: str | Path,
    *,
    job_states: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    rows = []
    job_states = job_states or {}
    for row in manifest.get("rows", []):
        name = str(row["name"])
        engine = str(row["engine"])
        train_dir = Path(row["train_dir"])
        job_state = job_states.get(name)
        if engine == "deepmd":
            status = _deepmd_status(row, train_dir, job_state)
        elif engine == "nep":
            status = _nep_status(row, train_dir, job_state)
        else:
            status = {"training_status": "unsupported", "job_state": dict(job_state or {})}
        rows.append({"name": name, "engine": engine, **status})
    payload = {
        "schema_version": "community_baselines_stage145_training_status.v1",
        "stage": "community_baselines_stage145",
        "rows": rows,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "stage145_training_status.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "stage145_training_status.md").write_text(render_training_status_markdown(payload), encoding="utf-8")
    return payload


def _fmt(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render_training_status_markdown(status: Mapping[str, Any]) -> str:
    lines = [
        "# Stage145 Training Status",
        "",
        "| row | engine | training | job | latest step | F RMSE val | F RMSE train |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for row in status.get("rows", []):
        job = row.get("job_state") or {}
        job_label = job.get("job_id") or job.get("state") or "NA"
        lines.append(
            "| {name} | {engine} | {training} | {job} | {step} | {f_val} | {f_train} |".format(
                name=row.get("name"),
                engine=row.get("engine"),
                training=row.get("training_status"),
                job=job_label,
                step=_fmt(row.get("latest_step")),
                f_val=_fmt(row.get("latest_rmse_f_val")),
                f_train=_fmt(row.get("latest_rmse_f_train")),
            )
        )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    print(json.dumps(collect_stage145_training_status(manifest, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
