# Community Baselines Stage145 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible stage145 baseline pipeline that trains and benchmarks NEP and DeepMD/DPA-like community models on the same stage142 weighted mixed-label contract used by recent rTECE experiments.

**Architecture:** Add one focused stage145 materializer that creates a manifest, conversion commands, and no-export sbatch wrappers. Keep NEP conversion, DeepMD conversion, and benchmark summarization as separate scripts so each format-specific risk is isolated and testable. Reuse the existing rTECE benchmark metric names and stage manifest test style.

**Tech Stack:** Python 3, ASE extxyz, GPUMD `gpumd/4.8-cuda12.4`, DeepMD-kit `deepmd-kit/3.1.2`, Slurm sbatch, pytest.

## Global Constraints

- Primary train source is `runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/weighted_train_base2048_plus_teacher_relax640_forceonly_eanchor.extxyz`.
- First train label target is mixed `energy` and `forces`; DFT and teacher/mixed errors are benchmarked when source fields are available.
- Report E/F MAE, E/F RMSE, E/F max error, E bias, atoms/s, dimer scan, and rattle-relax behavior.
- RMSE is the first ranking metric; MAE alone is not sufficient.
- Sbatch wrappers must contain no `--export`, no `--mem`, and no `--cpus-per-task`.
- Sbatch wrappers use `--nodes`, `--ntasks`, and `--gpus-per-node`; environment variables are set inside the script body with ordinary `export VAR=value`.
- NEP must run on a GPU compute node; login-node CUDA runtime mismatch is not a NEP availability failure.
- Outputs are rooted at `runs/oc20neb_tace_mace/community-baselines-stage145/`.
- Do not commit large converted datasets, checkpoints, frozen models, or raw training outputs.

---

### Task 1: Stage145 Manifest And Wrapper Materializer

**Files:**
- Create: `benchmarks/oc20neb_tace_mace/make_community_baselines_stage145.py`
- Modify: `test/test_rtece_scalar.py`
- Create by command: `runs/oc20neb_tace_mace/community-baselines-stage145/stage145_manifest.json`
- Create by command: `runs/oc20neb_tace_mace/community-baselines-stage145/stage145_manifest_audit.json`
- Create by command: `runs/oc20neb_tace_mace/community-baselines-stage145/stage145_plan.md`

**Interfaces:**
- Produces: `make_stage145_manifest(output_root: str | Path, train_file: str | Path = DEFAULT_TRAIN_FILE, limit_configs: int = 2688, valid_limit_configs: int = 256, bench_limit_configs: int = 1024) -> dict[str, Any]`
- Produces: `audit_stage145_manifest(payload: dict[str, Any]) -> dict[str, Any]`
- Produces: `materialize_stage145(payload: dict[str, Any]) -> dict[str, Any]`
- Later tasks rely on manifest rows with `engine` values `nep` and `deepmd`, row names `nep4_mixed_smoke` and `deepmd_dpa_like_mixed_smoke`, and artifact paths under `runs/oc20neb_tace_mace/community-baselines-stage145/`.

- [ ] **Step 1: Write failing manifest test**

Append this test to `test/test_rtece_scalar.py` near the stage144 manifest test:

```python
def test_community_baselines_stage145_manifest_materializes_no_export_wrappers(tmp_path):
    from benchmarks.oc20neb_tace_mace.make_community_baselines_stage145 import (
        audit_stage145_manifest,
        make_stage145_manifest,
        materialize_stage145,
    )

    payload = make_stage145_manifest(
        output_root=tmp_path / "stage145",
        limit_configs=16,
        valid_limit_configs=4,
        bench_limit_configs=8,
        nep_generations=20,
        deepmd_stop_batch=20,
    )

    assert payload["schema_version"] == "community_baselines_stage145.v1"
    assert payload["stage"] == "community_baselines_stage145"
    assert payload["train_contract"]["label_target"] == "mixed_energy_forces"
    assert payload["train_contract"]["energy_key"] == "energy"
    assert payload["train_contract"]["forces_key"] == "forces"
    assert payload["train_contract"]["dft_energy_key"] == "dft_energy"
    assert payload["train_contract"]["teacher_forces_key"] == "teacher_forces"

    rows = {row["name"]: row for row in payload["rows"]}
    assert set(rows) == {"nep4_mixed_smoke", "deepmd_dpa_like_mixed_smoke"}
    assert rows["nep4_mixed_smoke"]["engine"] == "nep"
    assert rows["nep4_mixed_smoke"]["module"] == "gpumd/4.8-cuda12.4"
    assert rows["deepmd_dpa_like_mixed_smoke"]["engine"] == "deepmd"
    assert rows["deepmd_dpa_like_mixed_smoke"]["module"] == "deepmd-kit/3.1.2"
    assert rows["deepmd_dpa_like_mixed_smoke"]["descriptor_label"] in {
        "dpa1_zero_attention",
        "dpa_like_low_attention",
    }

    audit = audit_stage145_manifest(payload)
    assert audit["contract_pass"], audit["failed_checks"]

    materialized = materialize_stage145(payload)
    assert Path(materialized["manifest"]).exists()
    assert Path(materialized["audit"]).exists()
    assert Path(materialized["stage_plan"]).exists()
    assert set(materialized["wrappers"]) == {"nep4_mixed_smoke", "deepmd_dpa_like_mixed_smoke"}

    for wrapper_path in materialized["wrappers"].values():
        text = Path(wrapper_path).read_text()
        assert "#SBATCH --nodes=1" in text
        assert "#SBATCH --ntasks=1" in text
        assert "#SBATCH --gpus-per-node=1" in text
        assert "--export" not in text
        assert "--mem" not in text
        assert "--cpus-per-task" not in text
        assert "community-baselines-stage145" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py::test_community_baselines_stage145_manifest_materializes_no_export_wrappers -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks.oc20neb_tace_mace.make_community_baselines_stage145'`.

- [ ] **Step 3: Implement materializer**

Create `benchmarks/oc20neb_tace_mace/make_community_baselines_stage145.py` with:

```python
#!/usr/bin/env python3
"""Create Stage145 NEP and DeepMD community baseline wrappers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_TRAIN_FILE = Path(
    "runs/oc20neb_tace_mace/rtece-stage142-teacher-relax-coverage/"
    "weighted_train_base2048_plus_teacher_relax640_forceonly_eanchor.extxyz"
)
DEFAULT_OUTPUT_ROOT = Path("runs/oc20neb_tace_mace/community-baselines-stage145")


def make_stage145_manifest(
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    train_file: str | Path = DEFAULT_TRAIN_FILE,
    limit_configs: int = 2688,
    valid_limit_configs: int = 256,
    bench_limit_configs: int = 1024,
    nep_generations: int = 20000,
    deepmd_stop_batch: int = 20000,
) -> dict[str, Any]:
    root = Path(output_root)
    if int(limit_configs) < 1:
        raise ValueError("limit_configs must be positive")
    if int(valid_limit_configs) < 1:
        raise ValueError("valid_limit_configs must be positive")
    if int(bench_limit_configs) < 1:
        raise ValueError("bench_limit_configs must be positive")
    rows = [
        {
            "name": "nep4_mixed_smoke",
            "engine": "nep",
            "module": "gpumd/4.8-cuda12.4",
            "descriptor_label": "nep4",
            "train_dir": str(root / "nep4_mixed_smoke"),
            "generation": int(nep_generations),
            "wrapper": str(root / "wrappers" / "nep4_mixed_smoke_no_export.sbatch"),
        },
        {
            "name": "deepmd_dpa_like_mixed_smoke",
            "engine": "deepmd",
            "module": "deepmd-kit/3.1.2",
            "descriptor_label": "dpa_like_low_attention",
            "train_dir": str(root / "deepmd_dpa_like_mixed_smoke"),
            "stop_batch": int(deepmd_stop_batch),
            "wrapper": str(root / "wrappers" / "deepmd_dpa_like_mixed_smoke_no_export.sbatch"),
        },
    ]
    return {
        "schema_version": "community_baselines_stage145.v1",
        "stage": "community_baselines_stage145",
        "row_set": "community-baselines-stage145",
        "output_root": str(root),
        "train_contract": {
            "train_file": str(train_file),
            "label_target": "mixed_energy_forces",
            "energy_key": "energy",
            "forces_key": "forces",
            "dft_energy_key": "dft_energy",
            "dft_forces_key": "dft_forces",
            "teacher_energy_key": "teacher_energy",
            "teacher_forces_key": "teacher_forces",
            "energy_weight_key": "energy_weight",
            "forces_weight_key": "forces_weight",
            "limit_configs": int(limit_configs),
            "valid_limit_configs": int(valid_limit_configs),
            "bench_limit_configs": int(bench_limit_configs),
        },
        "metrics_contract": [
            "mae_e_mev_atom",
            "rmse_e_mev_atom",
            "max_abs_e_mev_atom",
            "bias_e_mev_atom",
            "mae_f_mev_a",
            "rmse_f_mev_a",
            "max_abs_f_mev_a",
            "atoms_per_second",
            "dimer_scan",
            "rattle_relax",
        ],
        "comparison_anchors": ["stage140_forceonly_ew2", "stage142_teacher_relax_coverage", "stage144_t3_cavity_vector"],
        "rows": rows,
        "artifacts": {
            "manifest": str(root / "stage145_manifest.json"),
            "audit": str(root / "stage145_manifest_audit.json"),
            "stage_plan": str(root / "stage145_plan.md"),
            "wrapper_root": str(root / "wrappers"),
        },
    }


def audit_stage145_manifest(payload: dict[str, Any]) -> dict[str, Any]:
    rows = list(payload.get("rows") or [])
    row_names = {str(row.get("name")) for row in rows}
    wrappers = [str(row.get("wrapper", "")) for row in rows]
    checks = {
        "schema_version": payload.get("schema_version") == "community_baselines_stage145.v1",
        "stage": payload.get("stage") == "community_baselines_stage145",
        "row_set": payload.get("row_set") == "community-baselines-stage145",
        "has_nep_and_deepmd": row_names == {"nep4_mixed_smoke", "deepmd_dpa_like_mixed_smoke"},
        "mixed_label_contract": (payload.get("train_contract") or {}).get("label_target") == "mixed_energy_forces",
        "rmse_metrics_present": "rmse_e_mev_atom" in payload.get("metrics_contract", [])
        and "rmse_f_mev_a" in payload.get("metrics_contract", []),
        "no_export_wrapper_names": all(path.endswith("_no_export.sbatch") for path in wrappers),
    }
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "schema_version": "community_baselines_stage145_audit.v1",
        "contract_pass": not failed,
        "failed_checks": failed,
        "checks": checks,
    }


def _write_wrapper(row: dict[str, Any], payload: dict[str, Any]) -> str:
    wrapper = Path(row["wrapper"])
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    train_file = payload["train_contract"]["train_file"]
    limit = payload["train_contract"]["limit_configs"]
    train_dir = row["train_dir"]
    if row["engine"] == "nep":
        command = f"""module load {row['module']}
export STAGE145_TRAIN_FILE="{train_file}"
export STAGE145_OUTPUT_DIR="{train_dir}"
export STAGE145_LIMIT_CONFIGS="{limit}"
python benchmarks/oc20neb_tace_mace/convert_stage145_nep.py --manifest {payload['artifacts']['manifest']} --row {row['name']}
cd "{train_dir}"
nep
"""
    else:
        command = f"""module load {row['module']}
export STAGE145_TRAIN_FILE="{train_file}"
export STAGE145_OUTPUT_DIR="{train_dir}"
export STAGE145_LIMIT_CONFIGS="{limit}"
python benchmarks/oc20neb_tace_mace/convert_stage145_deepmd.py --manifest {payload['artifacts']['manifest']} --row {row['name']}
cd "{train_dir}"
dp --pt train input.json
dp --pt freeze -o frozen_model.pth
"""
    text = f"""#!/bin/bash
#SBATCH --job-name={row['name']}
#SBATCH --partition=4V100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=rush-1o2gpu
#SBATCH --output=logs/{row['name']}-%j.out
#SBATCH --error=logs/{row['name']}-%j.err

set -euo pipefail
cd "$(pwd)"
{command}
"""
    wrapper.write_text(text, encoding="utf-8")
    return str(wrapper)


def _write_stage_plan(payload: dict[str, Any]) -> str:
    path = Path(payload["artifacts"]["stage_plan"])
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Stage145 Community Baselines Plan",
        "",
        f"- train file: `{payload['train_contract']['train_file']}`",
        "- label target: mixed `energy` and `forces`",
        "- rows: `nep4_mixed_smoke`, `deepmd_dpa_like_mixed_smoke`",
        "- report DFT, teacher, mixed RMSE/MAE/max/bias and atoms/s.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def materialize_stage145(payload: dict[str, Any]) -> dict[str, Any]:
    audit = audit_stage145_manifest(payload)
    if not audit["contract_pass"]:
        raise ValueError(f"Stage145 manifest audit failed: {audit['failed_checks']}")
    manifest_path = Path(payload["artifacts"]["manifest"])
    audit_path = Path(payload["artifacts"]["audit"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    wrappers = {row["name"]: _write_wrapper(row, payload) for row in payload["rows"]}
    stage_plan = _write_stage_plan(payload)
    return {"manifest": str(manifest_path), "audit": str(audit_path), "stage_plan": stage_plan, "wrappers": wrappers}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--limit-configs", type=int, default=2688)
    parser.add_argument("--valid-limit-configs", type=int, default=256)
    parser.add_argument("--bench-limit-configs", type=int, default=1024)
    parser.add_argument("--nep-generations", type=int, default=20000)
    parser.add_argument("--deepmd-stop-batch", type=int, default=20000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = make_stage145_manifest(
        output_root=args.output_root,
        train_file=args.train_file,
        limit_configs=args.limit_configs,
        valid_limit_configs=args.valid_limit_configs,
        bench_limit_configs=args.bench_limit_configs,
        nep_generations=args.nep_generations,
        deepmd_stop_batch=args.deepmd_stop_batch,
    )
    print(json.dumps(materialize_stage145(payload), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py::test_community_baselines_stage145_manifest_materializes_no_export_wrappers -q
```

Expected: PASS.

- [ ] **Step 5: Materialize full stage145 wrappers**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python benchmarks/oc20neb_tace_mace/make_community_baselines_stage145.py
```

Expected: JSON with paths for `stage145_manifest.json`, `stage145_manifest_audit.json`, `stage145_plan.md`, and two wrapper paths.

- [ ] **Step 6: Audit forbidden sbatch flags**

Run:

```bash
rg -n -- '--export|--mem|--cpus-per-task' runs/oc20neb_tace_mace/community-baselines-stage145
```

Expected: exit code 1 with no matches.

- [ ] **Step 7: Commit**

Run:

```bash
git add benchmarks/oc20neb_tace_mace/make_community_baselines_stage145.py test/test_rtece_scalar.py runs/oc20neb_tace_mace/community-baselines-stage145/stage145_manifest.json runs/oc20neb_tace_mace/community-baselines-stage145/stage145_manifest_audit.json runs/oc20neb_tace_mace/community-baselines-stage145/stage145_plan.md runs/oc20neb_tace_mace/community-baselines-stage145/wrappers/nep4_mixed_smoke_no_export.sbatch runs/oc20neb_tace_mace/community-baselines-stage145/wrappers/deepmd_dpa_like_mixed_smoke_no_export.sbatch
git commit -m "Plan rTECE stage145 community baselines"
```

Expected: commit containing only the materializer, test, manifest, plan, and wrappers.

---

### Task 2: NEP Data Conversion

**Files:**
- Create: `benchmarks/oc20neb_tace_mace/convert_stage145_nep.py`
- Modify: `test/test_rtece_scalar.py`

**Interfaces:**
- Consumes: stage145 manifest JSON with row `nep4_mixed_smoke`.
- Produces: `convert_extxyz_to_nep(train_file: Path, output_dir: Path, limit_configs: int, energy_key: str = "energy", forces_key: str = "forces") -> dict[str, Any]`
- Produces files in the row train dir: `train.xyz`, `nep.in`, `conversion_summary.json`.

- [ ] **Step 1: Write failing NEP conversion smoke test**

Append this test to `test/test_rtece_scalar.py`:

```python
def test_stage145_nep_converter_writes_gpumd_train_xyz(tmp_path):
    import ase.io
    from ase import Atoms
    from benchmarks.oc20neb_tace_mace.convert_stage145_nep import convert_extxyz_to_nep

    source = tmp_path / "input.extxyz"
    atoms = Atoms(
        "CN",
        positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    atoms.info["energy"] = -3.0
    atoms.info["dft_energy"] = -2.9
    atoms.info["teacher_energy"] = -3.1
    atoms.arrays["forces"] = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=float)
    atoms.arrays["dft_forces"] = atoms.arrays["forces"]
    atoms.arrays["teacher_forces"] = atoms.arrays["forces"]
    ase.io.write(source, [atoms], format="extxyz")

    summary = convert_extxyz_to_nep(source, tmp_path / "nep", limit_configs=1)

    assert summary["engine"] == "nep"
    assert summary["num_configs"] == 1
    assert summary["type_map"] == ["C", "N"]
    text = (tmp_path / "nep" / "train.xyz").read_text()
    assert "energy=-3.0" in text
    assert "Properties=species:S:1:pos:R:3:force:R:3" in text
    assert "C " in text and "N " in text
    nep_in = (tmp_path / "nep" / "nep.in").read_text()
    assert "type         2 C N" in nep_in
    assert "generation   20000" in nep_in
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py::test_stage145_nep_converter_writes_gpumd_train_xyz -q
```

Expected: FAIL with `ModuleNotFoundError` for `convert_stage145_nep`.

- [ ] **Step 3: Implement NEP converter**

Create `benchmarks/oc20neb_tace_mace/convert_stage145_nep.py` with:

```python
#!/usr/bin/env python3
"""Convert stage145 extxyz data to GPUMD NEP train.xyz format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_atoms(path: Path, limit_configs: int | None):
    import ase.io

    index = ":" if limit_configs is None else f":{int(limit_configs)}"
    atoms_list = ase.io.read(str(path), index=index)
    return atoms_list if isinstance(atoms_list, list) else [atoms_list]


def _type_map(atoms_list) -> list[str]:
    symbols = sorted({symbol for atoms in atoms_list for symbol in atoms.get_chemical_symbols()})
    return symbols


def _write_gpumd_extxyz(path: Path, atoms_list, *, energy_key: str, forces_key: str) -> None:
    import ase.io

    converted = []
    for atoms in atoms_list:
        item = atoms.copy()
        item.info["energy"] = float(atoms.info[energy_key])
        item.arrays["force"] = np.asarray(atoms.arrays[forces_key], dtype=np.float64)
        for key in list(item.arrays):
            if key not in {"numbers", "positions", "force"}:
                del item.arrays[key]
        converted.append(item)
    ase.io.write(path, converted, format="extxyz")


def convert_extxyz_to_nep(
    train_file: str | Path,
    output_dir: str | Path,
    *,
    limit_configs: int | None,
    energy_key: str = "energy",
    forces_key: str = "forces",
    generation: int = 20000,
) -> dict[str, Any]:
    train_path = Path(train_file)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    atoms_list = _read_atoms(train_path, limit_configs)
    if not atoms_list:
        raise ValueError(f"no configurations read from {train_path}")
    missing_energy = [idx for idx, atoms in enumerate(atoms_list) if energy_key not in atoms.info]
    missing_forces = [idx for idx, atoms in enumerate(atoms_list) if forces_key not in atoms.arrays]
    if missing_energy:
        raise KeyError(f"missing energy key {energy_key!r} in configs {missing_energy[:5]}")
    if missing_forces:
        raise KeyError(f"missing forces key {forces_key!r} in configs {missing_forces[:5]}")
    types = _type_map(atoms_list)
    train_xyz = out / "train.xyz"
    _write_gpumd_extxyz(train_xyz, atoms_list, energy_key=energy_key, forces_key=forces_key)
    nep_in = out / "nep.in"
    nep_in.write_text(
        f"type         {len(types)} {' '.join(types)}\n"
        f"generation   {int(generation)}\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": "stage145_nep_conversion.v1",
        "engine": "nep",
        "train_file": str(train_path),
        "output_dir": str(out),
        "train_xyz": str(train_xyz),
        "nep_in": str(nep_in),
        "num_configs": len(atoms_list),
        "num_atoms": int(sum(len(atoms) for atoms in atoms_list)),
        "type_map": types,
        "energy_key": energy_key,
        "forces_key": forces_key,
        "generation": int(generation),
    }
    (out / "conversion_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--row", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.manifest.read_text())
    row = next(row for row in payload["rows"] if row["name"] == args.row)
    summary = convert_extxyz_to_nep(
        payload["train_contract"]["train_file"],
        row["train_dir"],
        limit_configs=payload["train_contract"]["limit_configs"],
        energy_key=payload["train_contract"]["energy_key"],
        forces_key=payload["train_contract"]["forces_key"],
        generation=row["generation"],
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py::test_stage145_nep_converter_writes_gpumd_train_xyz -q
```

Expected: PASS.

- [ ] **Step 5: Run smoke conversion on real data**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python benchmarks/oc20neb_tace_mace/convert_stage145_nep.py --manifest runs/oc20neb_tace_mace/community-baselines-stage145/stage145_manifest.json --row nep4_mixed_smoke
```

Expected: writes `train.xyz`, `nep.in`, and `conversion_summary.json` under `runs/oc20neb_tace_mace/community-baselines-stage145/nep4_mixed_smoke/`.

- [ ] **Step 6: Commit**

Run:

```bash
git add benchmarks/oc20neb_tace_mace/convert_stage145_nep.py test/test_rtece_scalar.py
git commit -m "Add rTECE stage145 NEP conversion"
```

Expected: commit containing converter and test only. Do not add generated `train.xyz`.

---

### Task 3: DeepMD Data Conversion

**Files:**
- Create: `benchmarks/oc20neb_tace_mace/convert_stage145_deepmd.py`
- Modify: `test/test_rtece_scalar.py`

**Interfaces:**
- Consumes: stage145 manifest JSON with row `deepmd_dpa_like_mixed_smoke`.
- Produces: `convert_extxyz_to_deepmd(train_file: Path, output_dir: Path, limit_configs: int, energy_key: str = "energy", forces_key: str = "forces") -> dict[str, Any]`
- Produces files in the row train dir: `type_map.raw`, `mixed/set.000/coord.npy`, `box.npy`, `energy.npy`, `force.npy`, `type.raw`, `input.json`, `conversion_summary.json`.

- [ ] **Step 1: Write failing DeepMD conversion smoke test**

Append this test to `test/test_rtece_scalar.py`:

```python
def test_stage145_deepmd_converter_writes_system_and_input(tmp_path):
    import ase.io
    from ase import Atoms
    from benchmarks.oc20neb_tace_mace.convert_stage145_deepmd import convert_extxyz_to_deepmd

    source = tmp_path / "input.extxyz"
    atoms = Atoms(
        "CN",
        positions=[[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    atoms.info["energy"] = -3.0
    atoms.arrays["forces"] = np.array([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]], dtype=float)
    ase.io.write(source, [atoms], format="extxyz")

    summary = convert_extxyz_to_deepmd(source, tmp_path / "dp", limit_configs=1, stop_batch=20)

    assert summary["engine"] == "deepmd"
    assert summary["num_configs"] == 1
    assert summary["type_map"] == ["C", "N"]
    assert (tmp_path / "dp" / "type_map.raw").read_text().splitlines() == ["C", "N"]
    assert np.load(tmp_path / "dp" / "mixed" / "set.000" / "coord.npy").shape == (1, 6)
    assert np.load(tmp_path / "dp" / "mixed" / "set.000" / "force.npy").shape == (1, 6)
    payload = json.loads((tmp_path / "dp" / "input.json").read_text())
    assert payload["training"]["numb_steps"] == 20
    assert payload["model"]["type_map"] == ["C", "N"]
    assert payload["model"]["descriptor"]["type"] in {"dpa2", "se_atten_v2", "se_atten"}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py::test_stage145_deepmd_converter_writes_system_and_input -q
```

Expected: FAIL with `ModuleNotFoundError` for `convert_stage145_deepmd`.

- [ ] **Step 3: Implement DeepMD converter**

Create `benchmarks/oc20neb_tace_mace/convert_stage145_deepmd.py` with:

```python
#!/usr/bin/env python3
"""Convert stage145 extxyz data to a minimal DeepMD-kit numpy system."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_atoms(path: Path, limit_configs: int | None):
    import ase.io

    index = ":" if limit_configs is None else f":{int(limit_configs)}"
    atoms_list = ase.io.read(str(path), index=index)
    return atoms_list if isinstance(atoms_list, list) else [atoms_list]


def _type_map(atoms_list) -> list[str]:
    return sorted({symbol for atoms in atoms_list for symbol in atoms.get_chemical_symbols()})


def _write_deepmd_arrays(base: Path, atoms_list, type_to_id: dict[str, int], *, energy_key: str, forces_key: str) -> None:
    set_dir = base / "mixed" / "set.000"
    set_dir.mkdir(parents=True, exist_ok=True)
    coords = []
    boxes = []
    energies = []
    forces = []
    for atoms in atoms_list:
        coords.append(np.asarray(atoms.positions, dtype=np.float64).reshape(-1))
        boxes.append(np.asarray(atoms.cell.array, dtype=np.float64).reshape(-1))
        energies.append(float(atoms.info[energy_key]))
        forces.append(np.asarray(atoms.arrays[forces_key], dtype=np.float64).reshape(-1))
    np.save(set_dir / "coord.npy", np.asarray(coords, dtype=np.float64))
    np.save(set_dir / "box.npy", np.asarray(boxes, dtype=np.float64))
    np.save(set_dir / "energy.npy", np.asarray(energies, dtype=np.float64))
    np.save(set_dir / "force.npy", np.asarray(forces, dtype=np.float64))
    type_ids = [type_to_id[symbol] for symbol in atoms_list[0].get_chemical_symbols()]
    (base / "mixed" / "type.raw").write_text("\n".join(str(value) for value in type_ids) + "\n", encoding="utf-8")


def _deepmd_input(type_map: list[str], *, stop_batch: int) -> dict[str, Any]:
    return {
        "model": {
            "type_map": type_map,
            "descriptor": {
                "type": "dpa2",
                "rcut": 5.0,
                "rcut_smth": 4.5,
                "sel": "auto",
                "axis_neuron": 16,
                "attn": 0,
                "attn_layer": 0,
                "neuron": [16, 32, 64],
            },
            "fitting_net": {"neuron": [64, 64, 64], "resnet_dt": True},
        },
        "learning_rate": {"type": "exp", "start_lr": 0.001, "stop_lr": 1.0e-8, "decay_steps": 5000},
        "loss": {
            "type": "ener",
            "start_pref_e": 0.02,
            "limit_pref_e": 1.0,
            "start_pref_f": 1000.0,
            "limit_pref_f": 1.0,
        },
        "training": {
            "training_data": {"systems": ["mixed"], "batch_size": "auto"},
            "validation_data": {"systems": ["mixed"], "batch_size": "auto", "numb_btch": 1},
            "numb_steps": int(stop_batch),
            "seed": 145,
            "disp_file": "lcurve.out",
            "disp_freq": 100,
            "save_freq": 1000,
        },
    }


def convert_extxyz_to_deepmd(
    train_file: str | Path,
    output_dir: str | Path,
    *,
    limit_configs: int | None,
    energy_key: str = "energy",
    forces_key: str = "forces",
    stop_batch: int = 20000,
) -> dict[str, Any]:
    train_path = Path(train_file)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    atoms_list = _read_atoms(train_path, limit_configs)
    if not atoms_list:
        raise ValueError(f"no configurations read from {train_path}")
    natoms = len(atoms_list[0])
    if any(len(atoms) != natoms for atoms in atoms_list):
        raise ValueError("initial DeepMD stage145 converter requires fixed atom count per system")
    symbols0 = atoms_list[0].get_chemical_symbols()
    if any(atoms.get_chemical_symbols() != symbols0 for atoms in atoms_list):
        raise ValueError("initial DeepMD stage145 converter requires fixed atom ordering per system")
    types = _type_map(atoms_list)
    type_to_id = {symbol: idx for idx, symbol in enumerate(types)}
    _write_deepmd_arrays(out, atoms_list, type_to_id, energy_key=energy_key, forces_key=forces_key)
    (out / "type_map.raw").write_text("\n".join(types) + "\n", encoding="utf-8")
    input_payload = _deepmd_input(types, stop_batch=stop_batch)
    (out / "input.json").write_text(json.dumps(input_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "schema_version": "stage145_deepmd_conversion.v1",
        "engine": "deepmd",
        "descriptor_label": "dpa_like_low_attention",
        "train_file": str(train_path),
        "output_dir": str(out),
        "num_configs": len(atoms_list),
        "num_atoms_per_config": natoms,
        "type_map": types,
        "energy_key": energy_key,
        "forces_key": forces_key,
        "stop_batch": int(stop_batch),
        "fixed_atom_order_required": True,
    }
    (out / "conversion_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--row", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.manifest.read_text())
    row = next(row for row in payload["rows"] if row["name"] == args.row)
    summary = convert_extxyz_to_deepmd(
        payload["train_contract"]["train_file"],
        row["train_dir"],
        limit_configs=payload["train_contract"]["limit_configs"],
        energy_key=payload["train_contract"]["energy_key"],
        forces_key=payload["train_contract"]["forces_key"],
        stop_batch=row["stop_batch"],
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py::test_stage145_deepmd_converter_writes_system_and_input -q
```

Expected: PASS.

- [ ] **Step 5: Run fixed-order preflight on real data**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python benchmarks/oc20neb_tace_mace/convert_stage145_deepmd.py --manifest runs/oc20neb_tace_mace/community-baselines-stage145/stage145_manifest.json --row deepmd_dpa_like_mixed_smoke
```

Expected if stage142 file has fixed atom count/order: writes DeepMD numpy system and `input.json`.

Expected if stage142 file has multiple formulas or atom order changes: fail with `initial DeepMD stage145 converter requires fixed atom ordering per system`. In that case, add a follow-up task to split systems by formula/order before training rather than forcing an invalid DeepMD system.

- [ ] **Step 6: Commit**

Run:

```bash
git add benchmarks/oc20neb_tace_mace/convert_stage145_deepmd.py test/test_rtece_scalar.py
git commit -m "Add rTECE stage145 DeepMD conversion"
```

Expected: commit containing converter and test only. Do not add generated `.npy` or `input.json`.

---

### Task 4: Unified Stage145 Benchmark Summary

**Files:**
- Create: `benchmarks/oc20neb_tace_mace/summarize_community_baselines_stage145.py`
- Modify: `test/test_rtece_scalar.py`

**Interfaces:**
- Consumes: stage145 manifest and per-row `conversion_summary.json`, later training/benchmark JSON files when available.
- Produces: `summarize_stage145(manifest: Mapping[str, Any], output_root: Path) -> dict[str, Any]`
- Produces files: `stage145_results_summary.json`, `stage145_results_summary.md`.

- [ ] **Step 1: Write failing summary test**

Append this test to `test/test_rtece_scalar.py`:

```python
def test_stage145_summary_keeps_rmse_first_and_missing_outputs_explicit(tmp_path):
    from benchmarks.oc20neb_tace_mace.summarize_community_baselines_stage145 import (
        render_stage145_markdown,
        summarize_stage145,
    )

    manifest = {
        "schema_version": "community_baselines_stage145.v1",
        "rows": [
            {"name": "nep4_mixed_smoke", "engine": "nep", "train_dir": str(tmp_path / "nep")},
            {"name": "deepmd_dpa_like_mixed_smoke", "engine": "deepmd", "train_dir": str(tmp_path / "dp")},
        ],
    }
    (tmp_path / "nep").mkdir()
    (tmp_path / "nep" / "conversion_summary.json").write_text(
        json.dumps({"engine": "nep", "num_configs": 2, "type_map": ["C", "N"]})
    )

    summary = summarize_stage145(manifest, tmp_path)
    assert summary["schema_version"] == "community_baselines_stage145_results.v1"
    rows = {row["name"]: row for row in summary["rows"]}
    assert rows["nep4_mixed_smoke"]["conversion_status"] == "found"
    assert rows["deepmd_dpa_like_mixed_smoke"]["conversion_status"] == "missing"
    assert rows["deepmd_dpa_like_mixed_smoke"]["dft_benchmark_status"] == "missing"

    markdown = render_stage145_markdown(summary)
    assert "Primary ranking metric: DFT force RMSE" in markdown
    assert "| row | engine | conversion | DFT F RMSE | DFT E RMSE | atoms/s | physical |" in markdown
    assert "deepmd_dpa_like_mixed_smoke" in markdown
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py::test_stage145_summary_keeps_rmse_first_and_missing_outputs_explicit -q
```

Expected: FAIL with `ModuleNotFoundError` for `summarize_community_baselines_stage145`.

- [ ] **Step 3: Implement summary script**

Create `benchmarks/oc20neb_tace_mace/summarize_community_baselines_stage145.py` with:

```python
#!/usr/bin/env python3
"""Summarize Stage145 community baseline artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping


def _read_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def _metric(payload: dict[str, Any] | None, key: str) -> float | None:
    if not payload:
        return None
    value = payload.get(key)
    return float(value) if value is not None else None


def summarize_stage145(manifest: Mapping[str, Any], output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    rows = []
    for row in manifest.get("rows", []):
        train_dir = Path(row["train_dir"])
        conversion = _read_json(train_dir / "conversion_summary.json")
        dft = _read_json(train_dir / f"{row['name']}_dft_benchmark.json")
        physical = _read_json(train_dir / f"{row['name']}_physical_pareto.json")
        rows.append(
            {
                "name": row["name"],
                "engine": row["engine"],
                "conversion_status": "found" if conversion else "missing",
                "dft_benchmark_status": "found" if dft else "missing",
                "physical_status": "found" if physical else "missing",
                "num_configs": conversion.get("num_configs") if conversion else None,
                "type_map": conversion.get("type_map") if conversion else None,
                "dft_f_rmse_mev_a": _metric(dft, "rmse_f_mev_a"),
                "dft_e_rmse_mev_atom": _metric(dft, "rmse_e_mev_atom"),
                "dft_f_max_mev_a": _metric(dft, "max_abs_f_mev_a"),
                "dft_e_max_mev_atom": _metric(dft, "max_abs_e_mev_atom"),
                "atoms_per_second": _metric(dft, "atoms_per_second") or _metric(dft, "atoms_per_s"),
            }
        )
    summary = {
        "schema_version": "community_baselines_stage145_results.v1",
        "stage": "community_baselines_stage145",
        "primary_ranking_metric": "dft_f_rmse_mev_a",
        "rows": rows,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "stage145_results_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "stage145_results_summary.md").write_text(render_stage145_markdown(summary), encoding="utf-8")
    return summary


def _fmt(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def render_stage145_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Stage145 Community Baselines Summary",
        "",
        "Primary ranking metric: DFT force RMSE.",
        "",
        "| row | engine | conversion | DFT F RMSE | DFT E RMSE | atoms/s | physical |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in summary.get("rows", []):
        lines.append(
            "| {name} | {engine} | {conversion_status} | {f_rmse} | {e_rmse} | {atoms_s} | {physical_status} |".format(
                name=row["name"],
                engine=row["engine"],
                conversion_status=row["conversion_status"],
                f_rmse=_fmt(row.get("dft_f_rmse_mev_a")),
                e_rmse=_fmt(row.get("dft_e_rmse_mev_atom")),
                atoms_s=_fmt(row.get("atoms_per_second")),
                physical_status=row["physical_status"],
            )
        )
    lines.append("")
    lines.append("Missing values are explicit because conversion, training, benchmark, and physical triage complete at different times.")
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("runs/oc20neb_tace_mace/community-baselines-stage145/stage145_manifest.json"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/oc20neb_tace_mace/community-baselines-stage145"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text())
    print(json.dumps(summarize_stage145(manifest, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run summary test**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python -m pytest test/test_rtece_scalar.py::test_stage145_summary_keeps_rmse_first_and_missing_outputs_explicit -q
```

Expected: PASS.

- [ ] **Step 5: Generate current summary**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python benchmarks/oc20neb_tace_mace/summarize_community_baselines_stage145.py
```

Expected: writes summary JSON/MD with conversion statuses and missing training benchmark fields explicit.

- [ ] **Step 6: Commit**

Run:

```bash
git add benchmarks/oc20neb_tace_mace/summarize_community_baselines_stage145.py test/test_rtece_scalar.py runs/oc20neb_tace_mace/community-baselines-stage145/stage145_results_summary.json runs/oc20neb_tace_mace/community-baselines-stage145/stage145_results_summary.md
git commit -m "Summarize rTECE stage145 community baselines"
```

Expected: commit containing summary script, test, and current summary only.

---

### Task 5: Smoke Jobs And Stage145 Submission Record

**Files:**
- Modify: `runs/oc20neb_tace_mace/community-baselines-stage145/stage145_results_summary.md`
- Create: `runs/oc20neb_tace_mace/community-baselines-stage145/stage145_submission_log.md`

**Interfaces:**
- Consumes wrappers from Task 1 and converters from Tasks 2-3.
- Produces job IDs and first real NEP/DeepMD smoke statuses.

- [ ] **Step 1: Submit NEP smoke**

Run:

```bash
sbatch runs/oc20neb_tace_mace/community-baselines-stage145/wrappers/nep4_mixed_smoke_no_export.sbatch
```

Expected: prints `Submitted batch job <jobid>`.

- [ ] **Step 2: Submit DeepMD smoke**

Run:

```bash
sbatch runs/oc20neb_tace_mace/community-baselines-stage145/wrappers/deepmd_dpa_like_mixed_smoke_no_export.sbatch
```

Expected: prints `Submitted batch job <jobid>`.

- [ ] **Step 3: Monitor jobs**

Run:

```bash
sacct -j <nep_jobid>,<deepmd_jobid> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList%30,MaxRSS
```

Expected: either `COMPLETED 0:0` or a concrete failure recorded in logs. Do not infer algorithmic failure from launcher/environment failures.

- [ ] **Step 4: Record submission evidence**

Create `runs/oc20neb_tace_mace/community-baselines-stage145/stage145_submission_log.md` with:

```markdown
# Stage145 Submission Log

- NEP job: `<nep_jobid>`
- DeepMD job: `<deepmd_jobid>`
- sacct command: `sacct -j <nep_jobid>,<deepmd_jobid> --format=JobID,JobName,State,ExitCode,Elapsed,NodeList%30,MaxRSS`
- NEP state: `<state>`
- DeepMD state: `<state>`
- Interpretation: `<completed | launcher failure | converter failure | training failure>`
```

- [ ] **Step 5: Regenerate summary**

Run:

```bash
/home/gengjianrui/bin/.venvs/tace-mace-cu126/bin/python benchmarks/oc20neb_tace_mace/summarize_community_baselines_stage145.py
```

Expected: summary reflects conversion/training artifacts that exist after smoke jobs.

- [ ] **Step 6: Commit and push**

Run:

```bash
git add runs/oc20neb_tace_mace/community-baselines-stage145/stage145_submission_log.md runs/oc20neb_tace_mace/community-baselines-stage145/stage145_results_summary.json runs/oc20neb_tace_mace/community-baselines-stage145/stage145_results_summary.md
git commit -m "Record rTECE stage145 community baseline smoke"
git -c core.sshCommand="ssh -i ~/.ssh/id_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=20 -o ProxyCommand='nc -X connect -x localhost:9998 %h %p' -p 443" push phorbol tece-renorm-distill
```

Expected: push succeeds. Do not add converted datasets, checkpoints, `.npy` arrays, or raw GPUMD/DeepMD training outputs.

---

## Self-Review

Spec coverage:

- Approach A data contract is covered by Task 1 manifest and Tasks 2-3 converters.
- NEP row is covered by Tasks 1-2 and Task 5 smoke.
- DeepMD/DPA-like row is covered by Tasks 1 and 3, with explicit fixed-order preflight.
- RMSE-first benchmark summary is covered by Task 4.
- SAI no-export/no-mem/no-cpus constraints are covered by Task 1 tests and wrapper audit.
- Smoke submission and evidence recording are covered by Task 5.

Placeholder scan:

- No `TBD`, `TODO`, or `implement later` placeholders are intentionally used.
- The DeepMD exact descriptor uncertainty is represented as an explicit `dpa_like_low_attention` contract, not an unstated assumption.

Type consistency:

- Manifest functions use `dict[str, Any]`.
- Converter entry points return `dict[str, Any]`.
- Summary consumes manifest row keys produced by Task 1.
