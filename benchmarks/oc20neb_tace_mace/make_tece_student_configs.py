#!/usr/bin/env python3
"""Generate reduced TACE student configs for the TECE distillation matrix."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


VARIANTS = (
    "scalar_fast",
    "edge_min",
    "edge_min_mixed",
    "edge_radial_mixed",
    "edge_radial_active6_mixed",
    "edge_radial_active4_mixed",
    "edge_readout_mixed",
    "edge_radial_slim_mixed",
    "path_scalar",
)


def _set_model_common(model_cfg: dict[str, Any], *, num_channel: int, radial_hidden: list[int]) -> None:
    model_cfg["num_channel"] = num_channel
    radial_basis = model_cfg.setdefault("radial_basis", {})
    radial_basis["hidden"] = radial_hidden
    radial_basis.pop("active_indices", None)
    model_cfg.setdefault("readout_emlp", {})["hidden"] = [8]
    model_cfg.setdefault("atomic_basis", {})["type"] = "cgtp"
    model_cfg.setdefault("product_basis", {})["type"] = "cgtp"


def build_variant_config(
    base_cfg: dict[str, Any],
    variant_name: str,
    *,
    train_file: str,
    valid_file: str,
) -> dict[str, Any]:
    """Return a reduced student config selected along TECE projection axes."""

    if variant_name not in VARIANTS:
        raise ValueError(f"unknown variant {variant_name!r}; expected one of {', '.join(VARIANTS)}")

    cfg = copy.deepcopy(base_cfg)
    cfg.setdefault("misc", {})["project_name"] = f"oc20neb_fullcase200_{variant_name}"
    cfg.setdefault("dataset", {})["train_file"] = train_file
    cfg.setdefault("dataset", {})["valid_file"] = valid_file
    cfg["dataset"].setdefault("train_dataloader", {})["batch_size"] = 12
    cfg["dataset"].setdefault("valid_dataloader", {})["batch_size"] = 24
    cfg.setdefault("trainer", {})["max_steps"] = 20000
    cfg.setdefault("trainer", {})["max_epochs"] = 64
    cfg["trainer"]["check_val_every_n_epoch"] = 8
    cfg["trainer"]["log_every_n_steps"] = 200
    cfg.setdefault("scheduler", {}).setdefault("extra", {})["frequency"] = 8
    checkpoint_epoch = cfg.setdefault("callbacks", {}).setdefault("checkpoint_epoch", {})
    checkpoint_epoch["every_n_epochs"] = 8
    checkpoint_step = cfg["callbacks"].setdefault("checkpoint_step", {})
    checkpoint_step["every_n_train_steps"] = 5000

    model_cfg = cfg["model"]["config"]
    atomic_basis = model_cfg.setdefault("atomic_basis", {})
    product_basis = model_cfg.setdefault("product_basis", {})

    if variant_name == "scalar_fast":
        model_cfg.update({"Lmax": 0, "lmax": 1, "mmax": 0, "num_layers": 2})
        model_cfg.setdefault("edge_embedding", {})["type"] = "identity"
        model_cfg.setdefault("edge_update", {})["type"] = "identity"
        atomic_basis.update({"nonlinear": None, "edge_nonlinear": None, "edge_info_type": "mlp"})
        product_basis.update({"correlation": 2, "return_components": None})
        _set_model_common(model_cfg, num_channel=24, radial_hidden=[32, 32])
    elif variant_name in {
        "edge_min",
        "edge_min_mixed",
        "edge_radial_mixed",
        "edge_radial_active6_mixed",
        "edge_radial_active4_mixed",
        "edge_readout_mixed",
        "edge_radial_slim_mixed",
    }:
        model_cfg.update({"Lmax": 1, "lmax": 2, "mmax": 1, "num_layers": 2})
        model_cfg.setdefault("edge_embedding", {})["type"] = "nonlinear"
        model_cfg.setdefault("edge_update", {})["type"] = "element2"
        atomic_basis.update({"nonlinear": "sigmoid_gate", "edge_nonlinear": None, "edge_info_type": "mlp"})
        product_basis.update({"correlation": 2, "return_components": None})
        if variant_name == "edge_radial_mixed":
            _set_model_common(model_cfg, num_channel=32, radial_hidden=[64, 64])
            model_cfg.setdefault("readout_emlp", {})["hidden"] = [16]
        elif variant_name == "edge_radial_active6_mixed":
            _set_model_common(model_cfg, num_channel=32, radial_hidden=[64, 64])
            model_cfg.setdefault("radial_basis", {})["active_indices"] = [0, 1, 2, 3, 4, 5]
            model_cfg.setdefault("readout_emlp", {})["hidden"] = [16]
        elif variant_name == "edge_radial_active4_mixed":
            _set_model_common(model_cfg, num_channel=32, radial_hidden=[64, 64])
            model_cfg.setdefault("radial_basis", {})["active_indices"] = [0, 1, 2, 3]
            model_cfg.setdefault("readout_emlp", {})["hidden"] = [16]
        elif variant_name == "edge_readout_mixed":
            _set_model_common(model_cfg, num_channel=32, radial_hidden=[48, 48])
            model_cfg.setdefault("readout_emlp", {})["hidden"] = [16]
        elif variant_name == "edge_radial_slim_mixed":
            _set_model_common(model_cfg, num_channel=32, radial_hidden=[64, 32])
            model_cfg.setdefault("readout_emlp", {})["hidden"] = [16]
        else:
            _set_model_common(model_cfg, num_channel=32, radial_hidden=[48, 48])
    elif variant_name == "path_scalar":
        model_cfg.update({"Lmax": 1, "lmax": 2, "mmax": 1, "num_layers": 2})
        model_cfg.setdefault("edge_embedding", {})["type"] = "identity"
        model_cfg.setdefault("edge_update", {})["type"] = "identity"
        atomic_basis.update({"nonlinear": "sigmoid_gate", "edge_nonlinear": None, "edge_info_type": "mlp"})
        product_basis.update({"correlation": 3, "return_components": None})
        _set_model_common(model_cfg, num_channel=24, radial_hidden=[48, 48])

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--valid-file", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for variant_name in args.variants:
        cfg = build_variant_config(
            base_cfg,
            variant_name,
            train_file=args.train_file,
            valid_file=args.valid_file,
        )
        out = args.output_dir / f"tace_oc20neb_{variant_name}.yaml"
        out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        print(out)


if __name__ == "__main__":
    main()
