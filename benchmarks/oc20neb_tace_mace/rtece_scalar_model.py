from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RTECEScalarConfig:
    variant: str
    cutoff: float = 5.0
    num_radial: int = 8
    hidden_channels: tuple[int, ...] = (64, 64)
    max_atomic_number: int = 100
    use_atomic_moments: bool = False
    num_edge_sketches: int = 0


def build_rtece_config(variant: str) -> RTECEScalarConfig:
    if variant == "rtece_pair":
        return RTECEScalarConfig(variant=variant)
    if variant == "rtece_atomic_moments":
        return RTECEScalarConfig(variant=variant, use_atomic_moments=True)
    if variant == "rtece_edge_sketch8":
        return RTECEScalarConfig(variant=variant, use_atomic_moments=True, num_edge_sketches=8)
    if variant == "rtece_edge_sketch16":
        return RTECEScalarConfig(variant=variant, use_atomic_moments=True, num_edge_sketches=16)
    raise ValueError(f"unknown rTECE scalar variant {variant!r}")


def descriptor_dim(config: RTECEScalarConfig) -> int:
    dim = config.num_radial
    if config.use_atomic_moments:
        dim += 2 * config.num_radial
    dim += config.num_edge_sketches
    return dim
