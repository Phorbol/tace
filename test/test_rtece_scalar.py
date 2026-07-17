from __future__ import annotations

from benchmarks.oc20neb_tace_mace.rtece_scalar_model import (
    build_rtece_config,
    descriptor_dim,
)


def test_build_rtece_config_defines_ordered_variants():
    pair = build_rtece_config("rtece_pair")
    atomic = build_rtece_config("rtece_atomic_moments")
    sketch8 = build_rtece_config("rtece_edge_sketch8")
    sketch16 = build_rtece_config("rtece_edge_sketch16")

    assert pair.variant == "rtece_pair"
    assert pair.use_atomic_moments is False
    assert pair.num_edge_sketches == 0
    assert atomic.use_atomic_moments is True
    assert atomic.num_edge_sketches == 0
    assert sketch8.use_atomic_moments is True
    assert sketch8.num_edge_sketches == 8
    assert sketch16.use_atomic_moments is True
    assert sketch16.num_edge_sketches == 16
    assert descriptor_dim(pair) < descriptor_dim(atomic) < descriptor_dim(sketch8) < descriptor_dim(sketch16)
