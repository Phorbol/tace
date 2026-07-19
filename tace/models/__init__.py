from importlib import import_module

_OPTIONAL_MODEL_EXPORTS = {
    "cartTACE": ("._cart", "cartTACE"),
    "e3nnTACE": ("._e3nn", "e3nnTACE"),
    "TensorModel": (".adapter", "TensorModel"),
    "CompileTensorModel": (".compile", "CompileTensorModel"),
}


def __getattr__(name):
    if name not in _OPTIONAL_MODEL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, symbol_name = _OPTIONAL_MODEL_EXPORTS[name]
    try:
        module = import_module(module_name, __name__)
    except ImportError:
        value = None
    else:
        value = getattr(module, symbol_name)
    globals()[name] = value
    return value


from . import rtece_workflow as rtece_workflow

RTECEWorkflow = rtece_workflow

from .rtece_workflow import (
    evaluate_loss as evaluate_rtece_loss,
    load_checkpoint as load_rtece_checkpoint,
    loss_for_batch as rtece_loss_for_batch,
    predict as predict_rtece,
    save_checkpoint as save_rtece_checkpoint,
    train_steps as train_rtece_steps,
)
from .rtece_scalar import (
    RTECEGraph,
    RTECEScalarConfig,
    RTECEScalarModel,
    atomic_scalar_descriptors,
    available_rtece_variants,
    build_rtece_config,
    build_rtece_config_from_manifest,
    build_rtece_config_from_path_ids,
    cell_list_packed_element_density_descriptors,
    descriptor_dim,
    edge_relational_sketches,
    packed_element_density_descriptors,
    rtece_descriptors,
    rtece_path_manifest,
    rtece_route_contract,
    rtece_variant_registry,
)

__all__ = [
    "cartTACE",
    "e3nnTACE",
    "TensorModel",
    "CompileTensorModel",
    "RTECEGraph",
    "RTECEScalarConfig",
    "RTECEScalarModel",
    "RTECEWorkflow",
    "rtece_workflow",
    "atomic_scalar_descriptors",
    "available_rtece_variants",
    "build_rtece_config",
    "build_rtece_config_from_manifest",
    "build_rtece_config_from_path_ids",
    "cell_list_packed_element_density_descriptors",
    "descriptor_dim",
    "edge_relational_sketches",
    "packed_element_density_descriptors",
    "rtece_descriptors",
    "rtece_path_manifest",
    "rtece_route_contract",
    "rtece_variant_registry",
    "evaluate_rtece_loss",
    "load_rtece_checkpoint",
    "rtece_loss_for_batch",
    "predict_rtece",
    "save_rtece_checkpoint",
    "train_rtece_steps",
]
