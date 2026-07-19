from __future__ import annotations

__all__ = ["TACEAseCalc", "RTECEAseCalc", "atoms_to_rtece_graph", "add_dispersion"]


def __getattr__(name):
    if name in {"TACEAseCalc", "add_dispersion"}:
        from .calculator import TACEAseCalc, add_dispersion

        return {"TACEAseCalc": TACEAseCalc, "add_dispersion": add_dispersion}[name]
    if name in {"RTECEAseCalc", "atoms_to_rtece_graph"}:
        from .rtece_calculator import RTECEAseCalc, atoms_to_rtece_graph

        return {"RTECEAseCalc": RTECEAseCalc, "atoms_to_rtece_graph": atoms_to_rtece_graph}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
