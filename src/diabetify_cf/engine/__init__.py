from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CounterfactualEngine",
    "NearestNeighborCounterfactualEngine",
    "build_counterfactual_engine",
]


def __getattr__(name: str) -> Any:
    if name == "CounterfactualEngine":
        return getattr(import_module("diabetify_cf.engine.base"), name)
    if name in {"NearestNeighborCounterfactualEngine", "build_counterfactual_engine"}:
        module_name = (
            "diabetify_cf.engine.nn_engine"
            if name == "NearestNeighborCounterfactualEngine"
            else "diabetify_cf.engine.factory"
        )
        return getattr(import_module(module_name), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
