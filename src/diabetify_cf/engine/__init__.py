from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CounterfactualEngine",
    "NearestNeighborCounterfactualEngine",
]


def __getattr__(name: str) -> Any:
    if name == "CounterfactualEngine":
        return getattr(import_module("diabetify_cf.engine.base"), name)
    if name == "NearestNeighborCounterfactualEngine":
        return getattr(import_module("diabetify_cf.engine.nn_engine"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
