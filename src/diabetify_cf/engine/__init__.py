from typing import TYPE_CHECKING

from diabetify_cf.engine.base import CounterfactualEngine
from diabetify_cf.engine.factory import build_counterfactual_engine
from diabetify_cf.engine.nn_engine import NearestNeighborCounterfactualEngine

if TYPE_CHECKING:
    from diabetify_cf.engine.dice_engine import DiceCounterfactualEngine

__all__ = [
    "CounterfactualEngine",
    "DiceCounterfactualEngine",
    "NearestNeighborCounterfactualEngine",
    "build_counterfactual_engine",
]


def __getattr__(name: str) -> object:
    if name == "DiceCounterfactualEngine":
        from diabetify_cf.engine.dice_engine import DiceCounterfactualEngine

        return DiceCounterfactualEngine
    raise AttributeError(f"module 'diabetify_cf.engine' has no attribute {name!r}")
