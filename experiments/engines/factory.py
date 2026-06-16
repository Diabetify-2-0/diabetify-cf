from __future__ import annotations

from typing import Any

from diabetify_cf.config import Settings
from experiments.engines.base import ExperimentEngine
from experiments.engines.dice_adapter import (
    DiceConstrainedGatedExperimentAdapter,
    DiceExperimentAdapter,
)
from experiments.engines.nn_adapter import NearestNeighborExperimentAdapter

ENGINE_ADAPTERS = {
    "dice": DiceExperimentAdapter,
    "dice_constrained_native": DiceExperimentAdapter,
    "dice_constrained_gated": DiceConstrainedGatedExperimentAdapter,
    "nn": NearestNeighborExperimentAdapter,
    "nn_production": NearestNeighborExperimentAdapter,
}
SUPPORTED_ENGINES = set(ENGINE_ADAPTERS)


def build_experiment_engine(
    engine_name: str,
    settings: Settings,
    config: dict[str, Any] | None = None,
) -> ExperimentEngine:
    normalized = engine_name.strip().lower()
    adapter_cls = ENGINE_ADAPTERS.get(normalized)
    if adapter_cls is not None:
        return adapter_cls(settings=settings, config=config)

    supported = ", ".join(sorted(SUPPORTED_ENGINES))
    raise ValueError(
        f"Unsupported experiment engine '{engine_name}'. Supported engines: {supported}."
    )
