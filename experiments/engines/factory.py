from __future__ import annotations

from typing import Any

from diabetify_cf.config import Settings
from experiments.engines.base import ExperimentEngine
from experiments.engines.dice_adapter import DiceExperimentAdapter
from experiments.engines.nn_adapter import NearestNeighborExperimentAdapter
from experiments.engines.ocean_adapter import OceanExperimentAdapter

ENGINE_ADAPTERS = {
    "dice": DiceExperimentAdapter,
    "nn": NearestNeighborExperimentAdapter,
    "ocean": OceanExperimentAdapter,
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
