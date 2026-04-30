from __future__ import annotations

from diabetify_cf.config import Settings
from experiments.engines.base import ExperimentEngine
from experiments.engines.dice_adapter import DiceExperimentAdapter

ENGINE_ADAPTERS = {
    "dice": DiceExperimentAdapter,
}
SUPPORTED_ENGINES = set(ENGINE_ADAPTERS)


def build_experiment_engine(engine_name: str, settings: Settings) -> ExperimentEngine:
    normalized = engine_name.strip().lower()
    adapter_cls = ENGINE_ADAPTERS.get(normalized)
    if adapter_cls is not None:
        return adapter_cls(settings=settings)

    supported = ", ".join(sorted(SUPPORTED_ENGINES))
    raise ValueError(
        f"Unsupported experiment engine '{engine_name}'. Supported engines: {supported}."
    )
