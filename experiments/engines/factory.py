from __future__ import annotations

from diabetify_cf.config import Settings
from experiments.engines.base import ExperimentEngine
from experiments.engines.dice_adapter import DiceExperimentAdapter

SUPPORTED_ENGINES = {"dice"}


def build_experiment_engine(engine_name: str, settings: Settings) -> ExperimentEngine:
    normalized = engine_name.strip().lower()
    if normalized == "dice":
        return DiceExperimentAdapter(settings=settings)

    supported = ", ".join(sorted(SUPPORTED_ENGINES))
    raise ValueError(
        f"Unsupported experiment engine '{engine_name}'. Supported engines: {supported}."
    )
