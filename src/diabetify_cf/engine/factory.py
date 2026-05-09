from __future__ import annotations

from diabetify_cf.config import Settings
from diabetify_cf.engine.base import CounterfactualEngine
from diabetify_cf.engine.dice_engine import DiceCounterfactualEngine
from diabetify_cf.engine.nn_engine import (
    NearestNeighborCounterfactualEngine,
    NearestNeighborOptions,
)
from diabetify_cf.planner.base import PrescriptivePlanner


def build_counterfactual_engine(
    settings: Settings,
    *,
    planner: PrescriptivePlanner | None = None,
) -> CounterfactualEngine:
    provider = settings.engine_provider.strip().lower()
    common_kwargs = {
        "model_path": settings.model_path,
        "columns_path": settings.columns_path,
        "reference_data_path": settings.reference_data_path,
        "feature_registry_path": settings.feature_registry_path,
        "max_lof_score": settings.max_lof_score,
        "planner": planner,
    }

    if provider == "dice":
        return DiceCounterfactualEngine(**common_kwargs)
    if provider == "nn":
        return NearestNeighborCounterfactualEngine(
            **common_kwargs,
            options=NearestNeighborOptions.from_settings(settings),
        )

    raise ValueError(f"Unsupported counterfactual engine provider '{settings.engine_provider}'.")
