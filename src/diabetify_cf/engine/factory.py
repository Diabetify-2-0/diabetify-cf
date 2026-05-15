from __future__ import annotations

from importlib.util import find_spec

from diabetify_cf.config import Settings
from diabetify_cf.engine.base import CounterfactualEngine
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
        if find_spec("dice_ml") is None:
            raise ValueError(
                "CF_ENGINE_PROVIDER=dice requires the optional 'dice' dependency. "
                "Install it with `pip install -e \".[dice]\"` or use the "
                "`.[experiments]` extra for the research stack."
            )
        from diabetify_cf.engine.dice_engine import DiceCounterfactualEngine

        return DiceCounterfactualEngine(**common_kwargs)
    if provider == "nn":
        return NearestNeighborCounterfactualEngine(
            **common_kwargs,
            options=NearestNeighborOptions.from_settings(settings),
        )

    raise ValueError(f"Unsupported counterfactual engine provider '{settings.engine_provider}'.")
