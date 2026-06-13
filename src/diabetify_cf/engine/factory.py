from __future__ import annotations

from diabetify_cf.config import Settings
from diabetify_cf.engine.base import CounterfactualEngine
from diabetify_cf.engine.nn_engine import (
    NearestNeighborCounterfactualEngine,
    NearestNeighborOptions,
)


def build_counterfactual_engine(
    settings: Settings,
) -> CounterfactualEngine:
    provider = settings.engine_provider.strip().lower()
    if provider == "nn":
        return NearestNeighborCounterfactualEngine(
            model_path=settings.model_path,
            columns_path=settings.columns_path,
            reference_data_path=settings.reference_data_path,
            feature_registry_path=settings.feature_registry_path,
            artifact_manifest_path=settings.artifact_manifest_path,
            max_lof_score=settings.max_lof_score,
            options=NearestNeighborOptions.from_settings(settings),
        )

    raise ValueError(f"Unsupported counterfactual engine provider '{settings.engine_provider}'.")
