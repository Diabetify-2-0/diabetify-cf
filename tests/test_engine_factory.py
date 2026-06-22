import pytest

from diabetify_cf.config import Settings
from diabetify_cf.engine import (
    NearestNeighborCounterfactualEngine,
    build_counterfactual_engine,
)


def test_build_counterfactual_engine_returns_nn_by_default() -> None:
    engine = build_counterfactual_engine(
        Settings(
            model_path="missing-model.pkl",
            columns_path="missing-columns.pkl",
        )
    )

    assert isinstance(engine, NearestNeighborCounterfactualEngine)


def test_build_counterfactual_engine_passes_nn_settings() -> None:
    engine = build_counterfactual_engine(
        Settings(
            model_path="missing-model.pkl",
            columns_path="missing-columns.pkl",
            nn_candidate_pool_size=32,
            nn_max_neighbors=12,
            nn_min_reference_low_risk_probability=0.7,
        )
    )

    assert isinstance(engine, NearestNeighborCounterfactualEngine)
    assert engine.options.candidate_pool_size == 32
    assert engine.options.max_neighbors == 12
    assert engine.options.min_reference_low_risk_probability == 0.7


def test_build_counterfactual_engine_rejects_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported counterfactual engine provider"):
        build_counterfactual_engine(
            Settings(
                engine_provider="carla",
                model_path="missing-model.pkl",
                columns_path="missing-columns.pkl",
            )
        )
