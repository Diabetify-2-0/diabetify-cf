from types import SimpleNamespace

import pandas as pd

from diabetify_cf.engine.feature_registry import FeatureDefinition, FeatureRegistry
from diabetify_cf.engine.nn_engine import (
    NearestNeighborCounterfactualEngine,
    NearestNeighborOptions,
)


def _feature(
    name: str,
    *,
    feature_type: str = "continuous",
    global_min: float = 0.0,
    global_max: float = 50.0,
    cost_weight: float = 1.0,
) -> FeatureDefinition:
    return FeatureDefinition(
        name=name,
        feature_type=feature_type,
        immutable=False,
        actionable=True,
        default_mutable=True,
        global_min=global_min,
        global_max=global_max,
        cost_weight=cost_weight,
        preferred_direction="any",
        aliases=[],
    )


def _registry(features: list[FeatureDefinition]) -> FeatureRegistry:
    return FeatureRegistry(version="test_v1", features=features)


def test_nn_rank_changed_features_uses_weighted_normalized_distance() -> None:
    engine = NearestNeighborCounterfactualEngine.__new__(NearestNeighborCounterfactualEngine)
    prepared = SimpleNamespace(
        registry=SimpleNamespace(
            get=lambda name: {
                "BMI": _feature("BMI", global_min=10.0, global_max=60.0, cost_weight=1.0),
                "smoking_status": _feature(
                    "smoking_status",
                    feature_type="ordinal",
                    global_min=0.0,
                    global_max=2.0,
                    cost_weight=1.0,
                ),
            }.get(name)
        ),
    )
    neighbor = pd.Series({"BMI": 29.0, "smoking_status": 1.0})
    baseline = {"BMI": 31.0, "smoking_status": 0.0}

    ranked = engine._rank_changed_features(
        neighbor=neighbor,
        baseline=baseline,
        mutable_allowed=["BMI", "smoking_status"],
        prepared=prepared,
    )

    assert ranked == ["BMI", "smoking_status"]


def test_nn_rank_neighbors_prefers_heom_proximity_before_probability() -> None:
    engine = NearestNeighborCounterfactualEngine.__new__(NearestNeighborCounterfactualEngine)
    registry = _registry(
        [
            _feature("age", global_min=0.0, global_max=100.0),
            _feature("BMI", global_min=10.0, global_max=60.0),
            _feature("is_hypertension", feature_type="binary"),
        ]
    )
    prepared = SimpleNamespace(
        registry=registry,
        model_columns=["age", "BMI", "is_hypertension"],
    )
    baseline = {"age": 50, "BMI": 30.0, "is_hypertension": 0}
    eligible = pd.DataFrame(
        [
            {"age": 80, "BMI": 20.0, "is_hypertension": 1},
            {"age": 51, "BMI": 29.0, "is_hypertension": 0},
        ]
    )
    ranked = engine._rank_neighbors(
        eligible=eligible,
        eligible_probabilities=pd.Series([0.95, 0.55]).to_numpy(),
        baseline=baseline,
        mutable_allowed=["BMI"],
        prepared=prepared,
    )

    assert ranked[0]["BMI"] == 29.0


def test_nn_rank_neighbors_measures_heom_only_on_mutable_allowed_features() -> None:
    engine = NearestNeighborCounterfactualEngine.__new__(NearestNeighborCounterfactualEngine)
    registry = _registry(
        [
            _feature("age", global_min=0.0, global_max=100.0),
            _feature("BMI", global_min=10.0, global_max=60.0),
        ]
    )
    prepared = SimpleNamespace(registry=registry, model_columns=["age", "BMI"])
    baseline = {"age": 50.0, "BMI": 30.0}
    eligible = pd.DataFrame(
        [
            {"age": 100.0, "BMI": 29.0},
            {"age": 50.0, "BMI": 20.0},
        ]
    )

    ranked = engine._rank_neighbors(
        eligible=eligible,
        eligible_probabilities=pd.Series([0.55, 0.95]).to_numpy(),
        baseline=baseline,
        mutable_allowed=["BMI"],
        prepared=prepared,
    )

    assert ranked[0]["BMI"] == 29.0

def test_nn_rank_neighbors_uses_action_cost_as_first_tie_breaker() -> None:
    engine = NearestNeighborCounterfactualEngine.__new__(NearestNeighborCounterfactualEngine)
    registry = _registry(
        [
            _feature("low_cost", global_min=0.0, global_max=10.0, cost_weight=1.0),
            _feature("high_cost", global_min=0.0, global_max=10.0, cost_weight=5.0),
        ]
    )
    prepared = SimpleNamespace(registry=registry, model_columns=["low_cost", "high_cost"])
    baseline = {"low_cost": 0.0, "high_cost": 0.0}
    eligible = pd.DataFrame(
        [
            {"low_cost": 0.0, "high_cost": 1.0},
            {"low_cost": 1.0, "high_cost": 0.0},
        ]
    )
    ranked = engine._rank_neighbors(
        eligible=eligible,
        eligible_probabilities=pd.Series([0.9, 0.9]).to_numpy(),
        baseline=baseline,
        mutable_allowed=["low_cost", "high_cost"],
        prepared=prepared,
    )

    assert ranked[0]["low_cost"] == 1.0


def test_nn_rank_neighbors_uses_probability_as_final_tie_breaker() -> None:
    engine = NearestNeighborCounterfactualEngine.__new__(NearestNeighborCounterfactualEngine)
    registry = _registry(
        [
            _feature("first", global_min=0.0, global_max=10.0),
            _feature("second", global_min=0.0, global_max=10.0),
        ]
    )
    prepared = SimpleNamespace(registry=registry, model_columns=["first", "second"])
    baseline = {"first": 0.0, "second": 0.0}
    eligible = pd.DataFrame(
        [
            {"first": 1.0, "second": 0.0},
            {"first": 0.0, "second": 1.0},
        ]
    )

    ranked = engine._rank_neighbors(
        eligible=eligible,
        eligible_probabilities=pd.Series([0.95, 0.55]).to_numpy(),
        baseline=baseline,
        mutable_allowed=["first", "second"],
        prepared=prepared,
    )

    assert ranked[0]["first"] == 1.0


def test_nn_project_neighbors_keeps_immutable_features_from_baseline() -> None:
    engine = NearestNeighborCounterfactualEngine.__new__(NearestNeighborCounterfactualEngine)
    engine.options = NearestNeighborOptions()
    engine.artifacts = SimpleNamespace(feature_columns=["age", "BMI", "smoking_status"])
    prepared = SimpleNamespace(
        registry=SimpleNamespace(
            get=lambda name: {
                "BMI": _feature("BMI", global_min=10.0, global_max=60.0),
                "smoking_status": _feature(
                    "smoking_status",
                    feature_type="ordinal",
                    global_min=0.0,
                    global_max=2.0,
                ),
            }.get(name)
        ),
    )
    baseline = {"age": 45, "BMI": 31.0, "smoking_status": 0}
    neighbor = pd.Series({"age": 60, "BMI": 27.0, "smoking_status": 1})

    projected = engine._project_neighbors(
        ranked_neighbors=[neighbor],
        baseline=baseline,
        mutable_allowed=["BMI", "smoking_status"],
        prepared=prepared,
    )

    assert projected
    assert all(candidate["age"] == 45 for candidate in projected)
    assert any(candidate["BMI"] == 27.0 for candidate in projected)

