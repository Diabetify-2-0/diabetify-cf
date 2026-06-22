from types import SimpleNamespace

import pandas as pd

from diabetify_cf.engine.feature_registry import FeatureDefinition
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
        permitted_range={"BMI": [18.5, 35.0], "smoking_status": [0.0, 2.0]},
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
        permitted_range={"BMI": [18.5, 35.0], "smoking_status": [0.0, 2.0]},
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


def test_nn_project_neighbors_uses_mutable_allowed_count_as_dynamic_sparse_limit() -> None:
    engine = NearestNeighborCounterfactualEngine.__new__(NearestNeighborCounterfactualEngine)
    engine.options = NearestNeighborOptions()
    engine.artifacts = SimpleNamespace(feature_columns=["age", "BMI", "smoking_status", "is_hypertension"])
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
                "is_hypertension": _feature(
                    "is_hypertension",
                    feature_type="binary",
                    global_min=0.0,
                    global_max=1.0,
                ),
            }.get(name)
        ),
        permitted_range={
            "BMI": [18.5, 35.0],
            "smoking_status": [0.0, 2.0],
            "is_hypertension": [0.0, 1.0],
        },
    )
    baseline = {"age": 45, "BMI": 31.0, "smoking_status": 2, "is_hypertension": 1}
    neighbor = pd.Series({"age": 60, "BMI": 27.0, "smoking_status": 1, "is_hypertension": 0})

    projected = engine._project_neighbors(
        ranked_neighbors=[neighbor],
        baseline=baseline,
        mutable_allowed=["BMI", "smoking_status"],
        prepared=prepared,
    )

    changed_pairs = {
        (
            candidate["BMI"] != baseline["BMI"],
            candidate["smoking_status"] != baseline["smoking_status"],
            candidate["is_hypertension"] != baseline["is_hypertension"],
        )
        for candidate in projected
    }

    assert (True, False, False) in changed_pairs
    assert (False, True, False) in changed_pairs
    assert (True, True, False) in changed_pairs
    assert all(pair[2] is False for pair in changed_pairs)
